"""Directed-peel driver: peel-to-flat, then excise named parts.

Generalisation of the spike's arm_test.py monkeypatch. Two phases:

  1. FLATTEN — the standard VLM-guided stacking loop peels decorative detail
     (outlines, eyes, patterns). On detailed images the stacking prior overrides
     directed instructions, so this MUST run before directing. Stops when a
     VLM flat-check says only the base body silhouette remains (mode "auto"),
     or after a fixed number of steps (mode "steps"), or immediately ("none").

  2. DIRECTED — for each part in the plan (in order), the top-layer VLM step is
     bypassed and the caption is forced ("remove the arm on the left of the
     image"). Part names must be viewer-relative, never anatomical.

Every intermediate frame is kept (layer_png/layer_N.png); the local
diff -> vtracer -> part-registry stage consumes consecutive frames. A peel log
(peel_log.json) records phase, caption, timing and the frame indices each part
spans, plus an economics summary (GPU seconds, VLM calls, retries).

Plan schema (JSON, Fable-authored):
{
  "target": "fox",
  "flatten": {"mode": "auto", "max_steps": 6},     # or {"mode": "steps", "steps": 4} / {"mode": "none"}
  "parts": [
    {"name": "arm_right_raised", "caption": "the raised arm on the right side of the image"},
    ...
  ]
}

Usage (on the GPU box):
  python directed_peel.py --image data/fox512.png --plan plans/fox.json \
      --pretrained_model_name_or_path ../PhotoDoodle_Pretrain --output_folder outputs/fox_directed
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image
from transformers import set_seed

from inference import (
    Config,
    load_model,
    create_output_subfolders,
    warp_caption,
    _get_mask_response,
    _generate_next_image,
)
from utils.util import load_image, load_json, load_or_generate, save_image, save_json, setup_logging
from utils.vlm_util import detect_top_layer, extract_tag_content, DEFAULT_PROMPT
from utils.image_util import is_pure_white, pad_image


FLAT_CHECK_PROMPT = """You are inspecting an intermediate step of a layer-peeling process on a cartoon mascot image. Decorative detail layers (outlines, eyes, facial features, patterns, small decorations) are being removed one by one.

Answer whether the image is now a FLAT BASE: only large flat-colour or gradient regions forming the character's body silhouette remain (body, limbs, head, tail as plain colour shapes), with NO remaining outlines, facial features, patterns, or small decorative elements.

Small soft shading is acceptable in a flat base. If any eyes, outlines, or decorative details are still visible, it is NOT a flat base.

Respond with your reasoning in <think></think> tags, then exactly YES or NO in <answer></answer> tags."""


FLATTEN_CONSTRAINT = """

# CRITICAL CONSTRAINT — flatten phase:
This peeling run is a FLATTEN pass: only DECORATIVE DETAIL layers may be removed
(outlines, eyes, facial features, highlights, shadows, patches, patterns,
decorations). You must NEVER include a body part of the character — arm, hand,
leg, foot, head, ear, tail, torso, body — as an element to remove, even if it is
non-occluded. If the only non-occluded elements left are body parts, return an
empty caption: <caption></caption>."""

BODY_PART_RE = re.compile(
    r"\b(arms?|hands?|legs?|feet|foot|heads?|tails?|torso|body)\b", re.IGNORECASE
)


def get_flatten_caption(image_vlm: Image.Image, step: int, cfg: Config, output_folders: Dict[str, str], logger) -> Optional[str]:
    """Flatten-phase top-layer caption with a body-part guard.

    Uses the upstream DEFAULT_PROMPT plus a constraint forbidding articulation
    parts (run 1 finding: the stacking VLM bundled 'the raised right arm' into a
    detail peel, destroying the part's clean diff and orphaning its directed
    peel). Caches to the same vlm_response_{step}.json path as upstream. An
    empty caption means 'nothing but body parts left' -> treat as flat.
    """
    vlm_response_path = os.path.join(output_folders["vlm"], f"vlm_response_{step}.json")

    def generate():
        raw = detect_top_layer(
            image_vlm, cfg.vlm_model_name, False,
            prompt_override=DEFAULT_PROMPT + FLATTEN_CONSTRAINT, logger=logger,
        )
        return {
            "description": extract_tag_content("description", raw),
            "think": extract_tag_content("think", raw),
            "caption": extract_tag_content("caption", raw),
        }

    response = load_or_generate(
        file_path=vlm_response_path,
        generate_func=generate,
        save_func=save_json,
        load_func=load_json,
        logger=logger,
        description=f"flatten VLM response for step {step}",
    )
    caption = (response or {}).get("caption") or ""
    caption = caption.strip()
    if caption and BODY_PART_RE.search(caption):
        logger.warning(f"Step {step}: flatten caption mentions a body part despite constraint: {caption!r}")
    return caption or None


def is_flat_base(image_vlm: Image.Image, step: int, cfg: Config, output_folders: Dict[str, str], logger) -> bool:
    """Ask the VLM whether flattening is complete. Conservative on failure: not flat."""
    check_path = os.path.join(output_folders["vlm"], f"flat_check_{step}.json")
    if os.path.exists(check_path):
        with open(check_path) as f:
            cached = json.load(f)
        if cached.get("answer") in ("YES", "NO"):
            logger.info(f"Step {step}: flat-check cached -> {cached['answer']}")
            return cached["answer"] == "YES"

    try:
        raw = detect_top_layer(image_vlm, cfg.vlm_model_name, False, prompt_override=FLAT_CHECK_PROMPT, logger=logger)
        answer = (extract_tag_content("answer", raw) or "").strip().upper()
        if answer not in ("YES", "NO"):
            logger.warning(f"Step {step}: flat-check gave unparseable answer {answer!r}; treating as NO")
            answer = "NO"
        save_json({"answer": answer, "think": extract_tag_content("think", raw)}, check_path)
        logger.info(f"Step {step}: flat-check -> {answer}")
        return answer == "YES"
    except Exception as e:
        logger.warning(f"Step {step}: flat-check failed ({e}); treating as NO")
        return False


def peel_once(
    pipeline,
    image: Image.Image,
    caption: str,
    step: int,
    cfg: Config,
    output_folders: Dict[str, str],
    logger,
) -> Optional[Image.Image]:
    """One peel: (optional) mask from caption, warp, generate frame `step`."""
    image_vlm = image.resize((cfg.vlm_resolution, cfg.vlm_resolution))
    image_flux = pad_image(image_vlm, cfg.flux_width, cfg.flux_height, logger=logger)

    mask_image = None
    if cfg.enable_mask_detection:
        try:
            mask_image = _get_mask_response(image_vlm, caption, step - 1, cfg, output_folders, logger)
        except Exception as e:
            # A lost mask degrades one peel; a raised exception kills the run
            # (run 1 died here). Proceed maskless.
            logger.warning(f"Step {step}: mask detection failed ({e}); proceeding without mask.")
            mask_image = None

    prompt = warp_caption(caption)
    logger.info(f"Step {step}: prompt = {prompt}")
    return _generate_next_image(pipeline, image_flux, mask_image, prompt, step, cfg, output_folders, logger)


def run(pipeline, image_path: str, plan: Dict, cfg: Config, logger) -> Dict:
    target = plan.get("target") or Path(image_path).stem
    output_folders = create_output_subfolders(cfg.output_folder, target)
    log_path = os.path.join(output_folders["target"], "peel_log.json")

    peel_log: List[Dict] = []
    run_start = time.time()

    image = load_image(image_path)
    save_image(image, os.path.join(output_folders["png"], "layer_0.png"))
    step = 0

    # ---- Phase 1: flatten -------------------------------------------------
    flatten = plan.get("flatten", {"mode": "auto", "max_steps": 6})
    mode = flatten.get("mode", "auto")
    max_flatten = flatten.get("steps") if mode == "steps" else flatten.get("max_steps", 6)

    while mode != "none" and step < max_flatten and not is_pure_white(image):
        image_vlm = image.resize((cfg.vlm_resolution, cfg.vlm_resolution))

        if mode == "auto" and is_flat_base(image_vlm, step, cfg, output_folders, logger):
            logger.info(f"Step {step}: flat base reached; ending flatten phase.")
            break

        t0 = time.time()
        caption = get_flatten_caption(image_vlm, step, cfg, output_folders, logger)
        if not caption:
            logger.info(f"Step {step}: flatten VLM returned no caption (only body parts left, or failure). Ending flatten phase.")
            break

        step += 1
        out = peel_once(pipeline, image, caption, step, cfg, output_folders, logger)
        if out is None:
            logger.error(f"Step {step}: flatten generation failed. Aborting flatten phase.")
            step -= 1
            break

        image = out
        peel_log.append({
            "step": step, "phase": "flatten", "caption": caption,
            "seconds": round(time.time() - t0, 1),
        })
        save_json({"log": peel_log}, log_path)  # checkpoint

    flat_step = step
    logger.info(f"Flatten phase done at frame {flat_step}.")

    # ---- Phase 2: directed peels -----------------------------------------
    parts_out = []
    for part in plan.get("parts", []):
        name, caption = part["name"], part["caption"]
        t0 = time.time()
        before = step
        step += 1
        logger.info(f"Directed peel '{name}': frames {before} -> {step}")
        out = peel_once(pipeline, image, caption, step, cfg, output_folders, logger)
        if out is None:
            logger.error(f"Directed peel '{name}' failed to generate; skipping (frames unchanged).")
            step -= 1
            peel_log.append({"step": step, "phase": "directed", "name": name,
                             "caption": caption, "failed": True})
            continue

        image = out
        rec = {
            "step": step, "phase": "directed", "name": name, "caption": caption,
            "frame_before": before, "frame_after": step,
            "seconds": round(time.time() - t0, 1),
        }
        peel_log.append(rec)
        parts_out.append(rec)
        save_json({"log": peel_log}, log_path)  # checkpoint

    # ---- Summary ----------------------------------------------------------
    summary = {
        "target": target,
        "flat_frame": flat_step,
        "final_frame": step,
        "parts": parts_out,
        "log": peel_log,
        "economics": {
            "wall_seconds": round(time.time() - run_start, 1),
            "peels": step,
            "directed_peels": len(parts_out),
            "flatten_peels": flat_step,
        },
    }
    save_json(summary, log_path)
    logger.info(f"Done: {step} frames, {len(parts_out)}/{len(plan.get('parts', []))} directed parts. Log: {log_path}")
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(description="Directed-peel driver (flatten, then excise named parts)")
    parser.add_argument("--image", type=str, required=True, help="Source mascot PNG")
    parser.add_argument("--plan", type=str, required=True, help="Plan JSON (flatten policy + directed part list)")

    # Flux settings (mirror inference.py defaults)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="kingno/LayerPeeler")
    parser.add_argument("--lora_name", type=str, default="LayerPeeler_rank256_step20000")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    # VLM settings
    parser.add_argument("--vlm_model_name", type=str, default="gemini-pro-latest")
    parser.add_argument("--disable_mask_detection", action="store_true")
    parser.add_argument("--bbox_expansion", type=int, default=15)
    parser.add_argument("--mask_detection_temperature", type=float, default=0.5)

    parser.add_argument("--flux_width", type=int, default=512)
    parser.add_argument("--flux_height", type=int, default=512)
    parser.add_argument("--vlm_resolution", type=int, default=512)

    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.plan) as f:
        plan = json.load(f)

    cfg = Config(
        seed=args.seed,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        vlm_model_name=args.vlm_model_name,
        enable_mask_detection=not args.disable_mask_detection,
        bbox_expansion=args.bbox_expansion,
        mask_detection_temperature=args.mask_detection_temperature,
        use_layer_graph_reasoning=False,
        max_steps=99,
        flux_width=args.flux_width,
        flux_height=args.flux_height,
        vlm_resolution=args.vlm_resolution,
        output_folder=args.output_folder,
        input_folder="",
        max_images=1,
    )

    Path(args.output_folder, "config.yaml").write_text(yaml.dump({**vars(args), "plan": plan}))

    logger = setup_logging(cfg.output_folder)
    set_seed(cfg.seed)

    logger.info("===== Loading Model and LoRA weights =====")
    pipeline = load_model(cfg.pretrained_model_name_or_path, cfg.lora_path, cfg.lora_name)

    logger.info("===== Directed peel run =====")
    run(pipeline, args.image, plan, cfg, logger)


if __name__ == "__main__":
    main()
