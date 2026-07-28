"""Qwen-backed peel driver: the F0 pivot made production (manifest 2026-07-28).

Same architecture as directed_peel.py's independent mode — flatten to a flat
base, then peel every part INDEPENDENTLY from that base with verify-retry and
the semantic removal gate — but the peel primitive is fal-hosted
qwen-image-edit instead of FLUX+sksremovelayer LoRA. Consequences:

  * Runs entirely LOCALLY (fal + Gemini are HTTP APIs). No vast box, no
    provisioning, no destroy trap. A fox run costs ~$0.30-0.70 in fal calls.
  * Prompts are imperative editing instructions ("Remove the X. Fill the
    revealed area...") built from the SAME plan captions the FLUX driver used
    — plans carry over unchanged. A part may override with "qwen_prompt".
  * CHANGE-MASK COMPOSITING is built in (bon gen-4 §0(ii)): Qwen re-renders
    the whole canvas, so raw output carries global pixel drift + occasional
    out-of-region collateral (F0: fox arm peel also took the tail). After each
    edit, only changed pixels INSIDE the part's dilated detection bbox are
    taken from Qwen; everything else is hard-composited from the input frame.
    Raw drift/collateral stats are logged as diagnostics — the composite is
    the collateral control, the VLM removal gate is the honesty control.

This file deliberately does NOT import inference.py (torch/FluxPipeline at
module top — unimportable on the Mac). The small shared helpers are copied;
the FLUX driver remains untouched for GPU runs.

Usage (from LayerPeeler/inference/):
  uv run --with requests --with pillow --with numpy --with python-dotenv \
         --with pyyaml --with graphviz python qwen_peel.py \
         --image ../../spike/fox512.png --plan plans/fox_run7.json \
         --output_folder ../out/fox_run7
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageFilter

# ---- env: map ~/.env secret names before vlm_util reads them ---------------
def _load_home_env() -> None:
    env = Path.home() / ".env"
    if not env.exists():
        return
    vals = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"')
    os.environ.setdefault("GEMINI_API_KEY", vals.get("GEMINI_API_SECRET_KEY", ""))
    os.environ.setdefault("FAL_KEY", vals.get("FAL_KEY", ""))

_load_home_env()

import requests

from utils.util import load_image, load_json, load_or_generate, save_image, save_json, setup_logging
from utils.vlm_util import (
    DEFAULT_PROMPT, MASK_PROMPT, detect_top_layer, detect_mask, extract_tag_content,
    merge_bbox_as_mask, parse_json, parse_segmentation_masks, plot_segmentation_masks,
)
from utils.image_util import is_pure_white


# ---------------------------------------------------------------------------
# Config (local dataclass — upstream Config drags FLUX fields we don't have)
# ---------------------------------------------------------------------------
class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


FAL_URL = "https://fal.run/fal-ai/qwen-image-edit"

# F0-validated phrasing: imperative removal + explicit fill instruction.
DIRECTED_TEMPLATE = ("Remove {part}. Fill the revealed area with the "
                     "character's body and the white background. "
                     "Do not add any shadows or new elements. "
                     "Do not change anything else in the image.")
FLATTEN_TEMPLATE = ("Remove {part}, revealing the flat colors of the "
                    "character underneath. Do not change anything else in the image.")


# ---------------------------------------------------------------------------
# Prompts + guards copied from directed_peel.py (unimportable locally)
# ---------------------------------------------------------------------------
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

# Negative lookahead: "tail tip", "head outline" etc. are DETAIL layers, not
# body parts — run 7's fallback refused "the white tail tip" and ended flatten
# with the face still on the base.
BODY_PART_RE = re.compile(
    r"\b(arms?|hands?|legs?|feet|foot|heads?|tails?|torso|body)\b"
    r"(?!\s+(tips?|outlines?|highlights?|stripes?|patch(es)?|markings?|spots?))",
    re.IGNORECASE,
)

REMOVAL_CHECK_PROMPT = """You are inspecting a layer-peeling edit on a cartoon character. The image shows BEFORE (left) and AFTER (right) side by side. The edit was asked to REMOVE: "{part}".

Compare the two sides. The element counts as REMOVED only if its shape is gone from the AFTER side (the area may be filled with background or plain fill colour). If the element's shape is still visible on the AFTER side — even recoloured, flattened, or with different shading — it was NOT removed. If the element was removed but a REPLACEMENT version of the same kind of element has been drawn elsewhere on the character (e.g. a raised arm removed but a hanging arm added), it counts as NOT removed.

Respond with brief reasoning in <think></think> tags, then exactly REMOVED or PRESENT in <answer></answer> tags."""


def create_output_subfolders(output_folder: str, target: str) -> Dict[str, str]:
    folders = {
        "target": os.path.join(output_folder, target),
        "png": os.path.join(output_folder, target, "layer_png"),
        "svg": os.path.join(output_folder, target, "layer_svg"),
        "vlm": os.path.join(output_folder, target, "layer_vlm"),
        "mask": os.path.join(output_folder, target, "layer_mask"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


# ---------------------------------------------------------------------------
# Generator backend: fal qwen-image-edit
# ---------------------------------------------------------------------------
def _data_uri(img: Image.Image) -> str:
    import base64, io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def qwen_edit(image: Image.Image, prompt: str, seed: int, dest: str,
              cfg, logger, fal_stats: Dict) -> Optional[Image.Image]:
    """One fal qwen-image-edit call, cached on dest path. Retries transient
    HTTP failures (F0: one balance 403 fired mid-run and cleared)."""
    if os.path.exists(dest):
        logger.info(f"  qwen: cached {os.path.basename(dest)}")
        return Image.open(dest).convert("RGB")

    key = os.environ.get("FAL_KEY")
    if not key:
        raise SystemExit("FAL_KEY not set (expected in ~/.env)")

    for attempt in range(3):
        t0 = time.time()
        try:
            r = requests.post(
                FAL_URL,
                headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                json={"image_url": _data_uri(image), "prompt": prompt,
                      "num_inference_steps": cfg.num_inference_steps,
                      "guidance_scale": cfg.guidance_scale,
                      "seed": seed, "output_format": "png"},
                timeout=300,
            )
        except requests.RequestException as e:
            logger.warning(f"  qwen: request error ({e}); retry {attempt + 1}/3")
            time.sleep(10)
            continue
        if r.status_code != 200:
            logger.warning(f"  qwen: HTTP {r.status_code} — {r.text[:200]}; retry {attempt + 1}/3")
            time.sleep(10 * (attempt + 1))
            continue
        body = r.json()
        out = Image.open(requests.get(body["images"][0]["url"], timeout=120, stream=True).raw).convert("RGB")
        if out.size != image.size:
            out = out.resize(image.size, Image.LANCZOS)
        out.save(dest)
        Path(dest).with_suffix(".json").write_text(json.dumps(
            {"prompt": prompt, "seed": seed, "fal_seed": body.get("seed"),
             "timings": body.get("timings"), "seconds": round(time.time() - t0, 1)}, indent=1))
        fal_stats["calls"] += 1
        logger.info(f"  qwen: ok ({time.time() - t0:.0f}s) [{fal_stats['calls']} calls]")
        return out
    logger.error(f"  qwen: all retries failed for {os.path.basename(dest)}")
    return None


# ---------------------------------------------------------------------------
# Change-mask compositing (bon gen-4 §0(ii))
# ---------------------------------------------------------------------------
def composite_change_mask(before: Image.Image, raw: Image.Image,
                          bbox_mask: Optional[np.ndarray],
                          logger) -> Tuple[Image.Image, Dict]:
    """Hard-composite untouched regions from the input frame.

    keep = changed(raw vs before) ∧ dilated(bbox). Everything else comes from
    `before`, killing Qwen's global pixel drift and out-of-region collateral
    at source. The kept region is dilated + feathered a few px so a slightly
    tight detection bbox doesn't leave a hard seam. With no bbox (detection
    failure, or a flatten caption whose bbox covers the canvas) the raw output
    passes through — logged, never silent.
    """
    b = np.asarray(before.convert("RGB"), dtype=int)
    r = np.asarray(raw.convert("RGB"), dtype=int)
    changed = np.abs(b - r).sum(axis=2) > 30
    diag = {"raw_changed_px": int(changed.sum())}

    if bbox_mask is None:
        diag["composited"] = False
        return raw, diag

    keep = changed & bbox_mask
    diag["kept_px"] = int(keep.sum())
    diag["discarded_px"] = int((changed & ~bbox_mask).sum())  # drift + collateral repaired
    diag["composited"] = True

    m = Image.fromarray((keep * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(7))       # dilate ~3px
    m = m.filter(ImageFilter.GaussianBlur(2))    # feather the seam
    alpha = np.asarray(m, dtype=float)[..., None] / 255.0
    out = (b * (1 - alpha) + r * alpha).round().astype(np.uint8)
    return Image.fromarray(out), diag


# ---------------------------------------------------------------------------
# VLM helpers (Gemini via utils.vlm_util — works locally)
# ---------------------------------------------------------------------------
def get_bbox_mask(image: Image.Image, caption: str, idx: int, cfg,
                  output_folders: Dict[str, str], logger) -> Optional[np.ndarray]:
    """Detection bboxes for a caption -> boolean mask at image size. Cached.
    Saves bbox_{idx}.png (verify + post-mortem) and mask_{idx}.png (viz)."""
    mask_response_path = os.path.join(output_folders["vlm"], f"mask_response_{idx}.json")
    bbox_path = os.path.join(output_folders["mask"], f"bbox_{idx}.png")

    def generate():
        raw = detect_mask(image, cfg.vlm_model_name, MASK_PROMPT.format(layers=caption),
                          temperature=cfg.mask_detection_temperature, logger=logger)
        return json.loads(parse_json(raw))

    response = load_or_generate(file_path=mask_response_path, generate_func=generate,
                                save_func=save_json, load_func=load_json, logger=logger,
                                description=f"mask VLM response {idx}")
    if not response:
        logger.warning(f"  detection: no mask response for {caption!r}")
        return None
    try:
        masks = parse_segmentation_masks(response, img_height=image.size[1], img_width=image.size[0])
    except Exception as e:
        logger.warning(f"  detection: parse failed ({e})")
        return None
    if not masks:
        logger.warning(f"  detection: zero bboxes for {caption!r}")
        return None
    masks = [m.expand_bbox(cfg.bbox_expansion) for m in masks]
    merged = merge_bbox_as_mask(masks, image.size[0], image.size[1])
    Image.fromarray(merged).save(bbox_path)
    plot_segmentation_masks(image, masks).save(os.path.join(output_folders["mask"], f"mask_{idx}.png"))
    arr = merged > 0
    if not arr.any():
        logger.warning(f"  detection: bbox mask EMPTY for {caption!r}")
        return None
    return arr


def get_bbox_mask_multi(image: Image.Image, captions: List[str], idx: int, cfg,
                        output_folders: Dict[str, str], logger) -> Optional[np.ndarray]:
    """Detection retry chain over a part's caption list.

    Axolotl runs 1 AND 2 both got an empty bbox on gills_left's primary
    caption — a recurring Gemini grounding blind spot on character-relative
    phrasings. The edit prompts already had alt-caption diversity; detection
    did not. Each caption caches under its own suffix (idx_c0, idx_c1, ...);
    the winning caption's mask is ALSO saved under the bare bbox_{idx}.png
    path so verify/post-mortem tooling finds it unchanged.

    Returns None only if EVERY caption fails — callers must then treat the
    part as a DETECTION failure and reject it: run 2 showed a raw-accept
    without compositing poisons both the part cut and base_residual
    (gills_left took the right gills and a global re-render with it).
    """
    for ci, caption in enumerate(captions):
        arr = get_bbox_mask(image, caption, f"{idx}_c{ci}", cfg, output_folders, logger)
        if arr is not None:
            if ci > 0:
                logger.info(f"  detection: succeeded on alt caption {ci} ({caption!r})")
            Image.fromarray((arr * 255).astype(np.uint8)).save(
                os.path.join(output_folders["mask"], f"bbox_{idx}.png"))
            return arr
    logger.warning(f"  detection: ALL {len(captions)} captions returned empty bboxes for part idx {idx}.")
    return None


def get_flatten_caption(image: Image.Image, step: int, cfg, output_folders, logger,
                        guarded: bool = True) -> Optional[str]:
    suffix = "" if guarded else "_fallback"
    path = os.path.join(output_folders["vlm"], f"vlm_response_{step}{suffix}.json")
    prompt = DEFAULT_PROMPT + FLATTEN_CONSTRAINT if guarded else DEFAULT_PROMPT

    def generate():
        raw = detect_top_layer(image, cfg.vlm_model_name, False, prompt_override=prompt, logger=logger)
        return {"description": extract_tag_content("description", raw),
                "think": extract_tag_content("think", raw),
                "caption": extract_tag_content("caption", raw)}

    response = load_or_generate(file_path=path, generate_func=generate, save_func=save_json,
                                load_func=load_json, logger=logger,
                                description=f"{'flatten' if guarded else 'fallback'} VLM response step {step}")
    caption = ((response or {}).get("caption") or "").strip()
    if not caption:
        return None
    if BODY_PART_RE.search(caption):
        if guarded:
            logger.warning(f"Step {step}: flatten caption names a body part despite constraint: {caption!r}")
        else:
            logger.info(f"Step {step}: fallback caption names a body part ({caption!r}); refusing it.")
            return None
    return caption


def is_flat_base(image: Image.Image, step: int, cfg, output_folders, logger) -> bool:
    check_path = os.path.join(output_folders["vlm"], f"flat_check_{step}.json")
    if os.path.exists(check_path):
        cached = json.load(open(check_path))
        if cached.get("answer") in ("YES", "NO"):
            logger.info(f"Step {step}: flat-check cached -> {cached['answer']}")
            return cached["answer"] == "YES"
    try:
        raw = detect_top_layer(image, cfg.vlm_model_name, False, prompt_override=FLAT_CHECK_PROMPT, logger=logger)
        answer = (extract_tag_content("answer", raw) or "").strip().upper()
        if answer not in ("YES", "NO"):
            logger.warning(f"Step {step}: flat-check unparseable {answer!r}; treating as NO")
            answer = "NO"
        save_json({"answer": answer, "think": extract_tag_content("think", raw)}, check_path)
        logger.info(f"Step {step}: flat-check -> {answer}")
        return answer == "YES"
    except Exception as e:
        logger.warning(f"Step {step}: flat-check failed ({e}); treating as NO")
        return False


def verify_peel(before: Image.Image, after: Image.Image, bbox_mask: Optional[np.ndarray],
                min_part_px: int, max_collateral: float) -> Dict:
    """Acceptance check on the COMPOSITED frame: enough change inside the
    detection bbox (the part actually came off), bounded change outside it
    (feather slop only, post-composite — raw collateral is repaired upstream
    and logged separately)."""
    b = np.asarray(before.convert("RGB"), dtype=int)
    a = np.asarray(after.convert("RGB"), dtype=int)
    changed = np.abs(b - a).sum(axis=2) > 30
    if bbox_mask is None:
        inside, outside = int(changed.sum()), 0
    else:
        inside = int((changed & bbox_mask).sum())
        outside = int((changed & ~bbox_mask).sum())
    collateral = (outside / inside) if inside else float("inf")
    ok = inside >= min_part_px and collateral <= max_collateral
    return {"ok": bool(ok), "inside_px": inside, "outside_px": outside,
            "collateral": round(collateral, 3) if inside else None,
            "had_bbox": bbox_mask is not None}


def check_removed(before: Image.Image, after: Image.Image, caption: str, cfg,
                  output_folders, tag: str, logger) -> bool:
    """Semantic removal gate (before/after montage -> REMOVED/PRESENT).
    Validated 9/9 on runs fox-5 + axolotl-1; prompt extended for Qwen's
    F0 failure mode: part REPLACEMENT (raised arm -> hanging arm) counts as
    PRESENT. Conservative on API failure: accept with warning."""
    m = Image.new("RGB", (before.width + after.width, max(before.height, after.height)), "white")
    m.paste(before.convert("RGB"), (0, 0))
    m.paste(after.convert("RGB"), (before.width, 0))
    check_path = os.path.join(output_folders["vlm"], f"removal_check_{tag}.json")
    try:
        raw = detect_top_layer(m, cfg.vlm_model_name, False,
                               prompt_override=REMOVAL_CHECK_PROMPT.format(part=caption), logger=logger)
        answer = (extract_tag_content("answer", raw) or "").strip().upper()
        save_json({"answer": answer, "think": extract_tag_content("think", raw)}, check_path)
        if answer not in ("REMOVED", "PRESENT"):
            logger.warning(f"  removal-check '{tag}': unparseable {answer!r}; accepting with warning")
            return True
        logger.info(f"  removal-check '{tag}': {answer}")
        return answer == "REMOVED"
    except Exception as e:
        logger.warning(f"  removal-check '{tag}' failed ({e}); accepting with warning")
        return True


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def flatten_phase(image: Image.Image, plan: Dict, cfg, output_folders,
                  peel_log: List[Dict], log_path: str, logger, fal_stats) -> Tuple[Image.Image, int]:
    step = 0
    flatten = plan.get("flatten", {"mode": "auto", "max_steps": 6})
    mode = flatten.get("mode", "auto")
    max_flatten = flatten.get("steps") if mode == "steps" else flatten.get("max_steps", 6)

    while mode != "none" and step < max_flatten and not is_pure_white(image):
        if mode == "auto" and is_flat_base(image, step, cfg, output_folders, logger):
            logger.info(f"Step {step}: flat base reached; ending flatten phase.")
            break
        t0 = time.time()
        caption = get_flatten_caption(image, step, cfg, output_folders, logger, guarded=True)
        if not caption and mode == "auto":
            caption = get_flatten_caption(image, step, cfg, output_folders, logger, guarded=False)
        if not caption:
            logger.info(f"Step {step}: no flatten caption (only body parts left, or failure). Ending flatten.")
            break

        step += 1
        bbox = get_bbox_mask(image, caption, step - 1, cfg, output_folders, logger) \
            if cfg.enable_mask_detection else None
        prompt = FLATTEN_TEMPLATE.format(part=caption)
        raw_path = os.path.join(output_folders["png"], f"layer_{step}_raw.png")
        raw = qwen_edit(image, prompt, cfg.seed, raw_path, cfg, logger, fal_stats)
        if raw is None:
            logger.error(f"Step {step}: flatten generation failed. Ending flatten.")
            step -= 1
            break
        out, diag = composite_change_mask(image, raw, bbox, logger)
        save_image(out, os.path.join(output_folders["png"], f"layer_{step}.png"))
        image = out
        peel_log.append({"step": step, "phase": "flatten", "caption": caption,
                         "prompt": prompt, "composite": diag,
                         "seconds": round(time.time() - t0, 1)})
        save_json({"log": peel_log}, log_path)

    logger.info(f"Flatten phase done at frame {step}.")
    return image, step


def run_independent(image_path: str, plan: Dict, cfg, logger) -> Dict:
    target = plan.get("target") or Path(image_path).stem
    output_folders = create_output_subfolders(cfg.output_folder, target)
    log_path = os.path.join(output_folders["target"], "peel_log.json")

    peel_log: List[Dict] = []
    fal_stats = {"calls": 0}
    run_start = time.time()

    image = load_image(image_path).convert("RGB").resize((cfg.resolution, cfg.resolution))
    save_image(image, os.path.join(output_folders["png"], "layer_0.png"))

    flat, flat_step = flatten_phase(image, plan, cfg, output_folders, peel_log, log_path, logger, fal_stats)
    save_image(flat, os.path.join(output_folders["png"], "flat_base.png"))

    flat_np = np.asarray(flat.convert("RGB"), dtype=int)
    residual = np.asarray(flat.convert("RGB")).copy()

    parts_out = []
    for i, part in enumerate(plan.get("parts", [])):
        name = part["name"]
        captions = [part["caption"]] + part.get("alt_captions", [])
        mask_idx = 900 + i
        t0 = time.time()

        bbox = get_bbox_mask_multi(flat, captions, mask_idx, cfg, output_folders, logger) \
            if cfg.enable_mask_detection else None
        if bbox is None and cfg.enable_mask_detection:
            # No bbox -> no compositing -> a raw accept would poison the part
            # cut and base_residual (run 2 gills_left). Reject as DETECTION
            # failure; the part stays in base, which degrades gracefully.
            logger.error(f"Independent peel '{name}': detection failed on all captions; "
                         f"rejecting part WITHOUT peeling (detection failure, not peel failure).")
            peel_log.append({"phase": "independent", "name": name, "caption": captions[0],
                             "failed": True, "detection_failed": True, "attempts": []})
            save_json({"log": peel_log}, log_path)
            continue

        accepted, attempts = None, []
        for attempt in range(cfg.max_attempts):
            caption = captions[min(attempt, len(captions) - 1)]
            prompt = part.get("qwen_prompt") or DIRECTED_TEMPLATE.format(part=caption)
            seed = cfg.seed + attempt * 1009
            logger.info(f"Independent peel '{name}' attempt {attempt + 1}/{cfg.max_attempts} "
                        f"(seed={seed}): {caption!r}")
            raw_path = os.path.join(output_folders["png"], f"part_{name}_a{attempt + 1}_raw.png")
            raw = qwen_edit(flat, prompt, seed, raw_path, cfg, logger, fal_stats)
            if raw is None:
                attempts.append({"attempt": attempt + 1, "caption": caption, "generated": False})
                continue
            out, diag = composite_change_mask(flat, raw, bbox, logger)
            v = verify_peel(flat, out, bbox, cfg.min_part_px, cfg.max_collateral)
            v.update({"attempt": attempt + 1, "caption": caption, "generated": True,
                      "composite": diag})
            attempts.append(v)
            logger.info(f"  verify: inside={v['inside_px']}px outside={v['outside_px']}px "
                        f"collateral={v['collateral']} raw_drift={diag.get('discarded_px')}px "
                        f"-> {'ACCEPT' if v['ok'] else 'REJECT'}")
            if v["ok"]:
                v["removed"] = check_removed(flat, out, caption, cfg, output_folders,
                                             f"{name}_a{attempt + 1}", logger)
                if not v["removed"]:
                    v["ok"] = False
                else:
                    accepted = (out, v, caption)
                    break
            save_image(out, os.path.join(output_folders["png"],
                                         f"part_{name}_rejected{attempt + 1}.png"))

        if accepted is None:
            logger.error(f"Independent peel '{name}': all {cfg.max_attempts} attempts rejected; "
                         f"part stays in base.")
            peel_log.append({"phase": "independent", "name": name, "caption": captions[0],
                             "failed": True, "attempts": attempts})
            save_json({"log": peel_log}, log_path)
            continue

        out, verdict, used_caption = accepted
        part_file = f"part_{name}.png"
        save_image(out, os.path.join(output_folders["png"], part_file))
        o = np.asarray(out.convert("RGB"))
        changed = np.abs(flat_np - o.astype(int)).sum(axis=2) > 30
        residual[changed] = o[changed]

        rec = {"phase": "independent", "name": name, "caption": used_caption,
               "file": part_file, "mask_idx": mask_idx,
               "seconds": round(time.time() - t0, 1), "verify": verdict, "attempts": attempts}
        peel_log.append(rec)
        parts_out.append(rec)
        save_json({"log": peel_log}, log_path)

    save_image(Image.fromarray(residual), os.path.join(output_folders["png"], "base_residual.png"))

    summary = {
        "mode": "independent",
        "generator": "fal-ai/qwen-image-edit",
        "target": target,
        "flat_frame": flat_step,
        "base_file": "base_residual.png",
        "parts": parts_out,
        "log": peel_log,
        "economics": {
            "wall_seconds": round(time.time() - run_start, 1),
            "flatten_peels": flat_step,
            "parts_accepted": len(parts_out),
            "parts_requested": len(plan.get("parts", [])),
            "attempts_total": sum(len(p.get("attempts", [])) for p in peel_log if p.get("phase") == "independent"),
            "fal_calls": fal_stats["calls"],
            "fal_cost_est_usd": round(fal_stats["calls"] * 0.04, 2),
        },
    }
    save_json(summary, log_path)
    logger.info(f"Done (independent, qwen): {len(parts_out)}/{len(plan.get('parts', []))} parts accepted. "
                f"fal calls: {fal_stats['calls']} (~${fal_stats['calls'] * 0.04:.2f}). Log: {log_path}")
    return summary


def parse_arguments():
    p = argparse.ArgumentParser(description="Qwen-backed peel driver (local, fal-hosted)")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--plan", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--vlm_model_name", type=str, default="gemini-pro-latest")
    p.add_argument("--disable_mask_detection", action="store_true")
    p.add_argument("--bbox_expansion", type=int, default=15)
    p.add_argument("--mask_detection_temperature", type=float, default=0.5)
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--min_part_px", type=int, default=800)
    p.add_argument("--max_collateral", type=float, default=0.5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--output_folder", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)
    with open(args.plan) as f:
        plan = json.load(f)

    cfg = Cfg(
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        vlm_model_name=args.vlm_model_name,
        enable_mask_detection=not args.disable_mask_detection,
        bbox_expansion=args.bbox_expansion,
        mask_detection_temperature=args.mask_detection_temperature,
        max_attempts=args.max_attempts,
        min_part_px=args.min_part_px,
        max_collateral=args.max_collateral,
        resolution=args.resolution,
        output_folder=args.output_folder,
    )
    Path(args.output_folder, "config.yaml").write_text(yaml.dump({**vars(args), "plan": plan}))
    logger = setup_logging(cfg.output_folder)
    random.seed(cfg.seed)

    if plan.get("mode") != "independent":
        raise SystemExit("qwen_peel.py implements independent mode only; set \"mode\": \"independent\" in the plan.")
    logger.info("===== Independent peel run (qwen-image-edit via fal) =====")
    run_independent(args.image, plan, cfg, logger)


if __name__ == "__main__":
    main()
