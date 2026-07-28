"""Directed-peel diagnostic bench: N independent peel trials off ONE flat base.

Purpose (run-3 findings): map the model's capability boundary. Captions must be
short/in-distribution; hanging-arm and legs never peel at defaults; guidance /
seed / caption knobs untested. One boot amortises the model download over the
whole matrix (~25 s/trial), and every trial is verified with the same
inside/outside-bbox check as the production driver.

Each trial peels from the SAME base image (no chaining), so results are
independent and comparable. Masks are detected once per unique caption.

Bench spec JSON:
{
  "trials": [
    {"name": "head", "caption": "the head with ears at the top of the image",
     "guidance": 6.5, "seed": 42, "steps": 30},
    ...
  ]
}

Usage:
  python peel_bench.py --base_image flat.png --bench bench.json \
      --pretrained_model_name_or_path ../PhotoDoodle_Pretrain --output_folder outputs/bench
Outputs: bench_png/<name>_g<g>_s<seed>.png per trial + results.json + summary table.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image
from transformers import set_seed

from inference import Config, load_model, warp_caption, _get_mask_response
from directed_peel import verify_peel
from utils.util import load_image, save_image, setup_logging
from utils.image_util import pad_image, unpad_image


def caption_key(caption: str) -> int:
    """Stable cache index per unique caption (mask reused across its trials)."""
    return int(hashlib.md5(caption.encode()).hexdigest()[:6], 16) % 100000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_image", required=True, help="flat base PNG all trials peel from")
    ap.add_argument("--bench", required=True, help="bench spec JSON")
    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--lora_path", default="kingno/LayerPeeler")
    ap.add_argument("--lora_name", default="LayerPeeler_rank256_step20000")
    ap.add_argument("--vlm_model_name", default="gemini-pro-latest")
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--min_part_px", type=int, default=800)
    ap.add_argument("--max_collateral", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    folders = {
        "png": os.path.join(args.output_folder, "bench_png"),
        "vlm": os.path.join(args.output_folder, "layer_vlm"),
        "mask": os.path.join(args.output_folder, "layer_mask"),
    }
    for f in folders.values():
        os.makedirs(f, exist_ok=True)

    logger = setup_logging(args.output_folder)
    bench = json.loads(Path(args.bench).read_text())

    cfg = Config(
        seed=42, pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        lora_path=args.lora_path, lora_name=args.lora_name,
        num_inference_steps=30, guidance_scale=4.5,
        vlm_model_name=args.vlm_model_name, enable_mask_detection=True,
        bbox_expansion=15, mask_detection_temperature=0.5,
        use_layer_graph_reasoning=False, max_steps=99,
        flux_width=512, flux_height=512, vlm_resolution=512,
        output_folder=args.output_folder, input_folder="", max_images=1,
    )

    logger.info("===== Loading model =====")
    pipeline = load_model(cfg.pretrained_model_name_or_path, cfg.lora_path, cfg.lora_name)

    base = load_image(args.base_image)
    image_vlm = base.resize((cfg.vlm_resolution, cfg.vlm_resolution))
    image_flux = pad_image(image_vlm, cfg.flux_width, cfg.flux_height, logger=logger)

    results = []
    results_path = os.path.join(args.output_folder, "results.json")

    for i, trial in enumerate(bench["trials"]):
        name = trial["name"]
        caption = trial["caption"]
        guidance = trial.get("guidance", 4.5)
        seed = trial.get("seed", 42)
        steps = trial.get("steps", 30)
        tag = f"{name}_g{guidance}_s{seed}"
        logger.info(f"[{i + 1}/{len(bench['trials'])}] {tag}: {caption!r}")

        key = caption_key(caption)
        try:
            mask_image = _get_mask_response(image_vlm, caption, key, cfg, folders, logger)
        except Exception as e:
            logger.warning(f"  mask detection failed ({e}); maskless")
            mask_image = None

        set_seed(seed)
        t0 = time.time()
        out = pipeline(
            prompt=warp_caption(caption),
            condition_image=image_flux,
            mask_image=mask_image,
            height=cfg.flux_height, width=cfg.flux_width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            max_sequence_length=512,
        ).images[0]
        out = unpad_image(out, cfg.vlm_resolution, cfg.vlm_resolution, logger=logger)
        dt = time.time() - t0

        save_image(out, os.path.join(folders["png"], f"{tag}.png"))
        v = verify_peel(image_vlm, out, os.path.join(folders["mask"], f"bbox_{key}.png"),
                        args.min_part_px, args.max_collateral)
        rec = {"tag": tag, **trial, **v, "seconds": round(dt, 1)}
        results.append(rec)
        Path(results_path).write_text(json.dumps(results, indent=2))
        logger.info(f"  -> {'ACCEPT' if v['ok'] else 'REJECT'} inside={v['inside_px']} "
                    f"outside={v['outside_px']} collateral={v['collateral']} ({dt:.0f}s)")

    logger.info("===== BENCH SUMMARY =====")
    for r in results:
        logger.info(f"{'PASS' if r['ok'] else 'fail'}  {r['tag']:34s} "
                    f"in={r['inside_px']:6d} out={r['outside_px']:6d} col={r['collateral']}")


if __name__ == "__main__":
    main()
