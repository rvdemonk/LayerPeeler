"""Prompt bench: sweep the Qwen instruction layer, everything else frozen.

The peel pipeline has five prompt surfaces (four Gemini-facing, one
Qwen-facing). This benches the Qwen-facing one in isolation: same base
frame, same detection bbox, same compositing, same verify + removal gate —
only the edit instruction varies. $0.04/cell; a full arm matrix costs less
than one FLUX provisioning fizzle.

Motivating case (run 7): every arm peel removed the raised arm and drew a
HANGING replacement — rejected 3/3 by the gate. Before concluding that's a
fine-tune-only fix, falsify cheaply: maybe it's prompt-addressable. Captions
inherited from the FLUX plans obey the LoRA-era "short + located" rule; Qwen
is an instruction follower with no such constraint.

Spec JSON:
{
  "cases": [
    {
      "name": "fox_arm",
      "base": "../out/fox_run7/fox/layer_png/flat_base.png",
      "detect": "the raised arm on the right side of the image",   # bbox caption
      "gate_part": "the raised arm on the right side of the image", # removal-gate wording (held constant!)
      "prompts": [{"id": "baseline", "text": "Remove ..."}, ...]
    }
  ],
  "seeds": [42, 1051]
}

The removal-gate wording stays CONSTANT across variants — we are benching
the generator prompt, not gaming the judge.

Usage (from LayerPeeler/inference/):
  uv run --with requests --with pillow --with numpy --with python-dotenv \
         --with pyyaml --with graphviz python prompt_bench.py \
         --spec plans/prompt_bench_arm.json --output_folder ../out/prompt_bench_arm
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from qwen_peel import (
    Cfg, check_removed, composite_change_mask, create_output_subfolders,
    get_bbox_mask, qwen_edit, verify_peel,
)
from utils.util import load_image, save_image, save_json, setup_logging


def run_case(case, seeds, cfg, logger, fal_stats):
    name = case["name"]
    folders = create_output_subfolders(cfg.output_folder, name)
    base = load_image(case["base"]).convert("RGB").resize((cfg.resolution, cfg.resolution))
    save_image(base, os.path.join(folders["png"], "base.png"))

    bbox = get_bbox_mask(base, case["detect"], 900, cfg, folders, logger)
    gate_part = case.get("gate_part", case["detect"])

    cells = []
    for variant in case["prompts"]:
        vid = variant["id"]
        for seed in seeds:
            tag = f"{vid}_s{seed}"
            t0 = time.time()
            raw_path = os.path.join(folders["png"], f"{tag}_raw.png")
            raw = qwen_edit(base, variant["text"], seed, raw_path, cfg, logger, fal_stats)
            if raw is None:
                cells.append({"case": name, "variant": vid, "seed": seed, "generated": False})
                continue
            out, diag = composite_change_mask(base, raw, bbox, logger)
            save_image(out, os.path.join(folders["png"], f"{tag}.png"))
            v = verify_peel(base, out, bbox, cfg.min_part_px, cfg.max_collateral)
            removed = check_removed(base, out, gate_part, cfg, folders, tag, logger) if v["ok"] else None
            cell = {"case": name, "variant": vid, "seed": seed, "generated": True,
                    "verify": v, "composite": diag, "removed": removed,
                    "true_excision": bool(v["ok"] and removed),
                    "seconds": round(time.time() - t0, 1)}
            cells.append(cell)
            logger.info(f"[{name}/{tag}] inside={v['inside_px']} collateral={v['collateral']} "
                        f"gate={'REMOVED' if removed else ('PRESENT' if removed is False else 'n/a')} "
                        f"-> {'TRUE' if cell['true_excision'] else 'no'}")

    # contact sheet: base | one column per (variant, seed), gate verdict in label
    ok_cells = [c for c in cells if c.get("generated")]
    cell_px, pad = 200, 30
    sheet = Image.new("RGB", ((len(ok_cells) + 1) * cell_px, cell_px + pad), "white")
    d = ImageDraw.Draw(sheet)
    sheet.paste(base.resize((cell_px, cell_px)), (0, 0))
    d.text((4, cell_px + 4), "base", fill="black")
    for i, c in enumerate(ok_cells, start=1):
        tag = f"{c['variant']}_s{c['seed']}"
        im = Image.open(os.path.join(folders["png"], f"{tag}.png")).resize((cell_px, cell_px))
        sheet.paste(im, (i * cell_px, 0))
        verdict = "TRUE" if c["true_excision"] else ("PRESENT" if c.get("removed") is False else "rej")
        d.text((i * cell_px + 4, cell_px + 4), f"{tag} [{verdict}]", fill="black")
    sheet_path = os.path.join(folders["target"], "bench_sheet.png")
    sheet.save(sheet_path)
    logger.info(f"[{name}] sheet: {sheet_path}")
    return cells


def main():
    p = argparse.ArgumentParser(description="Qwen instruction-layer prompt bench")
    p.add_argument("--spec", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--seed_override", type=int, nargs="*", default=None)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--vlm_model_name", type=str, default="gemini-pro-latest")
    p.add_argument("--bbox_expansion", type=int, default=15)
    p.add_argument("--mask_detection_temperature", type=float, default=0.5)
    p.add_argument("--min_part_px", type=int, default=800)
    p.add_argument("--max_collateral", type=float, default=0.5)
    p.add_argument("--resolution", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    spec = json.load(open(args.spec))
    seeds = args.seed_override or spec.get("seeds", [42])

    cfg = Cfg(seed=seeds[0], num_inference_steps=args.num_inference_steps,
              guidance_scale=args.guidance_scale, vlm_model_name=args.vlm_model_name,
              enable_mask_detection=True, bbox_expansion=args.bbox_expansion,
              mask_detection_temperature=args.mask_detection_temperature,
              min_part_px=args.min_part_px, max_collateral=args.max_collateral,
              resolution=args.resolution, output_folder=args.output_folder)
    logger = setup_logging(args.output_folder)

    fal_stats = {"calls": 0}
    all_cells = []
    for case in spec["cases"]:
        all_cells.extend(run_case(case, seeds, cfg, logger, fal_stats))

    # scorecard: variant -> true_excisions/attempts
    score = {}
    for c in all_cells:
        key = f"{c['case']}:{c['variant']}"
        s = score.setdefault(key, {"true": 0, "attempts": 0})
        s["attempts"] += 1
        s["true"] += int(bool(c.get("true_excision")))
    summary = {"score": score, "cells": all_cells,
               "economics": {"fal_calls": fal_stats["calls"],
                             "fal_cost_est_usd": round(fal_stats["calls"] * 0.04, 2)}}
    out = os.path.join(args.output_folder, "bench_results.json")
    save_json(summary, out)
    logger.info("===== SCORECARD =====")
    for k, s in sorted(score.items(), key=lambda kv: -kv[1]["true"]):
        logger.info(f"  {k}: {s['true']}/{s['attempts']}")
    logger.info(f"fal calls: {fal_stats['calls']} (~${fal_stats['calls'] * 0.04:.2f}). Results: {out}")


if __name__ == "__main__":
    main()
