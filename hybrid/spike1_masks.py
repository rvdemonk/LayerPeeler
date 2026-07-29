"""Spike 1 — per-part masks via SAM2 video propagation (the real stack).

Frame-0 point prompts per part (hand-placed per clip, coordinates in the
512x512 prepped frames), all parts propagated jointly through the clip.
Outputs per clip:
  out/spike1_clips/<name>/part_masks.npz   — bool (T,H,W) per part
  out/spike1_clips/<name>/mask_sheet.png   — overlay contact sheet (eyeball
                                             gate: mask quality is THE
                                             load-bearing question here)

Usage: .venv-hybrid/bin/python hybrid/spike1_masks.py [--clip name]
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
CLIPS = REPO / "out" / "spike1_clips"
CKPT = os.path.expanduser("~/.cache/sam2/sam2.1_hiera_small.pt")

# part -> {pos: [(x,y)...], neg: [(x,y)...]}; z-order listed bottom -> top
PROMPTS = {
    "gatin": {
        "tail": {"pos": [(340, 440), (388, 435)], "neg": [(195, 360), (238, 443)]},
        "body": {"pos": [(195, 350), (215, 395)], "neg": [(195, 190), (340, 440)]},
        "legs": {"pos": [(118, 437), (238, 443)], "neg": [(340, 440), (195, 350)]},
        "head": {"pos": [(195, 190), (230, 230)], "neg": [(195, 360)]},
    },
    "adrock": {
        "leg_left": {"pos": [(205, 470), (197, 495)], "neg": [(255, 380)]},
        "leg_right": {"pos": [(293, 470), (298, 495)], "neg": [(255, 380)]},
        "arm_left": {"pos": [(130, 300), (127, 340)], "neg": [(200, 300)]},
        "arm_right": {"pos": [(390, 290), (405, 315)], "neg": [(310, 300)]},
        # no articulated head: one-piece hood+body suit (the "head" object
        # collapsed to the crown stripe — correct behavior, wrong roster)
        "body": {"pos": [(255, 300), (255, 380), (255, 120)],
                 "neg": [(127, 340), (405, 315), (197, 480), (298, 480)]},
    },
    # Two attempts at arm/torso/skirt separation flooded: SAM2-small cannot
    # split same-color gown fabric (a FINDING for the manifest — on
    # hand-drawn organic content the decomposition fails before the fit).
    # Reduced roster: head+wig / whole gown; gown-as-rigid failing is the
    # honest spectrum point.
    "cinderella": {
        "gown": {"pos": [(300, 450), (100, 320), (420, 340), (275, 330)],
                 "neg": [(265, 200), (185, 430)]},
        "head": {"pos": [(265, 160), (265, 230)],
                 "neg": [(275, 340), (300, 450), (100, 320)]},
    },
    "superman": {
        "torso": {"pos": [(400, 350), (170, 220), (255, 330)],
                  "neg": [(270, 130), (290, 265)]},
        "hand": {"pos": [(290, 265), (245, 280)], "neg": [(270, 130), (400, 350)]},
        "head": {"pos": [(270, 130), (260, 180)], "neg": [(400, 350), (290, 265)]},
    },
    # casper is a head-blob with face acting; no separable limbs on screen.
    # ONE part, deliberately: how far a single rigid transform explains a
    # talking ghost IS the measurement (face misfit -> pivot-B evidence).
    "casper": {
        "ghost": {"pos": [(240, 240), (170, 330), (280, 420)], "neg": []},
    },
}


def run_clip(name, pred):
    frames = sorted(glob.glob(str(CLIPS / name / "frames" / "frame_*.png")))
    jdir = f"/tmp/sam2_{name}"
    shutil.rmtree(jdir, ignore_errors=True)
    os.makedirs(jdir)
    for i, fp in enumerate(frames):
        cv2.imwrite(f"{jdir}/{i:05d}.jpg", cv2.imread(fp),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    state = pred.init_state(video_path=jdir)
    parts = list(PROMPTS[name].keys())
    for oid, part in enumerate(parts, 1):
        p = PROMPTS[name][part]
        pts = np.array(p["pos"] + p.get("neg", []), np.float32)
        lbl = np.array([1] * len(p["pos"]) + [0] * len(p.get("neg", [])),
                       np.int32)
        pred.add_new_points_or_box(state, frame_idx=0, obj_id=oid,
                                   points=pts, labels=lbl)
    T = len(frames)
    masks = {part: np.zeros((T, 512, 512), bool) for part in parts}
    for fidx, obj_ids, logits in pred.propagate_in_video(state):
        for k, oid in enumerate(obj_ids):
            masks[parts[oid - 1]][fidx] = (logits[k, 0] > 0).cpu().numpy()
    np.savez_compressed(CLIPS / name / "part_masks.npz",
                        **{p: masks[p] for p in parts})
    # contact sheet: every 6th frame, colored overlay per part
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
              (255, 0, 255), (255, 255, 0)]
    tiles = []
    for t in range(0, T, 6):
        im = cv2.imread(frames[t])
        ov = im.copy()
        for i, part in enumerate(parts):
            ov[masks[part][t]] = colors[i % len(colors)]
        tile = cv2.addWeighted(im, 0.55, ov, 0.45, 0)
        tile = cv2.resize(tile, (256, 256))
        cv2.putText(tile, str(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(CLIPS / name / "mask_sheet.png"), np.vstack(rows))
    areas = {p: (int(masks[p].sum(axis=(1, 2)).min()),
                 int(masks[p].sum(axis=(1, 2)).max())) for p in parts}
    print(name, "areas min/max per part:", areas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    args = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                      CKPT, device=dev)
    for name in ([args.clip] if args.clip else PROMPTS):
        run_clip(name, pred)


if __name__ == "__main__":
    main()
