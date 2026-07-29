"""Spike 1 — clip corpus prep: normalize heterogeneous sources to
out/spike1_clips/<name>/frames/frame_NNNN.png, 512x512, consecutive frames,
single shot (no cuts), character large in frame.

Sources (acquisition log lives in the workorder manifest):
- gatin, adrock: real lottie rigs (lottie-web demo) rendered headless at 512
  — the CUTOUT/PUPPET end, fitted from pixels only (rig never consulted).
- cinderella (Fleischer 1934), superman (Fleischer 1942), casper (1948):
  public-domain hand-drawn, archive.org 640x480 MPEG segments — the ORGANIC
  end (godmother full-figure gesture / talking-head close-up / blob ghost).

Each entry: src glob or dir, 1-based inclusive frame range in the source,
crop box (x0,y0,x1,y1) in source pixels (square-ish; resized to 512).
"""

import glob
import json
from pathlib import Path

import cv2

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
CLIPS = REPO / "out" / "spike1_clips"

SPEC = {
    "gatin": {"src": str(CLIPS / "gatin" / "frames_raw" / "frame_*.png"),
              "range": (1, 80), "crop": (140, 110, 420, 390)},
    # 161-232: upright idle (arms dangle, blinks) — on-domain mascot idle.
    # The higher-energy 37-108 window is a squash-stretch materialize
    # transition, not a performance.
    "adrock": {"src": "/tmp/adrock_all/frame_*.png",
               "range": (161, 232), "crop": (60, 60, 460, 460)},
    # trim to 72: a cut at ~f73 (Betty enters) — caught on the mask sheet
    "cinderella": {"src": str(CLIPS / "cinderella" / "raw" / "f_*.png"),
                   "range": (1, 72), "crop": (80, 0, 560, 480)},
    "superman": {"src": str(CLIPS / "superman" / "raw" / "f_*.png"),
                 "range": (1, 52), "crop": (150, 5, 625, 480)},
    "casper": {"src": str(CLIPS / "casper" / "raw" / "f_*.png"),
               "range": (1, 60), "crop": (110, 0, 590, 480)},
}


def main():
    for name, s in SPEC.items():
        files = sorted(glob.glob(s["src"]))
        if not files:
            print(f"{name}: SOURCE MISSING, skipped")
            continue
        lo, hi = s["range"]
        files = files[lo - 1:hi]
        x0, y0, x1, y1 = s["crop"]
        out = CLIPS / name / "frames"
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(files):
            img = cv2.imread(fp)[y0:y1, x0:x1]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"frame_{i:04d}.png"), img)
        (CLIPS / name / "prep.json").write_text(json.dumps(
            {"n_frames": len(files), **{k: v for k, v in s.items()}}, indent=1))
        print(f"{name}: {len(files)} frames -> {out}")


if __name__ == "__main__":
    main()
