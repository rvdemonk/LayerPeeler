#!/usr/bin/env python3
"""Frame-locked fidelity comparison: render the same animation frame in the
lossless-PNG control and each candidate rung, then diff the play area.

Liveness (verify.py) only proves something moved. This proves the webp rungs
render the SAME picture as the control, frame for frame.
"""

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = "/Users/lewis/research/image-to-lottie/LayerPeeler/ios-sandbox"
ART = os.path.join(ROOT, "artifacts", "frames")
BID = "com.layerpeeler.lottiesandbox"
SIM = open(os.path.join(ROOT, ".simid")).read().strip()
RECT = (123, 606, 1083, 1566)

FRAMES = [0, 20, 40, 60, 80, 100, 120, 140, 160]
SUBJECTS = ["raccoon-wave", "strawberry-wave"]
RUNGS = ["512", "512q", "512webp", "256webp"]


def capture(variant, frame, path):
    if os.path.exists(path):
        return
    subprocess.run(["xcrun", "simctl", "terminate", SIM, BID],
                   capture_output=True, text=True)
    subprocess.run(["xcrun", "simctl", "launch", SIM, BID,
                    "-variant", variant, "-frame", str(frame)],
                   capture_output=True, text=True, check=True)
    # The app renders asynchronously; poll until the play area stops changing.
    prev, stable = None, 0
    for _ in range(80):
        time.sleep(0.25)
        subprocess.run(["xcrun", "simctl", "io", SIM, "screenshot", path],
                       capture_output=True, text=True)
        cur = crop(path)
        if prev is not None and np.array_equal(cur, prev):
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        prev = cur
    print(f"WARN unstable render: {variant} f{frame}", file=sys.stderr)


def crop(path):
    a = np.array(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = RECT
    return a[y0:y1, x0:x1]


def ssim(a, b):
    """Global grayscale SSIM. Reported alongside worst-window SSIM, which is the
    number that actually catches local artifacts."""
    a = a.astype(float).mean(axis=2)
    b = b.astype(float).mean(axis=2)
    return _ssim_pair(a, b)


def _ssim_pair(a, b):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma**2 + mb**2 + C1) * (va + vb + C2))


def worst_window_ssim(a, b, win=64):
    ga = a.astype(float).mean(axis=2)
    gb = b.astype(float).mean(axis=2)
    h, w = ga.shape
    worst, at = 1.0, None
    for y in range(0, h - win + 1, win // 2):
        for x in range(0, w - win + 1, win // 2):
            wa, wb = ga[y:y + win, x:x + win], gb[y:y + win, x:x + win]
            if wa.std() < 2 and wb.std() < 2:
                continue  # flat background window; SSIM is meaningless there
            s = _ssim_pair(wa, wb)
            if s < worst:
                worst, at = s, (x, y)
    return worst, at


def main():
    os.makedirs(ART, exist_ok=True)
    rows = []
    for subj in SUBJECTS:
        for f in FRAMES:
            paths = {}
            for rung in RUNGS:
                p = os.path.join(ART, f"{subj}.{rung}.f{f:03d}.png")
                capture(f"{subj}.{rung}", f, p)
                paths[rung] = p
            ctrl = crop(paths["512"])
            for rung in RUNGS[1:]:
                cand = crop(paths[rung])
                d = np.abs(ctrl.astype(int) - cand.astype(int))
                ww, at = worst_window_ssim(ctrl, cand)
                rows.append({
                    "subject": subj, "frame": f, "rung": rung,
                    "mean_abs_diff": round(float(d.mean()), 3),
                    "max_abs_diff": int(d.max()),
                    "pct_px_diff_gt16": round(float((d.max(axis=2) > 16).mean() * 100), 3),
                    "global_ssim": round(float(ssim(ctrl, cand)), 5),
                    "worst_window_ssim": round(float(ww), 4),
                    "worst_window_xy": at,
                })
                print(json.dumps(rows[-1]))
    json.dump(rows, open(os.path.join(ROOT, "artifacts", "framecmp.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
