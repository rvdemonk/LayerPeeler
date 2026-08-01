#!/usr/bin/env python3
"""Mechanical verification harness for the lottie-ios frame-sequence smoke test.

Per variant: launch the app straight into the player (via the -variant launch
argument), wait for the app's own LOTTIE_METRIC log line, capture three
screenshots spaced across the animation, record a short video, then measure the
play area. Pixel counts are the verdict; the absence of an error is not.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = "/Users/lewis/research/image-to-lottie/LayerPeeler/ios-sandbox"
ART = os.path.join(ROOT, "artifacts")
BID = "com.layerpeeler.lottiesandbox"
SIM = open(os.path.join(ROOT, ".simid")).read().strip()

# Play area, verified against the control screenshot: a 320x320pt square at 3x.
RECT = (123, 606, 1083, 1566)  # x0, y0, x1, y1
TOL = 8  # per-channel distance from pure white that counts as "content"

VARIANTS = [
    "raccoon-wave.512",
    "raccoon-wave.512q",
    "raccoon-wave.512webp",
    "raccoon-wave.256webp",
    "strawberry-wave.512",
    "strawberry-wave.512q",
    "strawberry-wave.512webp",
    "strawberry-wave.256webp",
]


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def crop(path):
    a = np.array(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = RECT
    return a[y0:y1, x0:x1]


def rect_aligned(path):
    """The ring just outside the play area must still be the 0.92 grey chrome."""
    a = np.array(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = RECT
    ring = [a[y0 - 4, x0 + 5], a[y0 - 4, x1 - 5], a[y0 + 50, x0 - 4], a[y0 + 50, x1 + 3]]
    return all(abs(int(p[0]) - 235) <= 2 for p in ring)


def measure(arr):
    content = (np.abs(arr.astype(int) - 255) > TOL).any(axis=2)
    n = int(content.sum())
    mean = arr[content].mean(axis=0).tolist() if n else [0.0, 0.0, 0.0]
    return n, mean


def diff_px(a, b):
    return int((np.abs(a.astype(int) - b.astype(int)) > TOL).any(axis=2).sum())


def run_variant(v):
    out = {"variant": v}
    sh("xcrun", "simctl", "terminate", SIM, BID)
    time.sleep(1)

    logpath = os.path.join(ART, f"{v}.console.log")
    logf = open(logpath, "w")
    t_launch = time.time()
    proc = subprocess.Popen(
        ["xcrun", "simctl", "launch", "--console-pty", SIM, BID, "-variant", v],
        stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    )

    # Wait for the app to report that it drew its first frame.
    metric, deadline = None, time.time() + 120
    while time.time() < deadline:
        txt = open(logpath, errors="ignore").read()
        m = re.search(r"LOTTIE_METRIC (.+)", txt)
        if m:
            metric = m.group(1).strip()
            break
        time.sleep(0.25)
    out["wall_launch_to_firstframe_s"] = round(time.time() - t_launch, 2)
    out["metric_line"] = metric

    if metric:
        for k in ("parseMs", "firstFrameMs", "duration", "fps"):
            mm = re.search(rf"{k}=([\d.]+)", metric)
            if mm:
                out[k] = float(mm.group(1))
        am = re.search(r"asset=(.*)$", metric)
        out["assetProbe"] = am.group(1).strip() if am else None

    # Three shots spaced ~0.55s apart: at 32fps that is ~18 frames of motion.
    shots = []
    time.sleep(1.0)
    for i in range(3):
        p = os.path.join(ART, f"{v}.shot{i}.png")
        sh("xcrun", "simctl", "io", SIM, "screenshot", p)
        shots.append(p)
        time.sleep(0.55)

    out["rect_aligned"] = rect_aligned(shots[0])
    arrs = [crop(p) for p in shots]
    counts, means = zip(*(measure(a) for a in arrs))
    out["content_px"] = list(counts)
    out["mean_rgb"] = [[round(c, 1) for c in m] for m in means]
    out["interframe_diff_px"] = [diff_px(arrs[0], arrs[1]), diff_px(arrs[1], arrs[2]),
                                 diff_px(arrs[0], arrs[2])]

    # 5s video for human eyeballs.
    vid = os.path.join(ART, f"{v}.mov")
    if os.path.exists(vid):
        os.remove(vid)
    rec = subprocess.Popen(["xcrun", "simctl", "io", SIM, "recordVideo", "--codec=h264", vid],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=os.setsid)
    time.sleep(5.5)
    os.killpg(os.getpgid(rec.pid), signal.SIGINT)
    rec.wait(timeout=30)
    out["video"] = vid if os.path.exists(vid) else None

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    logf.close()
    return out


def main():
    targets = sys.argv[1:] or VARIANTS
    results = [run_variant(v) for v in targets]
    path = os.path.join(ART, "results.json")
    prev = {}
    if os.path.exists(path):
        prev = {r["variant"]: r for r in json.load(open(path))}
    for r in results:
        prev[r["variant"]] = r
    ordered = [prev[v] for v in VARIANTS if v in prev]
    json.dump(ordered, open(path, "w"), indent=2)
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
