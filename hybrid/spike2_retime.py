"""Post-stage retiming — the answer to wave-2's "gravity/pacing" verdict
(Lewis: motions "stuck in sand... slow loamy claymation"; global speed-up
would be "vulgar and unrefined").

Mechanism: arc-length reparameterization of the frame sequence. Per-frame
character motion (Farneback flow, character-masked) gives a cumulative
motion curve s(t). Uniform resampling in s instead of t makes motion-per-
output-frame constant: dead holds compress automatically (the too-long
rest gap dies as a side effect), action beats keep their frames. The
--strength dial blends between original timing (0) and full equalization
(1) — full equalization kills anticipation holds, which are LEGITIMATE
zero-motion, so default 0.65 keeps a breath of them. --max-hold caps any
surviving zero-motion run.

Output: <run>/retimed.gif (+ retime.json with the chosen source frames)
so Lewis can A/B against the original GIF in the sandbox.

Usage: spike2_retime.py out/spike2/<name> [--fps 16] [--frames 56]
       [--strength 0.65] [--max-hold 5]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def motion_curve(frames):
    vel = [0.0]
    prev, preva = None, None
    for fp in frames:
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        a = im[..., 3] > 128
        g = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vel.append(float(np.linalg.norm(flow, axis=-1)[a | preva].mean()))
        prev, preva = g, a
    return np.array(vel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--strength", type=float, default=0.65)
    ap.add_argument("--max-hold", type=int, default=5)
    args = ap.parse_args()
    rdir = Path(args.rundir)
    frames = sorted((rdir / "rgba").glob("frame_*.png"))
    T = len(frames)
    vel = motion_curve(frames)
    # cumulative arc length, with a small epsilon so pure holds still
    # advance (keeps SOME dwell); blend against uniform time by strength
    arc = np.cumsum(vel + 1e-3)
    arc = arc / arc[-1]
    tt = np.arange(T) / (T - 1)
    s = (1 - args.strength) * tt + args.strength * arc
    targets = np.linspace(0, 1, args.frames)
    picks = [int(np.searchsorted(s, x)) for x in targets]
    picks = [min(p, T - 1) for p in picks]
    # cap zero-motion runs (repeated identical picks = surviving hold)
    out, run = [], 0
    for i, p in enumerate(picks):
        if i and p == out[-1]:
            run += 1
            if run >= args.max_hold:
                continue
        else:
            run = 0
        out.append(p)
    from PIL import Image
    pil = []
    for p in out:
        im = cv2.imread(str(frames[p]), cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        checker = (((yy // 16 + xx // 16) % 2) * 40 + 200)[..., None]
        checker = np.repeat(checker, 3, -1).astype(np.float32)
        a = im[..., 3:4].astype(np.float32) / 255
        comp = (im[..., :3] * a + checker * (1 - a)).astype(np.uint8)
        pil.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(comp, (384, 384)), cv2.COLOR_BGR2RGB)))
    pil[0].save(str(rdir / "retimed.gif"), save_all=True,
                append_images=pil[1:], duration=int(1000 / args.fps), loop=0)
    (rdir / "retime.json").write_text(json.dumps(
        {"source_frames": out, "fps": args.fps,
         "strength": args.strength}, indent=1))
    print(f"{rdir.name}: {T} -> {len(out)} frames @ {args.fps}fps "
          f"(strength {args.strength}) -> retimed.gif")


if __name__ == "__main__":
    main()
