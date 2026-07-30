"""Spike 2 gate pass — mechanical instruments run on a matted run dir.

Born from ledger r001 calibration: Lewis's eyes caught a mid-clip flesh
darkening and an arm fidget that the original gates (loop/identity/
silhouette) missed. These instruments convert those temporal artifacts
into spatial/numeric ones Claude can actually perceive:

- color timeline: per-frame mean L*a*b* + saturation of character pixels,
  plotted as strips; drift shows as a visible ramp, the loop tell as a
  cliff. Metric: max |dL| and |dSat| excursion from frame-0, in Lab units.
- velocity strip: per-frame mean absolute pixel displacement (dense
  optical flow magnitude over the character), plotted; fidget shows as a
  comb, calm motion as smooth humps. Metrics: velocity p95, and "jerk" =
  RMS of frame-to-frame velocity delta (the fidget number).
- loop / identity / silhouette gates from r001, kept.

Writes gates.json + gates.png (the strips, human- and Claude-readable)
into the run dir, prints a one-line summary.

Usage: .venv-hybrid/bin/python hybrid/spike2_gates.py out/spike2/<name>
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HYBRID = Path(__file__).resolve().parent
sys.path.insert(0, str(HYBRID.parent / "local"))
from ssim_check import ssim  # noqa: E402


def character_stats(rgba):
    a = rgba[..., 3] > 128
    px = rgba[..., :3][a]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return a, lab.mean(0), float(hsv[:, 1].mean())


def run(rdir):
    rdir = Path(rdir)
    frames = sorted((rdir / "rgba").glob("frame_*.png"))
    T = len(frames)
    labs, sats, areas, vel, fvel = [], [], [], [0.0], [0.0]
    prev_gray, prev_a = None, None
    for fp in frames:
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        a, lab, sat = character_stats(im)
        labs.append(lab)
        sats.append(sat)
        areas.append(a.sum())
        g = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=-1)
            vel.append(float(mag[a | prev_a].mean()))
            # face region ~ upper-central band of the character bbox
            # (mascot heuristic; born from wave-1 raccoon-jig: dancing
            # body, dead face — Lewis caught it, this gate now does)
            ys, xs = np.nonzero(a)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            w, h = x1 - x0, y1 - y0
            fb = mag[y0 + int(.12 * h):y0 + int(.50 * h),
                     x0 + int(.25 * w):x0 + int(.75 * w)]
            fvel.append(float(fb.mean()) if fb.size else 0.0)
        prev_gray, prev_a = g, a
    labs = np.array(labs)
    sats = np.array(sats)
    vel = np.array(vel)
    fvel = np.array(fvel)
    dL = labs[:, 0] - labs[0, 0]
    dSat = sats - sats[0]
    jerk = float(np.sqrt(np.mean(np.diff(vel) ** 2)))
    # face liveness DURING action: face flow / body flow on the frames
    # where the body is actually moving (above-median velocity)
    active = vel > max(np.median(vel), 0.15)
    face_ratio = float(fvel[active].mean() / max(vel[active].mean(), 1e-6)) \
        if active.any() else 1.0

    f0 = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    fN = cv2.imread(str(frames[-1]), cv2.IMREAD_UNCHANGED)
    g0, gN = [cv2.cvtColor(f[..., :3], cv2.COLOR_BGR2GRAY) for f in (f0, fN)]
    a0, aN = f0[..., 3] > 128, fN[..., 3] > 128
    gates = {
        "frames": T,
        "loop_ssim": float(ssim(g0, gN)),
        "loop_alpha_iou": float((a0 & aN).sum() / (a0 | aN).sum()),
        "color_dL_max": float(np.abs(dL).max()),
        "color_dSat_max": float(np.abs(dSat).max()),
        "color_dL_cliff": float(np.abs(np.diff(dL)).max()),
        "velocity_p95": float(np.percentile(vel, 95)),
        "jerk_rms": jerk,
        "face_vel_p95": float(np.percentile(fvel, 95)),
        "face_body_ratio": face_ratio,
        "silhouette_rel_range": [float(min(areas) / np.median(areas)),
                                 float(max(areas) / np.median(areas))],
    }
    (rdir / "gates.json").write_text(json.dumps(gates, indent=1))

    # strips: three stacked plots as a plain image (no matplotlib dep)
    W, H = max(T * 4, 320), 120
    img = np.full((H * 3 + 40, W, 3), 255, np.uint8)

    def strip(row, series, label, lo, hi, color):
        y0 = row * (H + 13) + 10
        cv2.putText(img, label, (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1)
        pts = []
        for t, v in enumerate(series):
            x = int(t / max(T - 1, 1) * (W - 20)) + 10
            y = y0 + H - 20 - int(np.clip((v - lo) / (hi - lo), 0, 1) * (H - 35))
            pts.append((x, y))
        zero_y = y0 + H - 20 - int(np.clip((0 - lo) / (hi - lo), 0, 1) * (H - 35))
        cv2.line(img, (10, zero_y), (W - 10, zero_y), (200, 200, 200), 1)
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, color, 2)

    m = max(3, np.abs(dL).max() * 1.2)
    strip(0, dL, f"dL* vs f0 (max {np.abs(dL).max():.1f})", -m, m, (180, 60, 30))
    m = max(6, np.abs(dSat).max() * 1.2)
    strip(1, dSat, f"dSat vs f0 (max {np.abs(dSat).max():.1f})", -m, m,
          (30, 120, 180))
    strip(2, vel, f"velocity px/f (p95 {gates['velocity_p95']:.2f}, "
          f"jerk {jerk:.2f})", 0, max(2, vel.max() * 1.1), (60, 160, 60))
    cv2.imwrite(str(rdir / "gates.png"), img)
    print(f"{rdir.name}: loop {gates['loop_ssim']:.3f}/"
          f"{gates['loop_alpha_iou']:.3f} | dL max {gates['color_dL_max']:.1f}"
          f" cliff {gates['color_dL_cliff']:.1f} | dSat {gates['color_dSat_max']:.1f}"
          f" | vel p95 {gates['velocity_p95']:.2f} jerk {jerk:.2f}"
          f" | face/body {gates['face_body_ratio']:.2f}")
    return gates


if __name__ == "__main__":
    run(sys.argv[1])
