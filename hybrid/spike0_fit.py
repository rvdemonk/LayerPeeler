"""Spike 0b/0c — the fitter, validated against ground truth.

Stage A (0b): sample points in each part's frame-0 mask, push them through
the KNOWN per-frame affines (+ synthetic tracker noise), fit a similarity
per frame (Umeyama closed form), smooth with a truncated Fourier series
(periodic by construction — loop closure is a property of the basis, not a
post-hoc blend; stands in for the B-spline of the full design). PASS: fitted
transform positions within <0.5 px RMS of GT on the rigid variant.

Stage B (0c): for each skew variant, fit the (wrong-by-design) rigid model,
emit a raster-carrier Lottie from the fitted curves, render it headless, and
score SSIM against the GT frames — the residual→perception curve that
defines R_crit.

Usage: python3 spike0_fit.py [--noise 0.5] [--harmonics 4] [--pts 200]
Outputs: out/corpus_anim/<variant>/{fit_report.json, lottie/dino_fit.json}
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
ROOT = REPO.parent
SPIKE = ROOT / "spike"
OUT = REPO / "out" / "corpus_anim"

_NVM = sorted((Path.home() / ".nvm/versions/node").glob("v*/bin/node"))
NODE = str(_NVM[-1]) if _NVM else "node"

import sys
sys.path.insert(0, str(REPO / "local"))
from assemble_lottie import anim, val  # noqa: E402
from ssim_check import ssim as ssim_fn  # noqa: E402

Z_ORDER = ["tail", "arm_left", "legs", "body", "arm_right", "head"]  # bottom->top
PIVOTS = {"tail": (200, 365), "arm_left": (195, 300), "arm_right": (305, 290),
          "head": (256, 258), "body": (252, 330), "legs": (255, 420)}


def sample_points(mask_rgba, n):
    a = mask_rgba[:, :, 3] > 128
    ys, xs = np.nonzero(a)
    idx = np.random.default_rng(7).choice(len(xs), size=min(n, len(xs)),
                                          replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(float)


def umeyama_similarity(src, dst):
    """Closed-form similarity (sR|t) minimizing |dst - (sR src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def fourier_smooth(seq, k):
    """Truncated Fourier reconstruction of a periodic sequence (keep k harmonics)."""
    F = np.fft.rfft(seq)
    F[k + 1:] = 0
    return np.fft.irfft(F, n=len(seq))


def fit_variant(variant, args):
    vdir = OUT / variant
    gt = json.loads((vdir / "gt_params.json").read_text())
    T = len(gt)
    rng = np.random.default_rng(11)
    report, curves = {}, {}
    for part in Z_ORDER:
        mask = cv2.imread(str(vdir / "masks" / f"{part}.png"), cv2.IMREAD_UNCHANGED)
        p0 = sample_points(mask, args.pts)
        p0h = np.hstack([p0, np.ones((len(p0), 1))])
        thetas, scales, txs, tys, res_raw = [], [], [], [], []
        for t in range(T):
            M = np.array(gt[str(t)][part])
            pt = (M @ p0h.T).T[:, :2]
            pt_noisy = pt + rng.normal(0, args.noise, pt.shape)
            s, R, tr = umeyama_similarity(p0, pt_noisy)
            thetas.append(np.arctan2(R[1, 0], R[0, 0]))
            scales.append(s)
            txs.append(tr[0]); tys.append(tr[1])
            fit_pt = (s * (R @ p0.T)).T + tr
            res_raw.append(float(np.sqrt(((fit_pt - pt) ** 2).sum(1).mean())))
        # periodic smoothing — loop closure by construction
        thetas = fourier_smooth(np.array(thetas), args.harmonics)
        scales = fourier_smooth(np.array(scales), args.harmonics)
        txs = fourier_smooth(np.array(txs), args.harmonics)
        tys = fourier_smooth(np.array(tys), args.harmonics)
        # residual of the SMOOTHED similarity vs CLEAN GT positions
        res = []
        for t in range(T):
            M = np.array(gt[str(t)][part])
            pt = (M @ p0h.T).T[:, :2]
            R = np.array([[np.cos(thetas[t]), -np.sin(thetas[t])],
                          [np.sin(thetas[t]), np.cos(thetas[t])]])
            fit_pt = (scales[t] * (R @ p0.T)).T + np.array([txs[t], tys[t]])
            res.append(float(np.sqrt(((fit_pt - pt) ** 2).sum(1).mean())))
        diam = float(np.sqrt(((p0 - p0.mean(0)) ** 2).sum(1)).max() * 2)
        report[part] = {
            "rms_px": float(np.mean(res)), "rms_px_max": float(np.max(res)),
            "rms_raw_unsmoothed": float(np.mean(res_raw)),
            "rms_norm": float(np.mean(res) / diam),
            "loop_gap_px": float(abs(res[0] - res[-1])),
        }
        curves[part] = {"theta": thetas.tolist(), "scale": scales.tolist(),
                        "tx": txs.tolist(), "ty": tys.tolist()}
    return report, curves, T


# ------------------------------------------------------------- lottie emit

def dense_anim(values, dur):
    """Dense per-frame keyframes via the proven anim() helper — bare {t,s}
    keys without easing objects silently break lottie-web's property builder
    (first debug render: five of six layers vanished)."""
    return anim(list(enumerate(values)) + [(dur, values[0])])


def emit_lottie(variant, curves, T):
    vdir = OUT / variant
    assets, layers = [], []
    import base64, io
    from PIL import Image
    for i, part in enumerate(reversed(Z_ORDER)):  # top first in layer list
        rgba_pm = cv2.imread(str(vdir / "masks" / f"{part}.png"), cv2.IMREAD_UNCHANGED)
        a = rgba_pm[:, :, 3:].astype(float)
        rgb = np.where(a > 0, (rgba_pm[:, :, :3].astype(float) * 255 /
                               np.maximum(a, 1)).clip(0, 255), 0)
        rgba = np.dstack([cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB),
                          rgba_pm[:, :, 3]])
        ys, xs = np.nonzero(rgba[:, :, 3] > 8)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        buf = io.BytesIO()
        Image.fromarray(rgba[y0:y1, x0:x1]).save(buf, "PNG", optimize=True)
        assets.append({"id": f"img_{part}", "w": int(x1 - x0), "h": int(y1 - y0),
                       "u": "", "e": 1,
                       "p": "data:image/png;base64," +
                            base64.b64encode(buf.getvalue()).decode()})
        c = curves[part]
        px, py = PIVOTS[part]
        pos = []
        for t in range(T):
            R = np.array([[np.cos(c["theta"][t]), -np.sin(c["theta"][t])],
                          [np.sin(c["theta"][t]), np.cos(c["theta"][t])]])
            p = c["scale"][t] * (R @ np.array([px, py])) + np.array([c["tx"][t], c["ty"][t]])
            pos.append([round(float(p[0]), 2), round(float(p[1]), 2), 0])
        layers.append({
            "ddd": 0, "ind": i + 1, "ty": 2, "nm": part, "refId": f"img_{part}",
            "ks": {"a": val([px - int(x0), py - int(y0), 0]),
                   "p": dense_anim(pos, T),
                   "r": dense_anim([round(float(np.degrees(th)), 3)
                                    for th in c["theta"]], T),
                   "s": dense_anim([[round(float(s) * 100, 3)] * 2 + [100]
                                    for s in c["scale"]], T),
                   "o": val(100)},
            "ip": 0, "op": T, "st": 0, "sr": 1})
    doc = {"v": "5.9.0", "fr": 24, "ip": 0, "op": T, "w": 512, "h": 512,
           "nm": f"dino-fit-{variant}", "ddd": 0, "assets": assets,
           "layers": layers}
    ldir = vdir / "lottie"
    ldir.mkdir(exist_ok=True)
    p = ldir / "dino_fit.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    return p


def bench(variant, lottie_path, T):
    vdir = OUT / variant
    rdir = Path(f"/tmp/spike0_render_{variant}")
    subprocess.run([NODE, "render.js", "lottie", str(lottie_path), str(rdir),
                    str(T)], cwd=SPIKE, check=True, capture_output=True, text=True)
    ssims = []
    for t in range(T):
        f = cv2.imread(str(rdir / f"frame_{t:04d}.png"))
        g = cv2.imread(str(vdir / "frames" / f"frame_{t:04d}.png"))
        ssims.append(ssim_fn(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)))
    return {"ssim_mean": float(np.mean(ssims)), "ssim_min": float(np.min(ssims)),
            "ssim_worst_frame": int(np.argmin(ssims))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--harmonics", type=int, default=4)
    ap.add_argument("--pts", type=int, default=200)
    args = ap.parse_args()
    variants = sorted(d.name for d in OUT.iterdir() if (d / "gt_params.json").exists())
    for variant in variants:
        report, curves, T = fit_variant(variant, args)
        lp = emit_lottie(variant, curves, T)
        b = bench(variant, lp, T)
        worst = max(report, key=lambda p: report[p]["rms_px"])
        out = {"fit": report, "bench": b, "noise": args.noise,
               "harmonics": args.harmonics}
        (OUT / variant / "fit_report.json").write_text(json.dumps(out, indent=1))
        print(f"{variant}: worst part {worst} rms {report[worst]['rms_px']:.3f}px "
              f"(norm {report[worst]['rms_norm']:.4f}) | "
              f"SSIM mean {b['ssim_mean']:.4f} min {b['ssim_min']:.4f} "
              f"@f{b['ssim_worst_frame']}")


if __name__ == "__main__":
    main()
