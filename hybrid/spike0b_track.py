"""Spike 0 stage B — REAL tracking on synthetic frames with known answers.

CoTracker3 (offline) tracks points through the corpus animation; the fit
machinery from spike0_fit consumes the tracks; residuals are measured
against the exact GT affines. Because stage A established the noise floor
(0.033 px with ideal correspondences), any excess here is TRACKER error in
isolation — the number Spike 1/2 can't give us.

Seeding is gradient- + boundary-weighted (bon B1): uniform seeding inside
flat fills produces trackless points, and the dino is deliberately the
adversarial case (flat teal on flat teal).

Usage: .venv-hybrid/bin/python hybrid/spike0b_track.py [--variant rigid]
       [--pts 200] [--device mps|cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
OUT = REPO / "out" / "corpus_anim"
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(REPO / "local"))
from spike0_fit import Z_ORDER, fourier_smooth, umeyama_similarity  # noqa: E402


def seed_points(mask_rgba, frame0_bgr, n, rng):
    """Gradient- and boundary-weighted seeding inside the part mask."""
    a = (mask_rgba[:, :, 3] > 128).astype(np.uint8)
    grey = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    # boundary band: mask edge (where the silhouette lives)
    boundary = a - cv2.erode(a, np.ones((7, 7), np.uint8))
    w = (grad * a + 40.0 * boundary) * a
    w = w.flatten()
    if w.sum() <= 0:
        ys, xs = np.nonzero(a)
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return np.stack([xs[idx], ys[idx]], 1).astype(float)
    p = w / w.sum()
    idx = rng.choice(len(w), size=n, replace=False, p=p)
    ys, xs = np.unravel_index(idx, a.shape)
    return np.stack([xs, ys], 1).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rigid")
    ap.add_argument("--pts", type=int, default=200)
    ap.add_argument("--device", default="cpu",
                    help="cotracker3 grid ops can be flaky on mps; cpu is safe")
    ap.add_argument("--harmonics", type=int, default=4)
    args = ap.parse_args()
    vdir = OUT / args.variant
    gt = json.loads((vdir / "gt_params.json").read_text())
    T = len(gt)

    frames = [cv2.imread(str(vdir / "frames" / f"frame_{t:04d}.png"))
              for t in range(T)]
    video = torch.from_numpy(
        np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    ).permute(0, 3, 1, 2)[None].float().to(args.device)  # B T C H W, 0-255

    rng = np.random.default_rng(3)
    queries, part_slices, p0_all = [], {}, {}
    off = 0
    for part in Z_ORDER:
        mask = cv2.imread(str(vdir / "masks" / f"{part}.png"),
                          cv2.IMREAD_UNCHANGED)
        p0 = seed_points(mask, frames[0], args.pts, rng)
        p0_all[part] = p0
        part_slices[part] = slice(off, off + len(p0))
        off += len(p0)
        queries.append(np.hstack([np.zeros((len(p0), 1)), p0]))
    q = torch.from_numpy(np.concatenate(queries)).float()[None].to(args.device)

    cache = vdir / "tracks_cache.npz"
    if cache.exists():
        z = np.load(cache)
        tracks, vis = z["tracks"], z["vis"]
        print(f"loaded cached tracks {tracks.shape}")
    else:
        print(f"tracking {off} points over {T} frames ({args.device})...")
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(args.device).eval()
        with torch.no_grad():
            tracks, vis = model(video, queries=q)
        tracks = tracks[0].cpu().numpy()   # T N 2
        vis = vis[0].cpu().numpy()         # T N
        np.savez_compressed(cache, tracks=tracks, vis=vis)

    report = {}
    for part in Z_ORDER:
        sl = part_slices[part]
        p0 = p0_all[part]
        p0h = np.hstack([p0, np.ones((len(p0), 1))])
        # RELATIVE GT (M_t @ inv(M_0)): frame 0 and masks have the t=0
        # transform baked in — scoring vs absolute M_t was a frame-0
        # reference bug (at t=0 the tracker is exact yet the old metric
        # charged arm_right 10.3px). Corrected 2026-07-29.
        M0i = np.linalg.inv(np.array(gt["0"][part]))
        # tracker error vs GT positions (before any fitting)
        terr = []
        for t in range(T):
            gt_pt = ((np.array(gt[str(t)][part]) @ M0i) @ p0h.T).T[:, :2]
            terr.append(np.sqrt(((tracks[t, sl] - gt_pt) ** 2).sum(1)))
        terr = np.array(terr)  # T N
        # fit from tracks (visibility-weighted: drop low-vis points per frame)
        thetas, scales, txs, tys = [], [], [], []
        for t in range(T):
            keep = vis[t, sl] > 0.5
            if keep.sum() < 8:
                keep = np.ones(len(p0), bool)
            src, dst = p0[keep], tracks[t, sl][keep]
            # RANSAC, ABSOLUTE inlier threshold. IRLS with MAD scaling was a
            # measured no-op here: with a majority-bad point set (flat-fill
            # interiors lock to the nearest edge; arm_right median err 8.9px)
            # the robust scale inflates until nothing is rejected. RANSAC
            # finds the largest self-consistent subset regardless of its
            # share; 2px absolute tolerance ~ the tracker's good-point noise.
            rng_r = np.random.default_rng(t)
            best_inl, TOL = None, 2.0
            for _ in range(64):
                ij = rng_r.choice(len(src), 2, replace=False)
                if np.linalg.norm(src[ij[0]] - src[ij[1]]) < 8:
                    continue
                s, R, tr = umeyama_similarity(src[ij], dst[ij])
                r = np.sqrt((((s * (R @ src.T)).T + tr - dst) ** 2).sum(1))
                inl = r < TOL
                if best_inl is None or inl.sum() > best_inl.sum():
                    best_inl = inl
            if best_inl is not None and best_inl.sum() >= 6:
                src, dst = src[best_inl], dst[best_inl]
            s, R, tr = umeyama_similarity(src, dst)
            thetas.append(np.arctan2(R[1, 0], R[0, 0]))
            scales.append(s); txs.append(tr[0]); tys.append(tr[1])
        thetas = fourier_smooth(np.array(thetas), args.harmonics)
        scales = fourier_smooth(np.array(scales), args.harmonics)
        txs = fourier_smooth(np.array(txs), args.harmonics)
        tys = fourier_smooth(np.array(tys), args.harmonics)
        res, rot_err = [], []
        for t in range(T):
            M = np.array(gt[str(t)][part]) @ M0i
            gt_pt = (M @ p0h.T).T[:, :2]
            R = np.array([[np.cos(thetas[t]), -np.sin(thetas[t])],
                          [np.sin(thetas[t]), np.cos(thetas[t])]])
            fit_pt = (scales[t] * (R @ p0.T)).T + np.array([txs[t], tys[t]])
            res.append(float(np.sqrt(((fit_pt - gt_pt) ** 2).sum(1).mean())))
            gt_rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            rot_err.append(abs(np.degrees(thetas[t]) - gt_rot))
        report[part] = {
            "track_err_med_px": float(np.median(terr)),
            "track_err_p95_px": float(np.percentile(terr, 95)),
            "fit_rms_px": float(np.mean(res)),
            "fit_rms_max_px": float(np.max(res)),
            "rot_err_deg_mean": float(np.mean(rot_err)),
        }
        print(f"{part:10s} track med {report[part]['track_err_med_px']:.2f}px "
              f"p95 {report[part]['track_err_p95_px']:.2f}px | "
              f"fit rms {report[part]['fit_rms_px']:.3f}px "
              f"(max {report[part]['fit_rms_max_px']:.3f}) | "
              f"rot err {report[part]['rot_err_deg_mean']:.3f}°")
    (vdir / "track_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
