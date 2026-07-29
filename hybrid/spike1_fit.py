"""Spike 1 — decomposability spectrum: rigid-per-part fit on REAL 2D clips.

Per part: multi-start ECC (template = frame-0 crop, input = frame t masked
by the SAM2 per-frame mask), robust B-SPLINE curve fit (real clips don't
loop — the Fourier basis stays in spike0), one EM re-registration pass
(spike0c: 2x accuracy), then re-render the "rig" (frame-0 parts warped by
the fitted curves, z-order composite over a median background plate) and
score against the original frames.

No GT here, so three instruments, each calibrated on Spike 0:
- ECC cc per part: rigidity detector (rigid >= 0.97; tail-skew 4/8/12 deg
  degraded it 0.97/0.81/0.53).
- fit jitter: RMS px between per-frame ECC transforms and the smooth curve
  at probe points (a part that IS rigid + smooth sits near 0).
- photometric: region-local worst-frame SSIM of the re-render vs original
  (jank floor ~0.89, measured in Spike 0 R_crit calibration).

Report the SPECTRUM per part per clip (spec: not the mean).

Usage: .venv-hybrid/bin/python hybrid/spike1_fit.py [--clip name]
Outputs: out/spike1_clips/<name>/{spike1_report.json, render/, render_sheet.png}
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import BSpline

import sys
HYBRID = Path(__file__).resolve().parent
sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(HYBRID.parent / "local"))
from spike0c_ecc import MOTIONS, decompose  # noqa: E402
from ssim_check import ssim as ssim_fn  # noqa: E402

CLIPS = HYBRID.parent / "out" / "spike1_clips"
MARGIN = 48
MIN_VISIBLE_PX = 150
# SAM2 masks are TIGHT: on flat-fill cartoons the part's entire signal is
# the silhouette outline, which sits just OUTSIDE the mask — undilated,
# ECC starves and diverges (measured: gatin body FAIL -> cc 0.94 at 4px).
ECC_DILATE = 4


# ------------------------------------------------------------------ basis

def bspline_basis(T, K):
    """Clamped uniform cubic B-spline design matrix (T x K)."""
    K = max(K, 4)
    x = np.arange(T, dtype=float)
    interior = np.linspace(0, T - 1, K - 2)[1:-1]
    knots = np.r_[[0.0] * 4, interior, [float(T - 1)] * 4]
    return BSpline.design_matrix(x, knots, 3).toarray()


def robust_spline_fit(thetas, scales, txs, tys, K, ccs, probes, prior):
    """spike0c's IRLS curve fit with the periodic basis swapped for a
    clamped B-spline. cc gates hard failures only; the SAM2 mask area is
    the visibility prior (thin/occluded frames get less say); outlier
    frames are trimmed by position-space residual at probes."""
    T = len(thetas)
    seqs = [np.asarray(x, float) for x in (thetas, scales, txs, tys)]

    def transform_pts(th, sc, tx, ty):
        out = np.empty((T, len(probes), 2))
        for t in range(T):
            R = np.array([[np.cos(th[t]), -np.sin(th[t])],
                          [np.sin(th[t]), np.cos(th[t])]])
            out[t] = (sc[t] * (R @ probes.T)).T + np.array([tx[t], ty[t]])
        return out

    meas = transform_pts(*seqs)
    A = bspline_basis(T, K)
    cc = np.asarray(ccs)
    pw = np.asarray(prior, float)
    w = pw * np.where(cc > 0, np.clip((cc - 0.3) / 0.5, 0.05, 1.0), 0.0)
    fits = seqs
    for it in range(4):
        Aw = A * w[:, None]
        fits = [A @ np.linalg.lstsq(Aw, s * w, rcond=None)[0] for s in seqs]
        e = np.sqrt(((transform_pts(*fits) - meas) ** 2).sum(-1).mean(-1))
        med = np.median(e[w > 1e-3])
        if it == 0:
            w = w * (e < max(4 * med, 2.0))
        else:
            s = max(1.4826 * med, 0.05)
            r = e / (4.685 * s)
            w = pw * np.where(cc > 0, np.clip(1 - r ** 2, 0, 1) ** 2, 0.0)
    jitter = np.sqrt(((transform_pts(*fits) - meas) ** 2).sum(-1).mean(-1))
    return fits, w, jitter


# ------------------------------------------------------------------- ecc

def part_bbox(mask, shape, margin=MARGIN):
    ys, xs = np.nonzero(mask)
    h, w = shape
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(w, xs.max() + 1 + margin), min(h, ys.max() + 1 + margin))


def mask_moment_D(m0, mt):
    """Chain-independent euclidean init from SAM2 mask moments: centroid
    translation + principal-axis rotation (only when elongated enough for
    the axis to mean something; axis has a 180-deg ambiguity — wrapped to
    [-90, 90), fine for the small rotations of part motion)."""
    def moments(m):
        ys, xs = np.nonzero(m)
        c = np.array([xs.mean(), ys.mean()])
        d = np.stack([xs - c[0], ys - c[1]])
        cov = d @ d.T / len(xs)
        ev, evec = np.linalg.eigh(cov)
        elong = ev[1] / max(ev[0], 1e-6)
        ang = np.arctan2(evec[1, 1], evec[0, 1])
        return c, ang, elong
    c0, a0, e0 = moments(m0)
    ct, at, et = moments(mt)
    d = at - a0
    while d > np.pi / 2:
        d -= np.pi
    while d < -np.pi / 2:
        d += np.pi
    if min(e0, et) < 2.5:
        d = 0.0  # near-round part: axis is noise
    R = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
    t = ct - R @ c0
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]],
                     [0, 0, 1]])


def register(grays, tmpl_gray, masks_t, bbox, args, curve_Ds=None):
    x0, y0, x1, y1 = bbox
    tmpl = tmpl_gray[y0:y1, x0:x1]
    m0 = masks_t[0]
    Toff = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
    Toff_i = np.linalg.inv(Toff)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-7)
    W_prev = np.eye(2, 3, dtype=np.float32)
    Ds, ccs = [], []
    for t, g in enumerate(grays):
        mask = (masks_t[t][y0:y1, x0:x1]).astype(np.uint8) * 255
        if (mask > 0).sum() < MIN_VISIBLE_PX:
            Ds.append(Toff @ np.vstack([W_prev, [0, 0, 1]]) @ Toff_i)
            ccs.append(0.0)
            continue
        if curve_Ds is not None:
            cands = [(Toff_i @ curve_Ds[t] @ Toff)[:2].astype(np.float32)]
        else:
            cands = [W_prev, np.eye(2, 3, dtype=np.float32)]
            if m0.sum() > MIN_VISIBLE_PX and masks_t[t].sum() > MIN_VISIBLE_PX:
                Dm = mask_moment_D(m0, masks_t[t])
                cands.append((Toff_i @ Dm @ Toff)[:2].astype(np.float32))
        best_cc, best_W = -1.0, cands[0]
        for Wc in cands:
            try:
                cc, W = cv2.findTransformECC(tmpl, g[y0:y1, x0:x1], Wc.copy(),
                                             MOTIONS[args.motion], crit, mask,
                                             args.gauss)
            except cv2.error:
                continue
            if cc > best_cc:
                best_cc, best_W = cc, W
        if best_cc < 0:
            best_cc = 0.0
        else:
            W_prev = best_W
        Ds.append(Toff @ np.vstack([best_W, [0, 0, 1]]).astype(float) @ Toff_i)
        ccs.append(float(best_cc))
    return Ds, ccs


def compose_D(th, sc, tx, ty):
    c, s = np.cos(th), np.sin(th)
    return np.array([[sc * c, -sc * s, tx], [sc * s, sc * c, ty], [0, 0, 1]])


# ------------------------------------------------------------------- fit

def fit_clip(name, args):
    frames = sorted(glob.glob(str(CLIPS / name / "frames" / "frame_*.png")))
    imgs = [cv2.imread(fp) for fp in frames]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    T = len(imgs)
    z = np.load(CLIPS / name / "part_masks.npz")
    parts = list(z.files)  # insertion order = z-order bottom->top (PROMPTS)
    masks = {p: z[p] for p in parts}
    report, curves = {}, {}
    for part in parts:
        m0 = masks[part][0]
        if m0.sum() < MIN_VISIBLE_PX:
            report[part] = {"skipped": "empty frame-0 mask"}
            continue
        # bbox over the part's UNION across all frames — a frame-0 bbox
        # amputates far-swung parts (superman hand at f30: a third of its
        # mask fell outside the crop and ECC slid 120px)
        bbox = part_bbox(np.any(masks[part], axis=0), grays[0].shape,
                         margin=16)
        # neutralized template: part pixels (plus the outline band) from
        # frame 0, flat median-bg elsewhere — the raw crop's background/
        # neighbor pixels are what made ECC throw on far-swung parts
        # (spike0c's amodal-template lesson, poor-man's edition)
        kern = np.ones((2 * ECC_DILATE + 1,) * 2, np.uint8)
        m0_d = cv2.dilate(m0.astype(np.uint8), kern).astype(bool)
        tmpl = grays[0].copy()
        tmpl[~m0_d] = int(np.median(grays[0][~m0]))
        masks_ecc = np.stack([cv2.dilate(masks[part][t].astype(np.uint8),
                                         kern) for t in range(T)]).astype(bool)
        Ds, ccs = register(grays, tmpl, masks_ecc, bbox, args)
        seqs = list(map(np.array, zip(*[decompose(D) for D in Ds])))
        area = masks[part].sum(axis=(1, 2)).astype(float)
        prior = (area / max(area.max(), 1)) ** 2
        ys, xs = np.nonzero(m0)
        ctr = np.array([xs.mean(), ys.mean()])
        probes = ctr + np.array([[0, 0], [30, 0], [0, 30]])
        K = max(4, T // args.knot_stride)
        fits, _, _ = robust_spline_fit(*seqs, K, ccs, probes, prior)
        # EM pass: re-register from the curve (spike0c: the accuracy step)
        curve_Ds = [compose_D(*[f[t] for f in fits]) for t in range(T)]
        Ds, ccs = register(grays, tmpl, masks_ecc, bbox, args,
                           curve_Ds=curve_Ds)
        seqs = list(map(np.array, zip(*[decompose(D) for D in Ds])))
        fits, w_final, jitter = robust_spline_fit(*seqs, K, ccs, probes, prior)
        cc = np.asarray(ccs)
        report[part] = {
            "cc_min": float(cc[cc > 0].min()) if (cc > 0).any() else 0.0,
            "cc_med": float(np.median(cc[cc > 0])) if (cc > 0).any() else 0.0,
            "cc_p10": float(np.percentile(cc[cc > 0], 10)) if (cc > 0).any() else 0.0,
            "jitter_rms_px": float(np.sqrt((jitter ** 2).mean())),
            "jitter_p95_px": float(np.percentile(jitter, 95)),
            "frames_lost": int((cc == 0).sum()), "frames": T,
            "area_med_px": float(np.median(area)),
        }
        curves[part] = fits
    return imgs, grays, masks, parts, report, curves, T


# -------------------------------------------------------------- integrity
# Mechanical checks BEFORE any perceptual metric (doctrine 2026-07-29:
# "local SSIM 0.63 cannot distinguish a soft fit from a part that flew
# off-screen"). Thresholds calibrated on the first spike-1 run: clean
# parts sit at <= 4px jitter, garbage at 21-2112px — wide gap, 10px cut.
JITTER_MAX_PX = 10.0
IOU_MED_MIN = 0.5
IOU_BAD = 0.3          # per-frame IoU below this counts as a bad frame
IOU_BAD_FRAC = 0.2
INFRAME_MIN = 0.8      # warped-mask area / expected area (off-screen loss)
INFRAME_BAD_FRAC = 0.1
LOST_FRAC_MAX = 0.3
ADJ_BREAK_FRAC = 0.25


def warp_part_masks(masks, curves, T, shape=(512, 512)):
    """Frame-0 mask of each fitted part pushed through its curve."""
    H, W = shape
    out = {}
    for p, fits in curves.items():
        m0 = masks[p][0].astype(np.uint8) * 255
        out[p] = [cv2.warpAffine(
            m0, compose_D(*[c[t] for c in fits])[:2], (W, H)) > 128
            for t in range(T)]
    return out


def integrity(masks, parts, curves, warped, report, T):
    """Per part-frame: warped mask overlaps its SAM2 mask (IoU), lands
    inside frame, centroid tracks the mask centroid, and frame-0
    adjacencies survive the warp. Part verdict: fittable or not, with
    plain-language reasons."""
    kern = np.ones((7, 7), np.uint8)
    d0 = {p: cv2.dilate(masks[p][0].astype(np.uint8), kern).astype(bool)
          for p in curves}
    names = [p for p in parts if p in curves]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if (d0[a] & d0[b]).sum() > 30]
    adj_breaks = {p: 0 for p in curves}
    for t in range(T):
        wd = {p: cv2.dilate(warped[p][t].astype(np.uint8), kern)
              for p in curves}
        for a, b in pairs:
            if not (wd[a].astype(bool) & wd[b].astype(bool)).any():
                adj_breaks[a] += 1
                adj_breaks[b] += 1
    out = {}
    for p in curves:
        area0 = float(masks[p][0].sum())
        sc = curves[p][1]
        ious, inframe, cerr = [], [], []
        for t in range(T):
            w = warped[p][t]
            inframe.append(w.sum() / max(area0 * sc[t] ** 2, 1.0))
            mt = masks[p][t]
            if mt.sum() < MIN_VISIBLE_PX:
                ious.append(np.nan)
                cerr.append(np.nan)
                continue
            ious.append((w & mt).sum() / max((w | mt).sum(), 1))
            ys, xs = np.nonzero(mt)
            cm = np.array([xs.mean(), ys.mean()])
            if w.sum() > 0:
                ys, xs = np.nonzero(w)
                cerr.append(float(np.hypot(
                    *(np.array([xs.mean(), ys.mean()]) - cm))))
            else:
                cerr.append(512.0)
        iou = np.asarray(ious, float)
        inf = np.asarray(inframe, float)
        vis = ~np.isnan(iou)
        r = report[p]
        reasons = []
        if r["jitter_rms_px"] > JITTER_MAX_PX:
            reasons.append(f"jitter {r['jitter_rms_px']:.0f}px")
        if vis.any() and np.nanmedian(iou) < IOU_MED_MIN:
            reasons.append(f"IoU med {np.nanmedian(iou):.2f}")
        if vis.any() and (iou[vis] < IOU_BAD).mean() > IOU_BAD_FRAC:
            reasons.append(
                f"IoU<{IOU_BAD} on {(iou[vis] < IOU_BAD).mean():.0%} frames")
        if (inf < INFRAME_MIN).mean() > INFRAME_BAD_FRAC:
            reasons.append(f"off-frame on {(inf < INFRAME_MIN).mean():.0%}")
        if r["frames_lost"] / T > LOST_FRAC_MAX:
            reasons.append(f"ECC lost {r['frames_lost']}/{T} frames")
        if adj_breaks[p] / T > ADJ_BREAK_FRAC:
            reasons.append(f"detaches on {adj_breaks[p]}/{T} frames")
        out[p] = {
            "fittable": not reasons, "reasons": reasons,
            "iou_med": float(np.nanmedian(iou)) if vis.any() else None,
            "iou_min": float(np.nanmin(iou)) if vis.any() else None,
            "centroid_err_med_px":
                float(np.nanmedian(cerr)) if vis.any() else None,
            "inframe_min": float(inf.min()),
            "adjacency_breaks": int(adj_breaks[p]),
        }
    return out


# --------------------------------------------------------------- rerender

def build_plate(imgs, union):
    """Character-free background plate, TWO passes. Pass 1: per-pixel
    median over frames where the SAM2 union (dilated) is absent. Pass 2:
    pixels where any frame still differs strongly from the pass-1 plate
    are character pixels SAM2 MISSED (the adrock arm-residue bug) — add
    them to the union and re-median. Never-uncovered pixels inpainted."""
    stack = np.stack(imgs).astype(float)

    def median_plate(u):
        ma = np.ma.masked_array(stack, mask=np.repeat(u[..., None], 3, -1))
        pl = np.ma.median(ma, axis=0)
        hole = np.all(pl.mask, axis=-1) if np.ma.is_masked(pl) else \
            np.zeros(pl.shape[:2], bool)
        pl = pl.filled(0).astype(np.uint8)
        if hole.any():
            pl = cv2.inpaint(pl, hole.astype(np.uint8) * 255, 5,
                             cv2.INPAINT_TELEA)
        return pl

    plate = median_plate(union)
    aug = union.copy()
    pf = plate.astype(float)
    for t in range(len(imgs)):
        diff = (np.abs(stack[t] - pf).max(-1) > 40).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        aug[t] |= cv2.dilate(diff, np.ones((5, 5), np.uint8)).astype(bool)
    return median_plate(aug)


def rerender(name, imgs, masks, parts, curves, warped, integ, T):
    """The rig hypothesis made visible: frame-0 parts warped by their
    fitted curves, composited in z-order over a CHARACTER-FREE background
    plate. Region-local worst-frame SSIM per part = the perceptual verdict.

    Integrity-gated (doctrine 2026-07-29): a part whose curve failed its
    own mechanical checks is NOT drawn — the frame is stamped
    'OMITTED: part (reason)' so harness garbage can never masquerade as
    a thesis verdict. Every SSIM is reported next to the clip's own
    adjacent-frame ceiling: the fit can't beat the clip's noise floor."""
    T_ = len(imgs)
    union = np.stack([np.any([masks[p][t] for p in parts], axis=0)
                      for t in range(T_)])
    union = np.stack([cv2.dilate(u.astype(np.uint8),
                                 np.ones((5, 5), np.uint8)) for u in union]
                     ).astype(bool)
    plate = build_plate(imgs, union)
    rdir = CLIPS / name / "render"
    rdir.mkdir(exist_ok=True)
    f0 = imgs[0]
    drawn = [p for p in parts if p in curves and integ[p]["fittable"]]
    omitted = [p for p in curves if not integ[p]["fittable"]]
    stamp = "OMITTED: " + ", ".join(
        f"{p} ({integ[p]['reasons'][0]})" for p in omitted) if omitted else ""
    local = {p: [] for p in drawn}
    glob_ssim, coverage = [], []
    grays_og = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    for t in range(T):
        out = plate.copy()
        wunion = np.zeros((512, 512), bool)
        for p in drawn:
            D = compose_D(*[c[t] for c in curves[p]])[:2]
            wp = cv2.warpAffine(f0, D, (512, 512))
            wm = warped[p][t]
            out[wm] = wp[wm]
            wunion |= wm
        if union[t].any():
            coverage.append(float((union[t] & wunion).sum() / union[t].sum()))
        rg = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        glob_ssim.append(ssim_fn(rg, grays_og[t]))
        for p in local:
            m = masks[p][t] if masks[p][t].sum() > 100 else masks[p][0]
            x0, y0, x1, y1 = part_bbox(m, grays_og[t].shape, margin=16)
            local[p].append(ssim_fn(rg[y0:y1, x0:x1],
                                    grays_og[t][y0:y1, x0:x1]))
        if stamp:
            cv2.putText(out, stamp, (8, 502), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(rdir / f"frame_{t:04d}.png"), out)
    # per-clip ceilings: adjacent-frame SSIM med (global and per part
    # bbox) — the clip's own noise floor; report fits as delta from it
    adj_g, adj_l = [], {p: [] for p in local}
    for t in range(T - 1):
        adj_g.append(ssim_fn(grays_og[t + 1], grays_og[t]))
        for p in local:
            m = masks[p][t] if masks[p][t].sum() > 100 else masks[p][0]
            x0, y0, x1, y1 = part_bbox(m, grays_og[t].shape, margin=16)
            adj_l[p].append(ssim_fn(grays_og[t + 1][y0:y1, x0:x1],
                                    grays_og[t][y0:y1, x0:x1]))
    ceil_g = float(np.median(adj_g))
    bench = {"ssim_mean": float(np.mean(glob_ssim)),
             "ssim_min": float(np.min(glob_ssim)),
             "ceiling_global": ceil_g,
             "coverage_mean": float(np.mean(coverage)),
             "coverage_min": float(np.min(coverage)),
             "omitted": {p: integ[p]["reasons"] for p in omitted},
             "local": {p: {"ssim_min": float(np.min(v)),
                           "ssim_med": float(np.median(v)),
                           "ceiling": float(np.median(adj_l[p])),
                           "worst_frame": int(np.argmin(v))}
                       for p, v in local.items()}}
    # side-by-side sheet, every 6th frame: render | original
    tiles = []
    for t in range(0, T, 6):
        r = cv2.imread(str(rdir / f"frame_{t:04d}.png"))
        pair = np.vstack([cv2.resize(r, (256, 256)),
                          cv2.resize(imgs[t], (256, 256))])
        cv2.putText(pair, str(t), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        tiles.append(pair)
    rows = [np.hstack(tiles[i:i + 7]) for i in range(0, len(tiles), 7)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(CLIPS / name / "render_sheet.png"), np.vstack(rows))
    write_eyeball_artifacts(name, imgs, masks, rdir, drawn, T)
    return bench


def write_eyeball_artifacts(name, imgs, masks, rdir, drawn, T):
    """Claude's appraisal-gate inputs (watched BEFORE Lewis sees anything):
    render|original GIF at full cycle, plus a dense per-part crop sheet —
    EVERY frame, part bbox, 2x nearest-neighbour, render over original."""
    from PIL import Image
    pil = []
    for t in range(T):
        r = cv2.imread(str(rdir / f"frame_{t:04d}.png"))
        pair = np.hstack([r, imgs[t]])
        cv2.putText(pair, f"f{t}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        pil.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(pair, (768, 384)), cv2.COLOR_BGR2RGB)))
    pil[0].save(str(CLIPS / name / "eyeball.gif"), save_all=True,
                append_images=pil[1:], duration=80, loop=0)
    for p in drawn:
        x0, y0, x1, y1 = part_bbox(np.any(masks[p], axis=0),
                                   imgs[0].shape[:2], margin=12)
        cw, ch = x1 - x0, y1 - y0
        s = max(1, int(round(128 / max(cw, ch))))  # >=2x for small parts
        tiles = []
        for t in range(T):
            r = cv2.imread(str(rdir / f"frame_{t:04d}.png"))[y0:y1, x0:x1]
            o = imgs[t][y0:y1, x0:x1]
            pair = np.vstack([r, o])
            pair = cv2.resize(pair, (cw * s, 2 * ch * s),
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(pair, str(t), (2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
            tiles.append(pair)
        per_row = max(1, 2048 // (cw * s))
        rows = [np.hstack(tiles[i:i + per_row])
                for i in range(0, len(tiles), per_row)]
        w = max(r_.shape[1] for r_ in rows)
        rows = [cv2.copyMakeBorder(r_, 0, 0, 0, w - r_.shape[1],
                                   cv2.BORDER_CONSTANT) for r_ in rows]
        cv2.imwrite(str(CLIPS / name / f"crops_{p}.png"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--motion", default="euclidean", choices=MOTIONS)
    ap.add_argument("--gauss", type=int, default=3)
    ap.add_argument("--knot-stride", type=int, default=6,
                    help="frames per spline control point")
    args = ap.parse_args()
    names = [args.clip] if args.clip else \
        sorted(d.name for d in CLIPS.iterdir()
               if (d / "part_masks.npz").exists())
    for name in names:
        imgs, grays, masks, parts, report, curves, T = fit_clip(name, args)
        warped = warp_part_masks(masks, curves, T)
        integ = integrity(masks, parts, curves, warped, report, T)
        bench = rerender(name, imgs, masks, parts, curves, warped, integ, T)
        out = {"fit": report, "integrity": integ, "bench": bench,
               "motion": args.motion}
        (CLIPS / name / "spike1_report.json").write_text(
            json.dumps(out, indent=1))
        print(f"== {name} (T={T})  global SSIM {bench['ssim_mean']:.4f} "
              f"(clip ceiling {bench['ceiling_global']:.4f})")
        for p in parts:
            r = report[p]
            if "skipped" in r:
                print(f"  {p:10s} SKIPPED ({r['skipped']})")
                continue
            ig = integ[p]
            if not ig["fittable"]:
                print(f"  {p:10s} UNFITTABLE — {'; '.join(ig['reasons'])} "
                      f"(cc_med {r['cc_med']:.3f}, centroid err "
                      f"{ig['centroid_err_med_px']:.1f}px)")
                continue
            lb = bench["local"].get(p, {})
            print(f"  {p:10s} cc min/p10/med {r['cc_min']:.3f}/{r['cc_p10']:.3f}"
                  f"/{r['cc_med']:.3f} | jitter {r['jitter_rms_px']:.2f}px | "
                  f"IoU med {ig['iou_med']:.2f} | "
                  f"lSSIM min {lb.get('ssim_min', float('nan')):.3f} "
                  f"med {lb.get('ssim_med', float('nan')):.3f} "
                  f"(ceil {lb.get('ceiling', float('nan')):.3f}) "
                  f"@f{lb.get('worst_frame', -1)}")


if __name__ == "__main__":
    main()
