"""Spike 0c — ECC whole-part template registration, the promoted primary.

Point tracks were falsified as correspondence backbone (aperture problem on
uniform outlines — spike0b; corrected numbers still >= R_crit on every
moving part). ECC aligns the ENTIRE part appearance jointly, which has no
aperture ambiguity for non-circular parts.

Per part, per frame: findTransformECC(template=frame_t, input=frame_0,
inputMask=visible part pixels). OpenCV's W maps template coords -> input
coords, so the frame0->t motion is D = inv(W). All GT comparisons use the
RELATIVE motion D_gt = M_t @ inv(M_0): masks and frame 0 are rendered WITH
the t=0 transform baked in (sin(phase) != 0), so absolute M_t is the wrong
reference — the frame-0 bias that inflated spike0b's original table.

Visible mask: part alpha minus the alpha of every part above it in z-order,
eroded — occluded pixels belong to the occluder and move differently.

Gate (rigid variant): fit RMS < R_crit (1.0 px) on ALL parts.
Bench: raster-carrier Lottie emit from the fitted curves, headless render,
global SSIM + region-local worst-frame SSIM per part (the doctrine: global
SSIM dilutes local jank by design).

Usage: python3 hybrid/spike0c_ecc.py [--variant rigid] [--motion euclidean]
       [--harmonics 4] [--no-bench]
Outputs: out/corpus_anim/<variant>/{ecc_report.json, lottie/dino_ecc.json}
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
ROOT = REPO.parent
SPIKE = ROOT / "spike"
OUT = REPO / "out" / "corpus_anim"

sys.path.insert(0, str(HYBRID))
sys.path.insert(0, str(REPO / "local"))
from spike0_fit import (  # noqa: E402
    NODE, Z_ORDER, emit_lottie, fourier_smooth, sample_points,
)
from ssim_check import ssim as ssim_fn  # noqa: E402

MOTIONS = {"euclidean": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE}
MARGIN = 48  # crop margin around the part bbox; must exceed max motion (px)


def load_alphas(vdir):
    return {p: cv2.imread(str(vdir / "masks" / f"{p}.png"),
                          cv2.IMREAD_UNCHANGED)[:, :, 3]
            for p in Z_ORDER}


def part_bbox(alpha, shape, margin=MARGIN):
    ys, xs = np.nonzero(alpha > 8)
    h, w = shape
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(w, xs.max() + 1 + margin), min(h, ys.max() + 1 + margin))


def isolated_gray(vdir, part):
    """The part rendered ALONE (masks/<part>.png is premultiplied-on-black
    RGBA), un-premultiplied and composited over white -> grayscale. This is
    the template: the AMODAL part appearance, which the real pipeline gets
    from layer decomposition. The frame-0 composite is the wrong template —
    an 81%-occluded part contributes mostly occluder pixels there."""
    rgba = cv2.imread(str(vdir / "masks" / f"{part}.png"), cv2.IMREAD_UNCHANGED)
    a = rgba[:, :, 3:].astype(float) / 255.0
    rgb = np.where(a > 0, rgba[:, :, :3].astype(float) / np.maximum(a, 1e-3),
                   0).clip(0, 255)
    return cv2.cvtColor((rgb * a + 255.0 * (1 - a)).astype(np.uint8),
                        cv2.COLOR_BGR2GRAY)


def frame_visible_masks(alphas, gt, T, part):
    """Per-frame visible mask for `part`: its alpha warped by the GT motion,
    minus every warped part above it in z-order. This EMULATES SAM2 mask
    propagation — the per-frame masks the designed stack provides as input.
    The masks carry part location (as SAM2's would), not the fitted motion;
    mask quality on real footage is Spike 1's question, not this one's."""
    h, w = alphas[part].shape
    i = Z_ORDER.index(part)
    M0i = {p: np.linalg.inv(np.array(gt["0"][p])) for p in Z_ORDER[i:]}
    out = []
    for t in range(T):
        def warped(p):
            D = (np.array(gt[str(t)][p]) @ M0i[p])[:2].astype(np.float64)
            return cv2.warpAffine(alphas[p], D, (w, h)) > 128
        vis = warped(part)
        for above in Z_ORDER[i + 1:]:
            vis &= ~warped(above)
        # No erosion, no occluder dilation: both measured WORSE. The part's
        # own silhouette edge carries the signal (erosion starves thin
        # slivers), and trimming the occlusion boundary loses more signal
        # than whatever bias it removes (dilation sweep: monotonically bad).
        out.append(vis.astype(np.uint8) * 255)
    return out


MIN_VISIBLE_PX = 200


def register_part(grays, template, vis_masks, bbox, motion, args,
                  track_inits=None, curve_inits=None):
    """Multi-start ECC per frame: template = the isolated (amodal) part,
    input = frame t masked to the part's per-frame VISIBLE pixels. With this
    orientation OpenCV's W is D(frame0 -> frame_t) directly. Candidates:
    previous frame's warp, identity, and (when available) a track-derived
    init — keep the highest correlation. Frames with too few visible pixels
    are skipped as missing data (cc 0) for the weighted curve fit.

    Returns per-frame 3x3 D (full-image coords), correlation coefficients
    (0.0 where skipped/failed), and the failure count.
    """
    x0, y0, x1, y1 = bbox
    tmpl = template[y0:y1, x0:x1]
    Toff = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    Toff_i = np.linalg.inv(Toff)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-7)
    W_prev = np.eye(2, 3, dtype=np.float32)
    Ds, ccs, fails = [], [], 0
    for t, g in enumerate(grays):
        mask = vis_masks[t][y0:y1, x0:x1]
        if (mask > 0).sum() < MIN_VISIBLE_PX:
            fails += 1
            Ds.append(Toff @ np.vstack([W_prev, [0, 0, 1]]) @ Toff_i)
            ccs.append(0.0)
            continue
        if curve_inits is not None:
            # EM refinement pass: single init from the fitted curve. The
            # chained first pass over-rotates thin slivers; starting at the
            # curve's prediction lands ECC in the right basin (measured:
            # arm_left 1.05 -> 0.53 px). One pass only — a second oscillates.
            cands = [(Toff_i @ curve_inits[t] @ Toff)[:2].astype(np.float32)]
        else:
            cands = [W_prev, np.eye(2, 3, dtype=np.float32)]
            if track_inits is not None:
                cands.append((Toff_i @ track_inits[t] @ Toff)[:2]
                             .astype(np.float32))
        best_cc, best_W = -1.0, cands[0]
        for Wc in cands:
            try:
                cc, W = cv2.findTransformECC(tmpl, g[y0:y1, x0:x1], Wc.copy(),
                                             motion, crit, mask, args.gauss)
            except cv2.error:
                continue
            if cc > best_cc:
                best_cc, best_W = cc, W
        if best_cc < 0:
            fails += 1
            best_cc = 0.0  # unrecoverable frame: zero confidence
        else:
            W_prev = best_W
        # W maps template (frame0) coords -> frame_t coords in the crop
        Ds.append(Toff @ np.vstack([best_W, [0, 0, 1]]).astype(np.float64)
                  @ Toff_i)
        ccs.append(float(best_cc))
    return Ds, ccs, fails


def track_init_warps(vdir, gt, T):
    """Per-part per-frame euclidean inits from cached CoTracker tracks
    (tracks demoted to init — the bon B1 design). None if no cache."""
    cache = vdir / "tracks_cache.npz"
    if not cache.exists():
        return None
    from spike0b_track import seed_points  # replicate spike0b seeding exactly
    from spike0_fit import umeyama_similarity
    z = np.load(cache)
    tracks = z["tracks"]
    frame0 = cv2.imread(str(vdir / "frames" / "frame_0000.png"))
    rng = np.random.default_rng(3)
    inits, off = {}, 0
    for part in Z_ORDER:
        mask = cv2.imread(str(vdir / "masks" / f"{part}.png"),
                          cv2.IMREAD_UNCHANGED)
        p0 = seed_points(mask, frame0, 200, rng)
        sl = slice(off, off + len(p0))
        off += len(p0)
        Ds = []
        for t in range(T):
            _, R, tr = umeyama_similarity(p0, tracks[t, sl])  # drop scale
            Ds.append(np.array([[R[0, 0], R[0, 1], tr[0]],
                                [R[1, 0], R[1, 1], tr[1]], [0, 0, 1]]))
        inits[part] = Ds
    return inits


def fourier_basis(T, k):
    om = 2 * np.pi * np.arange(T) / T
    A = np.ones((T, 2 * k + 1))
    for j in range(1, k + 1):
        A[:, 2 * j - 1] = np.cos(j * om)
        A[:, 2 * j] = np.sin(j * om)
    return A


def robust_periodic_fit(thetas, scales, txs, tys, k, ccs, probes, prior=None):
    """IRLS fit of truncated Fourier curves to the per-frame transforms.

    cc is NOT trusted as a weight: a wrong ECC optimum can score high cc
    (the arm_left sliver locking onto the body edge), and low-cc frames can
    carry fine estimates (measured: cc-weighting made arm_right WORSE).
    Only total failures (cc == 0) are excluded a priori. Outlier frames are
    identified by their residual against the fitted curve — measured in
    POSITION space at probe points, jointly across all four params (a wrong
    theta also produces a wildly wrong tx/ty about the image origin) — then
    hard-trimmed and Tukey-downweighted. Majority-good frames + smooth
    periodic motion make this well-posed; occluded frames are interpolated
    by the periodic basis."""
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

    def fit_one(seq, w, kmax):
        """Weighted Fourier fit with leave-one-out model-order selection.
        Occlusion leaves CONTIGUOUS gaps (arm_left: 26 of 48 frames); a
        high-harmonic basis is ill-conditioned there and rings to hundreds
        of px (measured). LOO picks the k the confident frames actually
        support — gappy parts drop to k~1, fully-observed parts keep kmax."""
        best = None
        for kk in range(1, kmax + 1):
            if (w > 1e-3).sum() < 2 * kk + 2:
                break
            A = fourier_basis(T, kk)
            Aw = A * w[:, None]
            G = np.linalg.pinv(Aw.T @ Aw) @ Aw.T
            coef = G @ (seq * w)
            h = np.clip(np.einsum("ij,ji->i", Aw, G), 0, 0.95)
            r = (A @ coef - seq) * w
            loo = float((((r / (1 - h)) ** 2).sum()))
            if best is None or loo < best[0]:
                best = (loo, A @ coef)
        if best is None:
            A = fourier_basis(T, 1)
            Aw = A * w[:, None]
            coef, *_ = np.linalg.lstsq(Aw, seq * w, rcond=None)
            return A @ coef
        return best[1]

    # soft cc prior for the initial fit only: never hard-zero a converged
    # frame (low-cc frames can carry fine estimates — measured on
    # arm_right), but don't let dubious ones dominate the first pass.
    # `prior` (per-frame visible support) persists through every iteration:
    # a thin sliver's estimate is systematically biased (arm_left mid-cycle
    # over-rotates ~35% with cc still 0.997 — confidence does NOT see it),
    # and only the well-observed phases should set the curve.
    cc = np.asarray(ccs)
    pw = np.ones(T) if prior is None else np.asarray(prior, float)
    w = pw * np.where(cc > 0, np.clip((cc - 0.4) / 0.4, 0.05, 1.0), 0.0)
    fits = seqs
    for it in range(4):
        fits = [fit_one(s, w, k) for s in seqs]
        e = np.sqrt(((transform_pts(*fits) - meas) ** 2).sum(-1).mean(-1))
        med = np.median(e[w > 1e-3])
        if it == 0:
            w = w * (e < max(4 * med, 2.0))  # hard trim the garbage first
        else:
            s = max(1.4826 * med, 0.05)
            r = e / (4.685 * s)
            w = pw * np.where(cc > 0, np.clip(1 - r ** 2, 0, 1) ** 2, 0.0)
    return fits, w


def decompose(D):
    theta = np.arctan2(D[1, 0], D[0, 0])
    scale = np.sqrt(max(D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0], 1e-12))
    return theta, scale, D[0, 2], D[1, 2]


def fit_variant(variant, args):
    vdir = OUT / variant
    gt = json.loads((vdir / "gt_params.json").read_text())
    T = len(gt)
    grays = [cv2.cvtColor(cv2.imread(str(vdir / "frames" / f"frame_{t:04d}.png")),
                          cv2.COLOR_BGR2GRAY) for t in range(T)]
    alphas = load_alphas(vdir)
    motion = MOTIONS[args.motion]
    inits = track_init_warps(vdir, gt, T)
    report, curves = {}, {}
    for part in Z_ORDER:
        bbox = part_bbox(alphas[part], grays[0].shape)
        vis_t = frame_visible_masks(alphas, gt, T, part)
        Ds, ccs, fails = register_part(grays, isolated_gray(vdir, part),
                                       vis_t, bbox, motion, args,
                                       inits[part] if inits else None)
        thetas, scales, txs, tys = map(np.array, zip(*[decompose(D) for D in Ds]))
        # relative GT: masks/frame0 have M_0 baked in
        M0i = np.linalg.inv(np.array(gt["0"][part]))
        p0 = sample_points(np.dstack([np.zeros((*alphas[part].shape, 3),
                                                np.uint8), alphas[part]]), 200)
        p0h = np.hstack([p0, np.ones((len(p0), 1))])
        gt_pts, gt_rots = [], []
        for t in range(T):
            D_gt = np.array(gt[str(t)][part]) @ M0i
            gt_pts.append((D_gt @ p0h.T).T[:, :2])
            gt_rots.append(np.degrees(np.arctan2(D_gt[1, 0], D_gt[0, 0])))

        def residuals(th, sc, tx, ty):
            res, rot = [], []
            for t in range(T):
                R = np.array([[np.cos(th[t]), -np.sin(th[t])],
                              [np.sin(th[t]), np.cos(th[t])]])
                fp = (sc[t] * (R @ p0.T)).T + np.array([tx[t], ty[t]])
                res.append(float(np.sqrt(((fp - gt_pts[t]) ** 2).sum(1).mean())))
                rot.append(abs(np.degrees(th[t]) - gt_rots[t]))
            return res, rot

        res_raw, _ = residuals(thetas, scales, txs, tys)
        ys, xs = np.nonzero(alphas[part] > 128)
        ctr = np.array([xs.mean(), ys.mean()])
        probes = ctr + np.array([[0, 0], [30, 0], [0, 30]])
        vis_px = np.array([(m > 0).sum() for m in vis_t], float)
        prior = (vis_px / max(vis_px.max(), 1)) ** 2
        (thetas, scales, txs, tys), _ = robust_periodic_fit(
            thetas, scales, txs, tys, args.harmonics, ccs, probes, prior)
        # EM refinement: re-register every frame initialized from the fitted
        # curve, then refit. One pass (a second oscillates — measured).
        curve_Ds = []
        for t in range(T):
            c, s = np.cos(thetas[t]), np.sin(thetas[t])
            curve_Ds.append(np.array(
                [[scales[t] * c, -scales[t] * s, txs[t]],
                 [scales[t] * s, scales[t] * c, tys[t]], [0, 0, 1]]))
        Ds, ccs, _ = register_part(grays, isolated_gray(vdir, part), vis_t,
                                   bbox, motion, args, curve_inits=curve_Ds)
        thetas, scales, txs, tys = map(np.array,
                                       zip(*[decompose(D) for D in Ds]))
        (thetas, scales, txs, tys), w_final = robust_periodic_fit(
            thetas, scales, txs, tys, args.harmonics, ccs, probes, prior)
        res, rot = residuals(thetas, scales, txs, tys)
        n_conf = int((w_final > 0.5).sum())
        report[part] = {
            "rms_px": float(np.mean(res)), "rms_px_max": float(np.max(res)),
            "rms_raw_unsmoothed": float(np.mean(res_raw)),
            "rot_err_deg_mean": float(np.mean(rot)),
            "rot_err_deg_max": float(np.max(rot)),
            "ecc_cc_min": float(np.min(ccs)), "ecc_fails": fails,
            "frames_confident": n_conf, "frames_total": T,
        }
        curves[part] = {"theta": thetas.tolist(), "scale": scales.tolist(),
                        "tx": txs.tolist(), "ty": tys.tolist()}
    return report, curves, T


def bench_local(variant, lottie_path, T, alphas):
    """Headless render, global SSIM + per-part region-local worst-frame SSIM."""
    vdir = OUT / variant
    rdir = Path(f"/tmp/spike0c_render_{variant}")
    subprocess.run([NODE, "render.js", "lottie", str(lottie_path), str(rdir),
                    str(T)], cwd=SPIKE, check=True, capture_output=True, text=True)
    windows = {p: part_bbox(alphas[p], next(iter(alphas.values())).shape,
                            margin=24) for p in Z_ORDER}
    ssims, local = [], {p: [] for p in Z_ORDER}
    for t in range(T):
        f = cv2.cvtColor(cv2.imread(str(rdir / f"frame_{t:04d}.png")),
                         cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(cv2.imread(str(vdir / "frames" / f"frame_{t:04d}.png")),
                         cv2.COLOR_BGR2GRAY)
        ssims.append(ssim_fn(f, g))
        for p, (x0, y0, x1, y1) in windows.items():
            local[p].append(ssim_fn(f[y0:y1, x0:x1], g[y0:y1, x0:x1]))
    return {
        "ssim_mean": float(np.mean(ssims)), "ssim_min": float(np.min(ssims)),
        "ssim_worst_frame": int(np.argmin(ssims)),
        "local": {p: {"ssim_min": float(np.min(v)),
                      "worst_frame": int(np.argmin(v))}
                  for p, v in local.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="single variant; default: all with gt_params.json")
    ap.add_argument("--motion", default="euclidean", choices=MOTIONS)
    ap.add_argument("--harmonics", type=int, default=4)
    ap.add_argument("--gauss", type=int, default=3)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    variants = ([args.variant] if args.variant else
                sorted(d.name for d in OUT.iterdir()
                       if (d / "gt_params.json").exists()))
    for variant in variants:
        report, curves, T = fit_variant(variant, args)
        out = {"fit": report, "motion": args.motion,
               "harmonics": args.harmonics}
        if not args.no_bench:
            lp = emit_lottie(variant, curves, T, name="dino_ecc.json")
            out["bench"] = bench_local(variant, lp, T,
                                       load_alphas(OUT / variant))
        (OUT / variant / "ecc_report.json").write_text(json.dumps(out, indent=1))
        worst = max(report, key=lambda p: report[p]["rms_px"])
        gate = all(report[p]["rms_px"] < 1.0 for p in report)
        line = (f"{variant}: worst {worst} rms {report[worst]['rms_px']:.3f}px | "
                f"gate(<1.0px all parts): {'PASS' if gate else 'FAIL'}")
        if "bench" in out:
            b = out["bench"]
            wl = min(b["local"], key=lambda p: b["local"][p]["ssim_min"])
            line += (f" | SSIM {b['ssim_mean']:.4f} | worst-local {wl} "
                     f"{b['local'][wl]['ssim_min']:.4f}"
                     f"@f{b['local'][wl]['worst_frame']}")
        print(line)
        for p in Z_ORDER:
            r = report[p]
            print(f"  {p:10s} rms {r['rms_px']:.3f}px (raw {r['rms_raw_unsmoothed']:.3f}, "
                  f"max {r['rms_px_max']:.3f}) | rot {r['rot_err_deg_mean']:.3f}° | "
                  f"cc_min {r['ecc_cc_min']:.4f} fails {r['ecc_fails']}")


if __name__ == "__main__":
    main()
