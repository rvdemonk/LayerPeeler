"""Stage 2.5 — rim-only alpha anti-aliasing, applied after matte and before defringe.

The matte's morphology chain (open 3x3 -> close 5x5 -> largest-component
-> dilate 7x7) produces a near-binary stair-step alpha edge despite the
upstream smoothstep: measured on frog-wave-v7 (frame 81), alpha crosses
0->255 in ~1.5px with 72% solid just inboard and 91% zero just outboard
(CLAUDE.md handoff 2026-08-03-1823 §48-61). The defringe stage kills the
background-contaminated rim RGB, exposing this raw pixel-grid edge.

This stage rebuilds alpha coverage in the rim as a smooth function of
signed distance from the solid (alpha>=128) mask. Gaussian blur was
considered and rejected: an SDF ramp preserves the 128-mask pixel-identical
by construction (smoothstep symmetry proof — pixel centres only ever sit
at |signed| >= 1, so f(+1) > 0.5 and f(-1) < 0.5 for all width>0), which
means every mask-based gate downstream (holes, loop-IoU, silhouette)
sees bit-identical input. A Gaussian at thin features and convex corners
erodes below the threshold; SDF cannot. Additionally the ramp is locally
uniform width — no "fatter" edges where two boundaries are close, no
"skinnier" edges at convex corners. Parameterised by half-width `w`
(px), the width of the linear portion of the ramp on each side of the
contour. Interior alpha >3px from the surface is blended back to the
original — protecting any deep partial-alpha regions (eye whites against
a near-bg background) that exist in the general clip class.

Ordering matters: this runs BEFORE defringe. The new faint outward band
(alpha ~0-60) is caught by defringe's rim mask (a>0), and the bumped
MAX_PROPAGATE_DIST of 6.5 covers the worst-case source distance of
w + SOURCE_DEPTH = ~4.75 + margin. Without the defringe pass, those
low-alpha pixels would carry background-contaminated RGB and re-introduce
the fringe on exactly the band we just created — getting the order wrong
is the halo-class footgun.

Author: OpenCode (kimi-k3)
"""

import cv2
import numpy as np
from PIL import Image


def smoothstep(x):
    """House-style smoothstep: t²(3-2t), symmetric about 0.5.

    Same formula as matte.py, 0-1 clamped.
    """
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def _antialias_one(rgba, width=1.25):
    """Rebuild alpha coverage ramp in the rim of one frame.

    Returns uint8 RGBA array. The 128-mask is provably preserved for
    any width>0 — see module docstring for the proof.
    """
    h, w = rgba.shape[:2]
    a = rgba[..., 3].astype(np.float32)
    solid = (a >= 128).astype(np.uint8)

    d_in = cv2.distanceTransform(solid, cv2.DIST_L2, maskSize=5)
    d_out = cv2.distanceTransform(1 - solid, cv2.DIST_L2, maskSize=5)
    signed = d_in - d_out

    t = (signed + width) / (2 * width)
    f = smoothstep(t) * 255

    ad = np.abs(signed)
    blend = np.clip((3.0 - ad) / 1.0, 0, 1)
    result = rgba.copy()
    new_a = np.rint(f * blend + a * (1 - blend)).clip(0, 255).astype(np.uint8)
    result[..., 3] = new_a

    change = np.sum(new_a != a)
    if change == 0:
        return result, {"changed": 0, "mask_violations": 0,
                        "max_interior_delta": 0}

    violations = np.sum((new_a >= 128) != (solid.astype(bool)))
    interior = (ad > 3.0) & (solid == 1)
    max_int_delta = float(np.max(np.abs(new_a.astype(np.float32)[interior] -
                                        a[interior]))) if interior.any() else 0.0

    return result, {"changed": int(change),
                    "mask_violations": int(violations),
                    "max_interior_delta": max_int_delta}


def antialias_frames(frames, out_dir, width=1.25, log=None):
    """Anti-alias a whole sequence. Returns the written RGBA paths.

    Overwrites in place (same out_dir), matching defringe and matte convention.
    Frames are read and written as PIL RGBA PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    total_changed = 0
    max_violations = 0
    max_int_delta = 0.0
    for n, fp in enumerate(frames):
        im = np.array(Image.open(fp))
        clean, stats = _antialias_one(im, width=width)
        op = out_dir / fp.name
        Image.fromarray(clean).save(op, optimize=True)
        outs.append(op)
        total_changed += stats["changed"]
        max_violations = max(max_violations, stats["mask_violations"])
        max_int_delta = max(max_int_delta, stats["max_interior_delta"])
        if log and n and n % 40 == 0:
            log("    antialias %d/%d" % (n, len(frames)))
    if log:
        log("    antialias: %d px changed, %d mask-violating px (max), "
            "%.1f max interior delta" %
            (total_changed, max_violations, max_int_delta))
    return outs
