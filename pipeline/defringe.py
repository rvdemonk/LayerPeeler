"""Stage 2.5 — rim-RGB defringe, applied after matte and before gates.

The matte softens alpha across a colour-distance ramp (smoothstep).
Alpha saturates (a=255) at d≥ALPHA_HI=40, but the gen frame's
anti-aliased edge blends foreground and background colour far past that
point: solid-alpha pixels within ~2px of the silhouette carry
background-mixed RGB. The visible rim is significant on saturated
backgrounds (blue), invisible on pale ones (cream), but exists on ALL
backgrounds — see frog-eye-investigation-2026-08-02 §8.

This stage repaints the rim's RGB from deeper interior colour via
nearest-source lookup. Alpha is NEVER touched — every gate that reads
alpha (holes, loop-IoU) and every byte-size measurement is unchanged.
RGB is only changed for rim pixels, and only when a source exists within
MAX_PROPAGATE_DIST (otherwise the pixel is left as-is — a thin isolated
feature has no clean interior to pull from, and the fallback can't guess).

Author: DeepSeek (with the plan: §8.8-8.9)
"""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Rim mask: partial-alpha band + solid pixels within this many px of the
# silhouette (signed distance ≤ RIM_RADIUS).
RIM_RADIUS = 2.5      # px from solid boundary (both directions)

# Source pixel threshold: only solid pixels at least this far inboard.
SOURCE_DEPTH = 3.0     # px from solid boundary, inward

# Don't pull colour from sources further than this. Rim pixels without a
# nearby source are left unchanged (thin isolated features with no deep
# interior — those typically don't survive the matte anyway, and guessing
# their colour is worse than living with a faint rim).
MAX_PROPAGATE_DIST = 5.0

# Must match matte.py; the solid-alias threshold that gates and encode
# downstream depend on.
ALPHA_SOLID = 128


def _defringe_one(rgba, maskSize=5):
    """Repaint rim RGB in one frame. Returns uint8 BGRA array."""
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3]

    solid = (a >= ALPHA_SOLID).astype(np.uint8)
    d_in = cv2.distanceTransform(solid, cv2.DIST_L2, maskSize)
    d_out = cv2.distanceTransform(1 - solid, cv2.DIST_L2, maskSize)
    signed = d_in - d_out

    rim_mask = (signed <= RIM_RADIUS) & (a > 0)
    source_mask = (solid == 1) & (signed >= SOURCE_DEPTH)

    if source_mask.sum() == 0:
        return rgba

    src_bin = source_mask.astype(np.int32)
    dist_src, idx = distance_transform_edt(1 - src_bin, return_indices=True)

    ry, rx = np.where(rim_mask)
    sy, sx = idx[0][ry, rx], idx[1][ry, rx]
    valid = dist_src[ry, rx] <= MAX_PROPAGATE_DIST
    valid_ry, valid_rx = ry[valid], rx[valid]

    result = rgba.copy()
    result[valid_ry, valid_rx, :3] = rgba[sy[valid], sx[valid], :3]
    return result


def defringe_frames(frames, out_dir, log=None):
    """Defringe a whole sequence. Returns the written RGBA paths.

    Alpha is byte-for-byte identical to the input. Only rim-pixel RGB
    changes, and only when a source colour exists within the propagation
    radius.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for n, fp in enumerate(frames):
        im = np.array(Image.open(fp))
        clean = _defringe_one(im)
        op = out_dir / fp.name
        Image.fromarray(clean).save(op, optimize=True)
        outs.append(op)
        if log and n and n % 40 == 0:
            log("    defringed %d/%d" % (n, len(frames)))
    return outs
