"""Appraisal artifacts — the GIF, contact sheet, and edge verification sheet
Claude must watch.

Not optional decoration. Project doctrine (CLAUDE.md, "Appraisal gate"):
nothing goes to Lewis for appraisal without Claude's own animated-sequence
eyeball first, and a plain-language defect list written BEFORE anything is
shown. Numbers do not substitute for watching — Spike 1 shipped renders
whose own metrics said "garbage curve", and Lewis found an exploded tail
and a four-armed character.

The GIF plays at the clip's true fps. Hardcoding it is the half-speed bug
that voided three waves of pacing verdicts; `fps` has no default here for
that reason.

Alpha is composited over a checkerboard rather than white: a matte that
has eaten a limb and a limb that is white both look like white on white.
Edge sheets add light, dark, and magenta backgrounds at 2x nearest-neighbour
zoom — a full-frame sparse check cannot see a 3px doubled outline, and
unsampled frames hide phase-dependent defects.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def _checkerboard(h, w, bg=0):
    """Return a uint8 BGR checkerboard of size (h, w).

    bg=0: light (200/240), bg=1: dark (40/80), bg=2: magenta (128,0,128/255,0,255).
    """
    if bg == 0:
        lo, hi = np.array([200, 200, 200]), np.array([240, 240, 240])
    elif bg == 1:
        lo, hi = np.array([40, 40, 40]), np.array([80, 80, 80])
    else:
        lo, hi = np.array([128, 0, 128]), np.array([255, 0, 255])
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // 16 + xx // 16) % 2).astype(np.uint8)
    board = np.where(mask[..., None], hi[None, None, :].astype(np.float32),
                     lo[None, None, :].astype(np.float32))
    return board


def _composite(rgba, bg=0):
    """Composite RGBA over a checkerboard, return BGR uint8."""
    h, w = rgba.shape[:2]
    board = _checkerboard(h, w, bg)
    a = rgba[..., 3:4].astype(np.float32) / 255
    comp = (rgba[..., :3].astype(np.float32) * a + board * (1 - a))
    return comp.clip(0, 255).astype(np.uint8)


def eyeball(rgba_frames, out_dir, name, fps, sheet_every=4):
    gif = out_dir / ("%s.gif" % name)
    sheet = out_dir / ("%s_sheet.png" % name)
    tiles, pil = [], []
    for i, fp in enumerate(rgba_frames):
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        comp = _composite(im, 0)
        pil.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(comp, (384, 384)), cv2.COLOR_BGR2RGB)))
        if i % sheet_every == 0:
            t = cv2.resize(comp, (192, 192))
            cv2.putText(t, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            tiles.append(t)
    pil[0].save(str(gif), save_all=True, append_images=pil[1:],
                duration=int(1000 / fps), loop=0)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT)
            for r in rows]
    cv2.imwrite(str(sheet), np.vstack(rows))
    return gif, sheet


def _edge_regions(rgba_frames, n_regions=4, crop=96, grid=3):
    """Pick edge regions via bbox grid, sorted by boundary-pixel count.

    Returns list of (cx, cy) centroids — one per selected grid cell.
    """
    im0 = cv2.imread(str(rgba_frames[0]), cv2.IMREAD_UNCHANGED)
    h, w = im0.shape[:2]
    solid = (im0[..., 3] >= 128).astype(np.uint8)
    ys, xs = np.where(solid)
    if len(ys) == 0:
        return [(w // 2, h // 2)] * n_regions
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    dx, dy = (x1 - x0 + 1) / grid, (y1 - y0 + 1) / grid

    boundary = solid.astype(np.int16)
    boundary[1:-1, 1:-1] &= (
        (solid[1:-1, 1:-1] != solid[:-2, 1:-1]) |
        (solid[1:-1, 1:-1] != solid[2:, 1:-1]) |
        (solid[1:-1, 1:-1] != solid[1:-1, :-2]) |
        (solid[1:-1, 1:-1] != solid[1:-1, 2:])
    )

    scores = []
    for gy in range(grid):
        for gx in range(grid):
            cx0 = int(x0 + gx * dx)
            cx1 = int(x0 + (gx + 1) * dx + 1)
            cy0 = int(y0 + gy * dy)
            cy1 = int(y0 + (gy + 1) * dy + 1)
            count = int(boundary[cy0:cy1, cx0:cx1].sum())
            if count > 0:
                # np.where returns (rows, cols) = (y, x). Binding them the
                # other way round transposed every centroid and put the crops
                # off the boundary they were selected for.
                brows, bcols = np.where(boundary[cy0:cy1, cx0:cx1])
                scores.append((count, (cx0 + int(bcols.mean()),
                                       cy0 + int(brows.mean()))))

    scores.sort(reverse=True)
    return [c for _, c in scores[:n_regions]]


def edge_sheet(rgba_frames, out_dir, name,
               sample_every=20, dense_frames=None,
               n_regions=4, crop=96):
    """Generate the edge verification contact sheet.

    Composites `n_regions` edge-region crops (auto-selected by silhouette
    boundary density) at 2x nearest-neighbour zoom on light, dark, and
    magenta checkerboards. The default sweep is even across the FULL cycle
    (every `sample_every` source frames); `dense_frames` is an optional
    iterable of extra source frames to cover densely, for a band the
    operator already suspects. It has no default band: a hardcoded one
    (this was `range(148, 161)`, the frog-wave wink) silently makes every
    other clip's sheet claim coverage it does not have.

    Regions are labelled `region-N (x,y)` from the boundary-density search
    that produced them. They are NOT anatomy — nothing here determines
    which part of a character a crop lands on, and stamping "head"/"arm"
    on a derived centroid asserts a fact the code never established.

    Written to out_dir/verification/; this is the artefact the appraisal
    gate doctrine requires Claude to sweep before showing Lewis.
    """
    vdir = Path(out_dir) / "verification"
    vdir.mkdir(parents=True, exist_ok=True)
    regions = _edge_regions(rgba_frames, n_regions, crop)
    region_labels = ["region-%d (%d,%d)" % (i + 1, cx, cy)
                     for i, (cx, cy) in enumerate(regions)]

    full_set = set(list(range(0, len(rgba_frames), sample_every)) +
                   list(dense_frames or []))
    frame_indices = sorted(f for f in full_set if f < len(rgba_frames))

    bg_names = {0: "light", 1: "dark", 2: "magenta"}
    for bg, bg_label in bg_names.items():
        rows = []
        for fi in frame_indices:
            im = cv2.imread(str(rgba_frames[fi]), cv2.IMREAD_UNCHANGED)
            h, w = im.shape[:2]
            tiles = []
            for (cx, cy), rlabel in zip(regions, region_labels):
                x0 = max(0, cx - crop // 2)
                y0 = max(0, cy - crop // 2)
                x1 = min(w, x0 + crop)
                y1 = min(h, y0 + crop)
                x0 = max(0, x1 - crop)
                y0 = max(0, y1 - crop)
                patch = im[y0:y1, x0:x1]
                comp = _composite(patch, bg)
                zoomed = cv2.resize(comp, (crop * 2, crop * 2),
                                    interpolation=cv2.INTER_NEAREST)
                # Two lines: the derived region label does not fit beside the
                # frame number on a 2x-zoomed `crop`-wide tile.
                cv2.putText(zoomed, "f%d" % fi, (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
                cv2.putText(zoomed, rlabel, (4, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
                tiles.append(zoomed)
            rows.append(np.hstack(tiles))
        sheet = np.vstack(rows)
        cv2.imwrite(str(vdir / ("edges_%s_%s.png" % (name, bg_label))), sheet)

    return vdir
