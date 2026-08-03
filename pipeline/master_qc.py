"""Pre-spend master QC — verify the matte's one load-bearing assumption.

The matte (pipeline/matte.py) keys on RGB distance from the border-median
background: d<=12 is transparent, d>=40 is character, and below ~26 a
pixel falls under alpha 128 — counted as background downstream. Every
defect in the frog eye saga (r037-r042) was this assumption breaking:
eye whites at d~13 from a cream background were eaten in three runs, and
bg-coloured speckle enclosed by character survived as holes in another.
Generation costs money; this check costs nothing. It runs on the master
BEFORE Wan sees it, and it is also the intake check for user-supplied
masters, whose backgrounds we do not control.

One census, one screening band, both coordinate-free:

  LOSS MASS (fail). Pixels guaranteed to lose alpha: enclosed bg-coloured
  pixels (d < BG_D, not border-connected — re-key speckle or art that
  happens to match the background) PLUS character-interior pixels at
  d < HARD_D (alpha < ~75 at the matte). "Interior" means INTERIOR_MARGIN
  px clear of border-connected background, so the anti-aliased silhouette
  ring (legitimately mid-distance) is excluded. Symmetric rule, light and
  dark alike: pale eyes on cream and dark pupils on charcoal are the same
  failure. Genuine background POCKETS (an armpit donut) are carved out:
  an enclosed component whose p90 distance is under POCKET_D is flat
  background shade, correctly transparent in the first place.

  FEATHER BAND (flag). Interior pixels at d in [HARD_D, matte ALPHA_HI):
  they keep some alpha but have no margin for Wan's master-to-render
  colour drift (measured a few units on the corpus).

Known blind spot, by construction: features thinner than ~2x
INTERIOR_MARGIN have no interior and are never distance-checked. A thin
arm near bg colour passes silently. The post-matte holes gate
(gates.holes) is the backstop for it.

CLI (also the product-intake shape):
    .venv-hybrid/bin/python -m pipeline.master_qc master.png [--map out.png]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BORDER_PX = 8           # same convention as matte.py
BG_D = 10.0             # below this a pixel IS background-coloured
POCKET_D = 5.0          # enclosed component p90 below this = genuine bg pocket
HARD_D = 20.0           # alpha < ~75 at the matte: guaranteed lost
FEATHER_D = 40.0        # matte ALPHA_HI: below this, alpha < 255
INTERIOR_MARGIN = 4     # px from border-connected bg that define "interior"
SPECKLE_MIN_PX = 8      # enclosed components smaller than this are noise
# Thresholds at 1024x1024, scaled linearly by area. Calibration set
# 2026-08-02, nine corpus masters against ledger verdicts:
#   FAIL side:  concept-03 ~10.7k loss px, blob ~7.5k (loop IoU 0.804)
#   PASS side:  strawberry ~1.05k, star ~0.2k, raccoon ~1px, frog.jpg ~30px
# The gap between strawberry (1054) and blob (7500) holds the line.
LOSS_FAIL_PX = 2000
FEATHER_FLAG_PX = 5000  # raccoon 4857 known-good max, concept-03 5904


def _load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA"):
        im = im.convert("RGBA")
        a = np.asarray(im)
        if a[..., 3].min() < 250:
            # Flatten transparent masters onto their own border colour,
            # not black: distance-to-bg on a black fill would read every
            # transparent pixel as maximally distant and pass blind.
            op = a[a[..., 3] > 250][:, :3]
            fill = np.median(op, axis=0) if len(op) else np.array([255] * 3)
            a[a[..., 3] <= 250, :3] = fill
        return a[..., :3].astype(np.float32)
    return np.asarray(im.convert("RGB")).astype(np.float32)


def _bbox(mask):
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def qc_master(path, map_path=None):
    """Check one master image. Returns a gates.py-shaped record.

    verdict semantics follow gates.py: `fails` = will ship broken,
    `flags` = screening signal for a human.
    """
    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    scale = (h * w) / (1024.0 * 1024.0)
    border = np.concatenate([rgb[:BORDER_PX].reshape(-1, 3),
                             rgb[-BORDER_PX:].reshape(-1, 3),
                             rgb[:, :BORDER_PX].reshape(-1, 3),
                             rgb[:, -BORDER_PX:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(rgb - bg, axis=-1).astype(np.float32)

    is_bg = d < BG_D
    ncc, cc, stats, _ = cv2.connectedComponentsWithStats(is_bg.astype(np.uint8))
    border_ids = set(np.unique(np.concatenate(
        [cc[0], cc[-1], cc[:, 0], cc[:, -1]]))) - {0}
    border_bg = np.isin(cc, list(border_ids)) if border_ids else np.zeros_like(is_bg)

    # enclosed bg-coloured components: speckle (counted) vs genuine flat
    # background pockets (p90 < POCKET_D, carved out)
    speckle = np.zeros_like(is_bg)
    pockets = np.zeros_like(is_bg)
    speckle_regions = []
    for i in range(1, ncc):
        if i in border_ids:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < SPECKLE_MIN_PX:
            continue
        comp = cc == i
        if np.percentile(d[comp], 90) < POCKET_D:
            pockets |= comp
            continue
        speckle |= comp
        x, y, cw, ch = (int(stats[i, k]) for k in
                        (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                         cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        speckle_regions.append({"bbox": [x, y, x + cw, y + ch], "px": area})
    speckle_regions.sort(key=lambda r: -r["px"])

    character = ~is_bg
    near_bg = cv2.dilate(border_bg.astype(np.uint8),
                         np.ones((2 * INTERIOR_MARGIN + 1,) * 2, np.uint8)).astype(bool)
    interior = character & ~near_bg

    hard = interior & (d < HARD_D)
    feather = interior & (d >= HARD_D) & (d < FEATHER_D)
    hard_px, feather_px = int(hard.sum()), int(feather.sum())
    speckle_px = int(speckle.sum())
    loss_px = hard_px + speckle_px

    fails, flags = [], []
    if loss_px > LOSS_FAIL_PX * scale:
        fails.append("master_loss_mass: %d px guaranteed transparent at the "
                     "matte (%d interior d<%.0f + %d enclosed bg-coloured; "
                     "bg=%s, bbox %s) — fix background/feature distance "
                     "before spending" % (loss_px, hard_px, HARD_D, speckle_px,
                                          tuple(int(x) for x in bg),
                                          _bbox(hard | speckle)))
    if feather_px > FEATHER_FLAG_PX * scale:
        flags.append("master_feather_band: %d interior px in d[%.0f,%.0f) of "
                     "background — keeps alpha, no margin for Wan drift"
                     % (feather_px, HARD_D, FEATHER_D))

    if map_path:
        vis = np.full((h, w, 3), 160, np.uint8)        # background: grey
        vis[character] = (90, 200, 90)                  # character: green
        vis[pockets] = (120, 120, 120)                  # genuine pocket: dk grey
        vis[feather] = (60, 200, 240)                   # feather zone: amber
        vis[hard] = (60, 60, 240)                       # guaranteed loss: red
        vis[speckle] = (240, 120, 40)                   # enclosed speckle: blue
        Image.fromarray(vis[..., ::-1]).save(map_path)

    return {"bg": [int(x) for x in bg], "character_px": int(character.sum()),
            "interior_px": int(interior.sum()),
            "loss_px": loss_px, "hard_px": hard_px, "hard_bbox": _bbox(hard),
            "speckle_px": speckle_px, "speckle_regions": speckle_regions[:5],
            "pocket_px": int(pockets.sum()),
            "feather_px": feather_px, "feather_bbox": _bbox(feather),
            "fails": fails, "flags": flags}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="pre-spend master QC")
    ap.add_argument("image")
    ap.add_argument("--map", dest="map_path", help="write a verdict map PNG")
    args = ap.parse_args(argv)
    rec = qc_master(args.image, args.map_path)
    verdict = "FAIL" if rec["fails"] else ("FLAG" if rec["flags"] else "PASS")
    print("%s  bg=%s  character=%dpx  interior=%dpx"
          % (verdict, rec["bg"], rec["character_px"], rec["interior_px"]))
    print("  loss=%dpx (hard %d + speckle %d)  feather=%dpx  pockets=%dpx"
          % (rec["loss_px"], rec["hard_px"], rec["speckle_px"],
             rec["feather_px"], rec["pocket_px"]))
    for f in rec["fails"] + rec["flags"]:
        print("  - " + f)
    return 1 if rec["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
