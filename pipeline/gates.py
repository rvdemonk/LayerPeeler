"""Stage 3/5 — the mechanical instruments.

Doctrine (ledger, 2026-07-31): the gates ARE the product. Generation is a
commodity bought for five cents; what a customer pays for is never seeing
the rejects. Every gate here exists because Lewis's eye caught something
Claude's did not, and each one retires that defect class from needing his
minutes again.

Five gates, run in two places:

  BASIC INTEGRITY (on the matte, before anything perceptual). Cheap,
  mechanical, and first — "local SSIM 0.63" cannot distinguish a soft fit
  from a part that flew off-screen, and this can. Empty frames, silhouette
  collapse, the character drifting out of frame, frozen frames, loop
  closure.

  COLOUR / VELOCITY STRIPS (on the matte). Born from r001: Lewis caught a
  mid-clip flesh darkening and an arm fidget that the loop/identity/
  silhouette gates all passed. dL* drift shows as a ramp, the loop tell as
  a cliff, fidget as a high jerk. face_body_ratio came from wave-1's
  raccoon-jig: dancing body, dead face.

  IDENTITY DRIFT (on the matte). Born from pack #1: Wan draws a DIFFERENT
  character by late frames — the raccoon's eye-mask splits from one blob
  into two, white eye-rings appear, the head rounds. Every other gate was
  quiet (loop_ssim 0.974, colour stable, velocity clean) because none of
  them measures identity over time. Scored as a return distance: late
  frames are matched against a bank of early ones, so legitimate pose
  change (which recurs around a loop) cancels and design change (which
  does not) accumulates.

  POSTERIZATION PAIR (on the encoded assets). Colour-count ratio AND mean
  dE. Either alone is known-broken: pngquant crushes the palette by the
  same factor as octree (ratio 0.027 vs 0.031) at a quarter of the colour
  error, and only octree looks wrong. Both halves or it is not the gate.

  TEMPORAL SHIMMER (on the encoded assets). The risk stills cannot see: a
  codec that is clean per-frame can still make a flat region crawl. Per-
  pixel temporal std over flat areas in the quietest window, encoded
  against the lossless matte. Reported as a RATIO — an absolute number is
  meaningless because the h264 source has its own flicker (measured 3.09
  lossless, 3.33 webp = +8%, cleared).

VERDICTS. `fail` is reserved for breaches of physical integrity — things
that are broken regardless of taste. Everything else is `flag`: a
screening signal for a human or a reroll policy, deliberately stricter
than Lewis's eye on flat art (512q flags on raccoon-wave, which he
passed). A flag is not a verdict, and this module never renders one as
one.

What this module does NOT do: pick a profile. The velocity strip once
auto-selected frame rate per clip, and Lewis rejected the result on
clips the gate had cleared ("looks like a flipbook") — he was reporting
FLUIDITY where the instrument measured JUDDER. The fluidity floor is a
taste constant, not a per-clip measurement. Velocity stays a reroll and
diagnosis signal.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "local"))
from ssim_check import ssim  # noqa: E402

ALPHA_SOLID = 128

# --- thresholds -----------------------------------------------------------
# FAIL = physically broken. Set clear of the whole 31-run corpus, whose
# worst legitimate values are: silhouette 0.71/1.11 (raccoon-celebrate, a
# squashing jump Lewis passed), loop_alpha_iou 0.804 (blob-jig).
MIN_MASK_PX = 2000            # below this the frame has no character in it
SILH_MIN, SILH_MAX = 0.60, 1.60   # area / median area
BORDER_RING_PX = 2
BORDER_FRAC_FAIL = 0.005      # >0.5% of the mask on the outer ring
CENTROID_FRAC_FAIL = 0.25     # centroid travel, as a fraction of the edge
FROZEN_RUN_FAIL = 3           # N identical consecutive frames
LOOP_IOU_FAIL = 0.75
# Interior holes (see holes()). Corpus calibration 2026-08-02, eight runs:
# ship-breaking damage (frog r037/r041/r042 eaten eyes) sits at median
# >3000 px/frame; Lewis-passed clips sit at median <200 (raccoon-wave 174 —
# pale chest fur, real but invisible loss; r039 3). A transient pocket (an
# armpit closing mid-wave) spikes max without median, so BOTH fire-lines
# are needed: median for persistent feature loss, max for one-frame
# catastrophes.
HOLES_MEDIAN_FAIL_PX = 200    # persistent loss, px/frame (absolute @512)
HOLES_MAX_FAIL_PX = 1500      # one-frame catastrophe, px (absolute @512)
HOLES_MEDIAN_FAIL_FRAC = 0.003   # or these fractions of median mask area
HOLES_MAX_FAIL_FRAC = 0.02
HOLE_MIN_PX = 8               # components smaller than this are matte noise
# Pocket discrimination, measured on the corpus's enclosed components
# 2026-08-02: flat painted gaps (star leg-gap, strawberry shadow-gap)
# sit at laplacian p50 <=2.4 and d p50 <=5.4; eaten features (frog eyes,
# blob body, raccoon fur) at lap p50 >=4.2 and d p50 >=10. Both margins
# are thin; provisional until more mascots are scored.
HOLE_POCKET_LAP = 3.0         # eroded-interior laplacian p50 below = flat
HOLE_POCKET_D = 8.0           # eroded-interior d-to-bg p50 below = bg-like
HOLE_POCKET_MAX_PX = 500      # @512px, scaled by area: small solid-walled
                              # components are pockets regardless of texture

# FLAG = screening signal, calibrated against the corpus distribution, not
# against a spec. Provisional: only the posterization pair and the shimmer
# ratio have been checked against a Lewis verdict; the rest are set where
# they separate the corpus's known-good runs from its known-loud ones.
LOOP_SSIM_FLAG = 0.94
LOOP_IOU_FLAG = 0.90
DL_MAX_FLAG = 3.0             # colour-normed runs sit at ~1.0
DL_CLIFF_FLAG = 2.0
JERK_FLAG = 3.0               # fidget; the loud jigs measure 3.2-3.8
FACE_BODY_FLAG = 0.55         # dead face under body motion; corpus min 0.71
# Identity drift. CALIBRATION PROVENANCE, and its limits: set on n=10
# identity-appraised clips (3 known-bad, 7 known-good) spanning 2 mascot
# styles, in ONE session, 2026-08-02. It separates that set completely —
# worst bad 0.0575 (celebrate), worst good 0.0476 (strawberry-idle-720) —
# and the line sits at the geometric middle of a 21%-wide gap, so the
# margin is ~10% either way. Ten clips and a 10% margin is a provisional
# threshold by construction; widen the labelled set before trusting it to
# trigger a reroll. Known asymmetry: on the 22 spike2 runs that were never
# identity-appraised it fires 7 times, every one of them a raccoon and
# none of them a strawberry, star or blob. Unresolved whether Wan really
# does drift the grey-on-grey raccoon hardest or whether that palette
# quantises less stably.
# SEEDED 2026-08-02 (same day, later session). Until then the palette
# kmeans drew from OpenCV's unseeded per-process RNG, so a clip's drift
# depended on its POSITION in the process that scored it — the numbers
# above were never reproducible. Two checks, both against clips on disk:
# the recorded worst-good strawberry-idle-720 0.0476 re-scores at 0.0461
# scored alone, and the recorded worst-bad celebrate 0.0575 re-scores at
# 0.0583 alone but reproduces 0.0575 exactly at position 2 of a 4-emote
# pack. So the calibration sweep scored the set in ONE process in an
# order nobody wrote down, and neither endpoint of the 0.0476-0.0575 gap
# can be reproduced. The seed makes every future number a property of the
# clip; it does NOT recover the sweep. Re-verify the thresholds against a
# re-scored labelled set before trusting the ~10% margin claimed above —
# and note 5 of the 7 known-goods are not recorded anywhere in the repo.
IDENTITY_DRIFT_FLAG = 0.052   # median late-frame return distance
IDENTITY_PERSIST_FLAG = 0.50  # fraction of late frames over that line
IDENTITY_GRID = 64            # normalised crop, px
IDENTITY_K = 6                # palette clusters taken from frames 0-2
IDENTITY_WORK = 256           # working resolution before normalising
IDENTITY_SEED = 0             # see the kmeans call: pins the palette draw
DE_POSTERIZE = 2.0            # the dE half of the pair
RATIO_POSTERIZE = 0.05        # the colour-count half
RATIO_SOURCE_MIN = 2000       # below this the source is too flat to judge
# Shimmer is a PAIR too, and for the same reason posterization is — one
# number alone misreads. The ratio is denominator-fragile: on a genuinely
# quiet clip the lossless baseline is under one 8-bit level (raccoon-wave
# measured 0.89), so an added 0.28 of a level — imperceptible, and the
# same absolute excess the ledger cleared at +0.24 — reads as 1.31x. So a
# flag needs both a ratio above threshold AND an absolute excess worth
# more than a level.
SHIMMER_RATIO_FLAG = 1.25     # webp measured 1.08 over the lossless matte
SHIMMER_ABS_FLAG = 1.0        # levels of added temporal std, flat pixels
SHIMMER_WINDOW = 20           # frames in the quiet window


def _mask(rgba):
    return rgba[..., 3] > ALPHA_SOLID


def _load(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[-1] != 4:
        raise SystemExit("not an RGBA frame: %s" % path)
    return im


# ------------------------------------------------------------ integrity ---

def basic_integrity(frames, loop=True):
    """Mechanical checks on the matted sequence. Runs before any
    perceptual metric matters."""
    areas, cents, borders, digests = [], [], [], []
    h = w = None
    prev = None
    frozen_run, frozen_max = 1, 1
    for fp in frames:
        im = _load(fp)
        h, w = im.shape[:2]
        m = _mask(im)
        areas.append(int(m.sum()))
        if m.any():
            ys, xs = np.nonzero(m)
            cents.append((float(xs.mean()), float(ys.mean())))
            ring = np.zeros_like(m)
            ring[:BORDER_RING_PX] = ring[-BORDER_RING_PX:] = True
            ring[:, :BORDER_RING_PX] = ring[:, -BORDER_RING_PX:] = True
            borders.append(float((m & ring).sum()) / max(m.sum(), 1))
        else:
            cents.append((np.nan, np.nan))
            borders.append(0.0)
        d = int(im[::8, ::8].sum())
        digests.append(d)
        if prev is not None and d == prev:
            frozen_run += 1
            frozen_max = max(frozen_max, frozen_run)
        else:
            frozen_run = 1
        prev = d

    areas = np.array(areas, float)
    med = float(np.median(areas))
    cen = np.array(cents, float)
    valid = ~np.isnan(cen[:, 0])
    travel = (float(np.nanmax(np.linalg.norm(cen[valid] - cen[valid][0], axis=1)))
              if valid.any() else 0.0)

    f0, fN = _load(frames[0]), _load(frames[-1])
    a0, aN = _mask(f0), _mask(fN)
    g0, gN = [cv2.cvtColor(f[..., :3], cv2.COLOR_BGR2GRAY) for f in (f0, fN)]
    loop_ssim = float(ssim(g0, gN))
    loop_iou = float((a0 & aN).sum() / max((a0 | aN).sum(), 1))

    g = {
        "frames": len(frames), "edge": int(w),
        "empty_frames": int((areas < MIN_MASK_PX).sum()),
        "area_median_px": med,
        "silhouette_rel_range": [float(areas.min() / med),
                                 float(areas.max() / med)],
        "border_frac_max": float(max(borders)) if borders else 0.0,
        "centroid_travel_px": travel,
        "centroid_travel_frac": travel / max(w, 1),
        "frozen_run_max": int(frozen_max),
        "loop_ssim": loop_ssim,
        "loop_alpha_iou": loop_iou,
    }
    fails, flags = [], []
    if g["empty_frames"]:
        fails.append("%d frame(s) with no character" % g["empty_frames"])
    lo, hi = g["silhouette_rel_range"]
    if lo < SILH_MIN or hi > SILH_MAX:
        fails.append("silhouette %.2f-%.2f outside %.2f-%.2f"
                     % (lo, hi, SILH_MIN, SILH_MAX))
    if g["border_frac_max"] > BORDER_FRAC_FAIL:
        fails.append("character out of frame (%.1f%% of mask on border)"
                     % (100 * g["border_frac_max"]))
    elif g["border_frac_max"] > 0:
        flags.append("mask touches frame border")
    if g["centroid_travel_frac"] > CENTROID_FRAC_FAIL:
        fails.append("centroid travels %.0f%% of the canvas"
                     % (100 * g["centroid_travel_frac"]))
    if g["frozen_run_max"] >= FROZEN_RUN_FAIL:
        fails.append("%d identical consecutive frames" % g["frozen_run_max"])
    if loop:
        if loop_iou < LOOP_IOU_FAIL:
            fails.append("loop broken (alpha IoU %.3f)" % loop_iou)
        elif loop_iou < LOOP_IOU_FLAG:
            flags.append("loose loop (alpha IoU %.3f)" % loop_iou)
        if loop_ssim < LOOP_SSIM_FLAG:
            flags.append("loop seam (SSIM %.3f)" % loop_ssim)
    g["fails"], g["flags"] = fails, flags
    return g


def holes(frames):
    """Interior-hole census: transparent pixels ENCLOSED by character.

    Provenance: the frog eye saga (r037-r042, 2026-08-02). The matte keys
    on colour distance, so any feature within ~26 RGB units of the
    background loses alpha — the frog's shaded eye whites sat at d~13 from
    a cream background and were eaten in three runs, while global metrics
    (SSIM, silhouette) passed and the sandbox's light checkerboard hid the
    bite. Every one of those runs had thousands of interior-hole pixels in
    the eye band; the healthy run (r039) had 2-7.

    Two transparent components are distinguished by what the GENERATED
    frame painted inside them, because colour distance cannot separate
    them (star's genuine leg-gap sits at d=9 from bg; the frog's eaten
    eye whites at d=13 — overlapping ranges), and neither can topology
    (a closed leg-gap can be walled in by limbs thicker than an eye rim):

      POCKET (ignored) — genuine enclosed background: a between-legs gap,
      an armpit at a closed moment. The gen frame painted FLAT background
      there (the same painted surface as outside, locally shaded). Tested
      on the component's eroded interior, so a passing silhouette edge
      can't masquerade as texture.

      EATEN FEATURE (counted) — the gen frame painted a STRUCTURED surface
      there: eye-white gradient, fur, blob shading. Texture or colour
      shift beyond the flat-bg distribution = the matte removed art.

    …plus a size carve-out for the residue the flatness test misses:
    small solid-walled gaps (strawberry armpits read as textured because
    a passing limb edge dominates a 200px region). Corpus geometry says
    genuine pockets are small and feature loss is not: at 512px the
    strawberry gaps measure 198-237px, the frog's eaten eyes 1765px and
    blob's eaten body 700-820px. HOLE_POCKET_MAX_PX sits in that gap.

    The mean RGB of the hole pixels rides along: it names WHAT was eaten
    for the recovery decision, which lives upstream in master_qc — this
    gate detects, that one prevents.

    FAIL lines, provisional: corpus calibration 2026-08-02 — ship-breaking
    damage (frog r037/r041/r042) at median >3000 px/frame; Lewis-passed
    clips at median <200 (raccoon-wave 174, real-but-invisible fur loss).
    Median fires on persistent loss, max on one-frame catastrophes.
    """
    kern3 = np.ones((7, 7), np.uint8)
    per_frame, worst, worst_n = [], None, 0
    mask_areas = []
    for i, fp in enumerate(frames):
        im = _load(fp)
        m = _mask(im)
        mask_areas.append(int(m.sum()))
        transp = (~m).astype(np.uint8)
        ncc, cc, stats, _ = cv2.connectedComponentsWithStats(transp)
        border_ids = set(np.unique(
            np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))) - {0}
        # The RGBA frame's RGB channels ARE the (colour-normed) generated
        # pixels — the matte zeroes alpha, never colour — so the generated
        # surface inside a hole is readable right here.
        gen = im[..., :3].astype(np.float32)
        border = np.concatenate([gen[:8].reshape(-1, 3), gen[-8:].reshape(-1, 3),
                                 gen[:, :8].reshape(-1, 3), gen[:, -8:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        d = np.linalg.norm(gen - bg, axis=-1)
        lap = np.abs(cv2.Laplacian(cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY),
                                   cv2.CV_32F, ksize=3))
        eaten = np.zeros_like(transp)
        pocket_max = HOLE_POCKET_MAX_PX * (im.shape[0] * im.shape[1]) / (512.0 * 512.0)
        for j in range(1, ncc):
            if j in border_ids:
                continue
            if stats[j, cv2.CC_STAT_AREA] < HOLE_MIN_PX:
                continue
            comp = cc == j
            core = cv2.erode(comp.astype(np.uint8), kern3).astype(bool)
            sample = core if core.any() else comp
            if stats[j, cv2.CC_STAT_AREA] < pocket_max:
                continue                      # small solid-walled gap: pocket
            if (np.percentile(lap[sample], 50) < HOLE_POCKET_LAP
                    and np.percentile(d[sample], 50) < HOLE_POCKET_D):
                continue                      # flat painted background: pocket
            eaten |= comp.astype(np.uint8)
        n = int(eaten.sum())
        per_frame.append(n)
        if n > worst_n:
            worst_n = n
            ys, xs = np.nonzero(eaten)
            worst = {
                "frame": i, "px": n,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max()), int(ys.max())],
                "mean_rgb": [float(v) for v in im[..., :3][eaten.astype(bool)].mean(0)[::-1]],
            }
    med_mask = float(np.median(mask_areas)) if mask_areas else 0.0
    med = float(np.median(per_frame)) if per_frame else 0.0
    g = {"holes_max_px": int(max(per_frame) if per_frame else 0),
         "holes_median_px": int(med),
         "holes_worst": worst}
    med_thresh = max(HOLES_MEDIAN_FAIL_PX, HOLES_MEDIAN_FAIL_FRAC * med_mask)
    max_thresh = max(HOLES_MAX_FAIL_PX, HOLES_MAX_FAIL_FRAC * med_mask)
    fails = []
    if med > med_thresh:
        fails.append("persistent interior transparency: %d px/frame median "
                     "(worst %d px frame %d, bbox %s, mean RGB %s) — the "
                     "matte ate an enclosed feature"
                     % (med, worst["px"], worst["frame"], worst["bbox"],
                        [round(v) for v in worst["mean_rgb"] or []]))
    elif worst_n > max_thresh:
        fails.append("%d interior transparent px in frame %d (bbox %s, mean "
                     "RGB %s) — matte ate an enclosed feature"
                     % (worst["px"], worst["frame"], worst["bbox"],
                        [round(v) for v in worst["mean_rgb"] or []]))
    g["fails"] = fails
    g["flags"] = []
    return g


# ------------------------------------------------------ colour + motion ---

def _character_stats(rgba):
    a = _mask(rgba)
    px = rgba[..., :3][a]
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return a, lab.mean(0), float(hsv[:, 1].mean())


def strips(frames):
    """Colour timeline + velocity strip. Returns metrics and the per-frame
    series (the series feed the PNG strips and the shimmer window pick)."""
    labs, sats, vel, fvel = [], [], [0.0], [0.0]
    prev_gray = prev_a = None
    for fp in frames:
        im = _load(fp)
        a, lab, sat = _character_stats(im)
        labs.append(lab)
        sats.append(sat)
        g = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=-1)
            vel.append(float(mag[a | prev_a].mean()))
            # Face region ~ upper-central band of the character bbox. A
            # mascot heuristic, not a detector — it was enough to catch the
            # dead face Lewis saw on raccoon-jig.
            ys, xs = np.nonzero(a)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bw, bh = x1 - x0, y1 - y0
            fb = mag[y0 + int(.12 * bh):y0 + int(.50 * bh),
                     x0 + int(.25 * bw):x0 + int(.75 * bw)]
            fvel.append(float(fb.mean()) if fb.size else 0.0)
        prev_gray, prev_a = g, a

    labs, sats = np.array(labs), np.array(sats)
    vel, fvel = np.array(vel), np.array(fvel)
    dL = labs[:, 0] - labs[0, 0]
    dSat = sats - sats[0]
    active = vel > max(np.median(vel), 0.15)
    face_ratio = (float(fvel[active].mean() / max(vel[active].mean(), 1e-6))
                  if active.any() else 1.0)
    g = {
        "color_dL_max": float(np.abs(dL).max()),
        "color_dSat_max": float(np.abs(dSat).max()),
        "color_dL_cliff": float(np.abs(np.diff(dL)).max()),
        "velocity_p95": float(np.percentile(vel, 95)),
        "velocity_max": float(vel.max()),
        "jerk_rms": float(np.sqrt(np.mean(np.diff(vel) ** 2))),
        "face_vel_p95": float(np.percentile(fvel, 95)),
        "face_body_ratio": face_ratio,
    }
    flags = []
    if g["color_dL_max"] > DL_MAX_FLAG:
        flags.append("luminance drift dL* %.1f" % g["color_dL_max"])
    if g["color_dL_cliff"] > DL_CLIFF_FLAG:
        flags.append("colour cliff dL* %.1f" % g["color_dL_cliff"])
    if g["jerk_rms"] > JERK_FLAG:
        flags.append("fidget (jerk %.2f)" % g["jerk_rms"])
    if g["face_body_ratio"] < FACE_BODY_FLAG:
        flags.append("face underacting (face/body %.2f)" % g["face_body_ratio"])
    g["flags"] = flags
    return g, {"dL": dL, "dSat": dSat, "vel": vel}


# -------------------------------------------------------- identity drift ---

_SHIFTS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
           (1, 1), (-1, -1), (1, -1), (-1, 1)]


def _normalised_crop(rgba, m, radius):
    """A translation- and scale-normalised square crop of the character.

    Centred on the centroid of the ERODED mask and sized from sqrt(area),
    not on the mask bounding box: a raised arm moves a bbox edge by a
    third of the character's height, and every frame after that compares
    against a differently-framed anchor. The eroded centroid ignores thin
    limbs; sqrt(area) moves smoothly when they extend.
    """
    core = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(core if core.sum() > 200 else m)
    cy, cx = ys.mean(), xs.mean()
    y0, x0 = int(cy - radius), int(cx - radius)
    y1, x1 = int(cy + radius), int(cx + radius)
    pad = max(0, -y0, -x0, y1 - rgba.shape[0], x1 - rgba.shape[1])
    if pad:
        rgba = cv2.copyMakeBorder(rgba, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    c = rgba[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    if c.size == 0:
        return None
    return cv2.resize(c, (IDENTITY_GRID, IDENTITY_GRID),
                      interpolation=cv2.INTER_AREA)


def _design_maps(frames):
    """Each frame as a label map over the character's OWN frame-0 palette.

    Quantising to a fixed palette is what makes this a design comparison
    rather than a pixel one: soft shading, codec noise and a degree of
    lighting wander all collapse to the same label, while a region that
    changes shape or splits in two moves pixels between labels.
    """
    work = []
    for fp in frames:
        im = _load(fp)
        if im.shape[0] != IDENTITY_WORK:
            im = cv2.resize(im, (IDENTITY_WORK, IDENTITY_WORK),
                            interpolation=cv2.INTER_AREA)
        work.append(im)
    areas = np.array([float(_mask(im).sum()) for im in work])
    if (areas > 0).sum() < 10:
        return None
    radius = 1.25 * float(np.median(np.sqrt(areas[areas > 0])))

    crops, keep = [], []
    for i, im in enumerate(work):
        m = _mask(im)
        if not m.any():
            continue
        c = _normalised_crop(im, m, radius)
        if c is not None:
            crops.append(c)
            keep.append(i)
    if len(crops) < 10:
        return None

    labs = [cv2.cvtColor(c[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            for c in crops]
    masks = [c[..., 3] > ALPHA_SOLID for c in crops]
    anchor = np.concatenate([labs[j][masks[j]] for j in range(3)])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # KMEANS_PP_CENTERS seeds its centres from OpenCV's RNG, which is
    # per-process and advances with every call that draws from it. Unseeded,
    # a clip's drift therefore depended on how many clips had been scored
    # before it IN THE SAME PROCESS: the same pixels scored differently as
    # emote 1 and as emote 3 of a pack. Pinned here so the number is a
    # property of the clip alone.
    cv2.setRNGSeed(IDENTITY_SEED)
    _, _, cen = cv2.kmeans(anchor, IDENTITY_K, None, crit, 3,
                           cv2.KMEANS_PP_CENTERS)

    maps = []
    for lab, m in zip(labs, masks):
        d = np.linalg.norm(lab.reshape(-1, 1, 3) - cen[None], axis=2)
        q = d.argmin(1).reshape(IDENTITY_GRID, IDENTITY_GRID).astype(np.uint8)
        q[~m] = IDENTITY_K          # background is its own label
        maps.append(q)
    return np.array(maps), keep


def identity(frames):
    """Return distance: does the character late in the clip still match
    the one at the start?

    The hard part is that legitimate animation changes frames too. What
    separates the two is RECURRENCE. These clips loop, so a pose struck
    at frame 130 was struck somewhere in the opening fifth as well; a
    design change was not. So each frame is scored against a BANK of
    early frames and keeps its best match — pose is matched away, drift
    is not — and the verdict is the median over the last third, which no
    single odd frame can move.

    Not a ratio, so it takes no absolute-floor partner: the denominator
    is the fixed 64x64 grid, never small. It is still a pair, for the
    other reason the ledger keeps insisting on pairs — one number cannot
    tell a sustained redraw from one ugly frame. Magnitude AND the
    fraction of late frames holding it, or no flag.
    """
    built = _design_maps(frames)
    if built is None:
        return {"note": "too few usable frames to score identity",
                "flags": []}
    maps, keep = built
    T = len(maps)
    bank = list(range(0, max(T // 5, 2), 2))

    def d(a, b):
        return min(float((a != np.roll(np.roll(b, dy, 0), dx, 1)).mean())
                   for dy, dx in _SHIFTS)

    dist = np.array([min(d(maps[t], maps[b]) for b in bank) for t in range(T)])
    ctrl = dist[T // 5:2 * T // 5]      # early frames, same treatment
    late = dist[2 * T // 3:]
    drift = float(np.median(late))
    persist = float((late > IDENTITY_DRIFT_FLAG).mean())

    g = {
        "identity_drift": drift,
        "identity_drift_ctrl": float(np.median(ctrl)) if len(ctrl) else 0.0,
        "identity_drift_p90": float(np.percentile(late, 90)),
        "identity_persist": persist,
        "identity_scored_frames": int(T),
        "identity_worst_frame": int(keep[2 * T // 3 + int(np.argmax(late))]),
    }
    # ctrl is reported, never subtracted. Two of the three known-bads drift
    # early and hold it, so their late-minus-ctrl delta is ~0 — a gate built
    # on the delta would have passed them.
    g["flags"] = (["identity drift %.3f over %.0f%% of late frames "
                   "(worst f%d)" % (drift, 100 * persist,
                                    g["identity_worst_frame"])]
                  if drift > IDENTITY_DRIFT_FLAG
                  and persist >= IDENTITY_PERSIST_FLAG else [])
    return g


def draw_strips(series, metrics, path):
    """The strips as a plain image — no matplotlib dependency, and legible
    to Claude as well as to Lewis. Drift reads as a ramp, the loop tell as
    a cliff, fidget as a comb."""
    dL, dSat, vel = series["dL"], series["dSat"], series["vel"]
    T = len(dL)
    W, H = max(T * 4, 320), 120
    img = np.full((H * 3 + 40, W, 3), 255, np.uint8)

    def strip(row, s, label, lo, hi, color):
        y0 = row * (H + 13) + 10
        cv2.putText(img, label, (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1)
        pts = [(int(t / max(T - 1, 1) * (W - 20)) + 10,
                y0 + H - 20 - int(np.clip((v - lo) / (hi - lo), 0, 1) * (H - 35)))
               for t, v in enumerate(s)]
        zy = y0 + H - 20 - int(np.clip((0 - lo) / (hi - lo), 0, 1) * (H - 35))
        cv2.line(img, (10, zy), (W - 10, zy), (200, 200, 200), 1)
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, color, 2)

    m = max(3, np.abs(dL).max() * 1.2)
    strip(0, dL, "dL* vs f0 (max %.1f)" % np.abs(dL).max(), -m, m, (180, 60, 30))
    m = max(6, np.abs(dSat).max() * 1.2)
    strip(1, dSat, "dSat vs f0 (max %.1f)" % np.abs(dSat).max(), -m, m,
          (30, 120, 180))
    strip(2, vel, "velocity px/f (p95 %.2f, jerk %.2f)"
          % (metrics["velocity_p95"], metrics["jerk_rms"]),
          0, max(2, vel.max() * 1.1), (60, 160, 60))
    cv2.imwrite(str(path), img)
    return path


# --------------------------------------------------- encoded-asset gates ---

def _ref_rgba(path, edge):
    im = Image.open(path).convert("RGBA")
    if im.size != (edge, edge):
        im = im.resize((edge, edge), Image.LANCZOS)
    return np.array(im)


def posterization(ref_paths, var_paths, edge, sample=8):
    """The PAIR, scored on a sample of frames; worst frame decides.

    Comparison is at the variant's own resolution against a same-
    resolution Lanczos reference, so the score isolates the codec instead
    of re-measuring the downscale. `mid` is reported separately because
    the spike-era ladder scored mid-cycle only and the numbers have to
    stay comparable to ladder_report.json.
    """
    n = len(var_paths)
    idxs = sorted(set(list(range(0, n, max(n // sample, 1))) + [n // 2]))
    mid = n // 2
    per_frame, worst = {}, None
    for i in idxs:
        ref = _ref_rgba(ref_paths[i], edge)
        var = np.array(Image.open(var_paths[i]).convert("RGBA"))
        mask = ref[..., 3] > ALPHA_SOLID
        if mask.sum() < 100:
            continue
        u_ref = int(len(np.unique(ref[mask][:, :3], axis=0)))
        u_var = int(len(np.unique(var[mask][:, :3], axis=0)))
        lab_r = cv2.cvtColor(ref[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_v = cv2.cvtColor(var[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
        d = np.sqrt(((lab_r - lab_v) ** 2).sum(-1))[mask]
        s = {"frame": i, "colors_ref": u_ref, "colors_var": u_var,
             "color_ratio": round(u_var / max(u_ref, 1), 4),
             "deltaE_mean": round(float(d.mean()), 2),
             "deltaE_p95": round(float(np.percentile(d, 95)), 2),
             "alpha_levels": int(len(np.unique(var[..., 3])))}
        s["flag_color_ratio"] = bool(u_ref > RATIO_SOURCE_MIN
                                     and s["color_ratio"] < RATIO_POSTERIZE)
        s["flag_posterized"] = bool(s["flag_color_ratio"]
                                    and s["deltaE_mean"] >= DE_POSTERIZE)
        per_frame[i] = s
        if worst is None or s["deltaE_mean"] > per_frame[worst]["deltaE_mean"]:
            worst = i
    if worst is None:
        return {"scored": 0, "flags": []}
    g = {"scored": len(per_frame), "mid_frame": per_frame.get(mid),
         "worst_frame": per_frame[worst], "per_frame": per_frame}
    g["flags"] = (["POSTERIZED at f%d (ratio %.3f, dE %.2f)"
                   % (worst, per_frame[worst]["color_ratio"],
                      per_frame[worst]["deltaE_mean"])]
                  if per_frame[worst]["flag_posterized"] else [])
    return g


def _flat_mask(ref_rgba, grad_thresh=5.0):
    """Interior pixels with little local gradient — the places banding and
    crawl are visible. Edges and detail are excluded because their own
    variance would swamp the measurement."""
    m = ref_rgba[..., 3] > 250
    g = cv2.cvtColor(ref_rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + \
        np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    flat = m & (grad < grad_thresh)
    return cv2.erode(flat.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def shimmer(ref_paths, var_paths, edge, vel, window=SHIMMER_WINDOW):
    """Temporal std over flat regions in the quietest window: encoded vs
    the lossless matte.

    Reported as a ratio because the absolute number is dominated by the
    source's own h264 flicker. A codec that BANDS can score LOWER than
    lossless (octree measured 2.64 vs 3.09) — flattening noise into
    plateaus is not stability, so a low ratio is not a pass signal and is
    reported, never rewarded.
    """
    n = len(var_paths)
    window = min(window, n)
    # vel is indexed on the source frames; the variant may be decimated.
    step = max(len(vel) // n, 1)
    v = np.array([vel[min(i * step, len(vel) - 1)] for i in range(n)])
    sums = np.convolve(v, np.ones(window), "valid")
    start = int(np.argmin(sums))
    idxs = list(range(start, start + window))

    ref0 = _ref_rgba(ref_paths[idxs[0]], edge)
    flat = _flat_mask(ref0)
    if flat.sum() < 500:
        return {"flat_px": int(flat.sum()), "note": "no flat region to score",
                "flags": []}

    def stack(paths, is_ref):
        arr = []
        for i in idxs:
            a = (_ref_rgba(paths[i], edge) if is_ref
                 else np.array(Image.open(paths[i]).convert("RGBA")))
            arr.append(a[..., :3][flat].astype(np.float32))
        return np.stack(arr)

    ref_std = float(stack(ref_paths, True).std(0).mean())
    var_std = float(stack(var_paths, False).std(0).mean())
    ratio = var_std / max(ref_std, 1e-6)
    excess = var_std - ref_std
    g = {"window_start": start, "window": window, "flat_px": int(flat.sum()),
         "window_velocity_mean": round(float(v[idxs].mean()), 3),
         "std_lossless": round(ref_std, 3), "std_encoded": round(var_std, 3),
         "ratio": round(ratio, 3), "excess_levels": round(excess, 3)}
    g["flags"] = (["temporal shimmer %.2fx lossless (+%.2f levels)"
                   % (ratio, excess)]
                  if ratio > SHIMMER_RATIO_FLAG and excess > SHIMMER_ABS_FLAG
                  else [])
    return g


# ------------------------------------------------------------- assembly ---

def verdict(*gate_dicts):
    fails = [f for g in gate_dicts for f in g.get("fails", [])]
    flags = [f for g in gate_dicts for f in g.get("flags", [])]
    return {"verdict": "FAIL" if fails else ("FLAG" if flags else "PASS"),
            "fails": fails, "flags": flags}


def write(record, path):
    path = Path(path)
    path.write_text(json.dumps(record, indent=1, default=float))
    return path
