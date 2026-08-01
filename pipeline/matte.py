"""Stage 2 — RGB frames -> RGBA frames on a transparent background.

Algorithm carried over UNCHANGED from hybrid/spike2_oracle.matte(), and
it must stay that way unless a verdict says otherwise: every appraisal in
the ledger, every ladder byte-size and every gate threshold was measured
downstream of exactly this matte. Changing it silently re-baselines the
whole corpus.

Two parts, both load-bearing:

  MATTE. Wan is prompted into a flat uniform background, so the key is a
  colour-distance matte rather than a learned one: background colour is
  the median of the frame's own border pixels (per frame, because the
  background is not guaranteed identical across frames), alpha is a
  smoothstep on distance to it, and a morphology + largest-component pass
  drops speckle. The soft ramp is the point — a hard threshold gives
  aliased edges that read as a cut-out, and the feathered alpha (227-250
  distinct levels on the corpus) is precisely what the palette codecs
  crushed and WebP preserved.

  COLOUR NORM. Character pixels are shifted so their mean Lab matches
  frame 0's. This exists because of Lewis's r001 verdict: a slow
  luminance drift through the clip ("there is a slight colour shift,
  where the flesh suddenly darkens... the sudden reversion is the only
  indication of a clear loop seam"). Normalising kills the drift and the
  loop tell together.
"""

import cv2
import numpy as np
from PIL import Image

# Smoothstep band on colour distance from the background, in 0-255 RGB
# euclidean units. Below `lo` is background, above `hi` is character, and
# the ramp between is the feathered edge.
ALPHA_LO, ALPHA_HI = 12.0, 40.0
BORDER_PX = 8          # border band sampled for the background colour
ALPHA_SOLID = 128      # alpha above which a pixel counts as "character"


def matte_frames(frames, mdir, log=None):
    """Matte a whole sequence. Returns the written RGBA paths.

    Sequence-stateful by design: the colour norm references frame 0, so
    frames cannot be matted independently or in a different order without
    changing the output.
    """
    mdir.mkdir(parents=True, exist_ok=True)
    outs = []
    ref_lab = None
    for n, fp in enumerate(frames):
        im = cv2.imread(str(fp)).astype(np.float32)
        border = np.concatenate([im[:BORDER_PX].reshape(-1, 3),
                                 im[-BORDER_PX:].reshape(-1, 3),
                                 im[:, :BORDER_PX].reshape(-1, 3),
                                 im[:, -BORDER_PX:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        d = np.linalg.norm(im - bg, axis=-1)
        a = np.clip((d - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0, 1)
        a = a * a * (3 - 2 * a)
        hard = (a > 0.5).astype(np.uint8)
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Keep only the largest connected component: drops background
        # speckle without eroding the character.
        ncc, cc, stats, _ = cv2.connectedComponentsWithStats(hard)
        if ncc > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hard = (cc == big).astype(np.uint8)
        a = a * cv2.dilate(hard, np.ones((7, 7), np.uint8))
        rgba = np.dstack([im, a * 255]).astype(np.uint8)

        ch = rgba[..., 3] > ALPHA_SOLID
        if ch.sum() > 100:
            lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            mean = lab[ch].mean(0)
            if ref_lab is None:
                ref_lab = mean
            else:
                lab[ch] = np.clip(lab[ch] + (ref_lab - mean), 0, 255)
                rgba[..., :3] = cv2.cvtColor(lab.astype(np.uint8),
                                             cv2.COLOR_LAB2BGR)
        op = mdir / fp.name
        # PIL with optimize=True rather than cv2.imwrite: identical pixels
        # (lossless either way), but cv2 writes PNG at compression level 1
        # and PIL picks better filters — measured 30-34% smaller across the
        # spike2 corpus. Free bytes off every downstream pack.
        Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(
            op, optimize=True)
        outs.append(op)
        if log and n and n % 40 == 0:
            log("    matted %d/%d" % (n, len(frames)))
    return outs
