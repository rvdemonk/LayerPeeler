"""Appraisal artifacts — the GIF and contact sheet Claude must watch.

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
"""

import cv2
import numpy as np
from PIL import Image


def eyeball(rgba_frames, out_dir, name, fps, sheet_every=4):
    gif = out_dir / ("%s.gif" % name)
    sheet = out_dir / ("%s_sheet.png" % name)
    tiles, pil = [], []
    for i, fp in enumerate(rgba_frames):
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        checker = np.repeat((((yy // 16 + xx // 16) % 2) * 40 + 200)[..., None],
                            3, -1).astype(np.float32)
        a = im[..., 3:4].astype(np.float32) / 255
        comp = (im[..., :3] * a + checker * (1 - a)).astype(np.uint8)
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
