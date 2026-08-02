"""Cut the seven images the judge sees, per bench design §3.2.

Anchor f0 plus six probes: the identity gate's argmax late frame, and five
frames spread from ~f40 to ~f160. Including the gate's worst frame means a PASS
is a PASS on the frame most likely to carry the defect.

Crops are character-normalised on the SAME geometry the identity gate uses
(eroded-centroid centre, sqrt-area radius — `gates._normalised_crop`), but
rendered at 512px instead of the gate's 64px grid. A 3px doubled outline is
texture at full-frame scale and a defect at character scale; the project has
already paid for that lesson once.

Runs on the system interpreter (needs cv2). The API runner reads the PNGs this
writes, so nothing in the request path depends on OpenCV.

This module READS `pipeline/gates.py` and modifies nothing in it.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import gates  # noqa: E402
import corpus  # noqa: E402

CROP_PX = 512          # what the judge sees; the gate scores at 64
N_PROBES = 6           # gate argmax + 5 spread
SPREAD = (40, 160)     # probe span, in frame index


def _frames(clip_dir):
    return sorted((clip_dir / "rgba").glob("frame_*.png"))


def _crop_radius(paths):
    """Radius in FULL-RESOLUTION pixels, matching the gate's rule.

    The gate computes 1.25 * median(sqrt(area)) at IDENTITY_WORK; area scales
    with the square of the resize factor, so sqrt(area) scales linearly and the
    radius transfers by the same factor.
    """
    areas = []
    for p in paths:
        im = gates._load(p)
        areas.append(float(gates._mask(im).sum()))
    areas = np.array(areas)
    areas = areas[areas > 0]
    if areas.size < 10:
        raise SystemExit("too few non-empty frames to size a crop")
    return 1.25 * float(np.median(np.sqrt(areas)))


def _crop(path, radius):
    im = gates._load(path)
    m = gates._mask(im)
    if not m.any():
        return None
    core = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(core if core.sum() > 200 else m)
    cy, cx = ys.mean(), xs.mean()
    y0, x0 = int(cy - radius), int(cx - radius)
    y1, x1 = int(cy + radius), int(cx + radius)
    pad = max(0, -y0, -x0, y1 - im.shape[0], x1 - im.shape[1])
    if pad:
        im = cv2.copyMakeBorder(im, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    c = im[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    if c.size == 0:
        return None
    # Composite onto white: every provider re-encodes, and a judge should not be
    # asked to reason about whatever each one does with an alpha channel.
    c = cv2.resize(c, (CROP_PX, CROP_PX), interpolation=cv2.INTER_LANCZOS4)
    a = (c[..., 3:4].astype(np.float32) / 255.0)
    rgb = c[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
    return rgb.astype(np.uint8)


def pick_frames(clip_dir):
    """Return (frame indices, gate metrics). Indices are 0-based into rgba/."""
    paths = _frames(clip_dir)
    g = gates.identity(paths)
    worst = g.get("identity_worst_frame")
    lo, hi = SPREAD
    spread = [int(round(x)) for x in np.linspace(lo, hi, N_PROBES - 1)]
    probes = sorted(set(spread + ([worst] if worst is not None else [])))
    # Keep exactly N_PROBES: the argmax is never the one dropped.
    while len(probes) > N_PROBES:
        drop = max((p for p in probes if p != worst),
                   key=lambda p: min(abs(p - q) for q in probes if q != p))
        probes.remove(drop)
    while len(probes) < N_PROBES:
        cand = [i for i in range(lo, hi + 1) if i not in probes]
        probes.append(max(cand, key=lambda p: min(abs(p - q) for q in probes)))
        probes.sort()
    return [0] + probes, g


def main():
    outdir = ROOT / "out" / "bench" / "vlm-identity-2026-08-02" / "frames"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, path, label, why in corpus.clips():
        paths = _frames(path)
        idxs, g = pick_frames(path)
        radius = _crop_radius(paths)
        d = outdir / name
        d.mkdir(exist_ok=True)
        written = []
        for slot, i in enumerate(idxs):
            img = _crop(paths[i], radius)
            if img is None:
                raise SystemExit("empty frame %s f%d" % (name, i))
            fn = d / ("%d_f%03d.png" % (slot, i))
            cv2.imwrite(str(fn), img)
            written.append({"slot": slot, "frame": i, "file": fn.name})
        manifest[name] = {
            "clip_dir": str(path), "label": label, "provenance": why,
            "frames": written, "crop_px": CROP_PX,
            "gate": {k: v for k, v in g.items() if k != "flags"},
            "gate_flags": g.get("flags", []),
        }
        print("%-22s label=%-5s worst=f%-4s frames=%s"
              % (name, label, g.get("identity_worst_frame"), idxs))
    (outdir.parent / "frames.json").write_text(json.dumps(manifest, indent=1))
    print("\nwrote", outdir.parent / "frames.json")


if __name__ == "__main__":
    main()
