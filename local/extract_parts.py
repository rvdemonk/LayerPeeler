"""Local stage: peel frames -> part rasters -> vtracer SVGs -> part registry.

Consumes a directed_peel.py output folder (layer_png/ + peel_log.json) and
produces, per part:
  parts/<name>.png        RGBA raster cut from the BEFORE frame where the
                          before/after diff exceeds threshold (morphology-cleaned)
  parts/<name>.svg        vtracer trace of that raster
plus:
  parts/details.png/.svg  everything removed during the flatten phase (overlay)
  parts/base.png/.svg     the final residual frame (bottom layer)
  part_registry.json      name, z-order, bbox, centroid, area, source frames
  reassembly_raster.png   parts stacked back in z-order (raster fidelity check)

Fidelity numbers printed at the end: raster-reassembly SSIM + mean |d| vs the
source frame, which bounds what the vector gate can achieve.

Transitions come from peel_log.json but can be remapped via --remap
(e.g. run-2 fox: the tail actually came off in the leg_left transition):
  --remap tail=6:7 --skip leg_left --skip leg_right
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


DIFF_THRESHOLD = 30          # per-pixel |d| sum over RGB to count as changed
MORPH_KERNEL = 3             # close-then-open kernel size
MIN_PART_AREA_PX = 200       # below this, the diff is considered a failed peel


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Mean SSIM over greyscale float images in [0,255]."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    var_a = gaussian_filter(a * a, sigma) - mu_a**2
    var_b = gaussian_filter(b * b, sigma) - mu_b**2
    cov = gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * cov + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2)
    return float((num / den).mean())


def diff_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Binary mask of changed pixels, morphology-cleaned."""
    d = np.abs(before.astype(int) - after.astype(int)).sum(axis=2)
    mask = (d > DIFF_THRESHOLD).astype(np.uint8) * 255
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def cut_part(before: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RGBA raster: before-frame pixels where mask, transparent elsewhere."""
    rgba = cv2.cvtColor(before, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    return rgba


def source_palette(source: np.ndarray, min_frac: float = 0.002) -> np.ndarray:
    """Dominant flat colours of the source (mascots are flat-art: few colours +
    antialias edge blends, which fall below min_frac and are excluded)."""
    px = source.reshape(-1, 3)
    colours, counts = np.unique(px, axis=0, return_counts=True)
    keep = counts >= min_frac * len(px)
    pal = colours[keep]
    print(f"palette: {len(pal)} colours (of {len(colours)} unique)")
    return pal


def snap_svg_fills(svg_path: Path, palette: np.ndarray) -> int:
    """Replace each vtracer fill colour with the nearest source-palette colour.

    Post-trace, not pre-trace: snapping PIXELS before tracing destroys the
    antialiasing vtracer needs (outlines thin to nothing, speckle-filtered away)
    and hard-misassigns blends (drifted cream went to white). Snapping the few
    traced FILL colours preserves geometry exactly and only corrects drift."""
    import re as _re
    svg = svg_path.read_text()
    fills = set(_re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg))
    n = 0
    for f in fills:
        rgb = np.array([int(f[i:i + 2], 16) for i in (1, 3, 5)])
        d = np.abs(palette.astype(int) - rgb).sum(axis=1)
        snapped = palette[d.argmin()]
        s = "#{:02X}{:02X}{:02X}".format(*snapped)
        if s.lower() != f.lower():
            svg = svg.replace(f'fill="{f}"', f'fill="{s}"')
            n += 1
    svg_path.write_text(svg)
    return n


def vtrace(png_path: Path, svg_path: Path) -> bool:
    r = subprocess.run(
        ["vtracer", "--input", str(png_path), "--output", str(svg_path),
         "--mode", "spline", "--filter_speckle", "6", "-p", "4",
         "--path_precision", "0", "--corner_threshold", "60", "--segment_length", "5"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  vtracer failed on {png_path.name}: {r.stderr.strip()}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_folder", help="directed_peel output folder containing layer_png/ and peel_log.json")
    ap.add_argument("--source", help="original source PNG (defaults to layer_0)")
    ap.add_argument("--remap", action="append", default=[],
                    help="name=before:after — override a part's frame span")
    ap.add_argument("--skip", action="append", default=[], help="part name to skip")
    ap.add_argument("--palette-snap", action="store_true",
                    help="snap part colours to the source palette (kills generator drift)")
    args = ap.parse_args()

    run = Path(args.run_folder)
    frames_dir = run / "layer_png"
    log = json.loads((run / "peel_log.json").read_text())

    def frame(i):
        img = cv2.imread(str(frames_dir / f"layer_{i}.png"))
        assert img is not None, f"missing frame {i}"
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    out = run / "parts"
    out.mkdir(exist_ok=True)

    # ---- part spans -------------------------------------------------------
    spans = {p["name"]: (p["frame_before"], p["frame_after"]) for p in log["parts"]}
    # Detection bboxes belong to the part NAME (the VLM located the named part
    # when its peel was requested), even if --remap moves the pixel change to a
    # later transition. Index them before remapping.
    det_index = {name: b for name, (b, a) in spans.items()}
    for m in args.remap:
        name, fr = m.split("=")
        b, a = fr.split(":")
        spans[name] = (int(b), int(a))
        print(f"remapped {name} -> frames {b}:{a}")
    for s in args.skip:
        spans.pop(s, None)
        print(f"skipped {s}")

    flat_frame = log["flat_frame"]
    final_frame = log["final_frame"]
    source = cv2.cvtColor(cv2.imread(args.source), cv2.COLOR_BGR2RGB) if args.source else frame(0)
    if source.shape[:2] != frame(0).shape[:2]:
        source = cv2.resize(source, frame(0).shape[:2][::-1])

    registry = []
    palette = source_palette(source) if args.palette_snap else None

    def emit(name, rgba, z, frames_used):
        mask = rgba[:, :, 3]
        area = int((mask > 0).sum())
        if area < MIN_PART_AREA_PX:
            print(f"  {name}: diff area {area}px < {MIN_PART_AREA_PX} — FAILED PEEL, not emitted")
            return None
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        png = out / f"{name}.png"
        cv2.imwrite(str(png), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        svg = out / f"{name}.svg"
        traced = vtrace(png, svg)
        # Snap only parts cut from GENERATED frames — anything cut from frame 0
        # is source pixels with no drift, and snapping its legitimate antialias
        # blends corrupts it (black outlines went brown).
        if traced and palette is not None and frames_used[0] != 0:
            n = snap_svg_fills(svg, palette)
            if n:
                print(f"  {name}: snapped {n} drifted fill colours to source palette")
        rec = {
            "name": name, "z": z, "bbox": bbox,
            "centroid": [float(xs.mean()), float(ys.mean())],
            "area_px": area, "frames": frames_used, "svg": traced,
        }
        registry.append(rec)
        print(f"  {name}: area {area}px bbox {bbox} svg={'ok' if traced else 'FAIL'}")
        return rec

    # ---- base (bottom) ----------------------------------------------------
    print("base (residual):")
    base = frame(final_frame)
    body_mask = diff_mask(np.full_like(base, 255), base)  # non-white = residual body
    emit("base", cut_part(base, body_mask), 0, [final_frame])

    # ---- directed parts (z above base, in reverse peel order) -------------
    # The generator perturbs regions OUTSIDE the named part (colour drift,
    # incidental morphing), so a naive consecutive-frame diff cross-contaminates
    # parts (run-2 fox: arm_left's diff was mostly tail underside). Constrain
    # each part's diff to its own VLM detection bbox (layer_mask/bbox_<b>.png,
    # dilated for slack) when available.
    mask_dir = run / "layer_mask"
    print("directed parts:")
    z = 1
    for name, (b, a) in reversed(list(spans.items())):
        m = diff_mask(frame(b), frame(a))
        bbox_file = mask_dir / f"bbox_{det_index.get(name, b)}.png"
        if bbox_file.exists():
            det = cv2.imread(str(bbox_file), cv2.IMREAD_GRAYSCALE)
            det = cv2.resize(det, m.shape[::-1], interpolation=cv2.INTER_NEAREST)
            det = cv2.dilate(det, np.ones((15, 15), np.uint8))
            outside = int(((m > 0) & (det == 0)).sum())
            m = cv2.bitwise_and(m, det)
            print(f"  {name}: constrained to detection bbox ({outside}px of diff fell outside)")
        emit(name, cut_part(frame(b), m), z, [b, a])
        z += 1

    # ---- details overlay (topmost): everything the flatten phase removed --
    print("details overlay:")
    emit("details", cut_part(frame(0), diff_mask(frame(0), frame(flat_frame))), z, [0, flat_frame])

    (run / "part_registry.json").write_text(json.dumps(registry, indent=2))

    # ---- raster reassembly fidelity --------------------------------------
    canvas = np.full_like(source, 255)
    for rec in sorted(registry, key=lambda r: r["z"]):
        rgba = cv2.cvtColor(
            cv2.imread(str(out / f"{rec['name']}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        alpha = (rgba[:, :, 3:] > 0)
        canvas = np.where(alpha, rgba[:, :, :3], canvas)
    cv2.imwrite(str(run / "reassembly_raster.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    grey_s = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    grey_c = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    fg = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY) < 250  # mascot region
    mean_abs = float(np.abs(source.astype(int) - canvas.astype(int)).mean(axis=2)[fg].mean())
    print(f"\nraster reassembly vs source: SSIM={ssim(grey_s, grey_c):.4f}  "
          f"mean|d| (fg)={mean_abs:.2f}  parts={len(registry)}")
    print(f"registry: {run / 'part_registry.json'}")


if __name__ == "__main__":
    main()
