"""Assemble a rigged Lottie with RASTER image layers (the gradient-class carrier).

The dragon verdict (manifest 2026-07-28): vtracer flat-fill splines cannot
represent the gummy/gradient class (axolotl vector SSIM 0.7527 vs gate 0.92,
posterized). Lottie image layers carry it at the same weight as the fox's
vector rig: parts cropped to bbox, WebP q90, base64 data URIs. Transforms
animate image layers identically to shape layers, so pivots/parenting/
keyframes carry over from assemble_lottie.py unchanged.

Raster-mode geometry policy (analogue of the vector thesis): every visible
pixel is a SOURCE pixel — each part layer is cut from the source image over
(part diff mask ∪ owned details mask). This sidesteps flatten colour drift
entirely: the drifted flat base is only used for the healed reveal regions of
the base residual, which are exactly the pixels that have no source truth.

Details ownership reuses assemble_lottie.py's component-voting logic (dominant
alpha-overlap -> whole component; shared -> pixel-nearest; specificity
override), minus the tracing.

Usage:
  python3 assemble_lottie_raster.py <run_folder> --source <source.png>
          [--fps 30] [--dur 90] [--format webp|png] [--quality 90]
Outputs: <run_folder>/lottie/<target>_idle.json, _wave.json
"""

import argparse
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from assemble_lottie import anim, val

# ------------------------------------------------------------------ pivots

def pivot_for_raster(name, bbox, base_centroid):
    """Anchor heuristic by part role. Gills join arms/legs/tails in the
    'appendage' class: pivot at the bbox corner nearest the body centroid
    (the anatomical attachment), so rotation flutters the free end."""
    x0, y0, x1, y1 = bbox
    if "head" in name:
        return ((x0 + x1) / 2, y1 - 8)
    if any(k in name for k in ("tail", "arm", "leg", "gill", "wing", "ear", "fin")):
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        return min(corners, key=lambda c: (c[0] - base_centroid[0]) ** 2 + (c[1] - base_centroid[1]) ** 2)
    if "feet" in name or "foot" in name:
        return ((x0 + x1) / 2, y0)                  # ankle: top-centre
    return ((x0 + x1) / 2, y1)                      # base: bottom-centre


# ------------------------------------------------------------------ layers

def encode_asset(rgba: np.ndarray, aid: str, fmt: str, quality: int):
    im = Image.fromarray(rgba)
    buf = io.BytesIO()
    if fmt == "webp":
        im.save(buf, "WEBP", quality=quality, method=6)
        mime = "image/webp"
    else:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    data = base64.b64encode(buf.getvalue()).decode()
    return {"id": aid, "w": im.width, "h": im.height, "u": "",
            "p": f"data:{mime};base64,{data}", "e": 1}, len(buf.getvalue())


def image_layer(ind, name, ref_id, pivot, crop_origin, dur, parent=None,
                parent_origin=(0, 0)):
    """ty:2 image layer. Anchor is in the layer's ASSET pixel coords (pivot
    minus its crop origin). Position is the pivot in the PARENT's coordinate
    space — comp coords for unparented layers, the parent's asset coords for
    parented ones (first build rendered every part offset by base's crop
    origin (21,58): white gaps at the feet, drifted gills)."""
    ax, ay = pivot[0] - crop_origin[0], pivot[1] - crop_origin[1]
    px, py = pivot[0] - parent_origin[0], pivot[1] - parent_origin[1]
    L = {"ddd": 0, "ind": ind, "ty": 2, "nm": name, "refId": ref_id,
         "ks": {"a": val([round(ax, 1), round(ay, 1), 0]),
                "p": val([round(px, 1), round(py, 1), 0]),
                "s": val([100, 100, 100]), "r": val(0), "o": val(100)},
         "ip": 0, "op": dur, "st": 0, "sr": 1}
    if parent:
        L["parent"] = parent
    return L


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_folder")
    ap.add_argument("--source", required=True, help="original source PNG (pixel truth)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dur", type=int, default=90)
    ap.add_argument("--format", choices=["webp", "png"], default="webp")
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    run = Path(args.run_folder)
    parts_dir = run / "parts"
    registry = json.loads((run / "part_registry.json").read_text())
    by_name = {r["name"]: r for r in registry}
    target = run.name
    W = H = 512
    DUR = args.dur

    source = cv2.cvtColor(cv2.imread(args.source), cv2.COLOR_BGR2RGB)
    if source.shape[:2] != (H, W):
        source = cv2.resize(source, (W, H))

    riggable = [r for r in registry if r["name"] not in ("details", "base")]
    base_rec = by_name["base"]
    base_c = base_rec["centroid"]

    # ---- details ownership (component voting, from assemble_lottie.py) ----
    det_rgba = cv2.cvtColor(cv2.imread(str(parts_dir / "details.png"), cv2.IMREAD_UNCHANGED),
                            cv2.COLOR_BGRA2RGBA)
    owners = ["base"] + [r["name"] for r in riggable]
    raw_alpha, alphas, dists = [], [], []
    for name in owners:
        a = cv2.imread(str(parts_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3]
        raw_alpha.append(a)
        alphas.append(cv2.dilate((a > 0).astype(np.uint8), np.ones((9, 9), np.uint8)))
        dists.append(cv2.distanceTransform((a == 0).astype(np.uint8), cv2.DIST_L2, 3))
    n_comp, comp = cv2.connectedComponents((det_rgba[:, :, 3] > 0).astype(np.uint8))
    pixel_nearest = np.argmin(np.stack(dists), axis=0)
    areas = [int((a > 0).sum()) for a in raw_alpha]
    for k in sorted(range(len(owners)), key=lambda k: -areas[k]):
        pixel_nearest[dists[k] == 0] = k          # specificity: smallest covering part wins
    owner_idx = np.zeros(comp.shape, dtype=np.int32)
    for c in range(1, n_comp):
        cmask = comp == c
        overlaps = [int(a[cmask].sum()) for a in alphas]
        total = sum(overlaps)
        if total > 0 and max(overlaps) / total >= 0.7:
            owner_idx[cmask] = int(np.argmax(overlaps))
        elif total > 0:
            owner_idx[cmask] = pixel_nearest[cmask]
        else:
            owner_idx[cmask] = int(np.argmin([d[cmask].mean() for d in dists]))

    det_alpha = det_rgba[:, :, 3] > 0

    # ---- per-layer rasters: SOURCE pixels over (part mask | owned details) --
    def owned_mask(k):
        return det_alpha & (owner_idx == k)

    assets, layer_rasters = [], {}
    total_bytes = 0
    for k, name in enumerate(owners):
        if name == "base":
            # base residual pixels (healed reveals have no source truth),
            # then source pixels wherever the base owns details (this is what
            # restores the drifted body colour to source truth)
            base_png = cv2.cvtColor(cv2.imread(str(parts_dir / "base.png"), cv2.IMREAD_UNCHANGED),
                                    cv2.COLOR_BGRA2RGBA)
            rgba = base_png.copy()
            om = owned_mask(k)
            rgba[om, :3] = source[om]
            rgba[om, 3] = 255
        else:
            m = (raw_alpha[k] > 0) | owned_mask(k)
            rgba = np.dstack([source, (m * 255).astype(np.uint8)])
        ys, xs = np.nonzero(rgba[:, :, 3])
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crop = rgba[y0:y1, x0:x1]
        asset, nbytes = encode_asset(crop, f"img_{name}", args.format, args.quality)
        assets.append(asset)
        total_bytes += nbytes
        layer_rasters[name] = {"crop_origin": (int(x0), int(y0)),
                               "bbox": [int(x0), int(y0), int(x1), int(y1)]}
        print(f"{name}: crop {crop.shape[1]}x{crop.shape[0]} {args.format} {nbytes//1024}KB "
              f"(+{int(owned_mask(k).sum())}px owned details)")

    # ---- layers: parts by z desc (top first), base last, parts parented ----
    layers, name_to_ind = [], {}
    base_origin = layer_rasters["base"]["crop_origin"]
    ind = 0
    for rec in sorted(riggable, key=lambda r: -r["z"]):
        ind += 1
        lr = layer_rasters[rec["name"]]
        piv = pivot_for_raster(rec["name"], lr["bbox"], base_c)
        layers.append(image_layer(ind, rec["name"], f"img_{rec['name']}", piv,
                                  lr["crop_origin"], DUR, parent_origin=base_origin))
        name_to_ind[rec["name"]] = ind
    ind += 1
    lr = layer_rasters["base"]
    layers.append(image_layer(ind, "base", "img_base",
                              pivot_for_raster("base", lr["bbox"], base_c),
                              lr["crop_origin"], DUR))
    name_to_ind["base"] = ind
    for rec in riggable:
        layers[[l["nm"] for l in layers].index(rec["name"])]["parent"] = name_to_ind["base"]

    # ---- animation variants ----------------------------------------------
    def doc(name):
        return {"v": "5.9.0", "fr": args.fps, "ip": 0, "op": DUR, "w": W, "h": H,
                "nm": name, "ddd": 0, "assets": assets,
                "layers": json.loads(json.dumps(layers))}

    def dl(d, nm):
        names = [l["nm"] for l in d["layers"]]
        return d["layers"][names.index(nm)] if nm in names else None

    part_names = [r["name"] for r in riggable]
    gills = sorted(n for n in part_names if "gill" in n)
    arm = next((n for n in part_names if "arm" in n), None)

    out_dir = run / "lottie"
    out_dir.mkdir(exist_ok=True)
    variants = {}

    # idle: breathe + gill flutter (antiphase) + tail sway
    idle = doc(f"{target}-idle-raster")
    dl(idle, "base")["ks"]["s"] = anim([(0, [100, 100, 100]), (45, [101.5, 102.5, 100]),
                                        (DUR, [100, 100, 100])])
    for gi, g in enumerate(gills):
        # antiphase flutter; left gills mirror the sign so both wave outward
        sign = -1 if "left" in g else 1
        ph = 0 if gi % 2 == 0 else 11
        dl(idle, g)["ks"]["r"] = anim([(0, 0), (11 + ph, sign * 5), (22 + ph, 0),
                                       (33 + ph, sign * -3), (45 + ph, 0), (DUR, 0)])
    if dl(idle, "tail"):
        dl(idle, "tail")["ks"]["r"] = anim([(0, 0), (22, -6), (45, 0), (68, 6), (DUR, 0)])
    variants["idle"] = idle

    # wave: arm rotates outward (positive; negative crosses the face — fox
    # run-4 defect), gentler flutter + breathe underneath
    if arm:
        wave = doc(f"{target}-wave-raster")
        dl(wave, "base")["ks"]["s"] = anim([(0, [100, 100, 100]), (45, [101, 101.8, 100]),
                                            (DUR, [100, 100, 100])])
        dl(wave, arm)["ks"]["r"] = anim([(0, 0), (12, 14), (24, -4), (36, 14),
                                         (48, 0), (DUR, 0)])
        for g in gills:
            sign = -1 if "left" in g else 1
            dl(wave, g)["ks"]["r"] = anim([(0, 0), (22, sign * 3), (45, 0), (DUR, 0)])
        if dl(wave, "tail"):
            dl(wave, "tail")["ks"]["r"] = anim([(0, 0), (30, 5), (60, -3), (DUR, 0)])
        variants["wave"] = wave

    for vname, d in variants.items():
        p = out_dir / f"{target}_{vname}.json"
        p.write_text(json.dumps(d, separators=(",", ":")))
        kb = p.stat().st_size / 1024
        print(f"{p.name}: {kb:.1f} KB (assets {total_bytes//1024} KB {args.format})")


if __name__ == "__main__":
    main()
