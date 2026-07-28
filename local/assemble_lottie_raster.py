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

from assemble_lottie import anim, blink, val

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
        cx, cy = min(corners, key=lambda c: (c[0] - base_centroid[0]) ** 2 + (c[1] - base_centroid[1]) ** 2)
        # Pivot depth (Lewis appraisal 2026-07-28, "tail lift-off"): the bbox
        # corner sits ON the join's edge, so a wide join sweeps hard at its
        # far side. Push the pivot 15% toward the body centroid — behind the
        # join line — so displacement spreads across the join width instead
        # of hinging at one corner.
        return (cx + 0.15 * (base_centroid[0] - cx), cy + 0.15 * (base_centroid[1] - cy))
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
    # Despeckle details (2026-07-28 jank fix): extraction leaves junk shards
    # in the details overlay (dashed debris that rode the forearm, the belly
    # "fish" speckle). Morph-open + drop small components — same treatment
    # the vector assembler applies before tracing.
    d_open = cv2.morphologyEx((det_rgba[:, :, 3] > 0).astype(np.uint8),
                              cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_d, c_d, st_d, _ = cv2.connectedComponentsWithStats(d_open, connectivity=8)
    keep_d = np.isin(c_d, [i for i in range(1, n_d)
                           if st_d[i, cv2.CC_STAT_AREA] >= 60])
    dropped_d = int((det_rgba[:, :, 3] > 0).sum() - keep_d.sum())
    det_rgba[:, :, 3] = np.where(keep_d, det_rgba[:, :, 3], 0)
    if dropped_d:
        print(f"details despeckle: dropped {dropped_d}px of junk shards")
    owners = ["base"] + [r["name"] for r in riggable]
    raw_alpha, alphas, dists = [], [], []
    for name in owners:
        a = cv2.imread(str(parts_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3]
        if name == "base":
            # Despeckle the healed residual's alpha: the peel leaves junk
            # fragments floating in vacated part regions (found 2026-07-28:
            # arm-region debris made dilate(base) touch the whole arm
            # perimeter, so its "seam" cover traced the arm outline — a
            # frozen ghost — and reveals showed dirty speckle). Keep only
            # substantial components (body + ground shadow).
            n_b, c_b, st_b, _ = cv2.connectedComponentsWithStats(
                (a > 0).astype(np.uint8), connectivity=8)
            keep_b = np.isin(c_b, [i for i in range(1, n_b)
                                   if st_b[i, cv2.CC_STAT_AREA] >= 300])
            dropped = int((a > 0).sum() - (keep_b & (a > 0)).sum())
            a = np.where(keep_b, a, 0).astype(np.uint8)
            if dropped:
                print(f"base alpha despeckle: dropped {dropped}px of residual junk")
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

    # ---- eyes: raster port of the fox eye-split (assemble_lottie.py) -------
    # The details layer is ONE connected component here (outlines touch
    # everything), so the vector version's per-group pairing can't transfer.
    # Instead find the eye pair straight in the SOURCE: the two compact DARK
    # blobs (lum < 0.45) at similar height above the body centroid. The mouth
    # is dark but fails compactness and has no partner at its height. Each
    # eye's filled contour (dilated 2px) captures the iris ring AND the white
    # highlight inside it. The pair is pulled OUT of the base raster onto its
    # own blinkable layer; base keeps the healed residual beneath, which is
    # what a closed lid reveals.
    eyes_mask = np.zeros_like(det_alpha)
    eyes_bbox = None
    lum_img = source.mean(axis=2) / 255
    n_e, comp_e, stats_e, _ = cv2.connectedComponentsWithStats(
        (lum_img < 0.45).astype(np.uint8), connectivity=8)
    cands = []
    for c in range(1, n_e):
        x, y, w, h, area = stats_e[c]
        if not (6 <= w <= 80 and 6 <= h <= 80 and area >= 60):
            continue
        if area / (w * h) < 0.45:
            continue  # not compact: outline stretch, open mouth
        if y + h / 2 >= base_c[1]:
            continue  # eyes sit above the body centroid
        cands.append((c, (x, y, x + w, y + h), int(area)))
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            (ca, ba, aa), (cb, bb_, ab) = cands[i], cands[j]
            cya, cyb = (ba[1] + ba[3]) / 2, (bb_[1] + bb_[3]) / 2
            cxa, cxb = (ba[0] + ba[2]) / 2, (bb_[0] + bb_[2]) / 2
            if abs(cya - cyb) < 18 and abs(cxa - cxb) > 25 and \
                    max(aa, ab) / max(min(aa, ab), 1) < 3:
                score = aa + ab - 2 * abs(cya - cyb)
                if best is None or score > best[0]:
                    best = (score, (ca, ba), (cb, bb_))
    if best:
        for c, bb_ in (best[1], best[2]):
            em = cv2.dilate((comp_e == c).astype(np.uint8), np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(em, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(em)
            cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
            eyes_mask |= filled.astype(bool)
        ebbs = [best[1][1], best[2][1]]
        x0 = min(b[0] for b in ebbs); y0 = min(b[1] for b in ebbs)
        x1 = max(b[2] for b in ebbs); y1 = max(b[3] for b in ebbs)
        eyes_bbox = (x0, y0, x1, y1)
        print(f"eyes: {int(eyes_mask.sum())}px, bbox {[int(v) for v in eyes_bbox]}")

    # ---- per-layer rasters: SOURCE pixels over (part mask | owned details) --
    def owned_mask(k):
        m = det_alpha & (owner_idx == k)
        if k == 0:
            m = m & ~eyes_mask  # eyes live on their own blinkable layer
        return m

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
            rgba[:, :, 3] = raw_alpha[0]  # despeckled residual alpha
            om = owned_mask(k)
            rgba[om, :3] = source[om]
            rgba[om, 3] = 255
        else:
            # Edge treatment (Lewis appraisal 2026-07-28, "arm ghost outline"):
            # the diff mask's perimeter carries the source's AA halo + shared
            # outline strokes — a travelling dark rim once the part moves.
            # Shave 1px (erode) + feather ~1px (gaussian on alpha), but ONLY
            # on INTERIOR edges — where the part contacts the rest of the
            # figure and the halo would travel over body pixels. The outer
            # silhouette (against background) keeps the hard source edge:
            # there is no ghost there, and eroding it just thins outline art
            # (first cut dropped frame-0 SSIM 0.988->0.967, all outer rim).
            m = ((raw_alpha[k] > 0) | owned_mask(k)).astype(np.uint8)
            hard = m.astype(np.float32) * 255
            soft = cv2.GaussianBlur(
                cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32) * 255,
                (5, 5), 1.0)
            others = np.zeros_like(m)
            for j in range(len(owners)):
                if j != k:
                    others |= (raw_alpha[j] > 0).astype(np.uint8)
            contact = cv2.GaussianBlur(
                cv2.dilate(others, np.ones((7, 7), np.uint8)).astype(np.float32),
                (7, 7), 1.5)
            a = soft * contact + hard * (1 - contact)
            rgba = np.dstack([source, np.clip(a, 0, 255).astype(np.uint8)])
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

    # ---- seam covers: static source-pixel bands over each part/base seam ----
    # The generalized joint-cover patch (Lewis appraisal 2026-07-28). The base
    # residual carries a healed-reveal outline that MISMATCHES the source at
    # every part boundary; the part's eroded edge (above) would expose it even
    # at rest. A cover = source pixels over a ~5px band straddling the part's
    # contact seam WITH BASE ONLY (seams against other moving parts can't be
    # covered statically), parented to base, stacked under every part:
    #   at rest      -> band IS the source: frame-0 exactness restored
    #   part swings out -> reveal gap fills with source truth, not healed guess
    #   part swings in  -> part slides over the cover, overlap line hidden
    base_alpha_mask = (raw_alpha[0] > 0)
    cover_assets_meta = []
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    for k, name in enumerate(owners):
        if name == "base":
            continue
        P = raw_alpha[k] > 0
        seam = cv2.dilate(P.astype(np.uint8), k3).astype(bool) & \
            cv2.dilate(base_alpha_mask.astype(np.uint8), k3).astype(bool)
        if seam.sum() < 40:
            continue
        # Cover anatomy (2026-07-28 jank fixes, two failed cuts first):
        # - straddling band everywhere -> frozen part-outline ghosts (the
        #   junk-seam bug) and a "zipper" at the tail root;
        # - pure base-side band -> WHITE GAP at the tail root: the healed
        #   base is TRANSPARENT behind part roots (the peel diff owns those
        #   pixels), so there is nothing behind the part to reveal.
        # The working shape is the classic puppet patch, two pieces:
        # 1. thin seam band: base side + 1px fringe under the part edge
        #    (compensates the erode+feather, hides the healed outline);
        # 2. ROOT DISC: the part's own resting pixels within R of the pivot,
        #    parented to base — near the pivot displacement is tiny, so the
        #    static copy backfills the vacated root instead of white.
        #    R adapts to the join: ~90th pct of seam-pixel distance to pivot.
        band = cv2.dilate(seam.astype(np.uint8), k5).astype(bool) \
            & (base_alpha_mask | P) \
            & ~cv2.erode(P.astype(np.uint8), k3).astype(bool)
        #    Constrained to a ~12px collar along the seam: max reveal depth
        #    is r*theta (~100px * 7deg = 12px), so a deeper backfill is
        #    dead weight (unconstrained discs blew the gate: 105 KB).
        piv = pivot_for_raster(name, by_name[name]["bbox"], base_c)
        sy, sx = np.nonzero(seam)
        R = 1.1 * np.percentile(np.hypot(sx - piv[0], sy - piv[1]), 90)
        yy, xx = np.mgrid[0:H, 0:W]
        collar = cv2.dilate(seam.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        root_disc = P & collar.astype(bool) & (np.hypot(xx - piv[0], yy - piv[1]) <= R)
        band = band | root_disc
        a = cv2.GaussianBlur(band.astype(np.float32) * 255, (5, 5), 1.0)
        rgba = np.dstack([source, np.clip(a, 0, 255).astype(np.uint8)])
        ys, xs = np.nonzero(rgba[:, :, 3])
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crop = rgba[y0:y1, x0:x1]
        asset, nbytes = encode_asset(crop, f"img_cover_{name}", args.format, args.quality)
        assets.append(asset)
        total_bytes += nbytes
        cover_assets_meta.append({"name": f"cover_{name}",
                                  "crop_origin": (int(x0), int(y0)),
                                  "bbox": [int(x0), int(y0), int(x1), int(y1)]})
        print(f"cover_{name}: {int(band.sum())}px band, {nbytes//1024}KB")

    # eyes asset: source pixels over the eye components, 1px feather
    eyes_meta = None
    if eyes_bbox:
        em = cv2.dilate(eyes_mask.astype(np.uint8), k3)
        a = cv2.GaussianBlur(em.astype(np.float32) * 255, (5, 5), 1.0)
        rgba = np.dstack([source, np.clip(a, 0, 255).astype(np.uint8)])
        ys, xs = np.nonzero(rgba[:, :, 3])
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        asset, nbytes = encode_asset(rgba[y0:y1, x0:x1], "img_eyes",
                                     args.format, args.quality)
        assets.append(asset)
        total_bytes += nbytes
        eyes_meta = {"crop_origin": (int(x0), int(y0)),
                     "centre": ((eyes_bbox[0] + eyes_bbox[2]) / 2,
                                (eyes_bbox[1] + eyes_bbox[3]) / 2)}

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
    # seam covers: below every part, above base; parented to base (static)
    for cm in cover_assets_meta:
        ind += 1
        cx = (cm["bbox"][0] + cm["bbox"][2]) / 2
        cy = (cm["bbox"][1] + cm["bbox"][3]) / 2
        layers.append(image_layer(ind, cm["name"], f"img_{cm['name']}",
                                  (cx, cy), cm["crop_origin"], DUR,
                                  parent_origin=base_origin))
        name_to_ind[cm["name"]] = ind
    # eyes: above base, anchored at the pair's centre so lids meet mid-blink
    if eyes_meta:
        ind += 1
        layers.append(image_layer(ind, "eyes", "img_eyes", eyes_meta["centre"],
                                  eyes_meta["crop_origin"], DUR,
                                  parent_origin=base_origin))
        name_to_ind["eyes"] = ind
    ind += 1
    lr = layer_rasters["base"]
    layers.append(image_layer(ind, "base", "img_base",
                              pivot_for_raster("base", lr["bbox"], base_c),
                              lr["crop_origin"], DUR))
    name_to_ind["base"] = ind
    child_names = [r["name"] for r in riggable] + [cm["name"] for cm in cover_assets_meta]
    if eyes_meta:
        child_names.append("eyes")
    for nm in child_names:
        layers[[l["nm"] for l in layers].index(nm)]["parent"] = name_to_ind["base"]

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

    # idle: gill flutter (antiphase) + tail sway. NO breathe: whole-body scale
    # inflates silhouette+shadow (Lewis appraisal 2026-07-28) — wrong primitive,
    # killed rather than tuned. Breath needs a belly layer or nothing.
    idle = doc(f"{target}-idle-raster")
    for gi, g in enumerate(gills):
        # antiphase flutter; left gills mirror the sign so both wave outward
        sign = -1 if "left" in g else 1
        ph = 0 if gi % 2 == 0 else 11
        dl(idle, g)["ks"]["r"] = anim([(0, 0), (11 + ph, sign * 5), (22 + ph, 0),
                                       (33 + ph, sign * -3), (45 + ph, 0), (DUR, 0)])
    if dl(idle, "tail"):
        # asymmetric amplitude: -6/+6 -> -4/+7. The inward (CCW, lifting)
        # swing is what opens the join gap; the outward swing rides the
        # seam cover. Bias the sway outward, trim the lift.
        dl(idle, "tail")["ks"]["r"] = anim([(0, 0), (22, -4), (45, 0), (68, 7), (DUR, 0)])
    if dl(idle, "eyes"):
        dl(idle, "eyes")["ks"]["s"] = blink(DUR, at=[32, 74])
    variants["idle"] = idle

    # wave: arm rotates outward (positive; negative crosses the face — fox
    # run-4 defect), gentler flutter underneath (breathe killed, see idle)
    if arm:
        wave = doc(f"{target}-wave-raster")
        # amplitude trimmed 14/-4 -> 10/-2 (2026-07-28): at 14° the reveal
        # behind the arm outruns what any static cover can backfill
        dl(wave, arm)["ks"]["r"] = anim([(0, 0), (12, 10), (24, -2), (36, 10),
                                         (48, 0), (DUR, 0)])
        for g in gills:
            sign = -1 if "left" in g else 1
            dl(wave, g)["ks"]["r"] = anim([(0, 0), (22, sign * 3), (45, 0), (DUR, 0)])
        if dl(wave, "tail"):
            dl(wave, "tail")["ks"]["r"] = anim([(0, 0), (30, 5), (60, -3), (DUR, 0)])
        if dl(wave, "eyes"):
            dl(wave, "eyes")["ks"]["s"] = blink(DUR, at=[56])
        variants["wave"] = wave

    for vname, d in variants.items():
        p = out_dir / f"{target}_{vname}.json"
        p.write_text(json.dumps(d, separators=(",", ":")))
        kb = p.stat().st_size / 1024
        print(f"{p.name}: {kb:.1f} KB (assets {total_bytes//1024} KB {args.format})")


if __name__ == "__main__":
    main()
