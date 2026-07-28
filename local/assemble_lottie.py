"""Assemble a rigged Lottie from extracted parts (part_registry.json + parts/*.svg).

Geometry policy (the product thesis): every path in the output is a vtracer
trace of extracted pixels, converted verbatim to Lottie beziers — never drawn.
Only TRANSFORMS (anchors, parenting, keyframes) are authored.

Steps:
  1. Split the `details` overlay by ownership: each opaque detail pixel is
     assigned to the nearest riggable part's alpha (distance transform), so
     the face rides the head, the belly rides the base, etc.
  2. Parse each part SVG (vtracer spline mode: M/C/Z + translate transforms)
     into Lottie shape groups.
  3. Build layers: base (bottom) + one layer per part with its owned details
     merged on top; parts parented to base; pivot heuristics per part role.
  4. Emit animation variants (idle, wave) using eased keyframe templates
     ported from the spike's hand-rig (gen_fox.py).

Usage:
  python3 assemble_lottie.py <run_folder> [--fps 30] [--dur 90]
Outputs: <run_folder>/lottie/<target>_idle.json, _wave.json (+ report)
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- svg parsing

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_svg_paths(svg_text: str):
    """Yield (fill_rgba01, subpaths) per <path>; subpath = list of lottie-style
    vertex dicts. vtracer spline output: absolute M/C/Z only, plus an optional
    translate() on the element."""
    for m in re.finditer(r"<path([^>]*)>", svg_text):
        attrs = m.group(1)
        d_match = re.search(r'd="([^"]*)"', attrs)
        if not d_match or not d_match.group(1).strip():
            continue  # vtracer occasionally emits degenerate empty paths
        d = d_match.group(1)
        fill = re.search(r'fill="(#[0-9A-Fa-f]{6})"', attrs)
        fill = fill.group(1) if fill else "#000000"
        tr = re.search(r'translate\((' + _NUM + r'),(' + _NUM + r')\)', attrs)
        tx, ty = (float(tr.group(1)), float(tr.group(2))) if tr else (0.0, 0.0)
        rgba = [int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1]

        subpaths, cur, start = [], None, None
        tokens = re.findall(r"[MCLZmclz]|" + _NUM, d)
        i = 0
        pos = (0.0, 0.0)

        def pt(a, b):
            return (float(a) + tx, float(b) + ty)

        while i < len(tokens):
            t = tokens[i]
            if t in "Mm":
                if cur:
                    subpaths.append(cur)
                pos = pt(tokens[i + 1], tokens[i + 2])
                start = pos
                cur = [{"v": pos, "i": (0, 0), "o": (0, 0)}]
                i += 3
            elif t in "Cc":
                c1 = pt(tokens[i + 1], tokens[i + 2])
                c2 = pt(tokens[i + 3], tokens[i + 4])
                p1 = pt(tokens[i + 5], tokens[i + 6])
                cur[-1]["o"] = (c1[0] - pos[0], c1[1] - pos[1])
                cur.append({"v": p1, "i": (c2[0] - p1[0], c2[1] - p1[1]), "o": (0, 0)})
                pos = p1
                i += 7
            elif t in "Ll":
                p1 = pt(tokens[i + 1], tokens[i + 2])
                cur.append({"v": p1, "i": (0, 0), "o": (0, 0)})
                pos = p1
                i += 3
            elif t in "Zz":
                if cur and cur[0]["v"] == cur[-1]["v"] and len(cur) > 1:
                    # closing point duplicates start: merge tangents
                    cur[0]["i"] = cur[-1]["i"]
                    cur.pop()
                i += 1
            else:
                i += 1
        if cur:
            subpaths.append(cur)
        yield rgba, subpaths


def rnd(x):
    return int(x) if x == int(x) else round(x, 1)


def shape_group(name, rgba, subpaths):
    items = []
    for sp in subpaths:
        items.append({"ty": "sh", "ks": {"a": 0, "k": {
            "c": True,
            "v": [[rnd(p["v"][0]), rnd(p["v"][1])] for p in sp],
            "i": [[rnd(p["i"][0]), rnd(p["i"][1])] for p in sp],
            "o": [[rnd(p["o"][0]), rnd(p["o"][1])] for p in sp],
        }}})
    items.append({"ty": "fl", "c": {"a": 0, "k": [round(c, 3) for c in rgba]},
                  "o": {"a": 0, "k": 100}, "r": 2})
    items.append({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]},
                  "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}})
    return {"ty": "gr", "nm": name, "it": items}


MIN_SUBPATH_AREA = 30.0  # px^2 — smaller blobs are diff-edge speckle, not art


def poly_area(sp):
    v = [p["v"] for p in sp]
    return abs(sum(v[i][0] * v[(i + 1) % len(v)][1] - v[(i + 1) % len(v)][0] * v[i][1]
                   for i in range(len(v)))) / 2


def svg_to_groups(svg_path: Path, prefix: str):
    groups = []
    for j, (rgba, subpaths) in enumerate(parse_svg_paths(svg_path.read_text())):
        kept = [sp for sp in subpaths if len(sp) >= 3 and poly_area(sp) >= MIN_SUBPATH_AREA]
        if kept:
            groups.append(shape_group(f"{prefix}_{j}", rgba, kept))
    # SVG document order paints bottom-first; Lottie renders the FIRST group in
    # a shapes list on top. Reverse so stacking survives the conversion.
    return list(reversed(groups))


# --------------------------------------------------------------- lottie scaffold

def anim(keys):
    return {"a": 1, "k": [
        {"t": t, "s": v if isinstance(v, list) else [v],
         "i": {"x": [0.4], "y": [1]}, "o": {"x": [0.6], "y": [0]}}
        for t, v in keys]}


def val(v):
    return {"a": 0, "k": v}


def layer(ind, name, shapes, anchor, dur, parent=None):
    L = {"ddd": 0, "ind": ind, "ty": 4, "nm": name,
         "ks": {"a": val(list(anchor)), "p": val(list(anchor)),
                "s": val([100, 100, 100]), "r": val(0), "o": val(100)},
         "ip": 0, "op": dur, "st": 0, "sr": 1, "shapes": shapes}
    if parent:
        L["parent"] = parent
    return L


# ------------------------------------------------------------------ pivots

def pivot_for(name, bbox, base_centroid):
    """Anchor-point heuristic by part role (viewer coordinates)."""
    x0, y0, x1, y1 = bbox
    if "head" in name:
        return ((x0 + x1) / 2, y1 - 8)              # neck: bottom-centre
    if "tail" in name:
        # attachment: bbox corner nearest the body centroid
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    elif "arm" in name or "leg" in name:
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    else:
        return ((x0 + x1) / 2, y1)                  # base: bottom-centre
    return min(corners, key=lambda c: (c[0] - base_centroid[0]) ** 2 + (c[1] - base_centroid[1]) ** 2)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_folder")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dur", type=int, default=90)
    args = ap.parse_args()

    run = Path(args.run_folder)
    parts_dir = run / "parts"
    registry = json.loads((run / "part_registry.json").read_text())
    by_name = {r["name"]: r for r in registry}
    target = run.name
    W = H = 512
    DUR = args.dur

    riggable = [r for r in registry if r["name"] not in ("details", "base")]
    base_rec = by_name["base"]
    base_c = base_rec["centroid"]

    # ---- 1. split details by ownership -----------------------------------
    det_rgba = cv2.cvtColor(cv2.imread(str(parts_dir / "details.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    owners = ["base"] + [r["name"] for r in riggable]
    dists = []
    for name in owners:
        a = cv2.imread(str(parts_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)[:, :, 3]
        dists.append(cv2.distanceTransform((a == 0).astype(np.uint8), cv2.DIST_L2, 3))
    owner_idx = np.argmin(np.stack(dists), axis=0)

    det_groups = {}
    for k, name in enumerate(owners):
        sub = det_rgba.copy()
        sub[:, :, 3] = np.where((det_rgba[:, :, 3] > 0) & (owner_idx == k), det_rgba[:, :, 3], 0)
        if (sub[:, :, 3] > 0).sum() < 50:
            det_groups[name] = []
            continue
        png = parts_dir / f"details_{name}.png"
        cv2.imwrite(str(png), cv2.cvtColor(sub, cv2.COLOR_RGBA2BGRA))
        svg = parts_dir / f"details_{name}.svg"
        subprocess.run(["vtracer", "--input", str(png), "--output", str(svg),
                        "--mode", "spline", "--filter_speckle", "6", "-p", "4",
                        "--path_precision", "0", "--corner_threshold", "60",
                        "--segment_length", "5"], check=True, capture_output=True)
        det_groups[name] = svg_to_groups(svg, f"det_{name}")
        print(f"details_{name}: {(sub[:,:,3]>0).sum()}px, {len(det_groups[name])} traced groups")

    # ---- 2+3. layers ------------------------------------------------------
    layers = []
    ind = 0
    name_to_ind = {}
    # top of stack first: riggable parts by z desc, then base
    for rec in sorted(riggable, key=lambda r: -r["z"]):
        ind += 1
        shapes = det_groups.get(rec["name"], []) + svg_to_groups(parts_dir / f"{rec['name']}.svg", rec["name"])
        piv = pivot_for(rec["name"], rec["bbox"], base_c)
        layers.append(layer(ind, rec["name"], shapes, piv, DUR))
        name_to_ind[rec["name"]] = ind
    ind += 1
    base_shapes = det_groups.get("base", []) + svg_to_groups(parts_dir / "base.svg", "base")
    layers.append(layer(ind, "base", base_shapes, pivot_for("base", base_rec["bbox"], base_c), DUR))
    name_to_ind["base"] = ind
    for rec in riggable:
        layers[[l["nm"] for l in layers].index(rec["name"])]["parent"] = name_to_ind["base"]

    # ---- 4. animation variants -------------------------------------------
    def find(nm):
        for l in layers:
            if l["nm"] == nm:
                return l
        return None

    def doc(name):
        return {"v": "5.9.0", "fr": args.fps, "ip": 0, "op": DUR, "w": W, "h": H,
                "nm": name, "ddd": 0, "assets": [], "layers": json.loads(json.dumps(layers))}

    out_dir = run / "lottie"
    out_dir.mkdir(exist_ok=True)

    variants = {}

    # idle: breathe + head bob + tail sway
    idle = doc(f"{target}-idle-extracted")
    def dl(d, nm):
        return d["layers"][[l["nm"] for l in d["layers"]].index(nm)] if nm in [l["nm"] for l in d["layers"]] else None
    b = dl(idle, "base")
    b["ks"]["s"] = anim([(0, [100, 100, 100]), (45, [101.5, 102.5, 100]), (DUR, [100, 100, 100])])
    if dl(idle, "head"):
        dl(idle, "head")["ks"]["r"] = anim([(0, 0), (45, 2), (DUR, 0)])
    if dl(idle, "tail"):
        dl(idle, "tail")["ks"]["r"] = anim([(0, 0), (22, -6), (45, 0), (68, 6), (DUR, 0)])
    variants["idle"] = idle

    # wave: raised arm rotates, plus gentle idle underneath
    wave = doc(f"{target}-wave-extracted")
    b = dl(wave, "base")
    b["ks"]["s"] = anim([(0, [100, 100, 100]), (45, [101, 101.8, 100]), (DUR, [100, 100, 100])])
    arm = next((dl(wave, r["name"]) for r in riggable if "arm" in r["name"]), None)
    if arm:
        arm["ks"]["r"] = anim([(0, 0), (12, -16), (24, 8), (36, -16), (48, 0), (DUR, 0)])
    if dl(wave, "head"):
        dl(wave, "head")["ks"]["r"] = anim([(0, 0), (24, -2), (48, 0), (DUR, 0)])
    if dl(wave, "tail"):
        dl(wave, "tail")["ks"]["r"] = anim([(0, 0), (30, 5), (60, -3), (DUR, 0)])
    variants["wave"] = wave

    for vname, d in variants.items():
        p = out_dir / f"{target}_{vname}.json"
        p.write_text(json.dumps(d, separators=(",", ":")))
        kb = p.stat().st_size / 1024
        gate = "PASS" if kb < 100 else "FAIL"
        print(f"{p.name}: {kb:.1f} KB — weight gate (<100 KB): {gate}")


if __name__ == "__main__":
    main()
