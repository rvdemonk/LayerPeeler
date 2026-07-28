"""Spike 0a — corpus animator: layered SVG mascot -> frames with KNOWN transforms.

Ground truth by construction (bon 2026-07-29, branch 5 spike 0): each part
gets a periodic similarity transform (rotation about a hand-placed pivot,
or translation), plus a NON-RIGID knob — skewX on the tail — whose amplitude
sweeps to derive R_crit (the residual above which a rigid fit reads janky).

Emits per variant (rigid, skew<A>):
  out/corpus_anim/<variant>/frames/frame_NNNN.png   — flattened composites
  out/corpus_anim/<variant>/gt_params.json          — per-frame per-part affines
  out/corpus_anim/<variant>/masks/<part>.png        — frame-0 part alphas (RGBA)

The details group is REPARENTED: face features ride the head, belly rides
the body — the SVG's z-order grouping is not its articulation grouping.

Usage: python3 animate_corpus.py [--frames 48] [--skew 0,4,8,12]
"""

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

HYBRID = Path(__file__).resolve().parent
REPO = HYBRID.parent
ROOT = REPO.parent
SPIKE = ROOT / "spike"
SVG_PATH = ROOT / "corpus" / "mascots" / "dino.svg"
OUT = REPO / "out" / "corpus_anim"

_NVM = sorted((Path.home() / ".nvm/versions/node").glob("v*/bin/node"))
NODE = str(_NVM[-1]) if _NVM else "node"

# ---------------------------------------------------------------- rig spec
# pivot in SVG viewBox coords; motion = A_deg * sin(2*pi*(t/T) + phase)
RIG = {
    "tail":      {"pivot": (200, 365), "amp": 8.0,  "phase": 0.0},
    "arm_left":  {"pivot": (195, 300), "amp": 6.0,  "phase": 1.1},
    "arm_right": {"pivot": (305, 290), "amp": 10.0, "phase": 2.3},
    "head":      {"pivot": (256, 258), "amp": 3.0,  "phase": 0.6},
    "body":      {"pivot": None, "amp": 3.0, "phase": 0.0},   # y-bob, px
    "legs":      {"pivot": None, "amp": 0.0, "phase": 0.0},   # static
}

# details sub-elements (source line content match) -> articulation parent
FACE_MARKERS = ['cx="218" cy="168"', 'cx="296" cy="168"', 'cx="222" cy="163"',
                'cx="300" cy="163"', 'M 234 218', 'cx="196" cy="205"',
                'cx="318" cy="205"']
BELLY_MARKER = 'cx="252" cy="356"'


def split_details(svg: str):
    """Remove part_details; return (svg_without_details, face_elems, belly_elems)."""
    m = re.search(r'<g id="part_details">(.*?)</g>', svg, re.S)
    body_elems, face_elems = [], []
    for line in m.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if any(k in line for k in FACE_MARKERS):
            face_elems.append(line)
        elif BELLY_MARKER in line:
            body_elems.append(line)
        else:
            face_elems.append(line)
    return svg.replace(m.group(0), ""), face_elems, body_elems


def transform_str(part, t, T, skew_amp=0.0):
    spec = RIG[part]
    s = math.sin(2 * math.pi * t / T + spec["phase"])
    if spec["pivot"] is None:
        return f"translate(0 {spec['amp'] * s:.3f})" if spec["amp"] else ""
    px, py = spec["pivot"]
    tr = f"rotate({spec['amp'] * s:.4f} {px} {py})"
    if part == "tail" and skew_amp > 0:
        sk = skew_amp * s
        tr += f" translate({px} {py}) skewX({sk:.4f}) translate({-px} {-py})"
    return tr


def gt_affine(part, t, T, skew_amp=0.0):
    """3x3 GT matrix (SVG coords) matching transform_str exactly."""
    spec = RIG[part]
    s = math.sin(2 * math.pi * t / T + spec["phase"])
    if spec["pivot"] is None:
        M = np.eye(3)
        M[1, 2] = spec["amp"] * s if spec["amp"] else 0.0
        return M
    px, py = spec["pivot"]
    th = math.radians(spec["amp"] * s)
    R = np.array([[math.cos(th), -math.sin(th), 0],
                  [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    P = np.array([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=float)
    Pi = np.array([[1, 0, -px], [0, 1, -py], [0, 0, 1]], dtype=float)
    M = P @ R @ Pi
    if part == "tail" and skew_amp > 0:
        k = math.tan(math.radians(skew_amp * s))
        K = np.array([[1, k, 0], [0, 1, 0], [0, 0, 1]])
        M = M @ (P @ K @ Pi)
    return M


def build_frame_svg(base: str, face, belly, t, T, skew_amp, only_part=None,
                    black_bg=False):
    """Animated SVG for frame t. only_part: render that part alone (for masks)."""
    parts = {}
    for name in RIG:
        m = re.search(rf'<g id="part_{name}">(.*?)</g>', base, re.S)
        inner = m.group(1)
        if name == "head":
            inner += "\n" + "\n".join(face)
        if name == "body":
            inner += "\n" + "\n".join(belly)
        parts[name] = inner
    # z-order bottom->top: tail, arm_left, legs, body, arm_right, head
    z = ["tail", "arm_left", "legs", "body", "arm_right", "head"]
    if only_part:
        z = [only_part]
    groups = []
    for name in z:
        tr = transform_str(name, t, T, skew_amp)
        attr = f' transform="{tr}"' if tr else ""
        groups.append(f'<g id="anim_{name}"{attr}>{parts[name]}</g>')
    style = re.search(r"<defs>.*?</defs>", base, re.S).group(0)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" '
            'viewBox="0 0 512 512">' + style + "".join(groups) + "</svg>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--skew", default="0,4,8,12",
                    help="comma list of tail skew amplitudes (deg); 0 = rigid")
    args = ap.parse_args()
    T = args.frames
    base = SVG_PATH.read_text()
    base, face, belly = split_details(base)

    tmp = Path(tempfile.mkdtemp(prefix="corpus_anim_"))
    for skew in [float(x) for x in args.skew.split(",")]:
        variant = "rigid" if skew == 0 else f"skew{skew:g}"
        vdir = OUT / variant
        (vdir / "frames").mkdir(parents=True, exist_ok=True)
        (vdir / "masks").mkdir(exist_ok=True)
        manifest, gt = [], {}
        for t in range(T):
            svg = build_frame_svg(base, face, belly, t, T, skew)
            sp = tmp / f"{variant}_f{t:04d}.svg"
            sp.write_text(svg)
            manifest.append({"svg": str(sp),
                             "png": str(vdir / "frames" / f"frame_{t:04d}.png")})
            gt[str(t)] = {p: np.round(gt_affine(p, t, T, skew), 6).tolist()
                          for p in RIG}
        # frame-0 per-part masks via white/black double render
        for p in RIG:
            svg = build_frame_svg(base, face, belly, 0, T, skew, only_part=p)
            sp = tmp / f"{variant}_mask_{p}.svg"
            sp.write_text(svg)
            for bg, tag in (("#ffffff", "w"), ("#000000", "b")):
                manifest.append({"svg": str(sp), "bg": bg,
                                 "png": str(tmp / f"{variant}_{p}_{tag}.png")})
        mpath = tmp / f"{variant}_manifest.json"
        mpath.write_text(json.dumps(manifest))
        subprocess.run([NODE, str(HYBRID / "render_batch.js"), str(mpath)],
                       env={"NODE_PATH": str(SPIKE / "node_modules"),
                            "PATH": "/usr/bin:/bin"},
                       check=True, capture_output=True, text=True)
        # un-premultiply: alpha = 1 - (white - black)/255
        for p in RIG:
            w = cv2.imread(str(tmp / f"{variant}_{p}_w.png")).astype(int)
            b = cv2.imread(str(tmp / f"{variant}_{p}_b.png")).astype(int)
            alpha = (255 - (w - b).mean(axis=2)).clip(0, 255).astype(np.uint8)
            rgba = np.dstack([b.astype(np.uint8), alpha])  # b = premult on black
            cv2.imwrite(str(vdir / "masks" / f"{p}.png"), rgba)
        (vdir / "gt_params.json").write_text(json.dumps(gt))
        print(f"{variant}: {T} frames, {len(RIG)} masks, gt written")


if __name__ == "__main__":
    main()
