"""watch.py — watch a single matted loop: digest render + VLM judge.

The builder model (DeepSeek) is text-only, so it cannot watch animations.
This tool is its eyes: it turns one run's matted frames (``rgba/``) into a
single digest image a VLM can actually watch — frames sampled across the
cycle, composited over a checkerboard so the alpha matte is visible, each
cell labelled with its source frame index and overlaid with a 3x3
coordinate grid — then has the judge (Kimi K3, low reasoning effort) emit a
structured text digest the builder model can consume.

The judge is given as much context as possible, because the calibration
proved that its perception is accurate and its *interpretation* is what
fails when a fact is missing (the frog's front-facing loop edges are a
pipeline feature; K3 flagged them as a defect until told). Context layers,
in order of provenance:

1. Global pipeline design facts — alpha matte is intentional; loop edges
   return to the master pose by design (stitching); loop seam closed by
   construction; in-place walk cycles are standard.
2. Mascot identity invariants + expected micro-motion (blinks are expected;
   only asymmetric/sustained/jittery eye motion is a defect).
3. Per-run expected behaviour from ``--expect`` and the source prompt.

Alongside the digest and master anchor, the judge receives a BELLY STRIP:
the belly-band width vs whole-body area, both normalised to their median,
across the sampled cells, drawn as two lines. Born from the r063 belly gap
(2026-08-05): the belly class is not mechanically separable — the good
active clips swing their bodies wider than the pathological drowsy — so the
measurement is mechanical (a strip the judge can actually read) and the
*semantic call* (drowsy breath vs washing-machine belly) stays with the
judge against the intent.

K3 is a highly capable, strongly proactiveness-biased model: the prompt
therefore fences it hard — screening flag only, no scope expansion, no final
verdict, fixed JSON shape.

Layout, cell resolution, and sample stride are first-class parameters because
the "how does a model watch a video" question has to be benched, and the
answer is expected to differ from model to model.

Run:

    .venv-hybrid/bin/python -m pipeline.watch frog-wave-v8-aa-w075
    .venv-hybrid/bin/python -m pipeline.watch frog-walk --render-only
    .venv-hybrid/bin/python -m pipeline.watch r059 \
        --expect "returns to the front-facing master pose at both loop edges by design"

``--render-only`` renders the digest and skips the API call — the free way
to bench digest parameters before spending a judgement on them.
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.eyeball import _composite

DEFAULT_MODEL = "kimi-k3"
DEFAULT_EFFORT = "low"
MOONSHOT_BASE = "https://api.moonshot.ai/v1"
AUTH_JSON = Path.home() / ".local/share/opencode/auth.json"

# The shared grammar: judge output must speak these classes so it is directly
# comparable with the ledger and Lewis's verdicts. A judge that needs a new
# class says so in `evidence`, never in `class`.
DEFECT_VOCABULARY = [
    "detach", "disappear", "doubling", "seam", "matte-artifact",
    "feathering", "jitter", "teleport", "loop-seam", "shimmer",
    "posterization", "palette-injection", "identity-decay", "timing",
    "lifelessness", "semantic-mismatch", "unrealistic-rigging",
]

# Global pipeline design facts. These are intentional behaviour of the
# image-to-lottie pipeline, never defects. Missing this context is what made
# K3 flag the (intentional) alpha matte and the (intentional) front-facing
# loop edges in the first calibration.
GLOBAL_DESIGN_FACTS = [
    "The background is matted to transparent alpha by the pipeline. A transparent "
    "or checkerboard background is the product's shape, never a defect.",
    "Animations begin and end on the master asset image: the character returns to "
    "its canonical pose at both loop edges by design, so Lotties can be stitched "
    "into sequences of animations. A pose or facing change at the loop edges is "
    "expected, not a defect.",
    "The loop seam is closed by construction: the final frame equals the first frame.",
    "In-place loop cycles (the character does not traverse the frame) are standard "
    "game-asset behaviour; a walk intent is usually fulfilled by an in-place cycle "
    "that the host scene moves.",
]

# Per-mascot identity invariants (must not change) and expected micro-motion
# (expected, never defects). Extend as mascots are added.
MASCOT_FACTS = {
    "frog": {
        "invariants": [
            "oversized round white eyes with solid dark pupils",
            "glossy squishy gummy-clay body",
            "stubby arms",
            "small gentle closed smile",
        ],
        "expected": [
            "the frog blinks naturally: a quick, symmetric, soft close-and-open of "
            "both eyes. A blink is never a defect.",
            "the frog returns to its front-facing master pose at both loop edges.",
        ],
    },
}


def detect_mascot(name):
    for mascot in MASCOT_FACTS:
        if mascot in name.lower():
            return mascot
    return None


def resolve_run(run_arg):
    p = Path(run_arg)
    if p.is_dir():
        return p
    cand = Path("out/pipeline") / run_arg
    if cand.is_dir():
        return cand
    raise SystemExit(f"run not found: {run_arg!r} (tried cwd path and out/pipeline/)")


def load_rgba(run):
    rgba = run / "rgba"
    if not rgba.is_dir():
        raise SystemExit(f"no rgba/ dir in {run}")
    files = sorted(rgba.glob("frame_*.png"))
    if not files:
        raise SystemExit(f"no frame_*.png in {rgba}")
    return files


def run_meta(run):
    """Small, factual subset of run.json: nothing that launders the gate."""
    meta = {}
    rj = run / "run.json"
    if rj.exists():
        try:
            d = json.loads(rj.read_text())
            meta["name"] = d.get("name")
            profile = d.get("ship_profile") or ""
            m = re.search(r"-(\d+)$", profile)
            meta["fps"] = int(m.group(1)) if m else None
            meta["profile"] = profile or None
            meta["frames"] = d.get("gates", {}).get("frames")
        except Exception:
            pass
    return meta


def sample_indices(n, stride, include_last=True):
    idx = list(range(0, n, stride))
    if include_last and idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


BAND_LO, BAND_HI = 0.45, 0.80  # torso band as a fraction of mask bbox height


def _band_and_area_series(frames, indices):
    """Per-sampled-frame belly-band width + whole-mask area, both normalised
    to the clip median. The band is the middle torso (rows 45-80% of the
    mask bbox height) — the belly. The two series together are what separate
    intended breathing from pathological inflation: on r063 (bad, drowsy) the
    belly swings ~9% while the body moves ~4%; on the good clips (breathe,
    pop, walk) the two move in lockstep. Band width alone cannot separate the
    class — the good active clips swing wider than the bad drowsy — so the
    judge sees the strip and the body-area line as the contrast baseline.
    """
    band_w, areas = [], []
    for fp in frames:
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        m = im[..., 3] > 128
        ys, xs = np.nonzero(m)
        areas.append(int(m.sum()))
        if not ys.size:
            band_w.append(0.0)
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        widths = np.array([int((xs[ys == r]).size) for r in range(y0, y1)])
        band = slice(int(len(widths) * BAND_LO), int(len(widths) * BAND_HI))
        band_w.append(float(widths[band].mean()))
    band_w = np.array(band_w)[indices]
    areas = np.array(areas)[indices]
    med_b, med_a = np.median(band_w), np.median(areas)
    return band_w / med_b, areas / med_a


def _label_cell(img, text, color=(0, 0, 255), pos="tl"):
    h, w = img.shape[:2]
    org = (4, 18) if pos == "tl" else (4, h - 6)
    cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)


def _grid_overlay(img, label_letters=True):
    """3x3 coordinate grid, subtle, overlay-only."""
    h, w = img.shape[:2]
    for i in range(1, 3):
        x = w * i // 3
        cv2.line(img, (x, 0), (x, h), (90, 90, 90), 1)
        y = h * i // 3
        cv2.line(img, (0, y), (w, y), (90, 90, 90), 1)
    if label_letters:
        for j, col in enumerate("ABC"):
            cv2.putText(img, col, (w * j // 3 + 4, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
        for i, row in enumerate("123"):
            cv2.putText(img, row, (2, h * i // 3 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)


def _render_cell(fpath, cell, overlay):
    im = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    comp = _composite(im, 0)
    tile = cv2.resize(comp, (cell, cell), interpolation=cv2.INTER_NEAREST)
    if overlay:
        _grid_overlay(tile)
    return tile


def render_digest(frames, indices, layout, cols, cell, overlay, header_text):
    """Return a BGR uint8 digest image."""
    n = len(indices)
    if layout == "filmstrip":
        rows = 1
        use_cols = n
    else:
        rows = (n + cols - 1) // cols
        use_cols = cols
    header_h = 46
    margin = 4
    out_w = use_cols * (cell + margin) + margin
    out_h = header_h + rows * (cell + margin) + margin
    digest = np.full((out_h, out_w, 3), 245, np.uint8)

    cv2.putText(digest, header_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    k = 0
    for r in range(rows):
        for c in range(use_cols):
            # empty slots repeat frame 0 — the anchor — so the judge can
            # compare any late frame against the loop start in-place.
            idx = indices[k] if k < n else 0
            tile = _render_cell(frames[idx], cell, overlay)
            x0 = margin + c * (cell + margin)
            y0 = header_h + r * (cell + margin)
            digest[y0:y0 + cell, x0:x0 + cell] = tile
            _label_cell(digest[y0:y0 + cell, x0:x0 + cell], idx)
            k += 1
    return digest


def encode_png_b64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise SystemExit("failed to encode digest PNG")
    return base64.b64encode(buf.tobytes()).decode()


def render_band_strip(band, area, indices, run_name):
    """The belly-band vs body-area series as a plain-image chart.

    OpenCV-only (no matplotlib — the codebase doctrine). The judge reads
    SHAPE, not pixels: r063's belly line oscillates ~9% against a flat-ish
    body line; the good clips' lines move together. Y-axis is auto-scaled to
    the data so even a 1% calm-breathe wobble is visible as a line that is
    clearly *flat* against the 9% oscillation of the pathological clip.
    """
    x = np.arange(len(band), dtype=float)
    lo = min(float(band.min()), float(area.min()), 0.9)
    hi = max(float(band.max()), float(area.max()), 1.1)
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    W, H = max(len(band) * 24, 360), 200
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(img, f"{run_name} belly-band vs body area", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def px(t, v):
        return (int(10 + t / max(len(band) - 1, 1) * (W - 20)),
                int(H - 26 - (v - lo) / (hi - lo) * (H - 48)))

    zy = px(0, 1.0)[1]
    cv2.line(img, (8, zy), (W - 8, zy), (180, 180, 180), 1)
    cv2.putText(img, "1.0 = median", (6, zy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120, 120, 120), 1, cv2.LINE_AA)

    for t, a in enumerate(area):
        cv2.circle(img, px(t, a), 2, (200, 120, 40), -1)
    for t, b in enumerate(band):
        cv2.circle(img, px(t, b), 2, (40, 60, 200), -1)
    for t in range(len(band) - 1):
        cv2.line(img, px(t, area[t]), px(t + 1, area[t + 1]), (200, 120, 40), 1)
        cv2.line(img, px(t, band[t]), px(t + 1, band[t + 1]), (40, 60, 200), 2)

    for t in (0, len(band) // 2, len(band) - 1):
        cv2.putText(img, str(indices[t]), px(t, lo), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "orange = whole-body area, blue = belly band width", (6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def load_moonshot_key():
    key = os.environ.get("MOONSHOT_API_KEY")
    if key:
        return key
    if AUTH_JSON.exists():
        try:
            auth = json.loads(AUTH_JSON.read_text())
            if "moonshotai" in auth:
                return auth["moonshotai"]["key"]
        except Exception as e:
            raise SystemExit(f"could not read {AUTH_JSON}: {e}")
    raise SystemExit(
        "no Moonshot key: set MOONSHOT_API_KEY or connect Moonshot AI in opencode "
        "(`opencode auth login` -> Moonshot AI)")


JUDGE_SYSTEM = """You are a screening instrument in an animation QC pipeline: a FLAG, never a verdict. A lesser model may report numbers; it may never claim an animation looks right. Your output routes work to a human reviewer.

You are shown a digest of ONE animation loop, and (usually) the source MASTER asset image. In the digest, cells are frames in TIME ORDER (left to right, then top to bottom). Each cell is labelled with its source frame index; cell 0 is the loop's first frame, and the loop's final frame is the seam it must meet cleanly. The thin 3x3 grid inside each cell is a coordinate reference ONLY (columns A/B/C across, rows 1/2/3 down) — it is overlay, not content. Checkerboard areas are alpha transparency. The MASTER image is the identity ground truth: the character in the animation must match it in the IDENTITY INVARIANTS below, while pose and facing may legitimately change with the action.

DESIGN FACTS — intentional pipeline behaviour, never defects:
{facts}

IDENTITY INVARIANTS — these must NOT change, in any frame:
{invariants}

EXPECTED BEHAVIOUR — the animation is supposed to do this; micro-motion and the actions below are expected, never defects:
{expected}

INTENT — the prompt the animation is supposed to fulfil:
{intent}

DISCRIMINATION RULES — eye events:
- A blink is a quick, SINGLE, symmetric close-and-open of BOTH eyes that reopens fully by the next sampled cell. A blink touches AT MOST ONE sampled cell. A blink is never a defect.
- A DEFECT (flag it) is any eye event that is NOT a clean single blink: closure visible in TWO OR MORE CONSECUTIVE sampled cells (sustained, not a quick blink); asymmetric (one eye differs from the other); pupils shrinking or shifting inside a lowered lid; jitter or unsteadiness; or progressive narrowing across the clip. A jittery or slow wink reads as a sustained asymmetric partial close — that is a defect, not a blink.
- If you cannot cleanly classify an eye event as a single quick symmetric blink, FLAG it. A false FAIL costs a cheap reroll; a false PASS ships a broken emote.
- A clean render of the WRONG action is the worst defect class of all (semantic-mismatch).

DISCRIMINATION RULES — body proportion (the BELLY STRIP image):
- A part must keep its proportions across the loop except for natural micro-motion and the intended action. A belly or torso that widens, pulses, or churns by MORE than the rest of the body is inflation, not breath — flag it, even when rendering and attachment are otherwise clean.
- The belly strip plots the belly-band width (blue) against the whole-body area (orange) across the sampled cells, both relative to their median (the grey line at 1.0). When the blue line moves with the orange line — the same direction, similar size — the body is doing it: normal (breath, walk, a pop, a startle). When the blue line swings wider or faster than the orange line — or oscillates while the orange line is calm — the belly is acting on its own: that is the defect. A deep slow breath has a small blue wobble; a washing-machine belly has a large one.
- Body-wide deformation during a SINGLE quick action beat (one pop, one gasp, one startle) is expected, not a defect — the whole body moves together and returns. Only flag deformation that is disproportionate to the body's own motion, or longer-lived than one beat.
- If the belly line is clean (blue follows orange, both near 1.0), say so explicitly — do not invent an inflation.

REPORT, in order:
1. rendering integrity — pixels: detach, doubling, seam/zipper, matte-artifact (limb eaten by the matte, holes, fringe halo), feathering, posterization, palette-injection, coarse anti-aliasing.
2. parts integrity — for each part you can identify (head, eyes, arms, legs, tail, torso, ...): does it stay present, attached, and the same shape across the whole loop? Consult the BELLY STRIP for the torso's shape progression.
3. motion — reading cells in time order: is it continuous, or do parts jump, teleport, vanish, or flicker between neighbouring cells? Is the pacing plausible, or flipbook-stiff, or jittery?
4. prompt fidelity — does the motion actually perform the intended action? Only judge against the supplied intent.

RULES:
- class MUST be one of: {vocab}. New classes go in evidence, never class.
- every defect MUST carry the frame index where it is visible (and a cell reference like B2 when localisable).
- be conservative about flagging: a false FAIL costs a cheap reroll; a false PASS ships a broken emote. When genuinely uncertain, flag.
- stay strictly in scope: do not invent checks, do not extend the brief, do not make the final call. Respond with ONE JSON object exactly matching this shape:
{{"verdict": "pass"|"flag", "confidence": 0.0, "summary": "...", "defects": [{{"class": "...", "frames": [0], "cell": "...", "evidence": "..."}}], "parts": [{{"part": "...", "ok": true, "notes": "..."}}], "motion": {{"fluid": true, "notes": "..."}}, "prompt_fidelity": {{"judged": true, "matches": true, "notes": "..."}}}}"""


def build_system_prompt(intent, mascot, notes):
    if mascot and mascot in MASCOT_FACTS:
        invariants = MASCOT_FACTS[mascot]["invariants"]
        expected = list(MASCOT_FACTS[mascot]["expected"])
    else:
        invariants = ["the character as described in the INTENT"]
        expected = []
    expected.extend(notes or [])
    expected.append("natural micro-motion — blinks, breath, weight shifts, small sways — is expected and fine")
    facts = "\n".join(f"- {f}" for f in GLOBAL_DESIGN_FACTS)
    inv_lines = "\n".join(f"- {i}" for i in invariants)
    exp_lines = "\n".join(f"- {e}" for e in expected)
    vocab = ", ".join(DEFECT_VOCABULARY)
    return JUDGE_SYSTEM.format(
        facts=facts, invariants=inv_lines, expected=exp_lines,
        intent=intent or "(none supplied — judge rendering, parts and motion only)", vocab=vocab)


def judge(digest_b64, master_b64, strip_b64, system_prompt, key, model, effort, max_tokens=8192):
    import httpx

    def _post(payload):
        return httpx.post(f"{MOONSHOT_BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=300)

    content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{digest_b64}"}},
    ]
    if master_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{master_b64}", "detail": "high"},
        })
        content.append({"type": "text",
                        "text": "The MASTER image above is the identity ground truth. Judge the digest against it, per the system prompt."})
    if strip_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{strip_b64}"}})
        content.append({"type": "text",
                        "text": "The BELLY STRIP above plots belly-band width (blue) against whole-body area (orange) across the sampled cells, both relative to their median. Use it per the body-proportion discrimination rules."})
    content.append({"type": "text", "text": "Judge this loop. Reply with the JSON object."})
    body = {
        "model": model,
        "reasoning_effort": effort,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }

    resp = _post(body)
    if resp.status_code == 400 and "response_format" not in body:
        body["response_format"] = {"type": "json_object"}
        resp = _post(body)
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    content_text = msg.get("content") or ""
    try:
        parsed = json.loads(content_text)
    except json.JSONDecodeError:
        # tolerate fenced or trailing-text JSON
        start, end = content_text.find("{"), content_text.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(content_text[start:end + 1])
        else:
            raise SystemExit(f"judge returned unparseable JSON:\n{content_text[:2000]}")
    return parsed, data.get("usage")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run dir or name under out/pipeline/")
    ap.add_argument("--layout", choices=["grid", "filmstrip"], default="grid")
    ap.add_argument("--cols", type=int, default=4, help="grid columns")
    ap.add_argument("--stride", type=int, default=8, help="frame sample stride")
    ap.add_argument("--cell", type=int, default=256, help="cell px (square)")
    ap.add_argument("--no-overlay", action="store_true", help="skip 3x3 grid overlay")
    ap.add_argument("--intent", default=None, help="intended motion (default: from response.json)")
    ap.add_argument("--expect", action="append", default=[], help="per-run expected behaviour, repeatable")
    ap.add_argument("--no-anchor", action="store_true", help="don't send the master.png identity anchor")
    ap.add_argument("--no-strip", action="store_true", help="don't send the belly-band strip")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT, help="reasoning effort (low/high/max)")
    ap.add_argument("--render-only", action="store_true", help="render digest, skip the API call")
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    run = resolve_run(args.run)
    frames = load_rgba(run)
    n = len(frames)
    meta = run_meta(run)
    indices = sample_indices(n, args.stride)
    layout = args.layout
    header = f"{run.name} | {n} frames" + (f" @ {meta['fps']}fps" if meta.get("fps") else "") + \
        f" | {len(indices)} cells | stride {args.stride} | {layout}"

    digest = render_digest(frames, indices, layout, args.cols, args.cell,
                           not args.no_overlay, header)

    band, area = _band_and_area_series(frames, indices)
    strip = render_band_strip(band, area, indices, run.name)

    out_stem = f"watch_{run.name}_{layout}_c{args.cell}_s{args.stride}"
    out_dir = run / "verification"
    out_dir.mkdir(exist_ok=True)
    digest_path = out_dir / f"{out_stem}.png"
    cv2.imwrite(str(digest_path), digest)
    strip_path = out_dir / f"{out_stem}_bellystrip.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"digest: {digest_path} ({digest.shape[1]}x{digest.shape[0]})")
    print(f"belly strip: {strip_path}")

    if args.render_only:
        return

    intent = args.intent
    if not intent:
        rj = run / "response.json"
        if rj.exists():
            try:
                intent = json.loads(rj.read_text()).get("prompt")
            except Exception:
                pass

    master_b64 = None
    master_png = run / "master.png"
    if master_png.exists() and not args.no_anchor:
        master_b64 = encode_png_b64(cv2.imread(str(master_png)))

    strip_b64 = None
    if not args.no_strip:
        strip_b64 = encode_png_b64(strip)

    mascot = detect_mascot(run.name)
    key = load_moonshot_key()
    sysp = build_system_prompt(intent, mascot, args.expect)
    result, usage = judge(encode_png_b64(digest), master_b64, strip_b64, sysp,
                          key, args.model, args.effort, args.max_tokens)

    result["provenance"] = {
        "judge_model": args.model,
        "reasoning_effort": args.effort,
        "run": run.name,
        "digest": digest_path.name,
        "frames_total": n,
        "sampled_indices": indices,
        "layout": layout,
        "cell": args.cell,
        "stride": args.stride,
        "intent": intent,
        "expect": args.expect,
        "mascot": mascot,
        "master_anchor": bool(master_b64),
        "belly_strip": bool(strip_b64),
        "usage": usage,
    }
    result_path = out_dir / f"{out_stem}.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"digest text: {result_path}")
    print("----")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
