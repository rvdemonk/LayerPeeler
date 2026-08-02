"""Worst-case cost, from real token counts — the checkpoint-1 gate.

Input tokens are COUNTED, not guessed: Anthropic and Gemini both expose a free
count-tokens endpoint that runs no inference and bills nothing, and the assembled
prompt is sent to them verbatim. OpenAI has no such endpoint, so its image cost
is computed from the documented 32x32-patch scheme and its text from the
Anthropic count (the two tokenizers differ, but text is <5% of this prompt).

Output tokens are BOUNDED, not estimated: every adapter sets max_tokens, so
worst-case output cost is a hard ceiling rather than a forecast.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompt as P       # noqa: E402
import providers as PV   # noqa: E402
import run as R          # noqa: E402

REPEATS = 3


def _anthropic_count(images, client):
    content = [{"type": "text", "text": P.USER_PREFIX}]
    for slot, frame, path in images:
        content.append({"type": "text", "text": P.image_caption(slot, frame)})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": PV._b64(path)}})
    r = client.messages.count_tokens(
        model="claude-opus-5", system=P.SYSTEM,
        messages=[{"role": "user", "content": content}])
    return r.input_tokens


def _gemini_count(images, client):
    from google.genai import types
    parts = [types.Part.from_text(text=P.SYSTEM),
             types.Part.from_text(text=P.USER_PREFIX)]
    for slot, frame, path in images:
        parts.append(types.Part.from_text(text=P.image_caption(slot, frame)))
        parts.append(types.Part.from_bytes(data=path.read_bytes(),
                                           mime_type="image/png"))
    r = client.models.count_tokens(
        model="gemini-3.1-pro-preview",
        contents=[types.Content(role="user", parts=parts)])
    return r.total_tokens


def _openai_count(images, anthropic_tokens):
    # 32x32 patch scheme, capped at 1536 patches per image.
    per_image = min(math.ceil(512 / 32) * math.ceil(512 / 32), 1536)
    # Anthropic image cost is (w*h)/750 = 349 tokens per 512x512 crop.
    text_only = anthropic_tokens - len(images) * round(512 * 512 / 750)
    return text_only + len(images) * per_image


def main():
    clients = PV.make_clients()
    manifest = json.loads((R.OUTDIR / "frames.json").read_text())
    # Token count is a property of the prompt shape, not the clip: every
    # judgement is 7 crops of the same dimensions plus the same text.
    name = next(iter(manifest))
    images = R.images_for(name, manifest)

    a = _anthropic_count(images, clients["anthropic"])
    g = _gemini_count(images, clients["gemini"])
    o = _openai_count(images, a)
    counted = {"anthropic": a, "gemini": g, "openai": o}
    print("input tokens per judgement (counted, free endpoints):")
    for k, v in counted.items():
        print("  %-10s %6d %s" % (k, v, "(computed)" if k == "openai" else ""))

    n_clips = len(manifest)
    n = n_clips * REPEATS
    print("\n%d clips x %d repeats = %d judgements per model\n" % (
        n_clips, REPEATS, n))

    print("%-17s %8s %8s %10s %10s" % (
        "model", "in/judg", "out cap", "$/judg", "$ total"))
    total = 0.0
    for key, m in PV.MODELS.items():
        tin = counted[m["vendor"]]
        c = (tin / 1e6 * m["in"] + PV.MAX_OUTPUT / 1e6 * m["out"])
        total += c * n
        print("%-17s %8d %8d %10.5f %10.4f" % (
            key, tin, PV.MAX_OUTPUT, c, c * n))
    print("%-17s %8s %8s %10s %10.4f" % ("WORST CASE", "", "", "", total))
    print("\nCeiling $3.50 -> %s" % ("PASS" if total <= 3.50 else "EXCEEDS"))

    (R.OUTDIR / "cost-estimate.json").write_text(json.dumps({
        "input_tokens_per_judgement": counted,
        "max_output_tokens": PV.MAX_OUTPUT,
        "clips": n_clips, "repeats": REPEATS,
        "judgements_per_model": n,
        "worst_case_usd": round(total, 4),
    }, indent=1))


if __name__ == "__main__":
    main()
