"""One generation = one ledger row.

The run ledger (docs/workorders/hybrid-oracle-spikes/ledger.md) is the
project's memory: method -> output -> mechanical gates -> Lewis's verdict,
kept VERBATIM. Carried over from hybrid/spike2_oracle.ledger_row() so the
consolidated pipeline does not silently stop recording what the spike
harness recorded.

One change: the gates column is filled in at write time. The spike wrote
"—" and waited for a separate gate pass, which meant rows sat gateless
whenever the second command was forgotten. The verdict column stays
empty — that one is Lewis's, and nothing here may write it.

COLUMN SEMANTICS CHANGE, flagged rather than buried: the `lottie KB`
column held the RAW pack size in spike-era rows (17,000-86,000 KB — an
unshippable probe artifact). The consolidated pipeline never builds a raw
pack, so this writes the SHIPPING rung's gzipped size instead (~800-2000
KB). Rows above and below that boundary are not comparable in that one
column. Needs a supervisor ruling: either accept the break, or add a
column.

Only LIVE generations get a row. Re-running the local stages on an mp4
already on disk is not a new generation and must not inflate the ledger
(or the cost column, which is summed for unit economics).
"""

import time
from pathlib import Path

HEADER = (
    "# Run ledger — hybrid-oracle-spikes (Spike 2+)\n\n"
    "One row per oracle generation. Gates = Claude's mechanical "
    "instruments; verdict = Lewis, VERBATIM, filled after appraisal.\n\n"
    "| id | date | input | model | res | seed | loop | cost | frames | "
    "lottie KB | gen s | gates | Claude pre-verdict | Lewis verdict |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def gate_cell(gates):
    """The gates column: the numbers that decide a reroll, in one cell."""
    d = gates.get("detail", {})
    integ, strips = d.get("integrity", {}), d.get("strips", {})
    post = (d.get("posterization", {}) or {}).get("worst_frame") or {}
    shim = d.get("shimmer", {})
    bits = [
        "%s" % gates.get("verdict", "—"),
        "loop %.3f/%.3f" % (integ.get("loop_ssim", 0),
                            integ.get("loop_alpha_iou", 0)),
        "dL %.1f" % strips.get("color_dL_max", 0),
        "vel p95 %.1f jerk %.2f" % (strips.get("velocity_p95", 0),
                                    strips.get("jerk_rms", 0)),
        "face/body %.2f" % strips.get("face_body_ratio", 0),
    ]
    if post:
        bits.append("dE %.2f" % post.get("deltaE_mean", 0))
    if shim.get("ratio") is not None:
        bits.append("shimmer %.2fx" % shim["ratio"])
    issues = gates.get("fails", []) + gates.get("flags", [])
    if issues:
        bits.append("· " + "; ".join(issues))
    return " / ".join(bits)


def append_row(path, image, prompt, tier, resolution, seed, loop, cost,
               frames, gz_kb, gen_seconds, gates):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(HEADER)
    text = path.read_text()
    rid = "r%03d" % (text.count("\n| r") + 1)
    row = ("| %s | %s | %s: \"%s...\" | wan2.2-a14b-%s | %s | %s | %s | "
           "$%.3f | %d | %d | %.0f | %s | — | — |\n"
           % (rid, time.strftime("%Y-%m-%d"), Path(image).stem, prompt[:60],
              tier, resolution, seed, loop, cost, frames, gz_kb, gen_seconds,
              gate_cell(gates)))
    path.write_text(text + row)
    return rid
