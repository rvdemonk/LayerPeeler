"""Score the bench, per design §5.

The asymmetry in §5.2 is the whole point: a false PASS ships a broken emote or
slips one into Lewis's queue disguised as clean work, and a false FAIL costs one
reroll and a few seconds of his eye. So false-PASS count is reported first and
decides; false-FAIL is a secondary cost term.

Reported as COUNTS, never percentages alone — with 9 negatives a single clip
moves any rate by 11 points, and a percentage invites a confidence the corpus
cannot support.

`raccoon-look-bs` never enters the confusion matrix. It is reported separately as
a probe: two recorded verdicts disagree about it, and an informative judge should
side with one account.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import providers as PV  # noqa: E402
from run import OUTDIR, RESPONSES  # noqa: E402

# Recurring drift signatures the eye reported, mapped to the vocabulary. Used
# only to check whether a judge flagged the right clip for the right reason.
EXPECTED_CLASS = {
    "pack-wave": "identity-decay", "pack-celebrate": "identity-decay",
    "pack-sleepy": "mouth-deformity", "raccoon-sleepy": "mouth-deformity",
    "raccoon-jig": "identity-decay", "raccoon-jig-pace": "identity-decay",
    "raccoon-jig-bs-s3": "identity-decay", "raccoon-breathe": "identity-decay",
    "raccoon-jig-full": "feature-scale-drift",
}
FRAME_TOL = 15


def load():
    recs = [json.loads(p.read_text()) for p in sorted(RESPONSES.glob("*.json"))]
    return [r for r in recs if r.get("error") is None or r.get("parsed")], recs


def main():
    ok, allrecs = load()
    manifest = json.loads((OUTDIR / "frames.json").read_text())
    by = defaultdict(list)
    for r in allrecs:
        by[r["model"]].append(r)

    print("=" * 78)
    print("FALSE-PASS TABLE — the metric that rules (counts over all repeats)")
    print("=" * 78)
    print("%-18s %7s %7s %7s %7s %8s %7s" % (
        "model", "FALSE", "caught", "false", "correct", "unusable", "flip"))
    print("%-18s %7s %7s %7s %7s %8s %7s" % (
        "", "PASS", "neg", "FAIL", "pos", "resp", "rate"))
    summary = {}
    for model in PV.MODELS:
        rs = by.get(model, [])
        if not rs:
            continue
        fp = fn = tp = tn = bad = 0
        held_by_clip = defaultdict(list)
        for r in rs:
            p = r.get("parsed")
            if not p or "identity_held" not in (p or {}):
                bad += 1
                continue
            held = bool(p["identity_held"])
            held_by_clip[r["clip"]].append(held)
            if r["label"] == "bad":
                if held:
                    fp += 1      # shipped a broken clip — the expensive error
                else:
                    tp += 1
            elif r["label"] == "good":
                if held:
                    tn += 1
                else:
                    fn += 1      # one reroll and a few seconds of Lewis's eye
        flips = sum(1 for c, v in held_by_clip.items()
                    if r_label(manifest, c) and len(set(v)) > 1)
        scored_clips = sum(1 for c in held_by_clip if r_label(manifest, c))
        flip = flips / scored_clips if scored_clips else 0.0
        summary[model] = dict(false_pass=fp, caught=tp, false_fail=fn,
                              correct_pos=tn, unusable=bad, flip_rate=flip)
        print("%-18s %7d %7d %7d %7d %8d %6.0f%%" % (
            model, fp, tp, fn, tn, bad, 100 * flip))

    print("\n" + "=" * 78)
    print("DECISION RULE (§5.3) — flagging bar: zero false PASS across repeats")
    print("=" * 78)
    for model, s in summary.items():
        flag = s["false_pass"] == 0 and s["caught"] > 0
        auto = flag and s["flip_rate"] < 0.15
        print("%-18s flagging=%-4s  auto-reroll-eligible=%-4s  %s" % (
            model, "YES" if flag else "no", "YES" if auto else "no",
            "" if flag else "(%d false PASS)" % s["false_pass"]))

    print("\n" + "=" * 78)
    print("PER-DEFECT-CLASS AND FRAME LOCALISATION (on caught negatives)")
    print("=" * 78)
    for model in summary:
        cls_hit = cls_tot = frm_hit = frm_tot = 0
        for r in by[model]:
            p = r.get("parsed")
            if not p or r["label"] != "bad" or p.get("identity_held") is not False:
                continue
            defects = p.get("defects") or []
            if not defects:
                continue
            cls_tot += 1
            if EXPECTED_CLASS.get(r["clip"]) in {d.get("class") for d in defects}:
                cls_hit += 1
            worst = manifest[r["clip"]]["gate"].get("identity_worst_frame")
            frames = [d.get("frame") for d in defects
                      if isinstance(d.get("frame"), int)]
            if worst is not None and frames:
                frm_tot += 1
                if min(abs(f - worst) for f in frames) <= FRAME_TOL:
                    frm_hit += 1
        print("%-18s class match %d/%d   frame within +/-%d  %d/%d" % (
            model, cls_hit, cls_tot, FRAME_TOL, frm_hit, frm_tot))

    print("\n" + "=" * 78)
    print("PROBE — raccoon-look-bs (contested; excluded from scoring)")
    print("Fable read it as a TRUE POSITIVE; Lewis (ledger r019) recorded an")
    print("identity-hold PASS. An informative judge sides with one account.")
    print("=" * 78)
    for model in summary:
        for r in sorted((x for x in by[model] if x["clip"] == "raccoon-look-bs"),
                        key=lambda x: x["repeat"]):
            p = r.get("parsed") or {}
            held = p.get("identity_held")
            ev = ""
            if p.get("defects"):
                d = p["defects"][0]
                ev = " | %s f%s: %s" % (d.get("class"), d.get("frame"),
                                        (d.get("evidence") or "")[:70])
            print("%-18s r%d  held=%-5s conf=%-5s%s" % (
                model, r["repeat"], held, p.get("confidence"), ev))

    spent = sum(r.get("cost_usd") or 0 for r in allrecs)
    print("\nactual spend: $%.4f over %d judgements (%d unusable)"
          % (spent, len(allrecs), sum(s["unusable"] for s in summary.values())))

    (OUTDIR / "analysis.json").write_text(json.dumps(
        {"summary": summary, "actual_spend_usd": round(spent, 4)}, indent=1))


def r_label(manifest, clip):
    return manifest[clip]["label"] is not None


if __name__ == "__main__":
    main()
