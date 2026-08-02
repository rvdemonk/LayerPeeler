"""Drive the bench: every model, every clip, three repeats, everything kept.

Each judgement is written to disk the moment it returns — raw provider response,
parsed judgement, usage and cost. A crashed run resumes rather than re-spending,
and the responses survive as corpus regardless of whether this analysis is the
last one anyone runs on them.

Repeats exist to measure self-consistency: a judge that flips between PASS and
FAIL on identical inputs is unusable at any price, and a single pass cannot
detect that.
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import providers as PV  # noqa: E402

OUTDIR = (Path(__file__).resolve().parents[2]
          / "out" / "bench" / "vlm-identity-2026-08-02")
RESPONSES = OUTDIR / "responses"
REPEATS = 3


def images_for(name, manifest):
    """[(slot, frame_index, path)] for one clip, anchor first."""
    d = OUTDIR / "frames" / name
    return [(f["slot"], f["frame"], d / f["file"])
            for f in manifest[name]["frames"]]


def main():
    manifest = json.loads((OUTDIR / "frames.json").read_text())
    RESPONSES.mkdir(parents=True, exist_ok=True)
    clients = PV.make_clients()

    models = sys.argv[1:] or list(PV.MODELS)
    jobs = [(m, c, r) for m in models for c in manifest
            for r in range(REPEATS)]
    spent, done, failed = 0.0, 0, 0
    t0 = time.time()

    for i, (model, clip, rep) in enumerate(jobs, 1):
        dest = RESPONSES / ("%s__%s__r%d.json" % (model, clip, rep))
        if dest.exists():
            spent += json.loads(dest.read_text()).get("cost_usd", 0.0)
            done += 1
            continue
        rec = {"model": model, "model_id": PV.MODELS[model]["id"],
               "clip": clip, "repeat": rep,
               "label": manifest[clip]["label"],
               "frames": [f["frame"] for f in manifest[clip]["frames"]]}
        try:
            parsed, raw, usage = PV.judge(
                model, images_for(clip, manifest), clients)
            rec.update(parsed=parsed, raw=raw, usage=usage,
                       cost_usd=PV.cost(model, usage), error=None)
            spent += rec["cost_usd"]
        except Exception as e:
            failed += 1
            rec.update(parsed=None, raw=None, usage=None, cost_usd=0.0,
                       error="%s: %s" % (type(e).__name__, e),
                       traceback=traceback.format_exc())
        dest.write_text(json.dumps(rec, indent=1, default=str))
        done += 1
        held = (rec["parsed"] or {}).get("identity_held")
        print("[%3d/%3d] %-17s %-22s r%d  held=%-5s $%.4f  (tot $%.3f)%s"
              % (i, len(jobs), model, clip, rep, held, rec["cost_usd"],
                 spent, "  ERR " + rec["error"][:60] if rec["error"] else ""),
              flush=True)

    print("\n%d judgements, %d failed, $%.4f spent, %.1f min"
          % (done, failed, spent, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
