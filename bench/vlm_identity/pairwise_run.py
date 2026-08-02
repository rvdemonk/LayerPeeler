"""Pairwise identity bench — anchor vs each comparison frame, separate calls.

Each call: frame_0 (anchor) + one comparison frame, "same character?".
Any pair that says "no" flags the clip. 3 repeats per pair for consistency.
"""

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pairwise_prompt as P
import providers as PV

OUTDIR = (Path(__file__).resolve().parents[2]
          / "out" / "bench" / "vlm-pairwise-2026-08-02")
RESPONSES = OUTDIR / "responses"
REPEATS = 3


def pairwise_images(name, manifest):
    """Generator: (anchor_path, comp_path, frame_num)."""
    d = OUTDIR / "frames" / name
    for f in manifest[name]["frames"]:
        if f["slot"] == 0:
            anchor = d / f["file"]
        else:
            yield anchor, d / f["file"], f["frame"]


def main():
    # Copy frames from the all-at-once bench
    src_frames = (Path(__file__).resolve().parents[2]
                  / "out" / "bench" / "vlm-identity-2026-08-02" / "frames")
    dst_frames = OUTDIR / "frames"
    if not dst_frames.exists():
        import shutil
        shutil.copytree(str(src_frames), str(dst_frames))

    # Copy frames.json to OUTDIR root
    src_manifest = (Path(__file__).resolve().parents[2]
                    / "out" / "bench" / "vlm-identity-2026-08-02"
                    / "frames.json")
    if not (OUTDIR / "frames.json").exists():
        import shutil
        shutil.copy2(str(src_manifest), str(OUTDIR / "frames.json"))

    manifest = json.loads((OUTDIR / "frames.json").read_text())

    RESPONSES.mkdir(parents=True, exist_ok=True)
    clients = PV.make_clients()

    models = sys.argv[1:] or list(PV.MODELS)

    # Build job list: model x clip x pair x repeat
    jobs = []
    for model in models:
        for clip in sorted(manifest):
            pairs = list(pairwise_images(clip, manifest))
            for anchor_path, comp_path, frame_num in pairs:
                for rep in range(REPEATS):
                    jobs.append((model, clip, anchor_path, comp_path,
                                 frame_num, rep))

    spent, done, failed = 0.0, 0, 0
    t0 = time.time()

    for i, (model, clip, anchor_path, comp_path, frame_num, rep) \
            in enumerate(jobs, 1):
        dest = RESPONSES / ("%s__%s__f%03d__r%d.json"
                            % (model, clip, frame_num, rep))
        if dest.exists():
            spent += json.loads(dest.read_text()).get("cost_usd", 0.0)
            done += 1
            continue

        rec = {"model": model, "model_id": PV.MODELS[model]["id"],
               "clip": clip, "frame": frame_num, "repeat": rep,
               "label": manifest[clip]["label"]}
        images = [(0, 0, anchor_path),
                  (1, frame_num, comp_path)]
        try:
            parsed, raw, usage = PV.pairwise_judge(
                model, images, clients)
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
        print("[%3d/%3d] %-17s %-22s f%03d r%d  held=%-5s $%.4f  (tot $%.3f)%s"
              % (i, len(jobs), model, clip, frame_num, rep, held,
                 rec["cost_usd"], spent,
                 "  ERR " + rec["error"][:60] if rec["error"] else ""),
              flush=True)

    print("\n%d judgements, %d failed, $%.4f spent, %.1f min"
          % (done, failed, spent, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
