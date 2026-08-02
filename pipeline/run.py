"""The blessed path, end to end, as one command.

    mascot PNG + prompt
      -> Wan 2.2 i2v (fal, turbo, 480p, looped by construction)
      -> flat-background matte + colour norm
      -> basic-integrity + colour/velocity gates
      -> 512px WebP q65 @ 24fps frame-seq Lottie (+ gzip, + .lottie)
      -> posterization + shimmer gates on the encoded assets
      -> eyeball GIF + contact sheet, previewable in the sandbox

Every stage is timed and the timing record is a first-class output
(timings.json), because SLA design and unit economics are downstream of
it.

This supersedes the spike-era chain of hybrid/spike2_oracle.py +
spike2_gates.py + spike2_repack.py. Those stay on disk: they are the
ledger's driver and the size-ladder laboratory, and the ladder still
exists to measure rungs this pipeline deliberately does not ship.

Usage (from LayerPeeler/):
    .venv-hybrid/bin/python -m pipeline.run --name strawberry-wave \\
        --image ../corpus/mascots/strawberry.jpg --prompt "..."

    # everything below generation, on a clip already on disk — this is
    # how local changes are verified; fal credits are for verifying fal
    .venv-hybrid/bin/python -m pipeline.run --name repro \\
        --from-video out/spike2/raccoon-wave/oracle.mp4

    # show the request without spending anything
    .venv-hybrid/bin/python -m pipeline.run ... --dry-run

Env: FAL_KEY, only for live generation (set -a; source ~/.env; set +a).
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

from . import encode as enc
from . import gates as G
from . import generate as gen
from . import ledger
from .eyeball import eyeball
from .matte import matte_frames
from .timing import Timings

ROOT = Path(__file__).resolve().parent.parent          # LayerPeeler/
OUT = ROOT / "out" / "pipeline"
LEDGER = (ROOT.parent / "docs" / "workorders" / "hybrid-oracle-spikes"
          / "ledger.md")


def log(msg):
    print(msg, flush=True)


def stage_source(args, rdir, t):
    """Get the mp4 into the run dir: generate it, or adopt one on disk."""
    mp4 = rdir / "oracle.mp4"
    if args.from_video:
        src = Path(args.from_video).resolve()
        with t.stage("adopt_video", source=str(src)):
            if src != mp4.resolve():
                shutil.copy2(src, mp4)
        return mp4, {"source": str(src), "generated": False}
    with t.stage("generate", tier=args.tier, res=args.resolution) as rec:
        res = gen.generate(args.image, args.prompt, mp4, tier=args.tier,
                           resolution=args.resolution, seed=args.seed,
                           loop=not args.no_loop, log=log)
        rec["queue_seconds"] = res.get("queue_seconds")
    (rdir / "response.json").write_text(json.dumps(
        {**res, "prompt": args.prompt, "image": str(args.image)}, indent=1))
    return mp4, {"generated": True, "seed": res.get("seed"),
                 "request_id": res.get("request_id"),
                 "cost_usd": gen.cost(args.tier, args.resolution)}


def write_sandbox_report(record, out_root):
    """A ladder_report.json in the sandbox's shape, so the existing
    appraisal player can serve this tree with `serve.py --root`.

    Merged rather than overwritten: one run must not blank out the
    previous run's numbers, exactly as spike2_repack preserves partial
    ladder passes.
    """
    path = Path(out_root) / "ladder_report.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except ValueError:
            prev = {}
    runs = prev.get("runs", {})
    runs[record["name"]] = {
        "raw": {"frames": record["frames"], "edge": record["edge"],
                "fr": record["fps"]},
        "ladder": {e["profile"]: {
            "edge": e["edge"], "codec": e["codec"], "keep": e["keep"],
            "quality": e["quality"], "frames": e["frames"], "fr": e["fr"],
            "json": e["json_bytes"], "gz": e["gz_bytes"],
            "lottie": e["lottie_bytes"], "mime": e["codec"],
            "score": (record["gates"]["detail"].get("posterization", {})
                      .get("mid_frame") if e["profile"] ==
                      record["ship_profile"] else None),
            "rejected": False} for e in record["encodes"]},
    }
    path.write_text(json.dumps({"generator": "pipeline.run", "runs": runs},
                               indent=1))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="mascot PNG + prompt -> gated, shippable Lottie")
    ap.add_argument("--name", required=True, help="run dir name")
    ap.add_argument("--image", help="mascot PNG/JPG (required unless --from-video)")
    ap.add_argument("--prompt", help="animation prompt")
    ap.add_argument("--from-video", help="skip generation; use this mp4")
    ap.add_argument("--tier", default="turbo", choices=list(gen.ENDPOINTS))
    ap.add_argument("--resolution", default="480p",
                    choices=["480p", "580p", "720p"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-loop", action="store_true",
                    help="do not condition the end frame on the input image")
    ap.add_argument("--profiles", default=enc.SHIP_PROFILE,
                    help="comma-separated: %s" % ",".join(enc.PROFILES))
    ap.add_argument("--no-eyeball", action="store_true",
                    help="skip the GIF/contact sheet (they are doctrine; "
                         "skip only for a mechanical repro check)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="do not append a ledger row (live generations "
                         "append one by default)")
    ap.add_argument("--force", action="store_true",
                    help="encode even if basic integrity FAILS")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.from_video and not (args.image and args.prompt):
        ap.error("--image and --prompt are required unless --from-video")
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profiles if p not in enc.PROFILES]
    if unknown:
        ap.error("unknown profile(s) %s; have %s"
                 % (unknown, list(enc.PROFILES)))
    ship = profiles[0]

    if args.dry_run:
        payload = gen.build_payload(args.image, args.prompt, args.resolution,
                                    args.seed, not args.no_loop)
        print(json.dumps({
            "endpoint": gen.ENDPOINTS[args.tier],
            "cost_usd": gen.cost(args.tier, args.resolution),
            "payload": gen.redact(payload),
            "profiles": [{"profile": enc.PROFILES[p].name,
                          "quality": enc.PROFILES[p].quality,
                          "keep": enc.PROFILES[p].keep} for p in profiles],
        }, indent=1))
        return 0

    out_root = Path(args.out)
    rdir = out_root / args.name
    rdir.mkdir(parents=True, exist_ok=True)
    t = Timings(args.name, {"tier": args.tier, "resolution": args.resolution,
                            "profiles": profiles})
    scratch = Path(tempfile.mkdtemp(prefix="pipeline-%s-" % args.name))
    record = {"name": args.name, "prompt": args.prompt,
              "image": args.image, "ship_profile": enc.PROFILES[ship].name}

    try:
        mp4, src_meta = stage_source(args, rdir, t)
        record.update(src_meta)
        t.meta.update({k: v for k, v in src_meta.items() if k != "source"})

        with t.stage("extract") as rec:
            fps = gen.probe_fps(mp4)
            if not fps:
                sys.exit("fps unprobeable from %s — refusing to pack (a "
                         "guessed fr is the half-speed bug)" % mp4)
            frames = gen.extract_frames(mp4, rdir / "frames")
            rec["frames"] = len(frames)
            rec["fps"] = fps
        log("  %d frames @ %dfps (probed)" % (len(frames), fps))

        with t.stage("matte", frames=len(frames)):
            rgba = matte_frames(frames, rdir / "rgba", log=log)

        with t.stage("gate_integrity"):
            integ = G.basic_integrity(rgba, loop=not args.no_loop)
        with t.stage("gate_strips"):
            strip_g, series = G.strips(rgba)
            G.draw_strips(series, strip_g, rdir / "gates.png")

        with t.stage("gate_identity"):
            ident = G.identity(rgba)

        pre = G.verdict(integ, strip_g, ident)
        log("  integrity/strips: %s%s" % (pre["verdict"], "".join(
            "\n    - " + x for x in pre["fails"] + pre["flags"])))
        if pre["fails"] and not args.force:
            record["gates"] = {**pre, "detail": {"integrity": integ,
                                                 "strips": strip_g,
                                                 "identity": ident}}
            raise SystemExit("basic integrity FAILED — not encoding. "
                             "A known-broken clip must be rerolled, not "
                             "shipped as if fitted. (--force overrides)")

        native = cv2.imread(str(rgba[0]), cv2.IMREAD_UNCHANGED).shape[0]
        edge = enc.edge_for(native)
        encodes, ship_assets = [], []
        for p in profiles:
            keep_assets = (p == ship)
            with t.stage("encode", profile=enc.PROFILES[p].name) as rec:
                e, assets = enc.encode(rgba, native, fps, rdir / "ladder",
                                       args.name, p, scratch,
                                       keep_assets=keep_assets)
                rec["gz_kb"] = round(e["gz_bytes"] / 1024, 1)
            encodes.append(e)
            if keep_assets:
                ship_assets = assets
            log("  %-16s %3df @ %sfr  json %6.1fKB  gz %6.1fKB  .lottie %6.1fKB"
                % (e["profile"], e["frames"], e["fr"],
                   e["json_bytes"] / 1024, e["gz_bytes"] / 1024,
                   e["lottie_bytes"] / 1024))

        kept = enc.kept_indices(len(rgba), enc.PROFILES[ship].keep)
        ref_for_ship = [rgba[i] for i in kept]
        with t.stage("gate_posterization"):
            post = G.posterization(ref_for_ship, ship_assets, edge)
        with t.stage("gate_shimmer"):
            shim = G.shimmer(ref_for_ship, ship_assets, edge, series["vel"])
        for f in ship_assets:
            Path(f).unlink(missing_ok=True)

        final = G.verdict(integ, strip_g, ident, post, shim)
        # Flat scalars first: the appraisal sandbox reads specific keys off
        # the top level of gates.json, and nested detail would hide them.
        gates_out = {**{k: v for k, v in integ.items()
                        if k not in ("fails", "flags")},
                     **{k: v for k, v in strip_g.items() if k != "flags"},
                     **{k: v for k, v in ident.items() if k != "flags"},
                     "posterization_dE": (post.get("worst_frame") or {})
                     .get("deltaE_mean"),
                     "shimmer_ratio": shim.get("ratio"),
                     **final,
                     "detail": {"integrity": integ, "strips": strip_g,
                                "identity": ident,
                                "posterization": post, "shimmer": shim}}
        G.write(gates_out, rdir / "gates.json")
        record["gates"] = gates_out
        log("  gates: %s%s" % (final["verdict"], "".join(
            "\n    - " + x for x in final["fails"] + final["flags"])))

        if not args.no_eyeball:
            with t.stage("eyeball"):
                eyeball(rgba, rdir, args.name, fps)

        record.update({"frames": len(frames), "fps": fps, "edge": edge,
                       "native_edge": native, "encodes": encodes})
        ship_rec = next(e for e in encodes
                        if e["profile"] == record["ship_profile"])
        # Throughput lands in the timing record too: seconds alone do not
        # size an SLA, seconds per frame and bytes out do.
        t.meta.update({"frames": len(frames), "fps": fps,
                       "ship_gz_bytes": ship_rec["gz_bytes"],
                       "gates": final["verdict"]})

        if record.get("generated") and not args.no_ledger:
            with t.stage("ledger"):
                record["ledger_id"] = ledger.append_row(
                    LEDGER, args.image, args.prompt, args.tier,
                    args.resolution, record.get("seed"), not args.no_loop,
                    record.get("cost_usd", 0.0), len(frames),
                    round(ship_rec["gz_bytes"] / 1024), t.seconds_for(
                        "generate"), gates_out)
            log("  ledger: %s" % record["ledger_id"])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        t.write(rdir / "timings.json")
        record["timings"] = t.record()
        (rdir / "run.json").write_text(json.dumps(record, indent=1,
                                                  default=float))

    write_sandbox_report(record, out_root)
    log("\n== %s  %s  %.2f MB gz  %df @ %sfr  gates %s"
        % (args.name, ship_rec["profile"], ship_rec["gz_bytes"] / 1e6,
           ship_rec["frames"], ship_rec["fr"], final["verdict"]))
    log(t.summary())
    log("\n  artifacts: %s" % rdir)
    log("  preview:   python3 sandbox/serve.py --root %s" % out_root)
    return 0 if final["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
