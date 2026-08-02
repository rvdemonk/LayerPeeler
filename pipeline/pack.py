"""The batch driver — one mascot image + a pack spec -> N accepted emotes.

    pack spec (JSON)
      -> [base stage]  gemini-pro i2i edits for the emotes that need a new
                       pose/expression, reviewed one at a time by a human
      -> [clip stage]  each accepted base through pipeline.run (generate,
                       matte, gates, encode, eyeball) — unchanged
      -> [manifest]    pack.json: per-emote status, draw counts, timings,
                       machine cost, gate verdicts, and the operator's own
                       minutes

The manifest is the point. docs/business-model.md prices a pack off four
measurements — r_base (base draws per accepted base), r_anim (clips drawn
per accepted clip), M (Lewis-minutes) and machine wall clock — and none
of them can be recovered after the fact from a pile of PNGs. So every
draw, accept, reject and reroll is counted while it happens, and the
review prompts are timed.

This module is ADDITIVE: it calls pipeline.run.main() with the same argv
an operator would type, so timing, gates, ledger behaviour and the
sandbox report are whatever the single-clip path already does. Nothing
below it was modified to make packs work.

Usage (from LayerPeeler/):
    set -a; source ~/.env; set +a            # FAL_KEY, live runs only
    .venv-hybrid/bin/python -m pipeline.pack packs/raccoon-emotes.json

    # see every request and every cost, spend nothing
    ... --dry-run

    # rehearse the whole driver on clips already on disk (no fal, no gemini)
    ... --from-video-dir out/spike2 --accept-all

Reroll economics are the whole game: a reroll is cheap in dollars
(~$0.15 a base, $0.05 a clip) and expensive in minutes, which is why the
counts live in the manifest and the review clock runs.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import encode as enc
from . import generate as gen
from . import run as pipeline_run

ROOT = Path(__file__).resolve().parent.parent          # LayerPeeler/
PACKS_OUT = ROOT / "out" / "packs"

# Pack size is a PARAMETER, not a constant of the product: Layer 1 of the
# business model (how many emotes in a sellable pack) is a pricing
# decision Lewis has not made yet. 8 is the placeholder everywhere in
# docs/business-model.md, so it is the placeholder here too.
DEFAULT_N = 8

# gemini-pro: ruled the seam-frame / new-base model on 2026-08-01 (probe
# ledger entry). gemini-flash "scrapes through" with a hue drift and an
# unnamed lifelessness; nano-banana-2 was eliminated 0/3 (species swap).
# Do not make this a flag until another model wins a probe.
I2I_MODEL = "gemini-pro"
I2I_PRICE_USD = 0.15        # observed on the probe: 6 draws, $0.65 mixed

# The prompt shape that held identity across both probe mascots. The
# invariant clause is not decoration — nano-banana-2 failed with a short
# prompt and gemini-pro held with this one.
BASE_PROMPT = ("Edit: the {character} {edit}. Preserve the exact character "
               "design, flat style, colours, proportions, and background "
               "unchanged.")

# A clip whose gates FAIL is rerolled, never shipped (CLAUDE.md: a
# known-broken clip must be rerolled, not drawn as if fitted). Capped so a
# systematically-unfittable emote cannot burn the pack budget in a loop.
MAX_CLIP_ATTEMPTS = 3
MAX_BASE_ATTEMPTS = 6


def log(msg=""):
    print(msg, flush=True)


def die(msg):
    """Exit with an operator-facing message: what went wrong, what to do."""
    raise SystemExit("\nPACK ABORTED — %s\n" % msg)


# ---------------------------------------------------------------- spec

def load_spec(path):
    """Read and validate a pack spec, failing with what to fix.

    The operator after today has no Claude to ask what a field means, so
    every validation error names the field, the file, and the fix.
    """
    p = Path(path)
    if not p.exists():
        die("no pack spec at %s.\n  Copy the template: "
            "cp LayerPeeler/packs/raccoon-emotes.json mypack.json" % p)
    try:
        spec = json.loads(p.read_text())
    except ValueError as e:
        die("pack spec %s is not valid JSON (%s).\n  Check for a trailing "
            "comma or a missing quote." % (p, e))

    for field in ("pack", "mascot", "character", "emotes"):
        if not spec.get(field):
            die("pack spec %s is missing \"%s\".\n  Required fields: pack "
                "(name), mascot (image path), character (e.g. \"raccoon "
                "mascot\"), emotes (list)." % (p, field))

    mascot = (p.parent / spec["mascot"]).resolve()
    if not mascot.exists():
        mascot2 = (ROOT / spec["mascot"]).resolve()
        if not mascot2.exists():
            die("mascot image not found: \"%s\" in %s.\n  Tried %s and %s.\n"
                "  The path is relative to the spec file (or to "
                "LayerPeeler/)." % (spec["mascot"], p, mascot, mascot2))
        mascot = mascot2
    spec["_mascot_path"] = mascot
    spec["_spec_path"] = p

    names = set()
    for i, e in enumerate(spec["emotes"]):
        for field in ("name", "prompt"):
            if not e.get(field):
                die("emote #%d in %s is missing \"%s\".\n  Each emote needs: "
                    "name (short, used as the run/file name), base "
                    "(\"original\" or an edit instruction), prompt (the "
                    "animation prompt)." % (i + 1, p, field))
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", e["name"]):
            die("emote name %r in %s is not filename-safe.\n  Use lowercase "
                "letters, digits and hyphens." % (e["name"], p))
        if e["name"] in names:
            die("emote name %r appears twice in %s.\n  Names become run "
                "directories, so they must be unique." % (e["name"], p))
        names.add(e["name"])
        e.setdefault("base", "original")
    return spec


# ------------------------------------------------------------- bases

def base_prompt_for(spec, emote):
    """Assemble the i2i edit prompt.

    `base_prompt` in the spec overrides the template verbatim, for the
    case where the operator has a prompt that already works and does not
    want it wrapped.
    """
    if emote.get("base_prompt"):
        return emote["base_prompt"]
    return BASE_PROMPT.format(character=spec["character"], edit=emote["base"])


PIXERY_ID = re.compile(r"\(ID:\s*(\d+)\)")
PIXERY_COST = re.compile(r"Cost:\s*\$([0-9.]+)")


def pixery_generate(prompt, ref, dest, ratio=None, dry_run=False, tag=None):
    """One gemini-pro draw. Returns (path, meta) or (None, meta) on failure.

    `--copy-to` puts the image exactly where we want it (pixery creates
    the parent dir), so nothing has to be scraped out of the archive; the
    archive ID is still recorded so the operator can `pixery show <id>`.
    """
    cmd = ["pixery", "generate", "-m", I2I_MODEL, "--ref", str(ref),
           "-p", prompt, "--copy-to", str(dest)]
    if ratio:
        cmd += ["--ratio", ratio]
    if tag:
        cmd += ["-t", tag]
    if dry_run:
        log("    DRY-RUN would run: %s" % " ".join(
            ("'%s'" % c if " " in c else c) for c in cmd))
        return None, {"dry_run": True, "cost_usd": I2I_PRICE_USD}

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    secs = round(time.time() - t0, 2)
    out = (proc.stdout or "") + (proc.stderr or "")
    meta = {"seconds": secs, "model": I2I_MODEL,
            "cost_usd": I2I_PRICE_USD, "stdout_tail": out.strip()[-400:]}
    m = PIXERY_ID.search(out)
    if m:
        meta["pixery_id"] = int(m.group(1))
    m = PIXERY_COST.search(out)
    if m:
        meta["cost_usd"] = float(m.group(1))   # pixery's own number wins
    if proc.returncode != 0 or not Path(dest).exists():
        meta["error"] = "pixery exited %d" % proc.returncode
        return None, meta
    return Path(dest), meta


def open_file(path, enabled):
    """Show the operator the image. Best-effort — never fails a run."""
    if not enabled or not path:
        return
    try:
        subprocess.run(["open", str(path)], check=False,
                       capture_output=True)
    except OSError:
        pass


# ------------------------------------------------------------ prompts

class Clock:
    """Human seconds actually spent at review prompts.

    M (Lewis-minutes per pack) is the top-ranked sensitivity in the
    business model, and it is the one number nothing else on disk can
    reconstruct. Measured here, not estimated later.
    """

    def __init__(self):
        self.seconds = 0.0

    def ask(self, prompt, choices, default=None):
        while True:
            t0 = time.time()
            try:
                raw = input(prompt).strip().lower()
            except EOFError:
                raise SystemExit(
                    "\nno input available (stdin closed) — rerun with "
                    "--accept-all for a non-interactive pack.")
            self.seconds += time.time() - t0
            if not raw and default:
                return default
            if raw in choices:
                return raw
            log("    choose one of: %s" % ", ".join(choices))

    def ask_text(self, prompt):
        t0 = time.time()
        try:
            raw = input(prompt).strip()
        except EOFError:
            raw = ""
        self.seconds += time.time() - t0
        return raw

    @property
    def minutes(self):
        return round(self.seconds / 60.0, 1)


# --------------------------------------------------------------- pack

def clip_argv(name, image, prompt, out_dir, args, from_video=None):
    """The exact argv an operator would type for this one clip."""
    argv = ["--name", name, "--out", str(out_dir),
            "--tier", args.tier, "--resolution", args.resolution,
            "--profiles", args.profiles]
    if from_video:
        argv += ["--from-video", str(from_video)]
    else:
        argv += ["--image", str(image), "--prompt", prompt]
    if args.no_ledger:
        argv.append("--no-ledger")
    if args.no_eyeball:
        argv.append("--no-eyeball")
    return argv


def stage_bases(spec, emotes, pdir, args, clock, prior):
    """Draw and review one base image per emote that needs a new pose.

    Emotes whose base is "original" reuse the mascot untouched — no draw,
    no cost, and no review, because there is nothing new to look at.
    """
    bases_dir = pdir / "bases"
    records = {}
    for i, e in enumerate(emotes, 1):
        name = e["name"]
        if prior.get(name, {}).get("base", {}).get("status") == "accepted":
            rec = prior[name]["base"]
            if Path(rec.get("path", "")).exists():
                log("[base %d/%d] %-16s resumed (accepted earlier)"
                    % (i, len(emotes), name))
                records[name] = rec
                continue
        if args.from_video_dir:
            records[name] = {"status": "skipped", "draws": 0, "rejected": 0,
                             "path": None, "source": "from-video",
                             "reason": "clip adopted from disk; no base needed"}
            continue
        if e["base"] == "original":
            records[name] = {"status": "accepted", "draws": 0, "rejected": 0,
                             "path": str(spec["_mascot_path"]),
                             "source": "original", "cost_usd": 0.0}
            log("[base %d/%d] %-16s original mascot (no draw)"
                % (i, len(emotes), name))
            continue

        prompt = base_prompt_for(spec, e)
        rec = {"status": "pending", "draws": 0, "rejected": 0,
               "source": I2I_MODEL, "prompt": prompt, "cost_usd": 0.0,
               "attempts": []}
        while rec["draws"] < MAX_BASE_ATTEMPTS:
            k = rec["draws"] + 1
            dest = bases_dir / ("%s.draw%d.png" % (name, k))
            log("[base %d/%d] %-16s draw %d ($%.2f) via %s"
                % (i, len(emotes), name, k, I2I_PRICE_USD, I2I_MODEL))
            log("    prompt: %s" % prompt)
            path, meta = pixery_generate(prompt, spec["_mascot_path"], dest,
                                         ratio=spec.get("ratio"),
                                         dry_run=args.dry_run,
                                         tag=args.tag or spec.get("tag"))
            rec["draws"] += 1
            rec["cost_usd"] += meta.get("cost_usd", 0.0)
            rec["attempts"].append({"draw": k, "path": str(dest), **meta})
            if args.dry_run:
                rec["status"] = "dry-run"
                break
            if path is None:
                log("    FAILED: %s\n    %s" % (meta.get("error"),
                                                meta.get("stdout_tail", "")))
                if rec["draws"] >= MAX_BASE_ATTEMPTS:
                    rec["status"] = "failed"
                    break
                continue
            log("    image: %s" % path)
            if args.accept_all:
                rec.update(status="accepted", path=str(path))
                break
            open_file(path, not args.no_open)
            ans = clock.ask(
                "    [a]ccept  [r]eroll  [s]kip emote  [q]uit pack > ",
                ("a", "r", "s", "q"))
            if ans == "a":
                rec.update(status="accepted", path=str(path))
                break
            if ans == "r":
                rec["rejected"] += 1
                continue
            if ans == "s":
                rec["rejected"] += 1
                rec["status"] = "skipped"
                break
            rec["rejected"] += 1
            rec["status"] = "quit"
            records[name] = rec
            return records, True
        else:
            rec["status"] = "exhausted"
            log("    gave up after %d draws — the edit prompt is probably "
                "the problem, not the model.\n    Edit \"base\" for %r in %s "
                "and rerun with --resume."
                % (MAX_BASE_ATTEMPTS, name, spec["_spec_path"]))
        records[name] = rec
    return records, False


def stage_clips(spec, emotes, pdir, args, clock, bases, prior):
    """Run the single-clip pipeline per accepted base, with reroll.

    Two ways a clip is rejected, and both count as a draw for r_anim:
    the gates FAIL it (mechanical, automatic) or the operator does
    (taste). Rejected attempts are kept on disk as <name>.rej<k>/ — the
    rejects are the corpus for future gates, not garbage.
    """
    clips_dir = pdir / "clips"
    records = {}
    for i, e in enumerate(emotes, 1):
        name = e["name"]
        base = bases.get(name, {})
        if prior.get(name, {}).get("clip", {}).get("status") == "accepted":
            rec = prior[name]["clip"]
            if Path(rec.get("run_dir", "")).exists():
                log("[clip %d/%d] %-16s resumed (accepted earlier)"
                    % (i, len(emotes), name))
                records[name] = rec
                continue
        from_video = None
        if args.from_video_dir:
            vdir = Path(args.from_video_dir)
            stem = spec["_mascot_path"].stem
            cands = [vdir / name / "oracle.mp4",
                     vdir / ("%s-%s" % (stem, name)) / "oracle.mp4"]
            from_video = next((c for c in cands if c.exists()), None)
            if from_video is None:
                die("--from-video-dir %s has no clip for emote %r.\n  Tried "
                    "%s.\n  Rehearsal mode adopts <dir>/<emote>/oracle.mp4 "
                    "(or <dir>/<mascot>-<emote>/oracle.mp4); run with --only "
                    "on the emotes you have clips for."
                    % (vdir, name, " and ".join(str(c) for c in cands)))
        elif base.get("status") != "accepted" and not args.dry_run:
            records[name] = {"status": "no-base", "draws": 0, "rejected": 0,
                             "reason": "base stage produced no accepted image"}
            log("[clip %d/%d] %-16s SKIPPED (no accepted base)"
                % (i, len(emotes), name))
            continue

        rec = {"status": "pending", "draws": 0, "rejected": 0,
               "cost_usd": 0.0, "attempts": []}
        while rec["draws"] < MAX_CLIP_ATTEMPTS:
            k = rec["draws"] + 1
            rdir = clips_dir / name
            if rdir.exists() and rec["draws"]:
                # Rejects are kept (they are the corpus future gates get
                # calibrated on) but moved OUT of clips/, because the
                # sandbox lists every directory under its root and a
                # reject with nothing to play is noise in an appraisal.
                rej = pdir / "rejects" / ("%s.rej%d" % (name, k - 1))
                rej.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(rej, ignore_errors=True)
                shutil.move(str(rdir), str(rej))
                rec["attempts"][-1]["run_dir"] = str(rej)
            # In a dry run no base exists yet, so the mascot stands in as
            # the conditioning image: the payload's shape is what is being
            # inspected, and its image field is elided anyway.
            image = base.get("path") or spec["_mascot_path"]
            argv = clip_argv(name, image, e["prompt"], clips_dir,
                             args, from_video)
            log("[clip %d/%d] %-16s attempt %d  (pipeline.run %s)"
                % (i, len(emotes), name, k, " ".join(argv)))
            if args.dry_run:
                pipeline_run.main(argv + ["--dry-run"])
                rec["status"] = "dry-run"
                rec["draws"] += 1
                rec["cost_usd"] += gen.cost(args.tier, args.resolution)
                break
            rec["draws"] += 1
            t0 = time.time()
            try:
                rc = pipeline_run.main(argv)
                err = None
            except SystemExit as ex:                # gates FAIL, or a hard stop
                rc, err = 1, str(ex)
            secs = round(time.time() - t0, 2)
            run = read_run_json(rdir)
            verdict = (run.get("gates") or {}).get("verdict", "UNKNOWN")
            tm = run.get("timings") or {}
            attempt = {"attempt": k, "seconds": secs, "rc": rc,
                       "verdict": verdict, "run_dir": str(rdir),
                       "generation_seconds": tm.get("generation_seconds"),
                       "local_seconds": tm.get("local_seconds"),
                       "fails": (run.get("gates") or {}).get("fails", []),
                       "flags": (run.get("gates") or {}).get("flags", []),
                       "error": err,
                       "cost_usd": run.get("cost_usd", 0.0)}
            rec["attempts"].append(attempt)
            rec["cost_usd"] += attempt["cost_usd"] or 0.0
            if verdict == "FAIL" or rc != 0:
                rec["rejected"] += 1
                log("    gates %s%s — rerolling (a broken clip is rerolled, "
                    "not shipped)" % (verdict,
                                      "".join("\n      - " + f
                                              for f in attempt["fails"])))
                if err:
                    log("    %s" % err.strip())
                continue
            if args.accept_all:
                rec.update(status="accepted", run_dir=str(rdir),
                           verdict=verdict, run=run_summary(run))
                break
            log("    gates %s — watch it: %s"
                % (verdict, rdir / ("%s.gif" % name)))
            open_file(rdir / ("%s.gif" % name), not args.no_open)
            ans = clock.ask(
                "    [a]ccept  [r]eroll  [s]kip emote  [q]uit pack > ",
                ("a", "r", "s", "q"))
            if ans == "a":
                rec.update(status="accepted", run_dir=str(rdir),
                           verdict=verdict, run=run_summary(run))
                break
            if ans == "r":
                rec["rejected"] += 1
                continue
            if ans == "s":
                rec["rejected"] += 1
                rec["status"] = "skipped"
                break
            rec["rejected"] += 1
            rec["status"] = "quit"
            records[name] = rec
            return records, True
        else:
            rec["status"] = "exhausted"
            log("    %d attempts all failed their gates. This emote's "
                "animation prompt is the suspect — soften the motion or "
                "shorten it in %s, then rerun with --resume."
                % (MAX_CLIP_ATTEMPTS, spec["_spec_path"]))
        records[name] = rec
    return records, False


def read_run_json(rdir):
    p = Path(rdir) / "run.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def run_summary(run):
    """The few numbers from a clip run the pack manifest needs inline."""
    ship = next((e for e in run.get("encodes", [])
                 if e["profile"] == run.get("ship_profile")), {})
    t = run.get("timings", {})
    return {"profile": run.get("ship_profile"), "frames": run.get("frames"),
            "fps": run.get("fps"), "fr": ship.get("fr"),
            "gz_bytes": ship.get("gz_bytes"),
            "lottie_path": ship.get("lottie_path"),
            "json_path": ship.get("json_path"),
            "verdict": (run.get("gates") or {}).get("verdict"),
            "flags": (run.get("gates") or {}).get("flags", []),
            "total_seconds": t.get("total_seconds"),
            "generation_seconds": t.get("generation_seconds"),
            "local_seconds": t.get("local_seconds")}


# ----------------------------------------------------------- manifest

def build_manifest(spec, emotes, bases, clips, args, clock, wall_seconds,
                   human_minutes):
    """pack.json — the business model's data source.

    r_base and r_anim are ratios of draws to accepted outputs, defined
    exactly as docs/business-model.md defines them (draws per accepted
    output). They are null, not 1.0, when nothing was accepted: an
    undefined reroll rate must not read as a perfect one.
    """
    per = []
    for e in emotes:
        n = e["name"]
        b, c = bases.get(n, {}), clips.get(n, {})
        per.append({
            "name": n, "base_spec": e["base"], "prompt": e["prompt"],
            "base": b, "clip": c,
            "status": ("accepted" if c.get("status") == "accepted"
                       else c.get("status") or b.get("status") or "not-run"),
        })

    base_draws = sum(b.get("draws", 0) for b in bases.values())
    base_ok = sum(1 for b in bases.values() if b.get("status") == "accepted"
                  and b.get("source") != "original")
    base_drawn_ok = sum(1 for b in bases.values()
                        if b.get("status") == "accepted"
                        and b.get("source") == I2I_MODEL)
    clip_draws = sum(c.get("draws", 0) for c in clips.values())
    clip_ok = sum(1 for c in clips.values() if c.get("status") == "accepted")

    machine = (sum(b.get("cost_usd", 0.0) for b in bases.values())
               + sum(c.get("cost_usd", 0.0) for c in clips.values()))
    # Summed over EVERY attempt, not just the accepted one: a reroll's
    # seconds are machine time the pack really spent, and hiding them
    # would flatter exactly the number the SLA depends on.
    attempts = [a for c in clips.values() for a in c.get("attempts", [])]
    gen_secs = sum(a.get("generation_seconds") or 0.0 for a in attempts)
    local_secs = sum(a.get("local_seconds") or 0.0 for a in attempts)

    return {
        "pack": spec["pack"],
        "spec": str(spec["_spec_path"]),
        "mascot": str(spec["_mascot_path"]),
        "character": spec["character"],
        "n": len(emotes),
        "requested_n": args.n,
        "mode": ("dry-run" if args.dry_run else
                 "rehearsal (from-video)" if args.from_video_dir else
                 "live"),
        "accept_all": args.accept_all,
        "tier": args.tier, "resolution": args.resolution,
        "profile": enc.PROFILES[args.profiles.split(",")[0]].name,
        "i2i_model": I2I_MODEL,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "emotes": per,
        # --- the four business-model measurements, first-class ---
        "measurements": {
            "r_base": round(base_draws / base_drawn_ok, 2) if base_drawn_ok else None,
            "r_anim": round(clip_draws / clip_ok, 2) if clip_ok else None,
            "human_minutes": human_minutes,
            "human_minutes_at_prompts": clock.minutes,
            "machine_wall_clock_seconds": round(wall_seconds, 1),
            "machine_cost_usd": round(machine, 3),
            "accepted_emotes": clip_ok,
            "base_draws": base_draws, "accepted_bases": base_ok,
            "clip_draws": clip_draws,
            "generation_seconds": round(gen_secs, 1),
            "local_seconds": round(local_secs, 1),
            "cost_per_accepted_emote_usd": (round(machine / clip_ok, 3)
                                            if clip_ok else None),
        },
    }


def summary_table(man):
    w = max([len(e["name"]) for e in man["emotes"]] + [6])
    lines = ["", "== pack %s  (%s)" % (man["pack"], man["mode"]), "",
             "  %-*s %-10s %6s %6s %8s %9s" % (w, "emote", "status", "base",
                                               "clip", "gates", "gz KB"),
             "  " + "-" * (w + 43)]
    for e in man["emotes"]:
        run = (e["clip"].get("run") or {})
        gz = run.get("gz_bytes")
        lines.append("  %-*s %-10s %6s %6s %8s %9s"
                     % (w, e["name"], e["status"],
                        "%d/%d" % (e["base"].get("draws", 0),
                                   1 if e["base"].get("status") == "accepted"
                                   else 0),
                        "%d/%d" % (e["clip"].get("draws", 0),
                                   1 if e["clip"].get("status") == "accepted"
                                   else 0),
                        e["clip"].get("verdict") or "-",
                        "%.0f" % (gz / 1024) if gz else "-"))
    m = man["measurements"]
    fmt = lambda v, u="": ("%s%s" % (v, u)) if v is not None else "n/a"
    lines += [
        "",
        "  accepted emotes        %s of %s" % (m["accepted_emotes"], man["n"]),
        "  r_base (draws/base)    %s   (%d draws)" % (fmt(m["r_base"]),
                                                      m["base_draws"]),
        "  r_anim (draws/clip)    %s   (%d draws)" % (fmt(m["r_anim"]),
                                                      m["clip_draws"]),
        "  machine cost           $%.2f  (%s per accepted emote)"
        % (m["machine_cost_usd"],
           "$%.2f" % m["cost_per_accepted_emote_usd"]
           if m["cost_per_accepted_emote_usd"] is not None else "n/a"),
        "  machine wall clock     %.1f min  (gen %.0fs / local %.0fs)"
        % (m["machine_wall_clock_seconds"] / 60, m["generation_seconds"],
           m["local_seconds"]),
        "  human minutes          %s  (%.1f measured at prompts)"
        % (fmt(m["human_minutes"]), m["human_minutes_at_prompts"]),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="mascot + pack spec -> N accepted emotes, measured",
        epilog="See pipeline/README.md, section 'Packs', for the runbook.")
    ap.add_argument("spec", help="pack spec JSON (see packs/raccoon-emotes.json)")
    ap.add_argument("--n", type=int,
                    help="pack size: use the first N emotes of the spec "
                         "(default: every emote in the spec; the business "
                         "model's placeholder pack is %d)" % DEFAULT_N)
    ap.add_argument("--only", help="comma-separated emote names to run "
                                   "(overrides --n)")
    ap.add_argument("--out", default=str(PACKS_OUT),
                    help="pack output root (default out/packs)")
    ap.add_argument("--accept-all", action="store_true",
                    help="no prompts: accept every base and every "
                         "gate-passing clip (testing, and unattended runs)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse bases/clips already accepted in this pack's "
                         "pack.json instead of redrawing them")
    ap.add_argument("--from-video-dir",
                    help="rehearsal: adopt <dir>/<emote>/oracle.mp4 instead "
                         "of generating; skips the base stage. Spends nothing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every request that would be made, and its "
                         "price; spend nothing")
    ap.add_argument("--tier", default="turbo", choices=list(gen.ENDPOINTS))
    ap.add_argument("--resolution", default="480p",
                    choices=["480p", "580p", "720p"])
    ap.add_argument("--profiles", default=enc.SHIP_PROFILE,
                    help="comma-separated: %s" % ",".join(enc.PROFILES))
    ap.add_argument("--tag", help="pixery tag for the base draws")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open images/GIFs in Preview during review")
    ap.add_argument("--no-eyeball", action="store_true",
                    help="skip each clip's GIF/contact sheet (they are what "
                         "you review — skip only for mechanical checks)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="do not append ledger rows for live generations")
    ap.add_argument("--minutes", type=float,
                    help="your own minutes on this pack, if you would rather "
                         "state them than be asked at the end")
    args = ap.parse_args(argv)

    unknown = [p for p in args.profiles.split(",") if p not in enc.PROFILES]
    if unknown:
        ap.error("unknown profile(s) %s; have %s"
                 % (unknown, list(enc.PROFILES)))

    spec = load_spec(args.spec)
    emotes = spec["emotes"]
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        have = {e["name"] for e in emotes}
        missing = [w for w in want if w not in have]
        if missing:
            die("--only names %s, not in %s.\n  Spec has: %s"
                % (missing, spec["_spec_path"], ", ".join(sorted(have))))
        emotes = [e for e in emotes if e["name"] in want]
    elif args.n is not None:
        if len(emotes) < args.n:
            die("pack size --n %d but %s lists only %d emotes.\n  Add emotes "
                "to the spec, or run with --n %d."
                % (args.n, spec["_spec_path"], len(emotes), len(emotes)))
        emotes = emotes[:args.n]
    elif len(emotes) != DEFAULT_N:
        log("note: this spec is %d emotes; the business model's placeholder "
            "pack is %d. Pack size is a pricing decision, not a code one — "
            "--n picks a different size." % (len(emotes), DEFAULT_N))

    pdir = Path(args.out) / spec["pack"]
    pdir.mkdir(parents=True, exist_ok=True)
    manifest_path = pdir / "pack.json"
    prior = {}
    if args.resume and manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
            prior = {e["name"]: e for e in prev.get("emotes", [])}
            log("resuming %s: %d emotes on record" % (manifest_path, len(prior)))
        except ValueError:
            log("WARNING: %s unreadable; resuming from nothing" % manifest_path)

    log("pack %s — %d emotes, %s, mascot %s"
        % (spec["pack"], len(emotes),
           "DRY RUN (no spend)" if args.dry_run else
           "rehearsal from %s (no spend)" % args.from_video_dir
           if args.from_video_dir else "LIVE (spends money)",
           spec["_mascot_path"]))
    if not args.dry_run and not args.from_video_dir:
        n_draws = sum(1 for e in emotes if e["base"] != "original")
        log("  budget floor: %d bases x $%.2f + %d clips x $%.2f = $%.2f "
            "(before any reroll)"
            % (n_draws, I2I_PRICE_USD, len(emotes),
               gen.cost(args.tier, args.resolution),
               n_draws * I2I_PRICE_USD
               + len(emotes) * gen.cost(args.tier, args.resolution)))
    log("")

    clock = Clock()
    t0 = time.time()
    bases, quit_ = stage_bases(spec, emotes, pdir, args, clock, prior)
    clips = {}
    if not quit_:
        clips, quit_ = stage_clips(spec, emotes, pdir, args, clock, bases,
                                   prior)
    wall = time.time() - t0 - clock.seconds     # machine time, not ours

    human_minutes = args.minutes
    if human_minutes is None and not args.accept_all and not args.dry_run:
        raw = clock.ask_text(
            "\n  Your minutes on this pack (blank = use the %.1f measured at "
            "prompts): " % clock.minutes)
        try:
            human_minutes = float(raw) if raw else clock.minutes
        except ValueError:
            human_minutes = clock.minutes

    man = build_manifest(spec, emotes, bases, clips, args, clock, wall,
                         human_minutes)
    manifest_path.write_text(json.dumps(man, indent=1, default=float))
    log(summary_table(man))
    log("\n  manifest: %s" % manifest_path)
    if not args.dry_run:
        log("  preview:  python3 sandbox/serve.py --root %s" % (pdir / "clips"))
    if quit_:
        log("\n  quit early — rerun the same command with --resume to keep "
            "what you accepted.")
    accepted = man["measurements"]["accepted_emotes"]
    if args.dry_run:
        return 0
    return 0 if accepted == len(emotes) else 1


if __name__ == "__main__":
    sys.exit(main())
