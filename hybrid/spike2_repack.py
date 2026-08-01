"""Spike 2 — repack + size ladder for the frame-sequence Lotties.

Two jobs, one pass over out/spike2/<run>/:

  1. REPACK. Every pack written before 2026-07-31 carries fr:16 while the
     Wan clips are 32fps — the same half-speed bug that poisoned the wave
     1-3 pacing verdicts, frozen into the shipped artifact. fps is
     therefore never a constant here: it is probed from that run's own
     oracle.mp4 with ffprobe, exactly as spike2_oracle.main() does, and
     the run's <name>.json is rewritten in place at the true rate.

  2. LADDER. A raw pack is ~30MB of base64 PNG — unshippable; the target
     is well under 1MB. The ladder walks the three levers that trade
     bytes for fidelity, so the cost of each rung is measured rather than
     guessed:
       - resolution   512 (native) / 384 / 256, Lanczos, RGBA preserved
       - palette      256-colour quantization per PNG
       - frame rate   every-2nd frame with fr halved

     The decimated rungs halve fr as well as the frame count, so the clip
     still runs the same number of wall-clock seconds. This is not a
     nicety: every pacing verdict in the ledger is about wall-clock feel,
     and a rung that plays 2x fast would invalidate all of them.

Sizes are reported three ways because all three are real: the plain JSON
(what the sandbox loads), the same JSON gzipped (what a server actually
sends over the wire), and a .lottie zip (the shippable container). The
.lottie files exist for measurement and distribution; the sandbox still
plays plain JSON.

Variant PNGs are written to a scratch dir and deleted after packing —
only the JSON/.lottie survive, which keeps the ladder in the tens of MB
per run instead of hundreds. Nothing under a run dir is ever deleted.

Usage:
  .venv-hybrid/bin/python hybrid/spike2_repack.py                 # all runs
  .venv-hybrid/bin/python hybrid/spike2_repack.py --runs raccoon-wave
  .venv-hybrid/bin/python hybrid/spike2_repack.py --no-ladder     # repack only
  .venv-hybrid/bin/python hybrid/spike2_repack.py --rungs 256q,256q-half
"""

import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from spike2_oracle import OUT, pack_lottie

LADDER_REPORT_JSON = OUT / "ladder_report.json"
LADDER_REPORT_MD = OUT / "ladder_report.md"

# name -> (edge px, codec, keep every Nth frame)
#
# Codecs, and why there are four of them:
#   png      lossless re-encode; the fidelity ceiling and the size floor
#            for "no quality decision was made".
#   octree   PIL FASTOCTREE, 256 colours, NO dithering. Cheap and
#            dependency-free, and the reason this table grew: Lewis
#            appraised it as blotching "like an old GIF" on the
#            gradient-rich strawberry (9168 -> 127 colours in the body).
#            Flat-colour mascots like the raccoon survive it; shaded ones
#            do not. Kept as the measured baseline of that failure.
#   pngquant same 256-colour budget but a better palette AND Floyd-Steinberg
#            dithering, which trades banding for noise the eye integrates.
#   webp     lossy continuous-tone with alpha — no palette at all, so the
#            posterization failure mode cannot occur by construction.
# Frame keeping is (numerator, denominator): keep the first `num` of every
# `den` source frames, and scale fr by the same fraction so wall-clock
# duration is unchanged. (1,1) all, (1,2) every 2nd = 16fps, (3,4) drop
# every 4th = 24fps. Expressing it as a fraction rather than a stride is
# what makes 24fps sayable at all — and 24 is the interesting rung, since
# Lewis rejected 16fps outright as "a flipbook".
RUNGS = {
    # rung            edge  codec       q   keep
    "512":          (512, "png",      None, (1, 1)),
    "384":          (384, "png",      None, (1, 1)),
    "256":          (256, "png",      None, (1, 1)),
    "512webp":      (512, "webp",       80, (1, 1)),
    "448webp":      (448, "webp",       80, (1, 1)),
    "384webp":      (384, "webp",       80, (1, 1)),
    "256webp":      (256, "webp",       80, (1, 1)),
    "512webp-q65":  (512, "webp",       65, (1, 1)),
    "512webp-q50":  (512, "webp",       50, (1, 1)),
    "512webp-24":   (512, "webp",       80, (3, 4)),
    # Downscaling below native BACKFIRES on this art: Lanczos interpolation
    # roughly doubles the unique-colour count of flat vector-style fills
    # (raccoon-wave: 5571 colours at native 512, 9841 at 448), and WebP pays
    # more for that manufactured noise than it saves in pixels. Measured:
    # 448webp and 384webp are both LARGER than 512webp on every 512-native
    # run. So the size dial that works is quality x frame rate at native
    # resolution, not resolution.
    "512webp-q65-24": (512, "webp",     65, (3, 4)),
    "512webp-q50-24": (512, "webp",     50, (3, 4)),
    # --- rejected by Lewis; kept generatable for comparison, never default ---
    "512q":         (512, "octree",   None, (1, 1)),
    "384q":         (384, "octree",   None, (1, 1)),
    "256q":         (256, "octree",   None, (1, 1)),
    "512q-half":    (512, "octree",   None, (1, 2)),
    "384q-half":    (384, "octree",   None, (1, 2)),
    "256q-half":    (256, "octree",   None, (1, 2)),
    "512pq":        (512, "pngquant", None, (1, 1)),
    "256pq":        (256, "pngquant", None, (1, 1)),
    "512webp-half": (512, "webp",       80, (1, 2)),
    "256webp-half": (256, "webp",       80, (1, 2)),
}

# Ruled out by appraisal, not by measurement — so the files stay on disk
# and stay reachable, but nothing generates them by default and the
# sandbox hides them behind a toggle. Two separate verdicts:
#   octree/pngquant  posterization ("like an old GIF" on the strawberry)
#   any -half rung   16fps reads as "a flipbook" REGARDLESS of jump size;
#                    velocity-gate clearance does not overrule the eye.
REJECTED = {"512q", "384q", "256q", "512q-half", "384q-half", "256q-half",
            "512pq", "256pq", "512webp-half", "256webp-half"}
DEFAULT_RUNGS = [r for r in RUNGS if r not in REJECTED]

QUANT_COLORS = 256
PNGQUANT_QUALITY = "70-95"
WEBP_QUALITY = 80
# libwebp's effort dial. Measured on this corpus at 448px, 20 frames:
# method=6 took 59.06s and averaged 16.5 KB/frame; method=4 took 0.37s for
# 16.8 KB/frame. 160x the CPU to save 1.8% of the bytes — which priced a
# 31-run roll at ~13 hours instead of ~30 minutes. Not a close call.
# Rungs built before 2026-07-31 used method=6 and are ~2% smaller; they
# were regenerated at method=4 so the corpus is internally comparable.
WEBP_METHOD = 4
# Embedded MIME per codec — everything but webp rides in a PNG container.
CODEC_MIME = {"png": "png", "octree": "png", "pngquant": "png",
              "webp": "webp"}


def have_pngquant():
    return shutil.which("pngquant") is not None


def probe_fps(mp4):
    """Measured, never assumed. Returns int fps or None if unprobeable."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True).stdout.strip()
    if "/" not in out:
        return None
    num, den = out.split("/")
    if not int(den):
        return None
    return round(int(num) / int(den))


def quantize_png(im, colors=QUANT_COLORS):
    """RGBA-safe palette reduction, no dithering.

    PIL's default (median cut) refuses RGBA and would force a flatten,
    destroying the matte's feathered alpha — the thing the whole pipeline
    exists to produce. FASTOCTREE is the one PIL method that quantizes
    with the alpha channel in the octree, emitting a P-mode image whose
    transparency survives as a tRNS chunk.

    It also does not dither, which is exactly why it posterizes shaded
    art. See RUNGS; `pngquant` is the dithered alternative.
    """
    return im.quantize(colors=colors, method=Image.FASTOCTREE)


def encode_pngquant(im, op):
    """256-colour PNG with Floyd-Steinberg dithering, via the pngquant CLI.

    Returns False when pngquant declines the frame: exit 99 means it could
    not hit the quality floor, in which case the honest move is to keep
    the unquantized PNG rather than silently ship something worse than the
    rung claims. Callers must not treat a declined frame as a success.
    """
    tmp = op.with_suffix(".src.png")
    im.save(tmp, optimize=True)
    r = subprocess.run(
        ["pngquant", "--quality", PNGQUANT_QUALITY, "--force",
         "--output", str(op), "--", str(tmp)],
        capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if r.returncode == 0 and op.exists():
        return True
    im.save(op, optimize=True)
    return False


def kept_indices(n, keep):
    """Source frame indices this rung keeps. keep=(num, den)."""
    num, den = keep
    return [i for i in range(n) if i % den < num]


def scaled_fr(fps, keep):
    """fr scaled by the keep fraction, so wall-clock duration is unchanged.

    Kept exact: 32 * 3/4 is 24, but 30 * 3/4 is 22.5 and must NOT be
    rounded — a rounded fr silently retimes the clip, which is the same
    class of bug as the fr:16 half-speed one. Narrowed to int only when
    the value is exactly integral, so existing packs keep writing 32
    rather than 32.0.
    """
    num, den = keep
    fr = fps * num / den
    return int(fr) if float(fr).is_integer() else fr


def build_variant_frames(src_frames, edge, codec, quality, keep, dst):
    """Materialise one rung's frames into `dst`; returns (paths, declined)."""
    dst.mkdir(parents=True, exist_ok=True)
    outs, declined = [], 0
    ext = "webp" if codec == "webp" else "png"
    idxs = kept_indices(len(src_frames), keep)
    for j, i in enumerate(idxs):
        im = Image.open(src_frames[i]).convert("RGBA")
        if im.size != (edge, edge):
            im = im.resize((edge, edge), Image.LANCZOS)
        op = dst / ("frame_%04d.%s" % (j, ext))
        if codec == "webp":
            im.save(op, "WEBP", quality=quality or WEBP_QUALITY,
                    method=WEBP_METHOD)
        elif codec == "pngquant":
            if not encode_pngquant(im, op):
                declined += 1
        else:
            if codec == "octree":
                im = quantize_png(im)
            im.save(op, optimize=True)
        outs.append(op)
    return outs, declined


def score_posterization(ref_im, var_im):
    """How much colour did this rung destroy, and did it band?

    Born from a real miss: the ladder shipped FASTOCTREE rungs that
    measured fine on size and PSNR, and Lewis called the strawberry
    "like an old GIF". Two numbers, and they only mean anything together:

      color_ratio  unique RGB triples inside the character mask, variant
                   over reference. Detects that a palette was imposed.
      deltaE_mean  mean CIE76 dE inside the same mask. Detects that the
                   imposed palette is WRONG.

    Read alone, color_ratio cannot tell posterized from dithered — a
    dithered 256-colour image has the same crushed count as an undithered
    one, because dithering buys accuracy in the local mean, not in the
    count. Measured on strawberry-wave f81: octree 127 colours at dE 2.78
    (Lewis's blotching), pngquant 245 colours at dE 0.94 (acceptable).
    Nearly the same ratio, opposite verdicts. So the posterization flag
    requires BOTH a crushed ratio and a high dE; color_ratio alone is
    reported as the weaker spec'd signal.

    Comparison is at the rung's own resolution against the same-resolution
    Lanczos reference, so the score isolates the codec instead of
    re-measuring the downscale.
    """
    ref = np.array(ref_im.convert("RGBA"))
    var = np.array(var_im.convert("RGBA").resize(ref_im.size, Image.NEAREST))
    mask = ref[..., 3] > 128
    if mask.sum() < 100:
        return None

    def uniq(a):
        return int(len(np.unique(a[mask][:, :3], axis=0)))

    lab_r = cv2.cvtColor(ref[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_v = cv2.cvtColor(var[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.sqrt(((lab_r - lab_v) ** 2).sum(-1))[mask]
    u_ref, u_var = uniq(ref), uniq(var)
    ratio = u_var / max(u_ref, 1)
    dE = float(d.mean())
    return {
        "colors_ref": u_ref, "colors_var": u_var,
        "color_ratio": round(ratio, 4),
        "deltaE_mean": round(dE, 2),
        "deltaE_p95": round(float(np.percentile(d, 95)), 2),
        "alpha_levels": int(len(np.unique(var[..., 3]))),
        # The spec'd signal, kept literal so it can be audited...
        "flag_color_ratio": bool(u_ref > 2000 and ratio < 0.05),
        # ...and the one that actually matched Lewis's eye.
        "flag_posterized": bool(u_ref > 2000 and ratio < 0.05 and dE >= 2.0),
    }


def write_dotlottie(json_path, out_path, name):
    """Minimal dotLottie: manifest + animations/<id>.json, max deflate."""
    manifest = {
        "version": "1.0",
        "revision": 1,
        "keywords": "spike2",
        "author": "image-to-lottie",
        "generator": "spike2_repack",
        "animations": [{"id": name, "direction": 1, "speed": 1,
                        "playMode": "loop", "loop": True, "autoplay": True}],
    }
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        z.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        z.write(json_path, "animations/%s.json" % name)
    return out_path.stat().st_size


def gzip_size(path):
    return len(gzip.compress(path.read_bytes(), 9))


def kb(n):
    return round(n / 1024, 1)


def repack_run(rdir, name, fps):
    frames = sorted((rdir / "rgba").glob("frame_*.png"))
    edge = Image.open(frames[0]).size[0]
    path = rdir / ("%s.json" % name)
    size = pack_lottie(frames, edge, fps, path)
    return {"frames": len(frames), "edge": edge, "fr": fps,
            "json": size, "gz": gzip_size(path)}


def ladder_run(rdir, name, fps, rungs, scratch):
    frames = sorted((rdir / "rgba").glob("frame_*.png"))
    ldir = rdir / "ladder"
    ldir.mkdir(parents=True, exist_ok=True)
    rows = {}
    # Mid-cycle frame: scored on every rung so the numbers are comparable,
    # and snapped to the decimation grid so a -half rung is scored on a
    # frame it actually contains.
    for rung in rungs:
        edge, codec, quality, keep = RUNGS[rung]
        fr = scaled_fr(fps, keep)
        idxs = kept_indices(len(frames), keep)
        # Score the kept frame nearest mid-cycle, so every rung is scored
        # on comparable content even when its frame set differs.
        pos = min(range(len(idxs)), key=lambda k: abs(idxs[k] - len(frames) // 2))
        src_idx = idxs[pos]
        tmp = Path(tempfile.mkdtemp(dir=scratch))
        try:
            vf, declined = build_variant_frames(
                frames, edge, codec, quality, keep, tmp)
            jpath = ldir / ("%s.%s.json" % (name, rung))
            jsize = pack_lottie(vf, edge, fr, jpath, CODEC_MIME[codec])
            ref = Image.open(frames[src_idx]).convert("RGBA")
            if ref.size != (edge, edge):
                ref = ref.resize((edge, edge), Image.LANCZOS)
            score = score_posterization(ref, Image.open(vf[pos]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        lpath = ldir / ("%s.%s.lottie" % (name, rung))
        lsize = write_dotlottie(jpath, lpath, "%s-%s" % (name, rung))
        rows[rung] = {"edge": edge, "codec": codec, "keep": list(keep),
                      "quality": quality, "rejected": rung in REJECTED,
                      "frames": len(vf), "fr": fr, "json": jsize,
                      "gz": gzip_size(jpath), "lottie": lsize,
                      "mime": CODEC_MIME[codec], "score": score,
                      "pngquant_declined": declined,
                      "scored_frame": src_idx,
                      "json_path": str(jpath.relative_to(OUT)),
                      "lottie_path": str(lpath.relative_to(OUT))}
        s = score or {}
        print("    %-13s %3df @ %sfr  json %7.1fKB  gz %7.1fKB  .lottie %7.1fKB"
              "  colors %5d/%5d (%.3f)  dE %5.2f%s"
              % (rung, len(vf), fr, kb(jsize), kb(rows[rung]["gz"]), kb(lsize),
                 s.get("colors_var", 0), s.get("colors_ref", 0),
                 s.get("color_ratio", 0), s.get("deltaE_mean", 0),
                 "  POSTERIZED" if s.get("flag_posterized") else
                 ("  flag:ratio" if s.get("flag_color_ratio") else "")),
              flush=True)
        if declined:
            print("      pngquant declined %d frame(s); kept unquantized"
                  % declined, flush=True)
    return rows


def write_reports(report):
    runs = report["runs"]
    # Columns are the union of everything measured so far, in ladder
    # order — a partial invocation must not blank out rungs an earlier
    # pass already paid for.
    have = {r for rec in runs.values() for r in rec.get("ladder", {})}
    rungs = [r for r in RUNGS if r in have] or report["rungs"]
    report["rungs"] = rungs
    lines = [
        "# Spike 2 — Lottie size ladder",
        "",
        "Generated by `hybrid/spike2_repack.py`. Sizes in KB.",
        "",
        "- `raw` = the repacked full-fidelity pack in the run dir: the "
        "run's NATIVE resolution (`px` column), all frames, no "
        "quantization. Most runs are 512; `strawberry-idle-720` was "
        "generated at 720p and mattes out at 960, so its `512` rung is a "
        "downscale rather than a re-encode.",
        "- Quantization: **%s** (%d colours)." % (report["quant_method"],
                                                 QUANT_COLORS),
        "- Frame-decimated rungs scale `fr` by the same fraction they drop, "
        "so wall-clock duration and pacing are unchanged. `-half` = 16fps, "
        "`-24` = 24fps.",
        "- **Rejected** rungs are kept generatable and on disk for "
        "comparison, but are excluded from default generation and hidden "
        "in the sandbox behind *show rejected*. Octree/pngquant: "
        "posterization. Every `-half` (16fps) rung: Lewis reads it as "
        "\"a flipbook\" regardless of jump size — a fluidity verdict that "
        "velocity-gate clearance does not overrule.",
        "- `fr` is probed per run from `oracle.mp4`; never hardcoded.",
        "",
        "## Rung definitions",
        "",
        "| rung | px | codec | embedded as | frames kept | status |",
        "|---|---|---|---|---|---|",
    ]
    codec_note = {
        "png": "lossless re-encode",
        "octree": "PIL FASTOCTREE, 256 colours, **no dithering**",
        "pngquant": "pngquant q%s, 256 colours, Floyd-Steinberg dithering"
                    % PNGQUANT_QUALITY,
        "webp": "lossy WebP q%s, continuous tone + alpha",
    }
    for r in rungs:
        edge, codec, quality, keep = RUNGS[r]
        note = codec_note[codec]
        if codec == "webp":
            note = note % (quality or WEBP_QUALITY)
        num, den = keep
        kept = ("all" if (num, den) == (1, 1)
                else "%d of every %d" % (num, den))
        lines.append("| %s | %d | %s | image/%s | %s | %s |"
                     % (r, edge, note, CODEC_MIME[codec], kept,
                        "**rejected**" if r in REJECTED else "candidate"))

    for label, key in (("JSON", "json"), ("JSON gzipped", "gz"),
                       (".lottie (zip)", "lottie")):
        lines += ["", "## %s — KB" % label, "",
                  "| run | px | fr | raw | " + " | ".join(rungs) + " |",
                  "|---" * (len(rungs) + 4) + "|"]
        for name in sorted(runs):
            rec = runs[name]
            if not rec.get("ladder"):
                continue
            raw = rec["raw"].get(key)
            cells = [str(kb(rec["ladder"][r][key])) if r in rec["ladder"]
                     else "—" for r in rungs]
            lines.append("| %s | %d | %d | %s | %s |"
                         % (name, rec["raw"]["edge"], rec["raw"]["fr"],
                            kb(raw) if raw else "—", " | ".join(cells)))

    lines += [
        "", "## Posterization", "",
        "Scored on the mid-cycle frame, inside the character mask "
        "(alpha > 128), against the same-resolution Lanczos reference — so "
        "the score isolates the codec rather than re-measuring the "
        "downscale.",
        "",
        "- `colors` = unique RGB triples, variant / reference.",
        "- `dE` = mean CIE76 colour error. `p95` = worst 5%.",
        "- **POSTERIZED** = crushed palette *and* high error "
        "(ratio < 0.05 on a >2000-colour source, dE >= 2.0).",
        "- `ratio-flag` = the crushed palette alone. Dithered rungs trip "
        "this without being defective, which is the whole reason the "
        "flag needs both halves: on strawberry-wave, octree and pngquant "
        "crush the palette by the same factor and only octree looks "
        "wrong.",
        "",
        "| run | rung | colors | ratio | dE | dE p95 | alpha lv | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(runs):
        for rung in rungs:
            r = runs[name].get("ladder", {}).get(rung)
            if not r or not r.get("score"):
                continue
            s = r["score"]
            verdict = ("**POSTERIZED**" if s["flag_posterized"]
                       else "ratio-flag" if s["flag_color_ratio"] else "ok")
            lines.append("| %s | %s | %d / %d | %.3f | %.2f | %.2f | %d | %s |"
                         % (name, rung, s["colors_var"], s["colors_ref"],
                            s["color_ratio"], s["deltaE_mean"],
                            s["deltaE_p95"], s["alpha_levels"], verdict))

    if report["skipped"]:
        lines += ["", "## Skipped", ""]
        lines += ["- `%s` — %s" % (n, why) for n, why in report["skipped"]]

    lines += ["", "Total ladder bytes on disk: %.1f MB."
              % (report["ladder_bytes"] / 1e6), ""]
    LADDER_REPORT_MD.write_text("\n".join(lines))
    LADDER_REPORT_JSON.write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", help="comma-separated run dir names")
    ap.add_argument("--rungs", default=",".join(DEFAULT_RUNGS),
                    help="default = candidate rungs only; rejected rungs "
                         "(octree/pngquant/16fps) must be named explicitly")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--no-repack", action="store_true")
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    args = ap.parse_args()

    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    bad = [r for r in rungs if r not in RUNGS]
    if bad:
        raise SystemExit("unknown rung(s): %s" % bad)

    want = set(args.runs.split(",")) if args.runs else None
    dirs = sorted(p for p in OUT.iterdir() if p.is_dir())
    quant_method = "pngquant" if have_pngquant() else "PIL FASTOCTREE"

    report = {"quant_method": quant_method, "rungs": rungs,
              "runs": {}, "skipped": [], "ladder_bytes": 0}
    # Preserve any rungs measured in a previous partial run.
    if LADDER_REPORT_JSON.exists():
        prev = json.loads(LADDER_REPORT_JSON.read_text())
        report["runs"] = prev.get("runs", {})

    for rdir in dirs:
        name = rdir.name
        if want and name not in want:
            continue
        mp4 = rdir / "oracle.mp4"
        frames = sorted((rdir / "rgba").glob("frame_*.png"))
        if not frames:
            report["skipped"].append((name, "no rgba/ frames"))
            continue
        if not mp4.exists():
            report["skipped"].append((name, "no oracle.mp4"))
            continue
        fps = probe_fps(mp4)
        if not fps:
            report["skipped"].append((name, "fps unprobeable"))
            continue

        print("== %s  (%d frames, probed %dfps)" % (name, len(frames), fps),
              flush=True)
        rec = report["runs"].setdefault(name, {})
        if not args.no_repack:
            rec["raw"] = repack_run(rdir, name, fps)
            print("    raw        %3df @ %2dfr  json %7.1fKB  gz %7.1fKB"
                  % (rec["raw"]["frames"], fps, kb(rec["raw"]["json"]),
                     kb(rec["raw"]["gz"])), flush=True)
        if not args.no_ladder:
            rec.setdefault("ladder", {}).update(
                ladder_run(rdir, name, fps, rungs, args.scratch))

    for rec in report["runs"].values():
        for r in rec.get("ladder", {}).values():
            report["ladder_bytes"] += r["json"] + r["lottie"]

    write_reports(report)
    print("\nquantization: %s" % quant_method)
    print("ladder on disk: %.1f MB" % (report["ladder_bytes"] / 1e6))
    print("reports: %s\n         %s" % (LADDER_REPORT_JSON, LADDER_REPORT_MD))
    if report["skipped"]:
        print("skipped: %s" % report["skipped"])


if __name__ == "__main__":
    main()
