"""Appraisal sandbox server — a dev preview for eyeballing oracle runs.

WHY this exists and not spike/sandbox/build_sandbox.py: that builder
base64-embedded every GIF into a single 10MB index.html and overwrote it
each iteration, so (a) the browser reloaded 10MB per tweak and (b) every
past batch of experiments was destroyed by the next one. Appraisal is a
longitudinal act — Lewis needs to compare wave 3 against wave 1 — so the
model here is: media stays on disk and is served by relative path, the
page is a static shell, and the run list is a JSON manifest regenerated
on every request. Nothing is ever overwritten; new runs simply appear.

Two mount prefixes because the artifacts and the inputs live in
different trees: out/spike2/ is inside the LayerPeeler repo, while
corpus/mascots/ sits beside it in the research root. Symlinks would
either leak outside the repo or break on checkout, so the handler maps
URL prefixes to absolute roots and refuses anything that resolves out of
them.

The manifest merges three sources of truth, none of which is edited here:
  - sandbox/waves.json      which runs belong to which wave (curated, VERSIONED)
  - the out/spike2/ dir scan what actually exists on disk (ground truth)
  - docs/workorders/hybrid-oracle-spikes/ledger.md  the verdicts (prose)
A run dir in no wave lands in a synthetic "current" wave, so output can
never be invisible just because the manifest wasn't updated.

Usage: python3 sandbox/serve.py [port]
"""

import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

SANDBOX = Path(__file__).resolve().parent
LAYERPEELER = SANDBOX.parent
RESEARCH = LAYERPEELER.parent

SPIKE2 = LAYERPEELER / "out" / "spike2"
PACKS = LAYERPEELER / "out" / "packs"
MASCOTS = RESEARCH / "corpus" / "mascots"
LEDGER = RESEARCH / "docs" / "workorders" / "hybrid-oracle-spikes" / "ledger.md"

VENDOR = SANDBOX / "vendor"

# URL prefix -> absolute root. Everything else 404s.
# lottie-web is vendored, never a CDN: the sandbox has to work offline and
# an appraisal instrument that silently changes version under you is worse
# than no instrument.
MOUNTS = {
    "/media/spike2/": SPIKE2,
    "/media/packs/": PACKS,
    "/media/mascots/": MASCOTS,
    "/vendor/": VENDOR,
}

# 8646, not the more obvious 8642: a long-lived `python -m http.server 8642`
# of Lewis's already owns that port, and a preview that collides on launch
# every time is a preview he stops launching.
DEFAULT_PORT = 8646

# Display order for the player's variant dropdown: heaviest first, so the
# list reads as a descent. Mirrors RUNGS in hybrid/spike2_repack.py; a rung
# missing from here still shows, just last.
LADDER_ORDER = ["raw",
                "512", "512webp", "512webp-q65", "512webp-q50",
                "512webp-24", "512webp-q65-24", "512webp-q50-24",
                "448webp",
                "384", "384webp",
                "256", "256webp",
                # rejected, ordered last — reachable, never leading
                "512q", "384q", "256q", "512pq", "256pq",
                "512q-half", "384q-half", "256q-half",
                "512webp-half", "256webp-half"]

# Ruled out by Lewis's eye, not by measurement. Mirrors REJECTED in
# hybrid/spike2_repack.py. The files stay and stay playable — a rejected
# rung is the comparison you need when judging the one that replaced it —
# but they are hidden until "show rejected" is ticked.
REJECTED_RUNGS = {"512q", "384q", "256q", "512q-half", "384q-half",
                  "256q-half", "512pq", "256pq",
                  "512webp-half", "256webp-half"}

# Ledger header labels -> manifest keys. Kept explicit so a renamed or
# reordered ledger column degrades to a missing field rather than
# silently shifting every verdict one cell to the left.
LEDGER_FIELDS = {
    "id": "id",
    "date": "date",
    "input": "input",
    "model": "model",
    "res": "res",
    "seed": "seed",
    "loop": "loop",
    "cost": "cost",
    "frames": "frames",
    "lottie kb": "lottie_kb",
    "gen s": "gen_s",
    "gates": "gates_note",
    "claude pre-verdict": "claude",
    "lewis verdict": "lewis",
}


def strip_markdown(cell):
    """Ledger cells are prose with light markdown; the page renders text."""
    s = cell.strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    return s.strip()


def parse_ledger():
    """Parse the markdown run table into {id: {field: text}}.

    Defensive by construction: we key columns off the header row rather
    than by position, skip the alignment row, and ignore any row whose
    first cell is not an r-number. A malformed ledger must not take the
    appraisal instrument down — verdicts just show as missing.
    """
    if not LEDGER.exists():
        return {}
    rows, header = {}, None
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = [c.strip().lower() for c in cells]
            continue
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rec = {}
        for label, cell in zip(header, cells):
            key = LEDGER_FIELDS.get(label)
            if key:
                rec[key] = strip_markdown(cell)
        rid = rec.get("id", "")
        if re.fullmatch(r"r\d+", rid):
            rows[rid] = rec
    return rows


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


_head_cache = {}


def lottie_head(path):
    """Read fr / op / w out of a Lottie without parsing it.

    pack_lottie writes the scalar keys before the assets array, so the
    first few hundred bytes carry everything the player UI needs. This
    matters: the raw packs are ~30MB each and the manifest is rebuilt on
    every 4s poll — json.load on 31 of those would make the sandbox
    unusable to save one regex.
    """
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    hit = _head_cache.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    with path.open("rb") as fh:
        head = fh.read(512).decode("utf-8", "replace")

    def field(name):
        m = re.search(r'"%s"\s*:\s*(-?\d+(?:\.\d+)?)' % name, head)
        return float(m.group(1)) if m else None

    rec = {"fr": field("fr"), "frames": field("op"), "w": field("w"),
           "bytes": st.st_size}
    _head_cache[str(path)] = (key, rec)
    return rec


def ladder_of(rdir, name, base):
    """Playable variants for one run: the raw pack plus every ladder rung.

    Disk is the source of truth (a rung exists iff its JSON is there);
    ladder_report.json only contributes the gzip/.lottie numbers, which
    cannot be read off the filesystem. The player loads plain JSON — the
    .lottie zips are for size measurement and shipping, not playback.
    """
    rep = _ladder_report().get(name, {})
    sizes = rep.get("ladder", {})
    out = []
    raw = rdir / ("%s.json" % name)
    if raw.exists():
        rec = dict(lottie_head(raw), rung="raw", url=base + raw.name)
        rec["gz_bytes"] = (rep.get("raw") or {}).get("gz")
        out.append(rec)
    for jp in sorted((rdir / "ladder").glob("%s.*.json" % name)):
        rung = jp.name[len(name) + 1:-len(".json")]
        rec = dict(lottie_head(jp), rung=rung,
                   url=base + "ladder/" + jp.name)
        s = sizes.get(rung, {})
        rec["gz_bytes"] = s.get("gz")
        rec["lottie_bytes"] = s.get("lottie")
        # Codec and posterization score ride along so the player can name
        # what it is showing — a rung that Lewis rejected by eye should
        # say so on the card, not only in a report he has to go find.
        rec["codec"] = s.get("codec")
        rec["score"] = s.get("score")
        rec["quality"] = s.get("quality")
        rec["keep"] = s.get("keep")
        rec["rejected"] = rung in REJECTED_RUNGS
        if (jp.with_suffix(".lottie")).exists():
            rec["dotlottie_url"] = base + "ladder/" + jp.stem + ".lottie"
        out.append(rec)
    # Ladder order, not glob order: the dropdown is a ladder, and "256"
    # sorting above "512" would make it read as an arbitrary list.
    rank = {r: i for i, r in enumerate(LADDER_ORDER)}
    out.sort(key=lambda v: rank.get(v["rung"], len(LADDER_ORDER)))
    return out


_report_cache = {}


def _ladder_report():
    p = SPIKE2 / "ladder_report.json"
    if not p.exists():
        return {}
    key = p.stat().st_mtime_ns
    if _report_cache.get("key") != key:
        _report_cache["key"] = key
        _report_cache["val"] = (read_json(p) or {}).get("runs", {})
    return _report_cache["val"]


def mascot_of(run_dir):
    """Run dirs are named <mascot>-<motion>[n]; the input image is the
    mascot prefix. Matched against what is actually in corpus/mascots/ so
    a hyphenated future mascot name still resolves."""
    for img in sorted(MASCOTS.glob("*.jpg")) + sorted(MASCOTS.glob("*.png")):
        if run_dir == img.stem or run_dir.startswith(img.stem + "-"):
            return img.stem, "/media/mascots/" + img.name
    return run_dir.split("-")[0], None


def guess_ledger_id(ledger, mascot, prompt, seed):
    """Recover the ledger row for a run dir that no wave has claimed yet.

    The ledger records no run-dir name, so a freshly generated run would
    otherwise appear verdict-less until someone hand-maps it. Seed is the
    strong key (unique per generation); the mascot + truncated-prompt
    excerpt in the 'input' cell is the fallback. Returns None rather than
    guessing loosely — a wrong verdict on a card is worse than none.
    """
    if seed is not None:
        for rid, row in ledger.items():
            if row.get("seed", "").strip() == str(seed):
                return rid
    if not prompt:
        return None
    for rid, row in ledger.items():
        cell = row.get("input", "")
        if not cell.lower().startswith(mascot.lower() + ":"):
            continue
        m = re.search(r'"(.*?)(?:\.\.\.|…)?"\s*$', cell)
        excerpt = (m.group(1) if m else "").strip()
        if len(excerpt) >= 20 and prompt.startswith(excerpt):
            return rid
    return None


def describe_run(entry, ledger):
    """Build one manifest run record from disk + ledger."""
    name = entry.get("dir")
    rdir = SPIKE2 / name
    if not rdir.is_dir():
        return None

    gates = read_json(rdir / "gates.json") or {}
    resp = read_json(rdir / "response.json") or {}
    mascot, mascot_url = mascot_of(name)

    lid = entry.get("ledger")
    if not lid:
        lid = guess_ledger_id(ledger, mascot, resp.get("prompt") or "", resp.get("seed"))
    lrow = ledger.get(lid or "", {})

    prompt = resp.get("prompt")
    if not prompt:
        # Ledger 'input' cell is 'mascot: "excerpt…"' — better than nothing.
        m = re.search(r'"(.*)"', lrow.get("input", ""))
        prompt = m.group(1) if m else ""

    base = "/media/spike2/" + name + "/"

    def opt(fname):
        return base + fname if (rdir / fname).exists() else None

    return {
        "dir": name,
        "ledger_id": lid,
        "mascot": mascot,
        "mascot_url": mascot_url,
        "prompt": prompt,
        "seed": resp.get("seed", lrow.get("seed")),
        "gates": gates,
        "ledger": lrow,
        "gif": opt(name + ".gif"),
        "sheet": opt(name + "_sheet.png"),
        "gates_png": opt("gates.png"),
        "mp4": opt("oracle.mp4"),
        "lottie": opt(name + ".json"),
        "variants": ladder_of(rdir, name, base),
        "mtime": int((rdir / "gates.json").stat().st_mtime)
        if (rdir / "gates.json").exists()
        else int(rdir.stat().st_mtime),
    }


def build_manifest():
    """Merge waves.json, the dir scan and the ledger into render-ready JSON.

    Waves come back newest-first because appraisal starts from the latest
    batch; the synthetic 'current' wave (unassigned dirs) sorts first of
    all, since anything not yet curated is by definition the newest thing
    on disk and the thing awaiting an eyeball.
    """
    ledger = parse_ledger()
    decl = read_json(SANDBOX / "waves.json") or read_json(SPIKE2 / "waves.json") or {}
    waves_in = decl.get("waves", [])

    waves, claimed = [], set()
    for w in waves_in:
        entries = []
        for r in w.get("runs", []):
            entries.append({"dir": r} if isinstance(r, str) else dict(r))
        runs = []
        for e in entries:
            claimed.add(e.get("dir"))
            rec = describe_run(e, ledger)
            if rec:
                runs.append(rec)
        waves.append(
            {
                "id": w.get("id") or w.get("title", "wave"),
                "title": w.get("title", w.get("id", "wave")),
                "date": w.get("date", ""),
                "note": w.get("note", ""),
                "runs": runs,
                "missing": [e["dir"] for e in entries if not (SPIKE2 / e["dir"]).is_dir()],
            }
        )
    waves.reverse()

    loose = []
    for p in sorted(SPIKE2.iterdir()):
        if p.is_dir() and p.name not in claimed and not p.name.startswith("."):
            rec = describe_run({"dir": p.name}, ledger)
            if rec:
                loose.append(rec)
    if loose:
        loose.sort(key=lambda r: -r["mtime"])
        waves.insert(
            0,
            {
                "id": "current",
                "title": "Current — unassigned runs",
                "date": "",
                "note": "Not yet in waves.json. Add them to a wave entry to freeze this batch.",
                "runs": loose,
                "missing": [],
            },
        )

    # Signature lets the page skip re-render when nothing changed, so
    # looping GIFs are not restarted every poll.
    sig = [(r["dir"], r["mtime"],
            tuple((v["rung"], v["bytes"]) for v in r["variants"]))
           for w in waves for r in w["runs"]]
    return {
        "waves": waves,
        "signature": str(hash((tuple(sig), LEDGER.stat().st_mtime if LEDGER.exists() else 0))),
        "roots": {"spike2": str(SPIKE2), "mascots": str(MASCOTS), "ledger": str(LEDGER)},
    }


def pack_ladder_of(rdir, name, base):
    """Playable variants for one pack clip. Same shape as ladder_of()."""
    out = []
    raw = rdir / ("%s.json" % name)
    if raw.exists():
        rec = dict(lottie_head(raw), rung="raw", url=base + raw.name)
        out.append(rec)
    ladder_dir = rdir / "ladder"
    if ladder_dir.is_dir():
        for jp in sorted(ladder_dir.glob("%s.*.json" % name)):
            rung = jp.name[len(name) + 1:-len(".json")]
            rec = dict(lottie_head(jp), rung=rung,
                       url=base + "ladder/" + jp.name)
            if (jp.with_suffix(".lottie")).exists():
                rec["dotlottie_url"] = base + "ladder/" + jp.stem + ".lottie"
            out.append(rec)
    rank = {r: i for i, r in enumerate(LADDER_ORDER)}
    out.sort(key=lambda v: rank.get(v["rung"], len(LADDER_ORDER)))
    return out


def _pack_report(pack_dir):
    p = pack_dir / "clips" / "ladder_report.json"
    if not p.exists():
        return {}
    key = p.stat().st_mtime_ns
    if _report_cache.get("pk:" + str(p)) != key:
        _report_cache["pk:" + str(p)] = key
        _report_cache["val:" + str(p)] = (read_json(p) or {}).get("runs", {})
    return _report_cache.get("val:" + str(p), {})


def build_pack_manifest():
    packs = []
    if not PACKS.is_dir():
        return {"packs": packs, "roots": {"packs": str(PACKS), "mascots": str(MASCOTS)}}
    for pp in sorted(PACKS.iterdir()):
        if not pp.is_dir() or pp.name.startswith("."):
            continue
        clips_dir = pp / "clips"
        if not clips_dir.is_dir():
            continue
        pj = read_json(pp / "pack.json") or {}
        report = _pack_report(pp)
        mascot_img = pj.get("mascot", pj.get("image", ""))
        mascot_url = None
        if mascot_img and Path(mascot_img).exists():
            mascot_url = "/media/mascots/" + Path(mascot_img).name
        elif mascot_img:
            mascot_url = mascot_img
        clips = []
        for cd in sorted(clips_dir.iterdir()):
            if not cd.is_dir():
                continue
            name = cd.name
            gates = read_json(cd / "gates.json") or {}
            resp = read_json(cd / "response.json") or {}
            run_json = read_json(cd / "run.json") or {}
            emote_spec = {}
            for e in pj.get("emotes", []):
                if e.get("name") == name:
                    emote_spec = e
                    break
            base = "/media/packs/%s/clips/%s/" % (pp.name, name)

            def opt(fname):
                return base + fname if (cd / fname).exists() else None

            # Size info from ladder_report.json
            ladder_sizes = report.get(name, {})
            raw_bytes = None
            variants = pack_ladder_of(cd, name, base)
            if variants:
                raw_v = [v for v in variants if v.get("rung") == "raw"]
                if raw_v:
                    raw_bytes = raw_v[0].get("bytes")

            clips.append({
                "name": name,
                "dir": name,
                "prompt": emote_spec.get("prompt", resp.get("prompt", "")),
                "seed": resp.get("seed"),
                "gates": gates,
                "variants": variants,
                "raw_bytes": raw_bytes,
                "gif": opt(name + ".gif"),
                "sheet": opt(name + "_sheet.png"),
                "gates_png": opt("gates.png"),
                "mp4": opt("oracle.mp4"),
                "lottie_url": opt(name + ".json"),
                "ladder_sizes": ladder_sizes,
                "emote_status": emote_spec.get("status", ""),
            })
        packs.append({
            "name": pp.name,
            "title": pj.get("pack", pp.name),
            "character": pj.get("character", ""),
            "created": pj.get("created_at", ""),
            "mascot_url": mascot_url,
            "clips": clips,
        })
    return {"packs": packs, "roots": {"packs": str(PACKS), "mascots": str(MASCOTS)}}


class Handler(BaseHTTPRequestHandler):
    server_version = "appraisal-sandbox/1"

    # HEAD shares every route with GET and only suppresses the body, so a
    # link-integrity sweep over 60MB of GIFs costs headers instead of bytes.
    head_only = False

    def log_message(self, fmt, *args):
        # Media requests are noisy and useless; only surface API + errors.
        if "/api/" in self.path or not self.path.startswith("/media/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))

    def do_HEAD(self):
        self.head_only = True
        self.do_GET()

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.send_file(SANDBOX / "index.html", cache=False)
        if path in ("/packs", "/packs.html"):
            return self.send_file(SANDBOX / "packs.html", cache=False)
        if path == "/api/manifest":
            return self.send_json(build_manifest())
        if path == "/api/packs":
            return self.send_json(build_pack_manifest())
        for prefix, root in MOUNTS.items():
            if path.startswith(prefix):
                return self.send_mounted(root, path[len(prefix):])
        self.send_error(404, "no route")

    def send_mounted(self, root, rel):
        """Resolve inside the mount root only — a mounted dir must not be
        a hole through which the rest of the filesystem is readable."""
        target = (root / rel).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return self.send_error(403, "outside mount")
        if not target.is_file():
            return self.send_error(404, "not a file")
        self.send_file(target)

    def send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not self.head_only:
            self.wfile.write(body)

    def send_file(self, path, cache=True):
        """Serves whole files, plus single-range requests so the browser's
        video element can seek oracle.mp4 instead of refusing to scrub."""
        try:
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            rng = self.headers.get("Range", "")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip()) if rng else None
            start, end = 0, size - 1
            partial = False
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                else:
                    start = max(0, size - int(m.group(2)))
                partial = start <= end < size
            length = end - start + 1 if partial else size
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Cache-Control", "max-age=60" if cache else "no-store")
            self.end_headers()
            if self.head_only:
                return
            with path.open("rb") as fh:
                if partial:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                else:
                    while True:
                        chunk = fh.read(262144)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser cancelled a GIF/mp4 mid-flight; normal
        except FileNotFoundError:
            self.send_error(404, "gone")


def set_root(path):
    """Point the sandbox at a different tree of run dirs.

    The consolidated pipeline writes to out/pipeline/ rather than
    out/spike2/, in the same per-run layout (rgba/, ladder/, gates.json,
    the GIF and sheet) and with a ladder_report.json in the same shape.
    Rather than fork the player, the root is swappable: one appraisal
    instrument, two trees, no second copy to drift.
    """
    global SPIKE2
    SPIKE2 = Path(path).resolve()
    MOUNTS["/media/spike2/"] = SPIKE2
    _report_cache.clear()


def main():
    argv = sys.argv[1:]
    if "--root" in argv:
        i = argv.index("--root")
        set_root(argv[i + 1])
        del argv[i:i + 2]
    port = int(argv[0]) if argv else DEFAULT_PORT
    if not SPIKE2.is_dir():
        sys.exit("no run dir: %s" % SPIKE2)
    man = build_manifest()
    n = sum(len(w["runs"]) for w in man["waves"])
    print("appraisal sandbox — %d waves, %d runs" % (len(man["waves"]), n))
    for w in man["waves"]:
        print("  %-10s %-52s %2d runs" % (w["id"], w["title"][:52], len(w["runs"])))
        if w["missing"]:
            print("             MISSING ON DISK: %s" % ", ".join(w["missing"]))
    pm = build_pack_manifest()
    if pm["packs"]:
        print("\n  packs (%d)" % len(pm["packs"]))
    for p in pm["packs"]:
        print("  %-20s %2d clips  → /packs" % (p["name"], len(p["clips"])))
    print("\n  http://localhost:%d/   (ctrl-c to stop)\n" % port)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        sys.exit("port %d busy (%s) — try: ./serve.sh %d" % (port, e.strerror, port + 1))
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("")
