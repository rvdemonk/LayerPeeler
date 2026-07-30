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
MASCOTS = RESEARCH / "corpus" / "mascots"
LEDGER = RESEARCH / "docs" / "workorders" / "hybrid-oracle-spikes" / "ledger.md"

# URL prefix -> absolute root. Everything else 404s.
MOUNTS = {
    "/media/spike2/": SPIKE2,
    "/media/mascots/": MASCOTS,
}

# 8646, not the more obvious 8642: a long-lived `python -m http.server 8642`
# of Lewis's already owns that port, and a preview that collides on launch
# every time is a preview he stops launching.
DEFAULT_PORT = 8646

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
    sig = [(r["dir"], r["mtime"]) for w in waves for r in w["runs"]]
    return {
        "waves": waves,
        "signature": str(hash((tuple(sig), LEDGER.stat().st_mtime if LEDGER.exists() else 0))),
        "roots": {"spike2": str(SPIKE2), "mascots": str(MASCOTS), "ledger": str(LEDGER)},
    }


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
        if path == "/api/manifest":
            return self.send_json(build_manifest())
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


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    if not SPIKE2.is_dir():
        sys.exit("no run dir: %s" % SPIKE2)
    man = build_manifest()
    n = sum(len(w["runs"]) for w in man["waves"])
    print("appraisal sandbox — %d waves, %d runs" % (len(man["waves"]), n))
    for w in man["waves"]:
        print("  %-10s %-52s %2d runs" % (w["id"], w["title"][:52], len(w["runs"])))
        if w["missing"]:
            print("             MISSING ON DISK: %s" % ", ".join(w["missing"]))
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
