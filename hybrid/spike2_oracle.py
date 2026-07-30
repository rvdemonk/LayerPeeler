"""Spike 2 — maximalist oracle: mascot image -> Wan i2v -> matte -> frame-seq
Lottie ("the owl mechanism", decoded at workorder birth).

One invocation = one generation = one ledger row. The ledger
(docs/workorders/hybrid-oracle-spikes/ledger.md) records method -> output ->
mechanical gates -> Lewis verdict (verbatim, filled in later).

Pipeline per run:
  1. submit to queue.fal.run (Wan 2.2 A14B turbo; per-VIDEO pricing
     $0.05/480p $0.075/580p $0.10/720p). Same image as start AND end frame
     unless --no-loop: loop closure by construction.
  2. poll -> download mp4 -> extract frames (ffmpeg).
  3. matte: flat-background chroma distance + morphology + feathered alpha
     (background color sampled from frame borders).
  4. pack frame-seq Lottie (embedded base64 PNG assets, one image layer per
     frame) + eyeball GIF over checkerboard + contact sheet.
  5. append ledger row (gates left blank until gate pass runs).

Usage:
  .venv-hybrid/bin/python hybrid/spike2_oracle.py \
      --image ../corpus/mascots/strawberry.jpg --name strawberry-breathe \
      --prompt "..." [--resolution 480p] [--seed N] [--no-loop] [--dry-run]

Env: FAL_KEY (source ~/.env).
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ENDPOINT = "fal-ai/wan/v2.2-a14b/image-to-video/turbo"
QUEUE = "https://queue.fal.run"
ROOT = Path(__file__).resolve().parent.parent          # LayerPeeler/
OUT = ROOT / "out" / "spike2"
LEDGER = ROOT.parent / "docs" / "workorders" / "hybrid-oracle-spikes" / "ledger.md"
PRICE = {"480p": 0.05, "580p": 0.075, "720p": 0.10}


def api(url, payload=None, key=None):
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Key {key}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def data_uri(path):
    ext = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(
        Path(path).read_bytes()).decode()


def generate(args, key):
    payload = {
        "prompt": args.prompt,
        "image_url": data_uri(args.image),
        "resolution": args.resolution,
        "aspect_ratio": "1:1",
        "enable_safety_checker": False,
        "enable_output_safety_checker": False,
        "video_quality": "high",
    }
    if not args.no_loop:
        payload["end_image_url"] = payload["image_url"]
    if args.seed is not None:
        payload["seed"] = args.seed
    sub = api(f"{QUEUE}/{ENDPOINT}", payload, key)
    status_url, response_url = sub["status_url"], sub["response_url"]
    print(f"queued: {sub.get('request_id')}")
    t0 = time.time()
    while True:
        time.sleep(5)
        st = api(status_url, key=key)
        if st["status"] == "COMPLETED":
            break
        if st["status"] not in ("IN_QUEUE", "IN_PROGRESS"):
            sys.exit(f"FAILED: {json.dumps(st)[:500]}")
        print(f"  {st['status']} {time.time()-t0:.0f}s", flush=True)
    res = api(response_url, key=key)
    return res


def extract_frames(mp4, fdir):
    fdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                    str(fdir / "frame_%04d.png")], check=True)
    return sorted(fdir.glob("frame_*.png"))


def matte(frames, mdir):
    """Flat-bg matte: bg color = median of border pixels per frame; alpha =
    smoothstep on color distance; despeckle with morphology."""
    mdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for fp in frames:
        im = cv2.imread(str(fp)).astype(np.float32)
        border = np.concatenate([im[:8].reshape(-1, 3), im[-8:].reshape(-1, 3),
                                 im[:, :8].reshape(-1, 3),
                                 im[:, -8:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        d = np.linalg.norm(im - bg, axis=-1)
        # smoothstep between lo/hi color distance -> feathered alpha
        lo, hi = 12.0, 40.0
        a = np.clip((d - lo) / (hi - lo), 0, 1)
        a = a * a * (3 - 2 * a)
        hard = (a > 0.5).astype(np.uint8)
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        # keep only components touching the largest blob (drop bg speckle)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(hard)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hard = (lab == big).astype(np.uint8)
        a = a * cv2.dilate(hard, np.ones((7, 7), np.uint8))
        rgba = np.dstack([im, a * 255]).astype(np.uint8)
        op = mdir / fp.name
        cv2.imwrite(str(op), rgba)
        outs.append(op)
    return outs


def pack_lottie(rgba_frames, size, fps, path):
    """Frame-seq Lottie: one embedded PNG asset + one image layer per frame."""
    assets, layers = [], []
    for i, fp in enumerate(rgba_frames):
        b64 = base64.b64encode(fp.read_bytes()).decode()
        assets.append({"id": f"f{i}", "w": size, "h": size,
                       "u": "", "p": f"data:image/png;base64,{b64}", "e": 1})
        layers.append({"ddd": 0, "ind": i + 1, "ty": 2, "nm": f"f{i}",
                       "refId": f"f{i}", "ip": i, "op": i + 1, "st": 0,
                       "ks": {"o": {"a": 0, "k": 100},
                              "p": {"a": 0, "k": [size / 2, size / 2, 0]},
                              "a": {"a": 0, "k": [size / 2, size / 2, 0]},
                              "s": {"a": 0, "k": [100, 100, 100]}}})
    lot = {"v": "5.7.0", "fr": fps, "ip": 0, "op": len(rgba_frames),
           "w": size, "h": size, "nm": path.stem, "ddd": 0,
           "assets": assets, "layers": layers}
    path.write_text(json.dumps(lot, separators=(",", ":")))
    return path.stat().st_size


def eyeball(rgba_frames, rdir, name):
    """GIF over checkerboard (alpha made visible) + contact sheet."""
    from PIL import Image
    tiles, pil = [], []
    for i, fp in enumerate(rgba_frames):
        im = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        checker = (((yy // 16 + xx // 16) % 2) * 40 + 200)[..., None]
        checker = np.repeat(checker, 3, -1).astype(np.float32)
        a = im[..., 3:4].astype(np.float32) / 255
        comp = (im[..., :3] * a + checker * (1 - a)).astype(np.uint8)
        pil.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(comp, (384, 384)), cv2.COLOR_BGR2RGB)))
        if i % 4 == 0:
            t = cv2.resize(comp, (192, 192))
            cv2.putText(t, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            tiles.append(t)
    pil[0].save(str(rdir / f"{name}.gif"), save_all=True,
                append_images=pil[1:], duration=62, loop=0)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                               cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite(str(rdir / f"{name}_sheet.png"), np.vstack(rows))


def ledger_row(args, res, n_frames, lottie_kb, dur_s):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# Run ledger — hybrid-oracle-spikes (Spike 2+)\n\n"
            "One row per oracle generation. Gates = Claude's mechanical "
            "instruments; verdict = Lewis, VERBATIM, filled after appraisal."
            "\n\n| id | date | input | model | res | seed | loop | cost | "
            "frames | lottie KB | gen s | gates | Claude pre-verdict | "
            "Lewis verdict |\n|---|---|---|---|---|---|---|---|---|---|---|"
            "---|---|---|\n")
    rows = LEDGER.read_text().count("\n| r")
    rid = f"r{rows + 1:03d}"
    seed = res.get("seed", args.seed)
    LEDGER.write_text(LEDGER.read_text() + (
        f"| {rid} | {time.strftime('%Y-%m-%d')} | "
        f"{Path(args.image).stem}: \"{args.prompt[:60]}...\" | wan2.2-a14b-"
        f"turbo | {args.resolution} | {seed} | {not args.no_loop} | "
        f"${PRICE[args.resolution]:.3f} | {n_frames} | {lottie_kb} | "
        f"{dur_s:.0f} | — | — | — |\n"))
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--resolution", default="480p", choices=list(PRICE))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-loop", action="store_true",
                    help="don't condition end frame on the input image")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("FAL_KEY") or sys.exit("FAL_KEY not set")
    if args.dry_run:
        print(json.dumps({"endpoint": ENDPOINT, "res": args.resolution,
                          "cost": PRICE[args.resolution],
                          "loop": not args.no_loop}, indent=1))
        return
    rdir = OUT / args.name
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = generate(args, key)
    dur = time.time() - t0
    video_url = res["video"]["url"]
    mp4 = rdir / "oracle.mp4"
    urllib.request.urlretrieve(video_url, mp4)
    (rdir / "response.json").write_text(json.dumps(res, indent=1))
    frames = extract_frames(mp4, rdir / "frames")
    rgba = matte(frames, rdir / "rgba")
    size = cv2.imread(str(frames[0])).shape[0]
    kb = pack_lottie(rgba, size, 16, rdir / f"{args.name}.json") // 1024
    eyeball(rgba, rdir, args.name)
    rid = ledger_row(args, res, len(frames), kb, dur)
    print(f"== {rid} {args.name}: {len(frames)} frames, lottie {kb}KB, "
          f"gen {dur:.0f}s, seed {res.get('seed')}")
    print(f"   eyeball: {rdir / (args.name + '.gif')}")


if __name__ == "__main__":
    main()
