"""Stage 1 — mascot PNG + prompt -> looping mp4, via Wan 2.2 i2v on fal.

Lifted from hybrid/spike2_oracle.py (which stays as the spike-era ledger
driver). Behaviour is deliberately unchanged: same endpoints, same
payload, same per-video pricing table, same loop-by-construction trick of
passing the input image as BOTH start and end frame.

Two things this module adds over the spike version:
  - it can be dry-run, printing the exact payload it would post with the
    image data-URI elided, so the request can be code-reviewed without
    spending anything;
  - it can be bypassed entirely (`from_video`), so every stage below
    generation is testable on mp4s already on disk. Every local change to
    matting, gating or encoding should be verified that way — fal credits
    are for verifying fal, not for verifying our own arithmetic.

Env: FAL_KEY (set -a; source ~/.env; set +a).
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ENDPOINTS = {
    "turbo": "fal-ai/wan/v2.2-a14b/image-to-video/turbo",   # per-VIDEO price
    "full": "fal-ai/wan/v2.2-a14b/image-to-video",          # per-second price
}
QUEUE = "https://queue.fal.run"

# turbo = flat per video; full = per-second x ~5s clip.
# Turbo is the default on Lewis's 2026-07-30 ruling: full is "a bit
# smoother, like a higher frame rate, but that's about it... in terms of
# the mouth, the eyes, the coherence, honestly the turbo seems to win".
# Sample size 1 — the tier flag stays, the default does not move without
# a tailored rubric.
PRICE = {"turbo": {"480p": 0.05, "580p": 0.075, "720p": 0.10},
         "full": {"480p": 0.20, "580p": 0.30, "720p": 0.40}}

POLL_SECONDS = 5
POLL_TIMEOUT = 900


def api(url, payload=None, key=None, timeout=120):
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Key %s" % key)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def data_uri(path):
    ext = Path(path).suffix.lstrip(".").lower().replace("jpg", "jpeg")
    return "data:image/%s;base64," % ext + base64.b64encode(
        Path(path).read_bytes()).decode()


def build_payload(image, prompt, resolution="480p", seed=None, loop=True):
    """The request body. Kept separate from the call so --dry-run can show
    exactly what would go over the wire."""
    uri = data_uri(image)
    payload = {
        "prompt": prompt,
        "image_url": uri,
        "resolution": resolution,
        "aspect_ratio": "1:1",
        "enable_safety_checker": False,
        "enable_output_safety_checker": False,
        "video_quality": "high",
    }
    if loop:
        # Loop closure by construction: conditioning the last frame on the
        # input image is why loop_ssim comes back at .97+ without any
        # crossfade. Removing this makes the seam a real problem.
        payload["end_image_url"] = uri
    if seed is not None:
        payload["seed"] = seed
    return payload


def redact(payload):
    """Payload with the base64 images replaced by a size note."""
    out = dict(payload)
    for k in ("image_url", "end_image_url"):
        if k in out:
            out[k] = "<data-uri %.0f KB>" % (len(out[k]) / 1024)
    return out


def cost(tier, resolution):
    return PRICE[tier][resolution]


def generate(image, prompt, out_mp4, tier="turbo", resolution="480p",
             seed=None, loop=True, key=None, log=print):
    """Submit, poll to completion, download the mp4. Returns the fal
    response dict augmented with `queue_seconds`."""
    key = key or os.environ.get("FAL_KEY")
    if not key:
        raise SystemExit("FAL_KEY not set (set -a; source ~/.env; set +a)")
    payload = build_payload(image, prompt, resolution, seed, loop)
    sub = api("%s/%s" % (QUEUE, ENDPOINTS[tier]), payload, key)
    log("  queued: %s" % sub.get("request_id"))
    t0 = time.time()
    while True:
        time.sleep(POLL_SECONDS)
        st = api(sub["status_url"], key=key)
        if st["status"] == "COMPLETED":
            break
        if st["status"] not in ("IN_QUEUE", "IN_PROGRESS"):
            raise SystemExit("fal FAILED: %s" % json.dumps(st)[:500])
        if time.time() - t0 > POLL_TIMEOUT:
            raise SystemExit("fal timed out after %ds (request %s)"
                             % (POLL_TIMEOUT, sub.get("request_id")))
        log("  %s %.0fs" % (st["status"], time.time() - t0))
    res = api(sub["response_url"], key=key)
    res["queue_seconds"] = round(time.time() - t0, 2)
    res["request_id"] = sub.get("request_id")
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(res["video"]["url"], out_mp4)
    return res


def probe_fps(mp4):
    """Measured, never assumed.

    Hardcoding this is the 2026-07-30 half-speed bug: the packs claimed
    fr:16 while the clips were 32fps, and every pacing verdict in three
    waves was taken at 0.5x before Lewis caught it by eye. There is no
    default value for this function to fall back to — it returns None and
    the caller must refuse to pack.
    """
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


def extract_frames(mp4, fdir):
    fdir = Path(fdir)
    fdir.mkdir(parents=True, exist_ok=True)
    for stale in fdir.glob("frame_*.png"):
        stale.unlink()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                    str(fdir / "frame_%04d.png")], check=True)
    frames = sorted(fdir.glob("frame_*.png"))
    if not frames:
        sys.exit("ffmpeg extracted no frames from %s" % mp4)
    return frames
