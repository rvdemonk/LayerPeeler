"""Stage 4 — RGBA frames -> frame-sequence Lottie (+ gzip, + .lottie).

The carrier is a frame-seq Lottie: one embedded image asset and one image
layer per frame, each layer live for exactly one frame. It is not vector
and does not pretend to be — see the positioning note in the ledger
(2026-07-31): a hand-made vector Lottie at this complexity is 50-300KB
and no encoder closes that gap. What this carrier buys is a style palette
vector cannot reach, delivered in an afternoon, playable by the Lottie
players customers already ship.

PROFILES is the whole quality/size decision surface, and it is short on
purpose. Every dial that was tried and killed is recorded in DEAD below
rather than left generatable, because a rung that can still be selected
is a rung that will eventually be shipped by accident.
"""

import base64
import gzip
import json
import zipfile
from pathlib import Path

from PIL import Image

# libwebp's effort dial. Measured on this corpus at 448px, 20 frames:
# method=6 took 59.06s for 16.5 KB/frame, method=4 took 0.37s for
# 16.8 KB/frame — 160x the CPU to save 1.8% of the bytes, which prices a
# 31-run roll at ~13 hours instead of ~30 minutes. Never 6.
WEBP_METHOD = 4

# No profile may resample above this. It is a ceiling, not a target:
# `edge_for()` takes min(native, MAX_EDGE) and never upscales, because
# downscaling below native BACKFIRES on flat-vector art — Lanczos
# manufactures colour (raccoon-wave: 5571 unique at native 512, 9841 at
# 448) and WebP pays more to encode the manufactured noise than it saves
# in pixels. 448webp and 384webp both came out LARGER than 512webp.
MAX_EDGE = 512


class Profile:
    """One shipping configuration.

    keep=(num, den): keep the first `num` of every `den` source frames and
    scale `fr` by the same fraction, so wall-clock duration is unchanged.
    Expressed as a fraction rather than a stride because that is what makes
    24fps (3 of every 4) sayable at all.
    """

    def __init__(self, name, quality, keep, note):
        self.name = name
        self.quality = quality
        self.keep = keep
        self.note = note


PROFILES = {
    # THE SHIPPING SHAPE (Lewis, 2026-07-31, held tentatively):
    # strawberry-class ~1.33MB gz, star-class ~0.77MB.
    "ship": Profile("512webp-q65-24", 65, (3, 4),
                    "default: q65 at 24fps"),
    # Full rate. 24fps passed Lewis's eye at product scale but is a young
    # verdict; this is the fallback when a clip's velocity strip says the
    # motion is fast enough that dropping any frame is a risk.
    "full-rate": Profile("512webp-q65", 65, (1, 1),
                         "no frame drop"),
    # The per-clip squeeze. Lewis on q50 vs q65: "struggling to notice the
    # difference" — both passed. q65 is default only for style-margin on
    # unseen mascots (quantization damage is style-dependent), so q50 is
    # available but must be approved per clip by the posterization gate.
    "squeeze": Profile("512webp-q50-24", 50, (3, 4),
                       "gate-approvable per-clip squeeze"),
    # Lossless control. Not shippable (~20MB) and not meant to be — it is
    # the reference the posterization and shimmer gates score against.
    "lossless": Profile("512", None, (1, 1),
                        "reference for gates, never shipped"),
}
SHIP_PROFILE = "ship"

# Killed dials, kept as text so they are not rediscovered:
#   octree (PIL FASTOCTREE, 256c, no dither) — posterization; Lewis:
#       "blocky looking pixelation... like a compression artifact in
#       bootlegged dvds". Confirmed independently on iOS.
#   pngquant (256c, Floyd-Steinberg) — best colour accuracy (dE 0.94) but
#       2.8x webp's bytes; dither noise defeats PNG filters AND gzip.
#   16fps (-half rungs) — "seriously effects the premium feel. looks like
#       a flipbook". The velocity gate cleared it; the eye did not. That
#       was a fluidity verdict, and fluidity is a taste constant.
#   448/384/256 px — see MAX_EDGE.
DEAD = ("octree", "pngquant", "16fps/-half", "sub-native downscale")


def edge_for(native_edge, max_edge=MAX_EDGE):
    return min(native_edge, max_edge)


def kept_indices(n, keep):
    num, den = keep
    return [i for i in range(n) if i % den < num]


def scaled_fr(fps, keep):
    """fr scaled by the keep fraction so wall-clock duration is unchanged.

    Kept exact: 32 * 3/4 is 24, but 30 * 3/4 is 22.5 and must NOT be
    rounded — a rounded fr silently retimes the clip, the same class of
    bug as the fr:16 half-speed one.
    """
    num, den = keep
    fr = fps * num / den
    return int(fr) if float(fr).is_integer() else fr


def build_frames(src_frames, edge, quality, keep, dst):
    """Materialise one profile's frames into `dst`. Returns the paths.

    quality=None means lossless PNG (the gate reference); anything else is
    lossy WebP, which is the codec on all platforms — verified decoding in
    lottie-web by opaque-pixel census and on lottie-ios 4.6.1 by the same
    census plus frame-locked SSIM.
    """
    dst.mkdir(parents=True, exist_ok=True)
    outs = []
    ext = "png" if quality is None else "webp"
    for j, i in enumerate(kept_indices(len(src_frames), keep)):
        im = Image.open(src_frames[i]).convert("RGBA")
        if im.size != (edge, edge):
            im = im.resize((edge, edge), Image.LANCZOS)
        op = dst / ("frame_%04d.%s" % (j, ext))
        if quality is None:
            im.save(op, optimize=True)
        else:
            im.save(op, "WEBP", quality=quality, method=WEBP_METHOD)
        outs.append(op)
    return outs


def pack_lottie(frames, size, fr, path, mime="webp"):
    """Frame-seq Lottie: one embedded asset + one image layer per frame.

    `mime` only changes the data-URI prefix — the asset/layer structure is
    identical for PNG and WebP, because the player hands the URI to an
    <img> and never inspects the bytes. That is the entire reason the
    codec swap was possible without touching any player.
    """
    assets, layers = [], []
    for i, fp in enumerate(frames):
        b64 = base64.b64encode(Path(fp).read_bytes()).decode()
        assets.append({"id": "f%d" % i, "w": size, "h": size,
                       "u": "", "p": "data:image/%s;base64,%s" % (mime, b64),
                       "e": 1})
        layers.append({"ddd": 0, "ind": i + 1, "ty": 2, "nm": "f%d" % i,
                       "refId": "f%d" % i, "ip": i, "op": i + 1, "st": 0,
                       "ks": {"o": {"a": 0, "k": 100},
                              "p": {"a": 0, "k": [size / 2, size / 2, 0]},
                              "a": {"a": 0, "k": [size / 2, size / 2, 0]},
                              "s": {"a": 0, "k": [100, 100, 100]}}})
    lot = {"v": "5.7.0", "fr": fr, "ip": 0, "op": len(frames),
           "w": size, "h": size, "nm": Path(path).stem, "ddd": 0,
           "assets": assets, "layers": layers}
    Path(path).write_text(json.dumps(lot, separators=(",", ":")))
    return Path(path).stat().st_size


def write_dotlottie(json_path, out_path, name):
    """Minimal dotLottie container: manifest + animations/<id>.json.

    Measured equal to gzip within 0.5KB — the container is a distribution
    convenience, not a compression win. Both are emitted so the wire size
    and the shipped file are never confused for each other.
    """
    manifest = {
        "version": "1.0", "revision": 1,
        "author": "image-to-lottie", "generator": "pipeline.encode",
        "animations": [{"id": name, "direction": 1, "speed": 1,
                        "playMode": "loop", "loop": True, "autoplay": True}],
    }
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        z.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        z.write(json_path, "animations/%s.json" % name)
    return Path(out_path).stat().st_size


def gzip_size(path):
    """Bytes actually sent over the wire. This is the number that matters:
    a server serves the JSON gzipped, so the plain JSON size is a fiction
    and the .lottie size is a distribution choice."""
    return len(gzip.compress(Path(path).read_bytes(), 9))


def encode(rgba_frames, native_edge, fps, out_dir, name, profile,
           scratch, keep_assets=False):
    """Full encode for one profile. Returns a record dict.

    Variant frames go to a scratch dir and are deleted after packing
    unless `keep_assets` — the gates need them, a shipping run does not,
    and keeping them costs hundreds of MB per run.
    """
    p = PROFILES[profile] if isinstance(profile, str) else profile
    edge = edge_for(native_edge)
    fr = scaled_fr(fps, p.keep)
    mime = "png" if p.quality is None else "webp"
    adir = Path(scratch) / ("assets_%s" % p.name)
    frames = build_frames(rgba_frames, edge, p.quality, p.keep, adir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / ("%s.%s.json" % (name, p.name))
    jsize = pack_lottie(frames, edge, fr, jpath, mime)
    lpath = out_dir / ("%s.%s.lottie" % (name, p.name))
    lsize = write_dotlottie(jpath, lpath, "%s-%s" % (name, p.name))
    rec = {
        "profile": p.name, "note": p.note, "codec": mime,
        "quality": p.quality, "keep": list(p.keep),
        "edge": edge, "native_edge": native_edge, "fr": fr,
        "frames": len(frames), "source_frames": len(rgba_frames),
        "json_bytes": jsize, "gz_bytes": gzip_size(jpath),
        "lottie_bytes": lsize,
        "json_path": str(jpath), "lottie_path": str(lpath),
        "assets_dir": str(adir) if keep_assets else None,
    }
    if not keep_assets:
        for f in frames:
            f.unlink()
        adir.rmdir()
    return rec, (frames if keep_assets else [])
