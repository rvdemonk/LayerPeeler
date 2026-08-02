# pipeline/ — the blessed path, as one command

Consolidates the spike-2 harness into a single operable run:

```
mascot PNG + prompt
  -> Wan 2.2 A14B turbo i2v on fal, 480p, looped by construction   [generate]
  -> flat-background matte + Lab colour norm                        [matte]
  -> basic-integrity + colour/velocity gates                        [gates]
  -> 512px WebP q65 @ 24fps frame-seq Lottie, gzip + .lottie        [encode]
  -> posterization pair + temporal-shimmer gates on the assets      [gates]
  -> GIF + contact sheet, previewable in sandbox/                   [eyeball]
```

Every stage is wall-clock timed; `timings.json` is a first-class output.

## Run it

```bash
cd LayerPeeler
set -a; source ~/.env; set +a          # FAL_KEY, live generation only

# the whole thing
.venv-hybrid/bin/python -m pipeline.run \
    --name strawberry-wave \
    --image ../corpus/mascots/strawberry.jpg \
    --prompt "A cute flat-vector strawberry mascot performs one single clear
              friendly wave... The character stays anchored in place at the
              centre; the flat uniform background remains completely static.
              Smooth 2D flat-vector cartoon style, no camera movement,
              seamlessly loopable animation."

# show the request, spend nothing
... --dry-run

# everything below generation, on a clip already on disk
.venv-hybrid/bin/python -m pipeline.run --name repro \
    --from-video out/spike2/raccoon-wave/oracle.mp4

# appraise
python3 sandbox/serve.py --root out/pipeline
```

`--from-video` is the workhorse flag. **Verify local changes with it, not
with fal credits** — matting, gating and encoding are deterministic and
byte-reproducible from an mp4, so there is no reason to buy a new
generation to test our own arithmetic.

## What one run leaves on disk

```
out/pipeline/<name>/
  oracle.mp4            the generation (or the adopted clip)
  response.json         fal's response + the prompt (live runs only)
  frames/               ffmpeg RGB frames
  rgba/                 matted frames — the lossless reference for the gates
  ladder/<name>.<profile>.json / .lottie
  gates.json            flat scalars for the sandbox + nested detail
  gates.png             colour + velocity strips
  <name>.gif, <name>_sheet.png    the eyeball artifacts
  timings.json          per-stage wall clock
  run.json              everything above, in one record
```

## Profiles

| profile | rung | what it is |
|---|---|---|
| `ship` (default) | 512webp-q65-24 | the shipping shape |
| `full-rate` | 512webp-q65 | no frame drop, for motion too fast to decimate |
| `squeeze` | 512webp-q50-24 | per-clip squeeze, gate-approvable |
| `lossless` | 512 (PNG) | ~20MB reference, never shipped |

Dials that were tried and killed — octree, pngquant, 16fps, any downscale
below native — are **not selectable here**. They live in
`hybrid/spike2_repack.py`, which remains the size-ladder laboratory for
measuring rungs we do not ship. A rung that can be selected is a rung
that eventually ships by accident.

## Gates

`fail` means physically broken and stops the run before encoding (a
known-broken clip must be rerolled, not shipped as if fitted). `flag` is
a screening signal, deliberately stricter than Lewis's eye on flat art —
a flag is not a verdict.

| gate | catches | pass/flag basis |
|---|---|---|
| basic-integrity | empty frames, silhouette collapse, character leaving frame, frozen frames, broken loop | thresholds set clear of the whole 31-run corpus |
| colour + velocity strips | luminance drift, loop-seam colour cliff, fidget, dead face under body motion | corpus distribution |
| posterization PAIR | palette crush **and** colour error, worst of 8 sampled frames | matched to Lewis's octree verdict |
| temporal shimmer PAIR | flat-region crawl the stills cannot see: ratio **and** absolute excess vs the lossless matte | responds monotonically to webp quality; fires at q10, silent at q50/q65 (both of which Lewis passed) |

Both of the encoded-asset gates are PAIRS, and for the same reason: one
number alone misreads. Colour-count ratio cannot tell dithered from
posterized (pngquant crushes the palette as hard as octree at a quarter
of the error, and only octree looks wrong). Shimmer's ratio is
denominator-fragile — on a genuinely quiet clip the lossless baseline is
under one 8-bit level, so a sub-level addition reads as 1.3x.

Gates do **not** pick a profile. The velocity strip once auto-selected
frame rate per clip and Lewis rejected clips the gate had cleared
("looks like a flipbook") — he was reporting fluidity where the
instrument measured judder. The fluidity floor is a taste constant.

## Timing

`timings.json` per run: every stage, plus `total_seconds`,
`generation_seconds` (bought, cannot be engineered away) and
`local_seconds` (CPU we own and can parallelise). That split is what SLA
design and unit economics consume.

Measured on this machine (M3 Air, 161 source frames at 512px, one
profile, `--from-video` so generation is excluded):

| stage | sec |
|---|---|
| extract | 0.4 |
| matte | 6.3 |
| gate_integrity | 0.5 |
| gate_strips | 4.1 |
| encode | 1.8 |
| gate_posterization | 0.5 |
| gate_shimmer | 0.1 |
| eyeball (GIF + sheet) | 2.9 |
| **local total** | **~16.6** |

Generation across the ledger's 31 rows: min 19s, median 25s, mean 36s,
p90 59s, max 81s (turbo, 480p). So a clip is roughly **35s-100s wall
clock at $0.05**, generation dominating and highly variable (queue
depth, not our code), while the local half is steady and parallelisable
across clips.
Native-960 sources (720p generations) roughly triple the local half
(~45s) — matting and encoding scale with source pixels, not output
pixels.

## Packs — one mascot, N emotes, measured

`pipeline.pack` is the batch driver: it runs the base stage (new poses and
expressions via gemini-pro i2i), then the single-clip pipeline above once
per emote, and writes `pack.json` — the manifest the business model is
priced from.

```bash
cd LayerPeeler
set -a; source ~/.env; set +a                    # FAL_KEY

# rehearse the whole driver on clips already on disk. Spends nothing.
# (--from-video-dir adopts <dir>/<emote>/oracle.mp4, or
#  <dir>/<mascot>-<emote>/oracle.mp4, and skips the base stage)
.venv-hybrid/bin/python -m pipeline.pack packs/raccoon-emotes.json \
    --only wave,celebrate,sleepy --from-video-dir out/spike2 --accept-all

# see every request and its price, spend nothing
.venv-hybrid/bin/python -m pipeline.pack packs/raccoon-emotes.json --dry-run

# the real thing (asks you to accept/reroll each base and each clip)
.venv-hybrid/bin/python -m pipeline.pack packs/raccoon-emotes.json

# appraise the pack
python3 sandbox/serve.py --root out/packs/raccoon-emotes/clips
```

Start every new pack with `--dry-run`. It prints the exact gemini-pro
command, the exact fal payload, and the budget floor, and buys nothing.

### The spec

Copy `packs/raccoon-emotes.json` and edit it. Paths in it are relative to
the spec file. Required: `pack` (output dir name), `mascot` (the image),
`character` (a noun phrase — it is dropped into the i2i prompt), and
`emotes`. Per emote:

| field | meaning |
|---|---|
| `name` | lowercase-hyphen; becomes the run dir and the `.lottie` filename |
| `base` | `"original"` to animate the mascot as-is, or an edit instruction (`"raises both arms overhead"`), which gets wrapped in the identity-preserving prompt that held on the 2026-08-01 probe |
| `base_prompt` | optional: use this i2i prompt verbatim instead of the wrapped one |
| `prompt` | the animation prompt — beatsheet grammar: subject, invariants (`keeps its exact face…`), the eye rule, numbered beats that return to the exact resting pose, then the stage clause |

Pack size is **not** baked in. The spec's emote list is the pack; `--n K`
takes the first K; `--only wave,sleepy` runs named ones. The business
model's placeholder pack is 8, which is a pricing decision, not a code
one.

### Reviewing (this is the product)

Each base and each clip stops for you:

```
[base 4/8] thumbs-up   draw 1 ($0.15) via gemini-pro
    image: out/packs/…/bases/thumbs-up.draw1.png
    [a]ccept  [r]eroll  [s]kip emote  [q]uit pack >
```

The image (or, for clips, the GIF) opens in Preview — `--no-open` stops
that. `r` redraws with the same prompt and counts a reject; `s` abandons
that emote and keeps the pack going; `q` stops now and writes the
manifest for what you have. **Rerun the same command with `--resume` to
pick up where you quit** — accepted bases and clips are reused, not
redrawn.

A clip whose gates FAIL is rerolled automatically without asking you (up
to 3 attempts): a known-broken clip is rerolled, not shipped. A `FLAG` is
not a verdict — you still see it and decide. Rejected clip runs are moved
to `out/packs/<pack>/rejects/` rather than deleted; they are the corpus
the next gate gets calibrated on.

`--accept-all` takes every base and every gate-passing clip with no
prompts. Use it for rehearsals and unattended runs, not for a pack you
intend to sell.

### What a pack leaves on disk

```
out/packs/<pack>/
  pack.json               the manifest (below)
  bases/<emote>.drawK.png every i2i draw, accepted and not
  clips/<emote>/          exactly one single-clip run dir (see above)
  clips/ladder_report.json  so sandbox/serve.py --root clips/ works
  rejects/<emote>.rejK/   rejected clip runs, kept
```

### Reading pack.json

The four numbers `docs/business-model.md` needs are under
`measurements`, and nothing else on disk can reconstruct them:

| field | what it is |
|---|---|
| `r_base` | gemini-pro draws per accepted base — the i2i reroll rate |
| `r_anim` | clip draws per accepted clip — the animation reroll rate |
| `human_minutes` | your minutes. You are asked at the end; `--minutes 45` states it up front. `human_minutes_at_prompts` is the clock actually run while you were deciding — a floor, not the whole of it |
| `machine_wall_clock_seconds` | machine time only; the seconds you spent at prompts are subtracted, so this is the SLA number, not a stopwatch on the session |

Also there: `machine_cost_usd` (real prices, gemini-pro's own reported
cost when it reports one), `generation_seconds` vs `local_seconds` summed
over **every** attempt including rerolls, and per-emote gate verdicts and
flags. `r_base`/`r_anim` are `null`, never 1.0, when nothing was
accepted — an undefined reroll rate must not read as a perfect one.

### When it stops

Every failure message names the file to edit and the flag to rerun with.
Two worth knowing in advance:

- *"gave up after 6 draws"* — the edit instruction is the suspect, not
  the model. Rewrite that emote's `base` and rerun with `--resume`.
- *"3 attempts all failed their gates"* — the animation prompt is the
  suspect. Soften or shorten the motion and rerun with `--resume`.

## Superseded

This path supersedes, for shipping purposes:
`hybrid/spike2_oracle.py` (generate + matte + pack + ledger),
`hybrid/spike2_gates.py` (colour/velocity strips), and the repack half of
`hybrid/spike2_repack.py`. Those files stay: the ladder is still the
laboratory, and the wave scripts still drive corpus generation through
the oracle.
