"""Wave 3 — the BLIND test + pacing probes.

Enshrinement condition (agreed 2026-07-30): a beatsheet written cold for
a motion class we've never generated, first try, no retries, passing
Lewis's eyeball. Three unseen classes here (celebrate, sulk, sleepy) and
one fully-blind pairing (new star mascot x celebrate). These briefs are
committed BEFORE any generation and will not be edited after results.

Wave-2 verdicts encoded:
  - GRAVITY language in every action beat ("rises quickly, falls with
    real weight") — the stuck-in-sand fix, prompt side; retiming covers
    the post side.
  - EYES guideline in every brief (Lewis: slit-iris blinks are uncanny;
    r022 rick-and-morty pupils): lids sweep like soft curtains, eyes
    stay large and round, never slits.
  - Blob dropped from product corpus (Lewis: branding logic), kept out
    of this wave entirely.
"""

import subprocess
from pathlib import Path

HYBRID = Path(__file__).resolve().parent
PY = str(HYBRID.parent / ".venv-hybrid" / "bin" / "python")
M = str(HYBRID.parent.parent / "corpus" / "mascots")

STAGE = ("The character stays anchored in place at the center; the flat "
         "uniform background remains completely static. Smooth 2D "
         "flat-vector cartoon style, no camera movement, seamlessly "
         "loopable animation.")
EYES = ("Its eyes stay large and round with solid dark pupils; when it "
        "blinks, the lids sweep closed and open quickly and smoothly "
        "like soft curtains — the eyes never narrow into slits.")


def bs(subject, anchors, beats):
    return " ".join([subject.strip(), "Throughout, it " + anchors.strip(),
                     EYES, " ".join(b.strip() for b in beats), STAGE])


CELEBRATE_BEATS = [
    "First it crouches down in anticipation, fists tight, grin building.",
    "Then it leaps upward — rising quickly and falling back down with "
    "real weight — punching both arms into the air at the peak while "
    "its face bursts into open-mouthed joy.",
    "It lands with a springy squash, bounces once more, smaller,",
    "and finally straightens into a proud, beaming settle, back to its "
    "exact resting pose.",
]

RUNS = [
    ("strawberry-celebrate", f"{M}/strawberry.jpg", bs(
        "A cute flat-vector strawberry mascot celebrates a victory.",
        "keeps its exact face style — big oval eyes, rosy cheeks — and "
        "its leaf crown, crisp and on-model.", CELEBRATE_BEATS)),
    ("raccoon-celebrate", f"{M}/raccoon.jpg", bs(
        "A flat-vector cartoon raccoon mascot celebrates a victory.",
        "keeps its exact face — dark eye-mask, round white eyes, small "
        "black nose — and its striped tail, crisp and on-model.",
        CELEBRATE_BEATS)),
    ("star-celebrate", f"{M}/star.jpg", bs(
        "A cute flat-vector golden star mascot celebrates a victory.",
        "keeps its exact five-pointed star silhouette and its face — "
        "round eyes, peach cheeks, small smile — crisp and on-model.",
        CELEBRATE_BEATS)),
    ("strawberry-sulk", f"{M}/strawberry.jpg", bs(
        "A cute flat-vector strawberry mascot sulks, disappointed.",
        "keeps its exact face style crisp and on-model, with no tears.",
        ["First its smile fades and its shoulders slump as its gaze "
         "drops to the ground, leaf crown drooping a little.",
         "It gives one small sigh, body deflating slightly with it.",
         "Then it scuffs the ground once with a foot, slow and glum.",
         "Finally it glances up sheepishly and settles back to its "
         "exact resting pose."])),
    ("raccoon-sleepy", f"{M}/raccoon.jpg", bs(
        "A flat-vector cartoon raccoon mascot gets sleepy.",
        "keeps its exact face — dark eye-mask, round white eyes, small "
        "black nose — crisp and on-model.",
        ["First its eyelids grow heavy and its head begins to nod "
         "forward slowly.",
         "It catches itself with a tiny start, then yawns wide, "
         "covering the yawn politely with one paw, tail curling in.",
         "Its head dips once more, almost asleep,",
         "then it shakes awake briskly and settles back to its exact "
         "resting pose with bright reopened eyes."])),
    # pacing probe: wave-2 jig brief + explicit timing vocabulary (the
    # prompt-side gravity experiment; retiming is the post-side control)
    ("raccoon-jig-pace", f"{M}/raccoon.jpg", bs(
        "A flat-vector cartoon raccoon mascot does a joyful little dance "
        "with crisp, snappy comic timing — quick decisive movements, "
        "never slow motion, never floaty.",
        "keeps its exact face — dark eye-mask, big round white eyes, "
        "small black nose — crisp and on-model.",
        ["First it crouches briefly in anticipation, grin spreading.",
         "Then it snaps into two rhythmic bounces: each rise is quick "
         "and each fall lands with real weight and a springy squash, "
         "striped tail whipping a half-beat behind, the grin opening "
         "into a laugh on the first bounce and the eyes squeezing shut "
         "with delight on the second.",
         "Its arms swing loosely with the beat.",
         "Finally it lands crisply, straightens, and returns to its "
         "exact resting pose with a contented blink."])),
]


def run(name, image, prompt, resolution="480p"):
    subprocess.run([PY, str(HYBRID / "spike2_oracle.py"), "--image", image,
                    "--name", name, "--resolution", resolution,
                    "--prompt", prompt], check=True)
    subprocess.run([PY, str(HYBRID / "spike2_gates.py"),
                    str(HYBRID.parent / "out" / "spike2" / name)],
                   check=True)
    subprocess.run([PY, str(HYBRID / "spike2_retime.py"),
                    str(HYBRID.parent / "out" / "spike2" / name)],
                   check=True)


def main():
    for name, image, prompt in RUNS:
        run(name, image, prompt)
    # 720p face-fidelity probe: the eye/iris uncanny test at 2.25x pixels
    idle = ("A cute flat-vector strawberry mascot performs a living idle "
            "loop. Throughout, it keeps its exact face — big oval eyes "
            "with dark pupils, small smile, rosy cheeks — crisp and "
            f"unchanged in style. {EYES} First it breathes slowly, body "
            "softly rising and settling. On the second breath its leaf "
            "crown gives one gentle bob and its gaze drifts contentedly "
            "to one side. Then it blinks once — a quick, soft blink — "
            "and its smile widens a touch. Finally it settles back to "
            f"its exact resting pose. {STAGE}")
    run("strawberry-idle-720", f"{M}/strawberry.jpg", idle,
        resolution="720p")
    print("WAVE 3 DONE")


if __name__ == "__main__":
    main()
