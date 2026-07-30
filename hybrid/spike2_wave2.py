"""Wave 2 — beatsheet-driven prompting. WORKING VOCABULARY, not doctrine
(Lewis, 2026-07-30: grammar gets enshrined only at a watershed, obiter
dicta until then).

A BEATSHEET decomposes a motion the way an animation director briefs:
  layers : body | face | secondary (tail, leaf crown, ears...)
  beats  : anticipation -> action -> follow-through -> settle, on a
           rough timeline
  anchors: identity constraints stated positively ("keeps its exact
           simple face: two oval eyes and a small smile") — born from
           wave-1's blob, whose minimal face gave Wan nothing to hold.

compile() turns a beatsheet into flowing prose for Wan (the model reads
direction better as a performance script than as a list). Every prompt
ends with the same STAGE pin (anchored, static background, loopable).

Wave-1 lessons encoded here:
  - face gets explicit work during body action (raccoon-jig froze its
    face; Lewis caught it; face_body_ratio gate now watches)
  - eye beats described as lid/gaze acting, never "eyes look" alone
    (2/3 wave-1 'look' clips glitched the eyeballs)
  - jig seams: final beat returns to rest BEFORE the clip ends
"""

import subprocess
import sys
from pathlib import Path

HYBRID = Path(__file__).resolve().parent
PY = str(HYBRID.parent / ".venv-hybrid" / "bin" / "python")
M = str(HYBRID.parent.parent / "corpus" / "mascots")

STAGE = ("The character stays anchored in place at the center; the flat "
         "uniform background remains completely static. Smooth 2D "
         "flat-vector cartoon style, no camera movement, seamlessly "
         "loopable animation.")


def compile_bs(subject, anchors, beats):
    """Beatsheet -> performance-script prose."""
    parts = [subject.strip()]
    if anchors:
        parts.append("Throughout, it " + anchors.strip())
    parts.append(" ".join(b.strip() for b in beats))
    parts.append(STAGE)
    return " ".join(parts)


RUNS = [
    ("strawberry-idle-bs", f"{M}/strawberry.jpg", compile_bs(
        "A cute flat-vector strawberry mascot performs a living idle loop.",
        "keeps its exact face — big oval eyes with dark pupils, small "
        "smile, rosy cheeks — crisp and unchanged in style.",
        ["First it breathes slowly, body softly rising and settling.",
         "On the second breath its leaf crown gives one gentle bob and "
         "its gaze drifts contentedly to one side.",
         "Then it blinks once, slowly, and its smile widens a touch.",
         "Finally it settles back to its exact resting pose."])),
    ("strawberry-wave-bs", f"{M}/strawberry.jpg", compile_bs(
        "A cute flat-vector strawberry mascot greets the viewer.",
        "keeps its exact face style — big oval eyes, small smile, rosy "
        "cheeks — crisp and on-model.",
        ["First it perks up slightly, eyes brightening, as if noticing "
         "someone.",
         "Then it raises its right stub arm and gives two relaxed, "
         "friendly waves while its smile widens warmly.",
         "Its leaf crown bobs once with the motion.",
         "Finally it lowers the arm and settles back to its exact "
         "resting pose with a soft blink."])),
    ("strawberry-jig-bs", f"{M}/strawberry.jpg", compile_bs(
        "A cute flat-vector strawberry mascot does a joyful little jig.",
        "keeps its exact face style crisp and on-model.",
        ["First it crouches slightly in anticipation, grin spreading.",
         "Then it springs into two rhythmic bounces: body squashing on "
         "each landing and stretching at each peak, leaf crown bouncing "
         "a half-beat behind, eyes squeezing shut with joy on the second "
         "bounce while the mouth opens in a laugh.",
         "Its stub arms swing loosely with the rhythm.",
         "Finally it lands softly, straightens, and settles back to its "
         "exact resting pose with a happy sigh."])),
    ("raccoon-jig-bs", f"{M}/raccoon.jpg", compile_bs(
        "A flat-vector cartoon raccoon mascot does a joyful little dance.",
        "keeps its exact face — dark eye-mask, big round white eyes with "
        "black pupils, small black nose — crisp and on-model.",
        ["First it crouches a little in anticipation, grin spreading "
         "wide.",
         "Then it bounces twice in rhythm: knees springy, striped tail "
         "whipping a half-beat behind each bounce, and its FACE dances "
         "too — the grin opens into a laugh on the first bounce, the "
         "eyes squeeze shut with delight on the second.",
         "Its arms swing loosely with the beat.",
         "Finally it lands, straightens up, tail settling, and returns "
         "to its exact resting pose with a contented blink."])),
    ("raccoon-wave-bs", f"{M}/raccoon.jpg", compile_bs(
        "A flat-vector cartoon raccoon mascot greets the viewer.",
        "keeps its exact face — dark eye-mask, round white eyes, small "
        "black nose — crisp and on-model.",
        ["First its ears perk and its eyes brighten, noticing someone.",
         "Then its raised paw gives two warm, unhurried waves while the "
         "smile widens and the striped tail sways gently with the "
         "motion.",
         "Finally the paw comes to rest, the tail stills, and it "
         "settles back to its exact starting pose with one soft blink."])),
    ("raccoon-look-bs", f"{M}/raccoon.jpg", compile_bs(
        "A flat-vector cartoon raccoon mascot glances around curiously.",
        "keeps its eyes exactly as drawn — round white eyes with solid "
        "black pupils and a fixed catchlight — never changing their "
        "rendering style.",
        ["First its head turns slowly a little to the left, ears "
         "tilting, as its gaze follows something.",
         "It pauses, curious, tail giving one slow sway.",
         "Then the head turns gently to the right, eyebrows lifting "
         "with interest.",
         "Finally it returns to face forward and settles back to its "
         "exact resting pose with one calm blink."])),
    ("blob-breathe-bs", f"{M}/blob.jpg", compile_bs(
        "A minimal flat-vector blue blob mascot performs a calm living "
        "idle loop.",
        "keeps its exact simple face — two small white oval eyes and a "
        "tiny curved smile, nothing more — and its exact rounded "
        "teardrop silhouette with two small feet visible.",
        ["First it breathes slowly: the soft body gently inflates and "
         "settles, squashing a whisker on the exhale.",
         "On the second breath it blinks once, slowly.",
         "Finally it settles back to its exact resting shape."])),
    ("blob-jig-bs", f"{M}/blob.jpg", compile_bs(
        "A minimal flat-vector blue blob mascot does a happy bouncy jig.",
        "keeps its exact simple face — two small white oval eyes and a "
        "tiny curved smile, always visible, never changing style — and "
        "returns to its exact teardrop silhouette between bounces.",
        ["First it squashes down in anticipation.",
         "Then it springs into two rhythmic bounces with big squash and "
         "stretch, tilting playfully, its smile widening with joy.",
         "Finally it lands softly and settles back to its exact resting "
         "shape with one blink."])),
]


def run(name, image, prompt, tier="turbo", seed=None):
    cmd = [PY, str(HYBRID / "spike2_oracle.py"), "--image", image,
           "--name", name, "--resolution", "480p", "--tier", tier,
           "--prompt", prompt]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True)
    subprocess.run([PY, str(HYBRID / "spike2_gates.py"),
                    str(HYBRID.parent / "out" / "spike2" / name)],
                   check=True)


def main():
    for name, image, prompt in RUNS:
        run(name, image, prompt)
    # seed exploration: same beatsheet, 2 more rolls -> gates pick best
    jig = next(r for r in RUNS if r[0] == "raccoon-jig-bs")
    run("raccoon-jig-bs-s2", jig[1], jig[2], seed=1234)
    run("raccoon-jig-bs-s3", jig[1], jig[2], seed=987654)
    # tier comparison: the full (non-distilled) A14B on the same brief
    run("raccoon-jig-full", jig[1], jig[2], tier="full")
    print("WAVE 2 DONE")


if __name__ == "__main__":
    main()
