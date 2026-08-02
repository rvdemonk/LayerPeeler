"""The judge contract (design §3.1) and output schema (§3.4).

The judge answers Lewis's two-part question — what changes, and what must not
change — and a defect is a change that landed in the must-not-change set. The
contract halves are emitted FIRST so the verdict is conditioned on them rather
than rationalised after one.

§3.5 is load-bearing and enforced by omission: the judge is never shown gate
numbers, drift scores, or prior verdicts. A judge shown `identity_drift 0.070`
launders the existing instrument instead of adding an independent signal, and
the bench would measure an instrument against its own echo.
"""

# Constrained to the project's EXISTING defect vocabulary (ledger :339-342 plus
# the animation classes) so judge output is directly comparable with Lewis's
# verdicts. Free-text classes are rejected by the schema; a judge that wants to
# name something new says so in `evidence`.
DEFECT_CLASSES = [
    "species-swap",
    "appendage-invention",
    "palette-injection",
    "feature-scale-drift",
    "lifelessness",
    "photoreal-style-escape",
    "identity-decay",
    "mouth-deformity",
    "luminance-drift",
    "posterization",
]

GATE_TIERS = ["strict_spatial", "identity_invariants_only"]

SYSTEM = """\
You are a screening flag in a quality-control pipeline for animated mascot \
emotes, not the final arbiter. A human reviews every clip you flag; your job is \
to change the ORDER of that review and hand the reviewer a checkable evidence \
string, not to decide what ships.

You are shown frames sampled across one looping animation of a single character. \
The first image is the ANCHOR (frame 0) and defines the character's identity. \
Every later image is a candidate drift from it.

Answer two questions, in this order:

1. WHAT MUST NOT CHANGE — the character's identity invariants, read off the \
anchor: palette, proportions, face topology, marking geometry, silhouette class, \
rendering style.
2. WHAT CHANGES — the motion the animation is performing: pose, limb position, \
expression, squash and stretch.

A DEFECT is a change that landed in the must-not-change set. Legitimate \
animation is not a defect, however large: a raised arm, a turned head, a wide \
mouth mid-yawn, a squashing jump are motion. A defect is the character being \
DRAWN DIFFERENTLY — a marking that splits, fades or changes shape; a palette \
that shifts; a face that rounds or enlarges; a rendering style that flattens or \
gains texture; a feature that appears or disappears without the motion \
accounting for it.

Every defect you report must name the frame index you saw it in, taken from the \
label on the image. A defect claim without a frame is not checkable.

Report what you can see. Do not speculate about frames you were not shown.\
"""

USER_PREFIX = """\
Below are 7 frames from one looping animation of a single character.

Image 1 is the ANCHOR: frame 0. It defines the identity.
Images 2-7 are probe frames sampled across the rest of the cycle. Each is \
labelled with its frame index.

Judge whether this is still the same character, drawn the same way, by the late \
frames — or whether the character has drifted off-model across the clip.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "must_not_change": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The character's identity invariants, read off the anchor "
                "frame. Concrete and visual, e.g. 'one continuous dark "
                "eye-mask across both eyes'."
            ),
        },
        "what_changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The motion the animation performs across the clip, e.g. "
                "'one paw raises beside the head'."
            ),
        },
        "identity_held": {
            "type": "boolean",
            "description": (
                "true if the character is still the same character, drawn the "
                "same way, in the late frames as in the anchor. false if a "
                "change landed in the must_not_change set."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 confidence in identity_held.",
        },
        "defects": {
            "type": "array",
            "description": (
                "Empty when identity_held is true. One entry per distinct "
                "defect otherwise."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string", "enum": DEFECT_CLASSES},
                    "frame": {
                        "type": "integer",
                        "description": (
                            "Frame index where the defect is visible, taken "
                            "from the image label."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "What you see, contrasted against the anchor. "
                            "e.g. 'the dark eye-mask has separated into two "
                            "isolated patches; frame 0 shows one continuous "
                            "mask'."
                        ),
                    },
                },
                "required": ["class", "frame", "evidence"],
                "additionalProperties": False,
            },
        },
        "gate_tier": {
            "type": "string",
            "enum": GATE_TIERS,
            "description": (
                "strict_spatial when fine detail must hold exactly; "
                "identity_invariants_only when pose change is expected and "
                "only the invariants matter."
            ),
        },
    },
    "required": [
        "must_not_change", "what_changes", "identity_held",
        "confidence", "defects", "gate_tier",
    ],
    "additionalProperties": False,
}


def image_caption(slot, frame):
    if slot == 0:
        return "Image 1 - ANCHOR, frame %d:" % frame
    return "Image %d - probe, frame %d:" % (slot + 1, frame)
