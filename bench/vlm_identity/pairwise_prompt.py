"""Pairwise identity judge — two frames, one question.

The design trades 7-image context for 2-image precision: instead of showing
all frames at once and asking for a complex defect-class ruling, we show
frame_0 (anchor) + one comparison frame and ask a simple yes/no with evidence.
Any pair that says "no" flags the clip. This:
- Avoids the thinking-budget problem (simpler question per call)
- Gives frame-level precision (when does drift first appear?)
- Is trivially cheaper (2 images vs 7 per call)
- Aggregates cleanly across pairs
"""

SYSTEM = """\
You are a screening flag in a quality-control pipeline for animated mascot \
emotes. A human reviews every clip you flag; your job is to surface clips \
where the character's identity has changed, not to decide what ships.

You are shown TWO frames from the same looping animation. Frame 0 is the \
ANCHOR — it defines the character's identity. The second frame is from later \
in the animation. Answer one question: is the second frame still the SAME \
character as the anchor?

A character has changed identity if any of these hold:
- Face shape, eye shape, or eye spacing has changed
- Markings (stripes, patches, masks, spots) have moved, changed shape, \
  or changed color
- The character has acquired new features not present in the anchor \
  (extra appendages, whiskers, teeth, horns)
- The overall colour palette has shifted to a different scheme
- The design style has shifted (e.g. flat-vector to painterly, or \
  cartoon to photorealistic)

Do NOT flag changes that are expected in animation:
- Pose change (head turn, body rotation, limb movement)
- Expression change (smile, blink, open mouth)
- Slight lighting or compression artefacts

Frame 0 and the comparison frame are shown at the same crop size. \
Answer TRUTHFULLY — a false PASS is worse than a false FLAG.
"""

USER_PREFIX = "Compare these two frames and judge: is this the same character?"

SCHEMA = {
    "type": "object",
    "properties": {
        "identity_held": {
            "type": "boolean",
            "description": "true if the same character, false if identity has changed",
        },
        "evidence": {
            "type": "string",
            "description": "One sentence describing what changed, or 'no change detected'",
        },
    },
    "required": ["identity_held", "evidence"],
    "additionalProperties": False,
}


def image_caption(slot, frame_num):
    if slot == 0:
        return "Frame 0 (anchor — this IS the character):"
    return "Frame %d (comparison):" % frame_num
