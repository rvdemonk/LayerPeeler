"""The bench corpus: every clip carries a citable verdict or is a probe.

Labels come from `docs/identity-labelset-2026-08-02.md` and nowhere else. A clip
whose only recorded Lewis verdict is about MOTION is not an identity good and is
absent here — the manifest is explicit that treating one as such manufactures a
label the record does not contain.

`raccoon-look-bs` is a PROBE, not ground truth: the manifest records a Fable
TRUE POSITIVE and a Lewis identity-hold PASS on the same run (labelset §4). It is
judged and reported separately; scoring it either way would encode a call nobody
has made.

Every label here is PROVISIONAL — all ten negatives are Fable's eye, Lewis's
ratification pending (DECISION-QUEUE item 5).
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "out"

# (name, path relative to out/, label, provenance)
#   label: "bad" = identity broken, "good" = identity held, None = probe
CLIPS = [
    # --- negatives: identity broken -------------------------------------
    ("pack-wave", "packs/raccoon-emotes/clips/wave", "bad",
     "Fable gate entry, ledger 2026-08-02"),
    ("pack-celebrate", "packs/raccoon-emotes/clips/celebrate", "bad",
     "Fable gate entry, ledger 2026-08-02"),
    ("pack-sleepy", "packs/raccoon-emotes/clips/sleepy", "bad",
     "Fable gate entry, ledger 2026-08-02"),
    ("raccoon-sleepy", "spike2/raccoon-sleepy", "bad",
     "Fable retrospective, ledger :407"),
    ("raccoon-jig", "spike2/raccoon-jig", "bad",
     "Fable retrospective, ledger :408"),
    ("raccoon-jig-pace", "spike2/raccoon-jig-pace", "bad",
     "Fable adjudication 2026-08-02, VERDICTS.md"),
    ("raccoon-jig-bs-s3", "spike2/raccoon-jig-bs-s3", "bad",
     "Fable adjudication 2026-08-02, VERDICTS.md"),
    ("raccoon-breathe", "spike2/raccoon-breathe", "bad",
     "Fable adjudication 2026-08-02, VERDICTS.md"),
    ("raccoon-jig-full", "spike2/raccoon-jig-full", "bad",
     "Fable adjudication 2026-08-02, VERDICTS.md (mild)"),

    # --- positives: identity held ---------------------------------------
    ("pack-thumbs-up", "packs/raccoon-emotes/clips/thumbs-up", "good",
     "pack #1 docs; best of pack #1"),
    ("strawberry-idle-720", "spike2/strawberry-idle-720", "good",
     "named in gates.py :97 as the worst-scoring known-good"),
    ("strawberry-breathe", "spike2/strawberry-breathe", "good",
     "Lewis r001 (:7) identity PASS — 'same entity'"),

    # --- probe: two recorded verdicts disagree (labelset §4) -------------
    ("raccoon-look-bs", "spike2/raccoon-look-bs", None,
     "CONTESTED: Fable TP (VERDICTS.md) vs Lewis identity PASS (ledger r019)"),
]


def clips():
    """Yield (name, abs_path, label, provenance) for every clip in the corpus."""
    for name, rel, label, why in CLIPS:
        yield name, OUT / rel, label, why


def scored():
    """Clips that count toward the confusion matrix (probe excluded)."""
    return [c for c in clips() if c[2] is not None]
