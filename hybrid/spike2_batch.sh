#!/bin/zsh
# Spike 2 batch — 3 mascots x 4 motions at 480p ($0.05/video).
# Prompt doctrine from r001 (Lewis: "fidgety... arms and breath unnatural"):
# ONE clear action per clip, slow tempo, arms explicitly quiet unless the
# motion IS the arms. Every prompt pins: anchored in place, static uniform
# background, flat 2D style, no camera movement, seamlessly loopable.
set -e
set -a; source ~/.env; set +a
cd "$(dirname "$0")/.."
PY=.venv-hybrid/bin/python

STYLE="The character stays anchored in place at the center; the flat uniform background remains completely static. Smooth 2D flat-vector cartoon style, no camera movement, seamlessly loopable animation."

run() { # name image prompt
  $PY hybrid/spike2_oracle.py --image "$2" --name "$1" --resolution 480p --prompt "$3" \
    && $PY hybrid/spike2_gates.py "out/spike2/$1"
}

B=../corpus/mascots/blob.jpg
S=../corpus/mascots/strawberry.jpg
R=../corpus/mascots/raccoon.jpg

# --- strawberry (r001 was breathe; r002 chases Lewis's pacing note) ---
run strawberry-breathe2 $S "A cute flat-vector strawberry mascot performs an extremely calm, slow idle breathing loop: only its body gently rises and falls with one slow deep breath, twice. Its stubby arms rest still against its body the whole time. It blinks once, slowly. No other movement. $STYLE"
run strawberry-wave $S "A cute flat-vector strawberry mascot performs one single clear friendly wave: it raises its right stub arm and waves side to side twice at a relaxed pace, smiling warmly, then lowers the arm back to rest. Its other arm and body stay still. $STYLE"
run strawberry-jig $S "A cute flat-vector strawberry mascot does a happy little celebratory jig: it bounces up and down twice in rhythm, tilting slightly side to side, leaf crown bobbing, with a joyful open-mouth smile. The bounces are springy and clean. $STYLE"
run strawberry-look $S "A cute flat-vector strawberry mascot glances around curiously: its large eyes look slowly to the left, hold, then slowly to the right, then return to center with a soft blink. Head tilts very slightly with each glance. Arms and body stay completely still. $STYLE"

# --- blob ---
run blob-breathe $B "A minimal flat-vector blue blob mascot performs an extremely calm idle breathing loop: its soft teardrop body gently inflates and settles with slow breaths, squashing very slightly. It blinks once. No other movement. $STYLE"
run blob-wave $B "A minimal flat-vector blue blob mascot performs one single clear friendly wave: it raises its tiny stub arm and waves side to side twice at a relaxed pace with a cheerful smile, then lowers it back. The body stays still. $STYLE"
run blob-jig $B "A minimal flat-vector blue blob mascot does a happy bouncy jig: it bounces up and down twice in rhythm with squash and stretch, tilting playfully side to side, smiling. The bounces are springy and clean. $STYLE"
run blob-look $B "A minimal flat-vector blue blob mascot glances around curiously: its oval eyes look slowly left, hold, then slowly right, then return to center and blink once. The body sways very slightly with each glance. No other movement. $STYLE"

# --- raccoon ---
run raccoon-breathe $R "A flat-vector cartoon raccoon mascot performs an extremely calm idle breathing loop: its chest and belly gently rise and fall with slow breaths, its striped tail sways very slightly, and it blinks once. Both arms rest at its sides, still. No other movement. $STYLE"
run raccoon-wave $R "A flat-vector cartoon raccoon mascot performs one single clear friendly wave: its already-raised paw waves side to side twice at a relaxed pace with a warm smile, then holds. The other arm, body and tail stay still. $STYLE"
run raccoon-jig $R "A flat-vector cartoon raccoon mascot does a happy little dance: it bounces twice in rhythm, striped tail bouncing with it, arms swinging loosely, with a joyful grin. The movements are springy and clean. $STYLE"
run raccoon-look $R "A flat-vector cartoon raccoon mascot glances around curiously: its big masked eyes look slowly to the left, hold, then slowly to the right, then return to center with a blink. Its head turns slightly with each glance; body, arms and tail stay still. $STYLE"

echo "BATCH DONE"
