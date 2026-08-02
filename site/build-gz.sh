#!/usr/bin/env bash
# Pre-compress the served assets for nginx `gzip_static on`.
#
# Why precompress instead of letting nginx gzip on the fly: the clip JSONs are
# ~2 MB each and the droplet is a 1-vCPU box. On-the-fly gzip at level 6 would
# burn CPU on every request for a file that never changes; gzip_static serves
# the .gz straight off disk with zero CPU and a better ratio (-9).
#
# Also prints the byte table that CONFIG/CLIPS in index.html quotes. If you
# swap a clip, run this and paste the numbers into the CLIPS array — the page
# states those bytes as measured fact, so they must not drift.

set -euo pipefail
cd "$(dirname "$0")"

shopt -s nullglob
targets=(assets/clips/*.json assets/vendor/*.js index.html)

for f in "${targets[@]}"; do
  gzip -9 -k -f -c "$f" > "$f.gz"
done

printf '\n%-40s %12s %12s  %s\n' FILE RAW GZ RATIO
for f in "${targets[@]}"; do
  raw=$(wc -c < "$f" | tr -d ' ')
  gz=$(wc -c < "$f.gz" | tr -d ' ')
  printf '%-40s %12s %12s  %s\n' "$f" "$raw" "$gz" \
    "$(awk -v r="$raw" -v g="$gz" 'BEGIN{printf "%.0f%%", 100*g/r}')"
done
echo
echo "Paste the clip raw/gz numbers into the CLIPS array in index.html."
