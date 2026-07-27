#!/usr/bin/env bash
# vast_peel.sh — one-shot directed-peel run on vast.ai.
#   provision (A6000-class) -> setup -> run driver -> pull outputs -> DESTROY.
#
# Usage:
#   scripts/vast_peel.sh <image.png> <plan.json> [outdir]
#
# Env (sourced automatically if present):
#   GEMINI_API_KEY   — from ~/.env (GEMINI_API_SECRET_KEY is mapped)
#   HF_TOKEN         — from ~/code/sidequests/.env (FLUX.1-dev is gated)
#
# Budget guard: refuses to start if vast credit < $1.50; caps offer price at
# MAX_DPH; hard wall-clock cap on the remote run; instance is destroyed on any
# exit path (trap). Model download (~35 GB) dominates setup: expect ~20-30 min
# before first peel.
#
# NOTE: research-only outputs (FLUX.1-dev NC licence — see workorder scope).

set -euo pipefail

IMAGE_PNG="${1:?usage: vast_peel.sh <image.png> <plan.json> [outdir]}"
PLAN_JSON="${2:?usage: vast_peel.sh <image.png> <plan.json> [outdir]}"
OUTDIR="${3:-out/$(basename "${IMAGE_PNG%.*}")}"

REPO_URL="https://github.com/rvdemonk/LayerPeeler"
BRANCH="bitrot-2026"
DOCKER_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB=80
MAX_DPH=0.80              # $/hr ceiling when picking an offer
MIN_CREDIT=1.50           # refuse to start below this
MAX_RUN_SECONDS=7200      # hard cap on the remote run (2 h)
SEARCH_QUERY='gpu_name in [RTX_A6000,A40,L40,L40S] num_gpus=1 disk_space>=80 inet_down>=200 rentable=true verified=true'

log() { printf '\033[1;36m[vast_peel]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[vast_peel]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- keys -------------------------------------------------------------------
[ -f "$HOME/.env" ] && set -a && . "$HOME/.env" && set +a
[ -f "$HOME/code/sidequests/.env" ] && set -a && . "$HOME/code/sidequests/.env" && set +a
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GEMINI_API_SECRET_KEY:-}}"
[ -n "${GEMINI_API_KEY:-}" ] || die "GEMINI_API_KEY not found (expected GEMINI_API_SECRET_KEY in ~/.env)"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN not found (expected in ~/code/sidequests/.env)"
command -v vastai >/dev/null || die "vastai CLI not installed (pipx install vastai)"
command -v jq >/dev/null || die "jq required"
[ -f "$IMAGE_PNG" ] || die "image not found: $IMAGE_PNG"
[ -f "$PLAN_JSON" ] || die "plan not found: $PLAN_JSON"

# ---- budget guard -----------------------------------------------------------
CREDIT=$(vastai show user --raw | jq -r '.credit')
log "vast credit: \$$CREDIT"
awk -v c="$CREDIT" -v m="$MIN_CREDIT" 'BEGIN{exit !(c>=m)}' \
  || die "credit \$$CREDIT below floor \$$MIN_CREDIT — aborting"

# ---- provision --------------------------------------------------------------
log "searching offers: $SEARCH_QUERY (dph <= $MAX_DPH)"
OFFER=$(vastai search offers "$SEARCH_QUERY" -o 'dph' --raw \
  | jq -r --argjson m "$MAX_DPH" '[.[] | select(.dph_total <= $m)][0]')
[ "$OFFER" != "null" ] && [ -n "$OFFER" ] || die "no offer under \$$MAX_DPH/hr"
OFFER_ID=$(jq -r '.id' <<<"$OFFER")
log "offer $OFFER_ID: $(jq -r '"\(.gpu_name) @ $\(.dph_total)/hr, \(.inet_down) Mbps down"' <<<"$OFFER")"

INSTANCE_ID=$(vastai create instance "$OFFER_ID" --image "$DOCKER_IMAGE" \
  --disk "$DISK_GB" --ssh --direct --raw | jq -r '.new_contract')
[ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "null" ] || die "instance creation failed"
log "instance $INSTANCE_ID created"

destroy() {
  log "destroying instance $INSTANCE_ID"
  yes | vastai destroy instance "$INSTANCE_ID" || log "WARNING: destroy failed — check 'vastai show instances' NOW"
}
trap destroy EXIT

# ---- wait for running -------------------------------------------------------
log "waiting for instance to boot..."
for i in $(seq 1 60); do
  STATUS=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.actual_status // "loading"')
  [ "$STATUS" = "running" ] && break
  sleep 10
done
[ "$STATUS" = "running" ] || die "instance never reached running (last: $STATUS)"

INFO=$(vastai show instance "$INSTANCE_ID" --raw)
SSH_HOST=$(jq -r '.ssh_host' <<<"$INFO")
SSH_PORT=$(jq -r '.ssh_port' <<<"$INFO")
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $SSH_PORT root@$SSH_HOST"
log "up: root@$SSH_HOST:$SSH_PORT"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 10; done
$SSH true || die "ssh never came up"

# ---- setup + run ------------------------------------------------------------
TARGET=$(jq -r '.target // empty' "$PLAN_JSON")
[ -n "$TARGET" ] || TARGET=$(basename "${IMAGE_PNG%.*}")

log "uploading inputs"
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$IMAGE_PNG" "root@$SSH_HOST:/root/input.png"
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$PLAN_JSON" "root@$SSH_HOST:/root/plan.json"

log "remote setup + run (hard cap ${MAX_RUN_SECONDS}s)"
$SSH "GEMINI_API_KEY='$GEMINI_API_KEY' HF_TOKEN='$HF_TOKEN' timeout $MAX_RUN_SECONDS bash -s" <<'REMOTE'
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0 DEBIAN_FRONTEND=noninteractive
cd /root
git clone --depth 1 --branch bitrot-2026 https://github.com/rvdemonk/LayerPeeler
cd LayerPeeler
pip install -q -r requirements.txt
python -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
echo "== merging pretrained base (downloads FLUX.1-dev; longest step) =="
python merge_pretrain.py
cd inference
mkdir -p plans && cp /root/plan.json plans/plan.json
echo "== directed peel run =="
python directed_peel.py \
  --image /root/input.png \
  --plan plans/plan.json \
  --pretrained_model_name_or_path ../PhotoDoodle_Pretrain \
  --output_folder /root/outputs
REMOTE

# ---- pull -------------------------------------------------------------------
log "pulling outputs -> $OUTDIR"
mkdir -p "$OUTDIR"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
  "root@$SSH_HOST:/root/outputs/" "$OUTDIR/"

log "done. outputs in $OUTDIR (instance will now be destroyed)"
