#!/usr/bin/env bash
# Deploy the storefront to the droplet (170.64.189.221).
#
# ============================ GUARD =================================
# This script does NOT run as written. Delete the two lines marked
# GUARD below only when you have actually read the two documents named
# and set APP/DOMAIN. The droplet hosts pum.me, admin.pum.me,
# crashes.pum.me, brief.pum.me and the 3rigby apps — a careless rsync
# --delete against the wrong path takes real infrastructure down.
# ====================================================================
echo "review DROPLET.md + REGISTER.md first" && exit 1   # GUARD — delete to arm
# GUARD

set -euo pipefail
cd "$(dirname "$0")"

# --- Configure these two, then read the checklist at the bottom ------
APP=""                       # e.g. lottie-store   -> /var/www/$APP
DOMAIN=""                    # e.g. lottie.3rigby.xyz
HOST="root@170.64.189.221"
KEY="$HOME/.ssh/id_ed25519"
# --------------------------------------------------------------------

[ -n "$APP" ] && [ -n "$DOMAIN" ] || { echo "set APP and DOMAIN first"; exit 1; }

# One directory per app (DROPLET.md rule 3): everything served lives under
# /var/www/$APP and nothing is scattered into /var/www/html.
TARGET="/var/www/$APP"

echo "==> regenerating precompressed assets"
./build-gz.sh

echo "==> syncing to $HOST:$TARGET"
# --delete is safe ONLY because $TARGET holds nothing but this site's files.
# It must never be pointed at /var/www/html (pum.me's historical split lives
# there) or at any directory an app writes data into.
rsync -avz --delete \
  -e "ssh -i $KEY" \
  --exclude '.DS_Store' \
  ./index.html ./assets \
  "$HOST:$TARGET/"

ssh -i "$KEY" "$HOST" "chown -R www-data:www-data $TARGET"

echo "==> smoke test"
curl -sSI "https://$DOMAIN/" | head -1
curl -sS -H 'Accept-Encoding: gzip' -o /dev/null \
  -w 'clip: %{http_code}  %{size_download} bytes  %{content_type}\n' \
  "https://$DOMAIN/assets/clips/star-celebrate.json"

cat <<'EOF'

==> NOT DONE YET. Finish these by hand, this session:

  1. nginx (first deploy only)
       scp site/nginx.conf.example  ->  /etc/nginx/sites-available/$APP
       edit <APP> and <DOMAIN> tokens
       ln -s ../sites-available/$APP /etc/nginx/sites-enabled/$APP
       nginx -t && systemctl reload nginx     # never reload without -t
  2. TLS (first deploy only)
       certbot --nginx -d $DOMAIN
  3. Confirm gzip_static is actually serving the .gz twins:
       curl -sI -H 'Accept-Encoding: gzip' https://$DOMAIN/assets/clips/star-celebrate.json \
         | grep -i 'content-encoding\|content-length'
       Expect content-encoding: gzip and ~780 KB, not ~1.0 MB.
  4. Update /root/REGISTER.md — add the app row (no port: static, no
     systemd unit), and a Recent Changes entry. DROPLET.md is explicit
     that "later" does not happen. Do it before you close the session.

EOF
