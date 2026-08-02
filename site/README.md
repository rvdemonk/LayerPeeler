# site/ — the storefront

A single static page that sells mascot-to-Lottie packs, with four real clips
playing in `lottie-web` on the page. No framework, no build step, no CDN: hand
written HTML/CSS/JS plus a vendored copy of the player. Everything it serves
lives in this directory.

This README assumes you are working alone, with no Claude session open.

---

## What's here

| Path | What it is |
|---|---|
| `index.html` | The whole page. Config block at the top, styles, markup, gallery script. |
| `assets/vendor/lottie.min.js` | lottie-web 5.13.0, vendored from `sandbox/vendor/`. Not loaded from a CDN on purpose — the page must work if a CDN doesn't. |
| `assets/clips/*.json` | The four gallery clips. Copies of shipping artifacts (see below). |
| `assets/**/*.gz` | Precompressed twins for nginx `gzip_static`. Generated, not hand-edited. |
| `build-gz.sh` | Regenerates the `.gz` twins and prints the byte table the page quotes. |
| `deploy.sh` | rsync to the droplet. **Guarded — will not run until you disarm it.** |
| `nginx.conf.example` | Server block to copy onto the droplet. Not applied by anything here. |

The clips are copies, renamed for the URL. Originals:

```
out/spike2/strawberry-idle-720/ladder/strawberry-idle-720.512webp-q65-24.json  -> strawberry-idle.json
out/spike2/raccoon-wave/ladder/raccoon-wave.512webp-q65-24.json                -> raccoon-wave.json
out/spike2/star-celebrate/ladder/star-celebrate.512webp-q65-24.json            -> star-celebrate.json
out/packs/raccoon-emotes/clips/thumbs-up/ladder/thumbs-up.512webp-q65-24.json  -> raccoon-thumbs-up.json
```

---

## The TODO list before this goes live

Four edits, all in `index.html`, all inside the `CONFIG` block at the very top.

1. **Product name.** Set `brand`. It propagates to the page title, the header
   mark, the footer, and the email subject line. `BRANDNAME` appears nowhere
   else that matters — one edit is the whole rename.

2. **Stripe link.** In the Stripe dashboard: Payment Links → new link →
   one-off AU$79, name it "Founding pack". Copy the `https://buy.stripe.com/…`
   URL into `CONFIG.stripe`. Until you do, the buy button points at
   `#STRIPE_LINK` and goes nowhere. Two things worth setting on the link
   itself: collect the customer's email (that's your intake), and put "4–6
   Lottie emotes from one mascot, delivered within 48h" in the description so
   the promise is on the receipt.

3. **X handle.** Set `CONFIG.xHandle` (no `@`). If you'd rather not take DMs,
   delete the "X DMs" card in the `#intake` section instead — an empty
   placeholder looks worse than no card.

4. **Email.** Already `lewisthompson96@gmail.com`. Change it if you set up a
   business address.

Optional but cheap: the clip captions in the `CLIPS` array at the bottom of
`index.html`. They currently say true things about each clip; if you swap a
clip, rewrite its caption and re-run `./build-gz.sh` to get the new byte
counts, which the page states as measured fact.

---

## Deploying

The page is static files off disk — no port to claim, no systemd unit, no
build on the box. Read `~/claude-resources/DROPLET.md` and `/root/REGISTER.md`
first; the deploy script refuses to run until you have.

```bash
cd site
$EDITOR deploy.sh          # set APP and DOMAIN, delete the two GUARD lines
./deploy.sh
```

`deploy.sh` regenerates the `.gz` twins, rsyncs `index.html` + `assets/` to
`/var/www/$APP/`, chowns to `www-data`, and smoke-tests over HTTPS. It then
prints the four things it deliberately does **not** do for you:

1. Install the nginx server block (`nginx.conf.example`, edit two tokens,
   `nginx -t` before reload — never reload without it).
2. `certbot --nginx -d $DOMAIN`.
3. Confirm `gzip_static` is actually serving the `.gz` twins — the check is in
   the script's output. If `content-length` on `star-celebrate.json` comes back
   ~1.0 MB instead of ~780 KB, `gzip_static` is off and every visitor is paying
   for it.
4. **Update `/root/REGISTER.md`** in the same session. DROPLET.md is blunt
   about this: the window where "later" happens is about ten minutes.

`rsync --delete` is in the script. It is only safe because `/var/www/$APP/`
holds nothing but this site. Never point it at `/var/www/html/` — pum.me's
files are there.

---

## Local preview

```bash
cd site
python3 -m http.server 8791     # any free port; 8646 is taken
open http://127.0.0.1:8791/
```

Opening `index.html` as a `file://` URL will **not** work — the gallery
`fetch()`es the clip JSONs and the browser blocks that cross-origin. Use the
server.

---

## Things that will bite you if you forget them

- **The byte counts in the page are hand-maintained.** They're in the `CLIPS`
  array and quoted in the copy as measured fact. `build-gz.sh` prints the
  current table; if you swap a clip and don't update them, the page is lying
  about something a technical buyer can check with `curl -I` in ten seconds.
- **The clips are ~7.8 MB of JSON total.** The page loads them one at a time
  as they scroll into view and pauses off-screen animations. If you add a
  fifth clip, keep that behaviour — mounting four at once already means ~484
  embedded images live in the tab.
- **Canvas renderer, not SVG.** These are frame sequences (121 image layers
  each). The SVG renderer will mount them and then crawl. Same applies to any
  snippet you send a customer.
- **No caching header on `index.html`.** Intentional: the pricing and the
  Stripe link live in it. The `/assets/` cache header is 1 day; if you're
  iterating on clips, hard-refresh or drop it temporarily.

## What the page claims, and where each claim comes from

Everything on the page is either measured here or recorded in
`docs/workorders/hybrid-oracle-spikes/ledger.md`. Nothing is projected. If you
edit the copy, keep it that way — the buyers you're talking to will check.

- File sizes: measured by `build-gz.sh` off the exact files being served.
- "50–300 KB for hand-made vector": the positioning truth recorded in the
  ledger, 2026-07-31. Do not soften it; it's the sentence that makes a
  technical reader trust the rest of the page.
- "lottie-ios 4.6.1 verified": simulator smoke test, ledger 2026-07-31 — pixel
  census across all eight variants, WebP decode confirmed. The page says
  *simulator* and lists Android / React Native / older iOS as untested,
  because they are.
- "the single approved take out of four" (raccoon thumbs-up caption): pack #1,
  measured r_anim = 4.0, ledger 2026-08-02.
- No testimonials, no logos, no user counts, no "trusted by". The product is
  days old and the FAQ says so in as many words. Adding social proof you don't
  have is the one edit that would cost you the audience this page is aimed at.
