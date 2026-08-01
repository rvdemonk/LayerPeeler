# ios-sandbox

Instrument for answering one question: does `lottie-ios` play our frame-sequence
Lotties (161 embedded base64 image assets + 161 one-frame image layers), and does
it decode `data:image/webp;base64` assets?

Not a product. Nothing here is committed.

## Layout

- `project.yml` — xcodegen spec (lottie-ios 4.6.1 pinned via SPM).
- `Sources/` — SwiftUI app. `Resources/lotties/` — bundled test JSONs (56 MB).
- `scripts/verify.py` — liveness + pixel-count sweep, writes `artifacts/results.json`.
- `scripts/framecmp.py` — frame-locked control-vs-candidate diff, writes `artifacts/framecmp.json`.
- `.simid` — the dedicated simulator UDID both scripts read.

## Re-running

```sh
xcodegen generate
SIM=$(cat .simid)
xcrun simctl boot $SIM
xcodebuild -project LottieSandbox.xcodeproj -scheme LottieSandbox \
  -destination "id=$SIM" -derivedDataPath build build
xcrun simctl install $SIM build/Build/Products/Debug-iphonesimulator/LottieSandbox.app
python3 scripts/verify.py      # all variants, or pass specific ones
python3 scripts/framecmp.py
```

The simulator is a purpose-created device named `lottie-sandbox` (iPhone 17 Pro,
iOS 26.3). A pre-existing simulator popped a system Apple Account alert over the
play area mid-capture, which is why verification runs on a device with no account
state. Recreate with:

```sh
xcrun simctl create lottie-sandbox \
  com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro \
  com.apple.CoreSimulator.SimRuntime.iOS-26-3 > .simid
```

## Constraints the scripts depend on

- `RECT = (123, 606, 1083, 1566)` is the play area in screenshot pixels, derived
  from a 320pt square at 3x on this device. Change `Geometry.playSide`, the
  header height, or the device model and both scripts measure the wrong region —
  `verify.py` re-checks the surrounding chrome is still 0.92 grey and reports
  `rect_aligned: false` if not.
- The play area sits on pure white and all text lives outside it, so "content
  pixels" is just "pixels that are not white". Do not put UI inside the square.
- `-variant <name>` and `-frame <n>` land in the volatile NSArgumentDomain, so a
  plain launch still shows the variant list.
