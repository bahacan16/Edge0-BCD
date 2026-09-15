# Edge0 Demo (iOS)

This is **not** a port of the upstream [`Edge0-AI/edge0`](https://github.com/Edge0-AI/edge0)
Python framework. That project only runs on macOS with Apple Silicon (see the
top-level `README.md` of this repo) — it has no iOS target, and porting its
custom SSD expert-offload / prerouter / Recover-LoRA machinery to Swift is a
separate, much larger engineering effort.

What this **is**: a minimal SwiftUI chat app that runs a small language model
fully on-device on iPhone using Apple's official
[MLX Swift](https://github.com/ml-explore/mlx-swift) /
[MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) libraries — the
same underlying MLX engine the Instagram post referenced, used the way Apple
documents it for iOS. It downloads a small quantized model
(`gemma3-1B-qat-4bit`, a few hundred MB) from Hugging Face on first launch,
then generates replies locally with no network calls.

## Build locally (requires a Mac with Xcode)

```bash
brew install xcodegen
cd ios
xcodegen generate
open Edge0Demo.xcodeproj
```

## CI: unsigned IPA

`.github/workflows/build-unsigned-ipa.yml` builds this app on a GitHub-hosted
macOS runner with code signing disabled and uploads
`Edge0Demo-unsigned.ipa` as a workflow artifact.

An **unsigned** IPA cannot be installed by just tapping it — iOS refuses to
run unsigned code. To get it onto an iPhone 17 Pro Max you still need one of:

- **Sideloadly / AltStore** — resigns the IPA with your own free Apple ID
  (7-day install, no paid developer account needed).
- **A paid Apple Developer account** — resign with `codesign`/Xcode and
  install normally, no 7-day limit.
- **TrollStore** (only on exploitable iOS versions) — installs unsigned/
  permanently-signed IPAs with no resigning needed.

There is no way to make iOS run a truly unsigned binary without one of these
— that restriction is enforced by iOS itself, not by how the IPA was built.
