#!/bin/bash
# Fetches BeeWare's Python-Apple-support and the pure-Python wheels the app
# ships with, into ios/Python.xcframework and ios/Resources/app_packages.
#
# ezdxf is pinned to 1.0.3 deliberately: it is the last release that does not
# require numpy, which is a C extension and therefore cannot be installed at
# runtime on iOS at all. Verified here — 1.0.3 creates, writes, re-reads and
# measures a drawing on Python 3.14 with numpy never imported; 1.1.4, 1.2.0,
# 1.3.5 and 1.4.4 all die on `import ezdxf` without it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_SERIES="${PY_SERIES:-3.14}"

echo "== resolving Python-Apple-support ${PY_SERIES}"
curl -sSL --max-time 60 -H "Accept: application/vnd.github+json" \
  ${GH_TOKEN:+-H "Authorization: Bearer ${GH_TOKEN}"} \
  "https://api.github.com/repos/beeware/Python-Apple-support/releases?per_page=30" \
  > /tmp/releases.json

read -r TAG URL <<EOF
$(python3 - "$PY_SERIES" <<'PY'
import json, sys
series = sys.argv[1]
rels = json.load(open("/tmp/releases.json"))
if isinstance(rels, dict):
    raise SystemExit("github api: " + str(rels.get("message")))
for rel in rels:
    if not rel["tag_name"].startswith(series):
        continue
    for asset in rel.get("assets", []):
        name = asset["name"].lower()
        if "ios" in name and name.endswith(".tar.gz"):
            print(rel["tag_name"], asset["browser_download_url"])
            raise SystemExit
raise SystemExit(f"no iOS asset for {series}")
PY
)
EOF
echo "== tag ${TAG}"
echo "PYTHON_SUPPORT_TAG=${TAG}" >> "${GITHUB_ENV:-/dev/null}"

if [ -d "$ROOT/Python.xcframework" ]; then
  echo "== xcframework already present (cache hit)"
else
  rm -rf /tmp/pas && mkdir -p /tmp/pas
  curl -sSL --max-time 600 -o /tmp/pas/support.tar.gz "$URL"
  tar xzf /tmp/pas/support.tar.gz -C /tmp/pas
  # Only the framework is needed; the testbed app and the simulator slice are
  # not, and the simulator slice is half the 168 MB.
  # Both slices are kept. Deleting the simulator one to save 80 MB left the
  # xcframework's Info.plist still listing it, and Xcode reads that plist to
  # resolve which slice to use — the result was `Unable to find module
  # dependency: 'Python'`, which reads like a missing modulemap and was in fact
  # a manifest describing a directory that was no longer there. Only the
  # matching slice is embedded in the app either way, so the cost is CI cache,
  # not IPA size.
  mv /tmp/pas/Python.xcframework "$ROOT/Python.xcframework"
  echo "== xcframework $(du -sh "$ROOT/Python.xcframework" | cut -f1)"
fi

echo "== slices declared by Info.plist"
plutil -p "$ROOT/Python.xcframework/Info.plist" 2>/dev/null \
  | grep -E 'LibraryIdentifier' || true
echo "== slices on disk"
ls -1 "$ROOT/Python.xcframework" | grep -E '^ios' || true
echo "== modulemap (what `import Python` resolves against)"
cat "$ROOT/Python.xcframework/ios-arm64/Python.framework/Headers/module.modulemap" 2>/dev/null \
  || echo "(no modulemap at that path)"

PKGS="$ROOT/Resources/app_packages"
if [ -d "$PKGS" ] && [ -n "$(ls -A "$PKGS" 2>/dev/null)" ]; then
  echo "== app_packages already present (cache hit)"
else
  echo "== fetching pure-Python wheels"
  rm -rf "$PKGS" && mkdir -p "$PKGS"
  python3 -m pip download --no-deps --only-binary=:all: --python-version 3.14 \
    --implementation py --abi none --platform any \
    -d /tmp/wheels ezdxf==1.0.3 pyparsing typing_extensions
  for wheel in /tmp/wheels/*.whl; do
    echo "   unzip $(basename "$wheel")"
    unzip -q -o "$wheel" -d "$PKGS"
  done
  # Metadata the interpreter never reads, and megabytes of it.
  rm -rf "$PKGS"/*.dist-info/RECORD "$PKGS"/*.dist-info/licenses
  echo "== app_packages $(du -sh "$PKGS" | cut -f1)"
fi

echo "== contents"
ls -1 "$PKGS" | head -20
