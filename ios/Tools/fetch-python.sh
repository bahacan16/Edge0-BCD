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

# Clang looks for a framework's module map at `<name>.framework/Modules/
# module.modulemap`. This package ships one under `Headers/` instead, which is
# where a non-framework module map lives — so `import Python` finds nothing.
# One is written where Clang will look, naming Python.h directly rather than as
# an umbrella so the header-completeness check has nothing to complain about.
for slice_dir in "$ROOT"/Python.xcframework/ios-*; do
  framework="$slice_dir/Python.framework"
  [ -d "$framework" ] || continue
  mkdir -p "$framework/Modules"
  cat > "$framework/Modules/module.modulemap" <<'MAP'
framework module Python {
    header "Python.h"
    export *
    link "Python"
}
MAP
  echo "== wrote $(basename "$slice_dir")/Python.framework/Modules/module.modulemap"
done

echo "== slices declared by Info.plist"
plutil -p "$ROOT/Python.xcframework/Info.plist" 2>/dev/null | grep -E 'LibraryIdentifier' || true
echo "== slices on disk"
ls -1 "$ROOT/Python.xcframework" | grep -E '^ios' || true
echo "== shipped modulemap under Headers"
cat "$ROOT/Python.xcframework/ios-arm64/Python.framework/Headers/module.modulemap" 2>/dev/null \
  || echo "(none there)"
echo "== framework contents"
ls -1 "$ROOT/Python.xcframework/ios-arm64/Python.framework" 2>/dev/null || true
