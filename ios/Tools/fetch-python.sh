#!/bin/bash
# Fetches the embedded Python runtime and the pure-Python packages the app
# ships with, into ios/Python.xcframework and ios/Resources/app_packages.
#
# ezdxf is pinned to 1.0.3 deliberately: it is the last release that does not
# require numpy, and numpy is a C extension, which iOS will not load if it was
# downloaded at runtime. Verified — 1.0.3 creates, writes, re-reads and measures
# a drawing on Python 3.14 with numpy never imported, while 1.1.4, 1.2.0, 1.3.5
# and 1.4.4 all die on `import ezdxf` without it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_SERIES="${PY_SERIES:-3.14}"
XCFRAMEWORK="$ROOT/Python.xcframework"
PKGS="$ROOT/Resources/app_packages"

# ---------------------------------------------------------------- runtime ---

if [ -d "$XCFRAMEWORK" ]; then
  echo "== xcframework already present"
else
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
raise SystemExit("no iOS asset for " + series)
PY
)
EOF
  echo "== tag ${TAG}"
  echo "PYTHON_SUPPORT_TAG=${TAG}" >> "${GITHUB_ENV:-/dev/null}"

  rm -rf /tmp/pas && mkdir -p /tmp/pas
  curl -sSL --max-time 600 -o /tmp/pas/support.tar.gz "$URL"
  tar xzf /tmp/pas/support.tar.gz -C /tmp/pas
  # Both slices are kept. Deleting the simulator one to save 80 MB left the
  # xcframework's Info.plist still declaring it, and that plist is what Xcode
  # reads to resolve a slice — the result was `Unable to find module
  # dependency: 'Python'`, a manifest describing a directory that had been
  # removed. Only the matching slice is embedded in the app, so the cost was
  # never IPA size.
  mv /tmp/pas/Python.xcframework "$XCFRAMEWORK"
  echo "== xcframework $(du -sh "$XCFRAMEWORK" | cut -f1)"
fi

# Clang resolves a framework's module map at `<name>.framework/Modules/
# module.modulemap`. This package ships one under `Headers/`, which is where a
# *non*-framework module map lives, so nothing looks there. Python.h is named
# directly rather than declared an umbrella: with an umbrella, Clang checks that
# every header in the directory is reachable from it, and CPython ships two
# hundred internal ones that are not.
for slice_dir in "$XCFRAMEWORK"/ios-*; do
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
done
echo "== module maps written for: $(ls -1 "$XCFRAMEWORK" | grep -c '^ios') slice(s)"

# --------------------------------------------------------------- packages ---

if [ -d "$PKGS" ] && [ -n "$(ls -A "$PKGS" 2>/dev/null)" ]; then
  echo "== app_packages already present"
else
  echo "== fetching pure-Python wheels"
  rm -rf "$PKGS" && mkdir -p "$PKGS"
  rm -rf /tmp/wheels && mkdir -p /tmp/wheels
  python3 -m pip download --no-deps --only-binary=:all: \
    --python-version 3.14 --implementation py --abi none --platform any \
    -d /tmp/wheels ezdxf==1.0.3 pyparsing typing_extensions
  for wheel in /tmp/wheels/*.whl; do
    echo "   unzip $(basename "$wheel")"
    unzip -q -o "$wheel" -d "$PKGS"
  done
  echo "== app_packages $(du -sh "$PKGS" | cut -f1)"
fi

# Loudly, and here: an empty directory reaches XcodeGen as "missing source
# directory", which is a true statement about a completely different problem —
# and is exactly how a botched edit to this script presented last time.
for required in ezdxf pyparsing typing_extensions.py; do
  if [ ! -e "$PKGS/$required" ]; then
    echo "FATAL: $required missing from app_packages" >&2
    ls -1 "$PKGS" >&2
    exit 1
  fi
done
echo "== app_packages contents: $(ls -1 "$PKGS" | tr '\n' ' ')"
