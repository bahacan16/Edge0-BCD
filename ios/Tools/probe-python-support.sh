#!/bin/bash
# Prints what BeeWare's Python-Apple-support package actually contains.
#
# Embedding CPython means knowing this layout exactly: where the xcframework
# keeps its headers, whether Swift can `import Python` without a hand-written
# modulemap, and whether the stdlib's binary modules arrive as .so files that
# iOS refuses to load unsigned or as frameworks that it accepts. Each of those
# guessed wrong costs a build, a signing pass and an install on a real phone.
#
# The machine writing this code cannot reach github.com. The runner can. So the
# runner looks and the next commit is written against what it saw.
#
# Never fails the job: reconnaissance that can break a working build is worse
# than no reconnaissance.
set +e

api() {
  curl -sSL -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "Accept: application/vnd.github+json" "$1"
}

echo "======== iOS assets, recent releases ========"
api "https://api.github.com/repos/beeware/Python-Apple-support/releases?per_page=20" \
  > /tmp/releases.json
python3 - <<'PY'
import json
try:
    rels = json.load(open("/tmp/releases.json"))
except Exception as exc:
    print("could not parse releases:", exc)
    raise SystemExit
if isinstance(rels, dict):
    print("api said:", rels.get("message"))
    raise SystemExit
for rel in rels:
    for asset in rel.get("assets", []):
        if "iOS" in asset["name"]:
            print(f'{rel["tag_name"]:14} {asset["name"]:46} {asset["size"]/1e6:7.1f} MB')
PY

echo "======== newest 3.13 iOS asset ========"
URL=$(python3 - <<'PY'
import json
try:
    rels = json.load(open("/tmp/releases.json"))
except Exception:
    rels = []
if isinstance(rels, dict):
    rels = []
for rel in rels:
    if not rel["tag_name"].startswith("3.13"):
        continue
    for asset in rel.get("assets", []):
        if "iOS" in asset["name"]:
            print(asset["browser_download_url"])
            raise SystemExit
print("")
PY
)
echo "url: ${URL:-<none>}"
if [ -z "$URL" ]; then
  echo "no 3.13 iOS asset; stopping here"
  exit 0
fi

rm -rf /tmp/pas && mkdir -p /tmp/pas && cd /tmp/pas || exit 0
curl -sSL -o support.tar.gz "$URL" || exit 0
ls -lh support.tar.gz
tar xzf support.tar.gz || exit 0

echo "======== top level ========"
ls -la
echo "======== tree (stdlib contents elided) ========"
find . -maxdepth 4 -not -path '*/python-stdlib/*' | sort | head -100

echo "======== xcframework Info.plist ========"
for plist in $(find . -path '*xcframework/Info.plist' | head -3); do
  echo "--- $plist ---"
  plutil -p "$plist" 2>/dev/null | head -60
done

echo "======== headers and modulemap ========"
find . -name '*.modulemap' | head -20
echo "-- Python.h --"
find . -name 'Python.h' | head -5
echo "-- header dirs --"
find . -type d -name 'Headers' | head -10

echo "======== stdlib ========"
STD=$(find . -maxdepth 4 -type d -name 'python-stdlib' | head -1)
echo "stdlib dir: ${STD:-<none>}"
if [ -n "$STD" ]; then
  du -sh "$STD"
  ls "$STD" | head -40
fi

echo "======== binary modules ========"
find . -type d -name 'lib-dynload' | head
echo "-- .so count --"
find . -path '*lib-dynload*' -name '*.so' | wc -l
find . -path '*lib-dynload*' -name '*.so' | head -30
echo "-- .framework under lib-dynload --"
find . -path '*lib-dynload*' -name '*.framework' | head -30

echo "======== modules that decide the design ========"
for module in zlib binascii _struct array math _datetime _decimal _socket _ssl _hashlib; do
  hit=$(find . \( -name "${module}.*.so" -o -name "${module}.so" -o -name "${module}.framework" \) | head -1)
  printf '%-12s %s\n' "$module" "${hit:-<not a separate file: builtin or absent>}"
done

echo "======== pip / ensurepip ========"
find . -maxdepth 8 -type d \( -name 'ensurepip' -o -name 'pip' -o -name 'site-packages' \) | head

echo "======== unpacked size ========"
du -sh .
exit 0
