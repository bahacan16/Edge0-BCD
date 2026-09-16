#!/bin/bash
# Prints what BeeWare's Python-Apple-support package actually contains.
#
# Embedding CPython means knowing this layout exactly: where the xcframework
# keeps its headers, whether Swift can `import Python` without a hand-written
# modulemap, and whether the stdlib's binary modules arrive as .so files that
# iOS refuses to load unsigned or as frameworks that it accepts. Each of those
# guessed wrong costs a build, a signing pass and an install on a real phone.
#
# The machine writing this code cannot reach github.com. The runner can.
#
# Runs LAST and prints little: job logs are read from the end, and the first
# attempt at this put its findings behind half a megabyte of Swift build
# output where they could not be retrieved at all.
set +e
echo "@@@ PROBE START @@@"

# Authenticated: the runner's shared egress IP sits permanently over GitHub's
# anonymous API limit, and a rate-limit body parses as "no releases" — which is
# indistinguishable from the package having moved.
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${GH_TOKEN}")
echo "@@@ token present: $([ -n "${GH_TOKEN:-}" ] && echo yes || echo NO)"

curl -sSL --max-time 60 -H "Accept: application/vnd.github+json" "${AUTH[@]}" \
  "https://api.github.com/repos/beeware/Python-Apple-support/releases?per_page=30" \
  > /tmp/releases.json
echo "@@@ releases.json: $(wc -c < /tmp/releases.json) bytes"
echo "@@@ first 300 chars:"
head -c 300 /tmp/releases.json
echo

python3 - <<'PY' > /tmp/asset.txt
import json
try:
    rels = json.load(open("/tmp/releases.json"))
except Exception as exc:
    print("PARSE_FAIL", exc)
    raise SystemExit
if isinstance(rels, dict):
    print("API_MESSAGE", rels.get("message"))
    raise SystemExit
print("TAGS", " ".join(r["tag_name"] for r in rels[:12]))
for rel in rels:
    for asset in rel.get("assets", []):
        print("ASSET", rel["tag_name"], asset["name"], asset["size"])
PY
head -40 /tmp/asset.txt

URL=$(python3 -c "
import json
try: rels = json.load(open('/tmp/releases.json'))
except Exception: rels = []
if isinstance(rels, dict): rels = []
best = ''
for r in rels:
    for a in r.get('assets', []):
        n = a['name'].lower()
        if 'ios' in n and n.endswith('.tar.gz'):
            best = a['browser_download_url']
            break
    if best: break
print(best)
")
echo "@@@ chosen url: ${URL:-<none>}"
[ -z "$URL" ] && { echo "@@@ PROBE END (no asset) @@@"; exit 0; }

rm -rf /tmp/pas && mkdir -p /tmp/pas && cd /tmp/pas || exit 0
curl -sSL --max-time 300 -o support.tar.gz "$URL"
echo "@@@ downloaded: $(du -h support.tar.gz | cut -f1)"
tar xzf support.tar.gz || { echo "@@@ untar failed"; exit 0; }

echo "@@@ TOP LEVEL"
ls -1
echo "@@@ TREE depth 3 (stdlib elided)"
find . -maxdepth 3 -not -path '*/python-stdlib/*' | sort | head -60
echo "@@@ XCFRAMEWORK PLIST"
plutil -p "$(find . -path '*xcframework/Info.plist' | head -1)" 2>/dev/null | head -40
echo "@@@ MODULEMAPS"
find . -name '*.modulemap' | head -10
echo "@@@ Python.h"
find . -name 'Python.h' | head -3
echo "@@@ Headers dirs"
find . -type d -name 'Headers' | head -6
echo "@@@ STDLIB"
STD=$(find . -maxdepth 4 -type d -name 'python-stdlib' | head -1)
echo "dir=$STD  size=$([ -n "$STD" ] && du -sh "$STD" | cut -f1)"
[ -n "$STD" ] && ls -1 "$STD" | head -25
echo "@@@ LIB-DYNLOAD"
find . -type d -name 'lib-dynload' | head -3
echo "so=$(find . -path '*lib-dynload*' -name '*.so' | wc -l | tr -d ' ')  fw=$(find . -path '*lib-dynload*' -name '*.framework' | wc -l | tr -d ' ')"
find . -path '*lib-dynload*' \( -name '*.so' -o -name '*.framework' \) | head -25
echo "@@@ KEY MODULES"
for m in zlib binascii _struct array math _datetime _decimal _socket _ssl; do
  h=$(find . \( -name "${m}.*.so" -o -name "${m}.so" -o -name "${m}.framework" \) | head -1)
  printf '%-10s %s\n' "$m" "${h:-BUILTIN_OR_ABSENT}"
done
echo "@@@ PIP"
find . -maxdepth 8 -type d \( -name ensurepip -o -name pip -o -name site-packages \) | head -6
echo "@@@ TOTAL $(du -sh . | cut -f1)"
echo "@@@ PROBE END @@@"
exit 0
