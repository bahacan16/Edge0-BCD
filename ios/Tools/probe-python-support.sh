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

echo "@@@ ===== utils.sh (the build phase the reference app runs) ====="
cat ./Python.xcframework/build/utils.sh 2>/dev/null || echo "(not at that path)"
echo "@@@ ===== end utils.sh ====="

echo "@@@ ===== build/ directory ====="
find ./Python.xcframework/build -maxdepth 2 2>/dev/null | head -20

echo "@@@ ===== testbed main.m ====="
cat ./testbed/iOSTestbed/main.m 2>/dev/null || echo "(missing)"
echo "@@@ ===== end main.m ====="

echo "@@@ PROBE END @@@"
exit 0
