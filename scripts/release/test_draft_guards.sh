#!/usr/bin/env bash
# Tests the release job's draft step (.github/workflows/release.yml, "upload to a draft release for the tag"):
# its run: block, taken from the YAML and run as the workflow runs it (bash --noprofile --norc -eo pipefail),
# against a stub `gh`. Each case sets what GitHub answers, then checks the exit code, whether a draft was made,
# how often the step asked which commit the tag names, whether it retitled the draft, and what it said. Then the
# release job's wiring the step relies on (its concurrency group, BUILT). No network, unless --live.
#
# Usage: scripts/release/test_draft_guards.sh [--live] [release.yml]
#   --live  also run the step against the real API's commits/refs/tags endpoint (an annotated tag, a lightweight
#           one; the release listing and `gh release create` stay stubbed). Needs gh logged in.
set -euo pipefail

live=0
if [ "${1:-}" = --live ]; then live=1; shift; fi
yml=${1:-.github/workflows/release.yml}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# The step's run: block: the block scalar after its `run: |`, dedented. Fails if the step or its run: is missing.
python3 - "$yml" > "$T/step.sh" <<'EOF'
import sys
lines = open(sys.argv[1]).read().split("\n")
name = "- name: upload to a draft release for the tag"
start = [i for i, l in enumerate(lines) if l.strip() == name]
if len(start) != 1:
    sys.exit(f"{sys.argv[1]}: {len(start)} steps named {name[8:]!r}, want 1")
i = start[0]
step_indent = len(lines[i]) - len(lines[i].lstrip())
j = i + 1
while lines[j].strip() != "run: |":
    if lines[j].strip() and len(lines[j]) - len(lines[j].lstrip()) <= step_indent:
        sys.exit("the draft step has no `run: |`")
    j += 1
run_indent = len(lines[j]) - len(lines[j].lstrip())
body, block_indent = [], None
for l in lines[j + 1:]:
    if not l.strip():
        body.append("")
        continue
    ind = len(l) - len(l.lstrip())
    if ind <= run_indent:
        break
    block_indent = ind if block_indent is None else block_indent
    body.append(l[block_indent:])
print("\n".join(body).rstrip("\n"))
EOF
grep -q 'gh release create' "$T/step.sh" || { echo "test_draft_guards: no gh release create in the step"; exit 1; }

# The stub gh. Fixture files in $CASE_DIR: tagsha (one answer per line for commits/refs/tags/<tag>, the last one
# repeats: a SHA, FAIL = HTTP 422, NOSHA = a JSON answer without .sha, EMPTY = an empty .sha, REAL = ask the real
# GitHub), releases
# (the release listing's output, or FAIL). It applies the step's own --jq expression to the JSON it fakes.
mkdir "$T/bin"
cat > "$T/bin/gh" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$CASE_DIR/calls"
jq_expr=
args=("$@")
for ((k = 0; k < ${#args[@]}; k++)); do [ "${args[k]}" = --jq ] && jq_expr=${args[k + 1]}; done
answer() { if [ -n "$jq_expr" ]; then jq -r "$jq_expr"; else cat; fi; }
case "$1 $2" in
  "api repos/"*"/commits/refs/tags/"*)
    n=$(($(cat "$CASE_DIR/n" 2> /dev/null || echo 0) + 1))
    echo "$n" > "$CASE_DIR/n"
    total=$(wc -l < "$CASE_DIR/tagsha")
    a=$(sed -n "$((n < total ? n : total))p" "$CASE_DIR/tagsha")
    case "$a" in
      REAL) exec "$REAL_GH" "$@" ;;
      FAIL)
        echo '{"message":"No commit found for SHA: refs/tags/x","status":"422"}'
        echo "gh: No commit found for SHA: refs/tags/x (HTTP 422)" >&2
        exit 1
        ;;
      NOSHA) echo '{"message":"something else"}' | answer ;;
      EMPTY) echo '{"sha":""}' | answer ;;
      *) printf '{"sha":"%s","commit":{"message":"m"}}\n' "$a" | answer ;;
    esac
    ;;
  "api --paginate")
    if [ "$(cat "$CASE_DIR/releases")" = FAIL ]; then echo "gh: Server Error (HTTP 502)" >&2; exit 1; fi
    cat "$CASE_DIR/releases"
    ;;
  "release create")
    printf '%s\n' "$@" > "$CASE_DIR/created"
    exit "${CREATE_RC:-0}"
    ;;
  "release edit")
    printf '%s\n' "$@" > "$CASE_DIR/edited"
    exit "${EDIT_RC:-0}"
    ;;
  *) echo "stub gh: unexpected call: $*" >&2; exit 99 ;;
esac
EOF
chmod +x "$T/bin/gh"
mkdir "$T/dist"
echo "0123abcd  axon-x86_64-linux.tar.gz" > "$T/dist/axon-x86_64-linux.tar.gz.sha256"

A=1111111111111111111111111111111111111111
B=2222222222222222222222222222222222222222
fails=0 n=0
# check NAME TAG BUILT TAGSHA_ANSWERS RELEASES WANT_RC(0|nz) WANT_CREATED(yes|no) WANT_TAG_CALLS
#   WANT_RETITLE(no|a substring of the new title) STDERR [extra env...]
# STDERR: substrings separated by '|'; one that starts with '!' must not appear.
check() {
  local name=$1 tag=$2 built=$3 tagsha=$4 releases=$5 want_rc=$6 want_created=$7 want_calls=$8 want_retitle=$9
  local want_err=${10}
  shift 10
  local d="$T/case$n" rc=0 created=no retitled=no calls why='' answers wants w
  n=$((n + 1))
  mkdir "$d"
  read -ra answers <<< "$tagsha"
  printf '%s\n' "${answers[@]}" > "$d/tagsha"
  printf '%s' "$releases" > "$d/releases"
  : > "$d/calls"
  (cd "$T" && env -i PATH="$T/bin:$PATH" HOME="${LIVE_HOME:-$T}" CASE_DIR="$d" REAL_GH="$(command -v gh)" \
    GITHUB_REPOSITORY="${REPO:-HaseebKhalid1507/Axon}" GITHUB_REF_NAME="$tag" BUILT="$built" "$@" \
    bash --noprofile --norc -eo pipefail "$T/step.sh") > "$d/out" 2> "$d/err" || rc=$?
  [ -f "$d/created" ] && created=yes
  [ -f "$d/edited" ] && retitled=yes
  calls=$(grep -c '/commits/refs/tags/' "$d/calls" || true)
  if [ "$want_rc" = 0 ] && [ "$rc" != 0 ]; then why+=" rc $rc, want 0;"; fi
  if [ "$want_rc" = nz ] && [ "$rc" = 0 ]; then why+=" rc 0, want non-zero;"; fi
  [ "$created" = "$want_created" ] || why+=" draft created: $created, want $want_created;"
  [ "$calls" = "$want_calls" ] || why+=" asked the tag's commit $calls times, want $want_calls;"
  if [ "$want_retitle" = no ]; then
    [ "$retitled" = no ] || why+=" retitled the draft, want no;"
  elif [ "$retitled" = no ]; then
    why+=" didn't retitle the draft;"
  elif ! grep -qxF -- "--title" "$d/edited" || ! grep -qF -- "$want_retitle" "$d/edited"; then
    why+=" the new title lacks '$want_retitle';"
  fi
  IFS='|' read -ra wants <<< "$want_err"
  for w in "${wants[@]}"; do
    if [ "${w:0:1}" = '!' ]; then
      if grep -qF -- "${w:1}" "$d/err"; then why+=" stderr has '${w:1}';"; fi
    elif ! grep -qF -- "$w" "$d/err"; then
      why+=" stderr lacks '$w';"
    fi
  done
  if [ "$created" = yes ] && [ "$want_created" = yes ] && ! grep -qF "Built from commit $built" "$d/created"; then
    why+=" the notes don't name the commit;"
  fi
  if [ -z "$why" ]; then
    printf 'ok    %-58s rc=%s draft=%s tag-asks=%s retitled=%s\n' "$name" "$rc" "$created" "$calls" "$retitled"
  else
    printf 'FAIL  %-58s%s\n' "$name" "$why"
    sed 's/^/        stderr: /' "$d/err"
    fails=$((fails + 1))
  fi
}

none='[]'
others='[{"tag_name":"v0.0.9","draft":false,"html_url":"u/v0.0.9"}]'
page2='[{"tag_name":"v0.0.9","draft":false,"html_url":"u/v0.0.9"}][{"tag_name":"v0.1.0","draft":true,"html_url":"u/draft-v0.1.0"}]'

moved='the run for the commit the tag names makes the draft'
rerun='re-run this job'
# The tag names the commit this run built (lightweight or annotated: the endpoint answers the commit either way).
check "tag names the built commit: creates" v0.1.0 $A "$A" "$none" 0 yes 2 no ""
check "other releases only: creates" v0.1.0 $A "$A" "$others" 0 yes 2 no ""
# The tag was moved while the run built: the other run makes the draft, so no "re-run this job".
check "tag moved before the upload: refuses" v0.1.0 $A "$B" "$none" nz no 1 no \
  "v0.1.0 names $B now, but this run built $A|$moved|!$rerun"
# No commit from GitHub: the tag was deleted, or a transient error, when this may be the only run.
check "tag deleted / API error (HTTP 422): refuses, says re-run" v0.1.0 $A "FAIL" "$none" nz no 1 no \
  "can't ask GitHub which commit v0.1.0 names|if v0.1.0 still names $A (a transient error), $rerun|!$moved"
check "API answers without a SHA (null): refuses, says re-run" v0.1.0 $A "NOSHA" "$none" nz no 1 no \
  "GitHub's answer for v0.1.0 isn't a commit: 'null'|$rerun|!$moved"
check "API answers an empty SHA: refuses, says re-run" v0.1.0 $A "EMPTY" "$none" nz no 1 no \
  "GitHub's answer for v0.1.0 isn't a commit: ''|$rerun|!$moved"
# A broken job output or tag: refused before any API call.
check "BUILT empty (no job output): refuses, no API call" v0.1.0 "" "$A" "$none" nz no 0 no "isn't a SHA"
check "BUILT a short SHA: refuses, no API call" v0.1.0 1111111 "$A" "$none" nz no 0 no "isn't a SHA"
check "BUILT with a newline: refuses, no API call" v0.1.0 "$A
$A" "$A" "$none" nz no 0 no "isn't a SHA"
check "tag with '#': refuses, no API call" 'v0.1.0#x' $A "$A" "$none" nz no 0 no "isn't a version"
check "tag with '/': refuses, no API call" 'v0.1.0/../x' $A "$A" "$none" nz no 0 no "isn't a version"
# The tag moved, or went, during the upload: the draft exists. The job retitles it DO NOT PUBLISH and fails.
check "tag moved during the upload: retitles, fails" v0.1.0 $A "$A $B" "$none" nz yes 2 \
  "DO NOT PUBLISH: v0.1.0 moved during the upload" "delete it (gh release delete v0.1.0 --yes|!$rerun"
check "tag deleted during the upload: retitles, fails" v0.1.0 $A "$A FAIL" "$none" nz yes 2 \
  "DO NOT PUBLISH: v0.1.0 unchecked after the upload" \
  "delete the draft (gh release delete v0.1.0 --yes)|publishing it would create v0.1.0 at main's tip"
check "no SHA after the upload: retitles, fails" v0.1.0 $A "$A NOSHA" "$none" nz yes 2 \
  "DO NOT PUBLISH: v0.1.0 unchecked after the upload" "the draft is right: gh release edit v0.1.0 --title 'Axon v0.1.0'"
check "tag moved during the upload, retitle fails: still fails" v0.1.0 $A "$A $B" "$none" nz yes 2 \
  "DO NOT PUBLISH" "couldn't retitle the draft either|delete it (gh release delete v0.1.0 --yes" EDIT_RC=1
# #257's one release per tag, now behind the tag check.
check "a draft for the tag on page 2: refuses" v0.1.0 $A "$A" "$page2" nz no 1 no \
  "a release for v0.1.0 exists already: u/draft-v0.1.0"
check "release listing fails: refuses" v0.1.0 $A "$A" "FAIL" nz no 1 no ""
check "the upload fails: the step fails, no second check" v0.1.0 $A "$A" "$none" nz yes 1 no "" CREATE_RC=1

# The release job's wiring: one release job per tag at a time, never cancelled, and BUILT from the pack job.
n=$((n + 1))
if wiring=$(python3 - "$yml" 2>&1 <<'PY'
import sys
import yaml
jobs = yaml.safe_load(open(sys.argv[1]))["jobs"]
rel, bad = jobs["release"], []
c = rel.get("concurrency")
if not isinstance(c, dict) or "github.ref" not in str(c.get("group", "")):
    bad.append(f"concurrency group not keyed by github.ref: {c!r}")
elif c.get("cancel-in-progress") is not False:
    bad.append(f"cancel-in-progress is {c.get('cancel-in-progress')!r}, want false")
if "pack" not in rel.get("needs", []):
    bad.append(f"needs {rel.get('needs')!r} lacks pack")
if jobs["pack"].get("outputs", {}).get("commit") != "${{ steps.key.outputs.commit }}":
    bad.append("the pack job doesn't output commit from its key step")
step = [st for st in rel["steps"] if st.get("name") == "upload to a draft release for the tag"]
if len(step) != 1 or step[0].get("env", {}).get("BUILT") != "${{ needs.pack.outputs.commit }}":
    bad.append("the draft step's BUILT isn't the pack job's commit")
print("; ".join(bad))
sys.exit(1 if bad else 0)
PY
); then
  printf 'ok    %-58s\n' "release job wiring: concurrency, BUILT"
else
  printf 'FAIL  %-58s %s\n' "release job wiring: concurrency, BUILT" "$wiring"
  fails=$((fails + 1))
fi

if [ "$live" = 1 ]; then
  # The real endpoint. git/git v2.40.0 is an annotated tag: object d4ca2e31..., commit 73876f48...
  # cli/cli v2.40.0 is a lightweight tag: commit 54d56cab...
  export LIVE_HOME=$HOME
  REPO=git/git check "live: annotated tag, BUILT = its commit: creates" v2.40.0 \
    73876f4861cd3d187a4682290ab75c9dccadbc56 REAL "$none" 0 yes 2 no ""
  REPO=git/git check "live: annotated tag, BUILT = the tag object: refuses" v2.40.0 \
    d4ca2e3147b409459955613c152220f4db848ee1 REAL "$none" nz no 1 no "names 73876f4861cd3d187a4682290ab75c9dccadbc56 now"
  REPO=cli/cli check "live: lightweight tag, BUILT = its commit: creates" v2.40.0 \
    54d56cab3a0882b43ac794df59924dc3f93bb75c REAL "$none" 0 yes 2 no ""
  REPO=cli/cli check "live: lightweight tag, another commit: refuses" v2.40.0 $A REAL "$none" nz no 1 no "but this run built $A"
  REPO=HaseebKhalid1507/Axon check "live: a tag that doesn't exist: refuses" v9.9.9 $A REAL "$none" nz no 1 no \
    "can't ask GitHub which commit v9.9.9 names|$rerun"
fi

echo "test_draft_guards: $n cases, $fails failed"
[ "$fails" = 0 ]
