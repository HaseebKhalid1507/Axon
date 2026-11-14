#!/usr/bin/env bash
# Tests the release job's draft step (.github/workflows/release.yml, "upload to a draft release for the tag"):
# its run: block, taken from the YAML and run as the workflow runs it (bash --noprofile --norc -eo pipefail),
# against a stub `gh`. Each case sets what GitHub answers, then checks the exit code, whether a draft was made and
# with exactly which arguments, how often the step asked which commit the tag names, whether it retitled the draft
# (exactly to "Axon <tag>"), and what it said. Then the release job's wiring the step relies on (its concurrency
# group, BUILT). Then scripts/release/publish.sh against the same stub: what it refuses, and the exact publish call.
# No network, unless --live.
#
# Usage: scripts/release/test_draft_guards.sh [--live] [release.yml [publish.sh]]
#   --live  also run the step against the real API's commits/refs/tags endpoint (an annotated tag, a lightweight
#           one; the release listing and `gh release create` stay stubbed), and publish.sh --check against real
#           release listings (it refuses before any edit). Needs gh logged in.
set -euo pipefail

live=0
if [ "${1:-}" = --live ]; then live=1; shift; fi
yml=${1:-.github/workflows/release.yml}
publish=$(realpath "${2:-scripts/release/publish.sh}")
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
# (the release listing's output, FAIL, or REAL). It applies the caller's own --jq expression to the JSON it fakes.
# It appends each create's, edit's and PATCH's (`api -X`: publish.sh's revert) arguments, one per line, so a second
# call shows. From the environment: STUB_PUSH, repos/<repo>'s .permissions.push (default true; FAIL; REAL);
# STUB_AFTER, what repos/<repo>/releases/<id> answers (FAIL, or JSON; by default the release as the recorded edit
# left it: published under the edit's --title and --tag if it had --draft=false and EDIT_RC was 0, else a draft).
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
  "api repos/"*"/releases/"*)
    case ${STUB_AFTER:-} in
      FAIL) echo "gh: Server Error (HTTP 502)" >&2; exit 1 ;;
      "")
        draft=true title= edtag=
        if [ "${EDIT_RC:-0}" = 0 ] && grep -qx -- --draft=false "$CASE_DIR/edited" 2> /dev/null; then
          draft=false
          title=$(grep -x -A1 -- --title "$CASE_DIR/edited" | tail -1)
          edtag=$(grep -x -A1 -- --tag "$CASE_DIR/edited" | tail -1)
        fi
        jq -cn --argjson id "${2##*/}" --argjson d "$draft" --arg n "$title" --arg t "$edtag" \
          '{id: $id, draft: $d, name: $n, tag_name: $t}' | answer
        ;;
      *) printf '%s\n' "$STUB_AFTER" | answer ;;
    esac
    ;;
  "api repos/"*)
    case ${STUB_PUSH:-true} in
      REAL) exec "$REAL_GH" "$@" ;;
      FAIL) echo "gh: Server Error (HTTP 502)" >&2; exit 1 ;;
      *) printf '{"permissions":{"push":%s}}\n' "${STUB_PUSH:-true}" | answer ;;
    esac
    ;;
  "api -X")
    printf '%s\n' "$@" >> "$CASE_DIR/reverted"
    exit "${REVERT_RC:-0}"
    ;;
  "api --paginate")
    if [ "$(cat "$CASE_DIR/releases")" = FAIL ]; then echo "gh: Server Error (HTTP 502)" >&2; exit 1; fi
    if [ "$(cat "$CASE_DIR/releases")" = REAL ]; then exec "$REAL_GH" "$@"; fi
    cat "$CASE_DIR/releases"
    ;;
  "release create")
    printf '%s\n' "$@" >> "$CASE_DIR/created"
    exit "${CREATE_RC:-0}"
    ;;
  "release edit")
    printf '%s\n' "$@" >> "$CASE_DIR/edited"
    exit "${EDIT_RC:-0}"
    ;;
  *) echo "stub gh: unexpected call: $*" >&2; exit 99 ;;
esac
EOF
chmod +x "$T/bin/gh"
mkdir "$T/dist"
TARSHA=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
echo "$TARSHA  axon-x86_64-linux.tar.gz" > "$T/dist/axon-x86_64-linux.tar.gz.sha256"

A=1111111111111111111111111111111111111111
B=2222222222222222222222222222222222222222
fails=0 n=0

# What's wrong with a create's arguments (the stub's record, one per line), as " problem;" items. Want the tag, then
# exactly --repo REPO, --draft, --verify-tag, --title "DO NOT PUBLISH: Axon TAG (unchecked)" and --notes, in any
# order, then exactly the tarball, its .sha256 and install.sh. A second create shows up as extra files. The notes
# must say "SHA-256 of the tarball: <the .sha256's hash>." and "Built from commit <BUILT>." exactly once each:
# publish.sh reads both.
create_problems() {
  local tag=$1 repo=$2 built=$4 k flags='' files='' a line c
  mapfile -t a < "$3"
  [ "${a[0]:-} ${a[1]:-} ${a[2]:-}" = "release create $tag" ] || printf " the create starts '%s';" "${a[*]:0:3}"
  for ((k = 3; k < ${#a[@]}; k++)); do
    case ${a[k]} in
      --repo)
        k=$((k + 1)) flags+=" --repo"
        [ "${a[k]:-}" = "$repo" ] || printf " create --repo '%s', want %s;" "${a[k]:-}" "$repo"
        ;;
      --title)
        k=$((k + 1)) flags+=" --title"
        [ "${a[k]:-}" = "DO NOT PUBLISH: Axon $tag (unchecked)" ] || printf " create --title '%s';" "${a[k]:-}"
        ;;
      --notes)
        k=$((k + 1)) flags+=" --notes"
        for line in "SHA-256 of the tarball: $TARSHA." "Built from commit $built."; do
          c=$(grep -oF -- "$line" <<< "${a[k]:-}" | wc -l)
          [ "$c" = 1 ] || printf " the notes say '%s' %s times, want once;" "$line" "$c"
        done
        ;;
      -*) flags+=" ${a[k]}" ;;
      *) files+=" ${a[k]}" ;;
    esac
  done
  flags=$(tr ' ' '\n' <<< "$flags" | grep . | LC_ALL=C sort | paste -sd' ' -)
  [ "$flags" = "--draft --notes --repo --title --verify-tag" ] || printf " create's flags are '%s';" "$flags"
  [ "$files" = " dist/axon-x86_64-linux.tar.gz dist/axon-x86_64-linux.tar.gz.sha256 dist/install.sh" ] ||
    printf " create's files are '%s';" "${files# }"
}

# What's wrong with stderr ($1), as " problem;" items. $2: substrings separated by '|'; one that starts with '!'
# must not appear.
err_problems() {
  local wants w
  IFS='|' read -ra wants <<< "$2"
  for w in "${wants[@]}"; do
    if [ "${w:0:1}" = '!' ]; then
      if grep -qF -- "${w:1}" "$1"; then printf " stderr has '%s';" "${w:1}"; fi
    elif ! grep -qF -- "$w" "$1"; then
      printf " stderr lacks '%s';" "$w"
    fi
  done
}

# check NAME TAG BUILT TAGSHA_ANSWERS RELEASES WANT_RC(0|nz) WANT_CREATED(yes|no) WANT_TAG_CALLS
#   WANT_RETITLE(yes|no) STDERR [extra env...]
# WANT_RETITLE yes: exactly one edit, exactly `release edit TAG --repo REPO --title "Axon TAG (checked; publish with
# scripts/release/publish.sh)"`.
# STDERR: see err_problems. A created draft must have exactly the arguments create_problems wants.
check() {
  local name=$1 tag=$2 built=$3 tagsha=$4 releases=$5 want_rc=$6 want_created=$7 want_calls=$8 want_retitle=$9
  local want_err=${10}
  shift 10
  local d="$T/case$n" rc=0 created=no retitled=no calls why='' answers repo=${REPO:-HaseebKhalid1507/Axon}
  n=$((n + 1))
  mkdir "$d"
  read -ra answers <<< "$tagsha"
  printf '%s\n' "${answers[@]}" > "$d/tagsha"
  printf '%s' "$releases" > "$d/releases"
  : > "$d/calls"
  (cd "$T" && env -i PATH="$T/bin:$PATH" HOME="${LIVE_HOME:-$T}" CASE_DIR="$d" REAL_GH="$(command -v gh)" \
    GITHUB_REPOSITORY="$repo" GITHUB_REF_NAME="$tag" BUILT="$built" "$@" \
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
  elif ! diff -q <(printf '%s\n' release edit "$tag" --repo "$repo" --title "$(checked "$tag")") "$d/edited" > /dev/null; then
    why+=" the edits aren't exactly 'release edit $tag --repo $repo --title $(checked "$tag")': '$(tr '\n' ' ' < "$d/edited")';"
  fi
  [ "$created" = no ] || why+=$(create_problems "$tag" "$repo" "$d/created" "$built")
  why+=$(err_problems "$d/err" "$want_err")
  if [ -z "$why" ]; then
    printf 'ok    %-58s rc=%s draft=%s tag-asks=%s retitled=%s\n' "$name" "$rc" "$created" "$calls" "$retitled"
  else
    printf 'FAIL  %-58s%s\n' "$name" "$why"
    sed 's/^/        stderr: /' "$d/err"
    fails=$((fails + 1))
  fi
}

none='[]'
checked() { echo "Axon $1 (checked; publish with scripts/release/publish.sh)"; }
others='[{"tag_name":"v0.0.9","draft":false,"html_url":"u/v0.0.9"}]'
page2='[{"tag_name":"v0.0.9","draft":false,"html_url":"u/v0.0.9"}][{"tag_name":"v0.1.0","draft":true,"html_url":"u/draft-v0.1.0"}]'

moved='the run for the commit the tag names makes the draft'
rerun='re-run this job'
# The tag names the commit this run built (lightweight or annotated: the endpoint answers the commit either way):
# a draft created "DO NOT PUBLISH: ... (unchecked)", retitled "Axon v0.1.0 (checked; ...)" after the second check.
check "tag names the built commit: creates, then retitles" v0.1.0 $A "$A" "$none" 0 yes 2 yes ""
check "other releases only: creates, then retitles" v0.1.0 $A "$A" "$others" 0 yes 2 yes ""
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
# The tag moved, or went, during the upload: the draft exists and keeps its DO NOT PUBLISH title (no edit at
# all), and the job fails. None of these may say "re-run this job": the one-release check would refuse it.
check "tag moved during the upload: stays DO NOT PUBLISH, fails" v0.1.0 $A "$A $B" "$none" nz yes 2 no \
  "stays titled DO NOT PUBLISH|delete it (gh release delete v0.1.0 --yes|!$rerun"
check "tag deleted during the upload: stays DO NOT PUBLISH, fails" v0.1.0 $A "$A FAIL" "$none" nz yes 2 no \
  "stays titled DO NOT PUBLISH|delete the draft (gh release delete v0.1.0 --yes)|publishing it would create v0.1.0 at main's tip|!$rerun"
check "no SHA after the upload: stays DO NOT PUBLISH, fails" v0.1.0 $A "$A NOSHA" "$none" nz yes 2 no \
  "stays titled DO NOT PUBLISH|the draft is right: gh release edit v0.1.0 --repo HaseebKhalid1507/Axon --title '$(checked v0.1.0)'|!$rerun"
# The checks pass but the retitle fails: the draft is right, still titled DO NOT PUBLISH; the job fails.
check "the final retitle fails: fails, says how to retitle" v0.1.0 $A "$A" "$none" nz yes 2 yes \
  "couldn't be retitled, so it is still titled DO NOT PUBLISH|gh release edit v0.1.0 --repo HaseebKhalid1507/Axon --title '$(checked v0.1.0)'|!$rerun" \
  EDIT_RC=1
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

# scripts/release/publish.sh. pcheck NAME "ARGS" TAGSHA_ANSWERS RELEASES WANT_RC(0|nz|N) WANT_EDIT(yes|no)
#   WANT_REVERT(yes|no) WANT_TAG_CALLS(N|none) OUTPUT [extra env...]
# ARGS: split on spaces; '' is an empty argument. WANT_TAG_CALLS none: no gh call at all. OUTPUT: as STDERR
# (err_problems), on stdout and stderr together.
# WANT_EDIT yes: exactly one edit, exactly `release edit TAG --repo REPO --title "Axon TAG" --tag TAG --verify-tag
# --draft=false --prerelease=false`: the final title and the publish in one request (--verify-tag checks only
# alongside --tag, gh 2.102.1). no: no edit at all.
# WANT_REVERT yes: exactly one PATCH, exactly `api -X PATCH repos/REPO/releases/77 -F draft=true -f "name=DO NOT
# PUBLISH: Axon TAG (unpublished by publish.sh)"`. no: none.
pcheck() {
  local name=$1 args=$2 tagsha=$3 releases=$4 want_rc=$5 want_edit=$6 want_revert=$7 want_calls=$8 want_err=$9
  shift 9
  local d="$T/pcase$n" rc=0 edited=no reverted=no calls why='' answers argv k tag repo=${REPO:-HaseebKhalid1507/Axon}
  n=$((n + 1))
  mkdir "$d"
  read -ra answers <<< "$tagsha"
  printf '%s\n' "${answers[@]}" > "$d/tagsha"
  printf '%s' "$releases" > "$d/releases"
  : > "$d/calls"
  read -ra argv <<< "$args"
  for ((k = 0; k < ${#argv[@]}; k++)); do [ "${argv[k]}" != "''" ] || argv[k]=; done
  tag=
  for k in "${argv[@]}"; do [[ "$k" != v* ]] || tag=$k; done
  (cd "$T" && env -i PATH="$T/bin:$PATH" HOME="${LIVE_HOME:-$T}" CASE_DIR="$d" REAL_GH="$(command -v gh)" \
    GH_REPO="$repo" "$@" bash --noprofile --norc "$publish" "${argv[@]}") > "$d/out" 2> "$d/err" || rc=$?
  [ -f "$d/edited" ] && edited=yes
  [ -f "$d/reverted" ] && reverted=yes
  calls=$(grep -c '/commits/refs/tags/' "$d/calls" || true)
  case $want_rc in
    nz) [ "$rc" != 0 ] || why+=" rc 0, want non-zero;" ;;
    *) [ "$rc" = "$want_rc" ] || why+=" rc $rc, want $want_rc;" ;;
  esac
  if [ "$want_calls" = none ]; then
    [ ! -s "$d/calls" ] || why+=" called gh: '$(head -1 "$d/calls")', want no call;"
    calls=none
  else
    [ "$calls" = "$want_calls" ] || why+=" asked the tag's commit $calls times, want $want_calls;"
  fi
  if [ "$want_edit" = no ]; then
    [ "$edited" = no ] || why+=" edited (published): '$(tr '\n' ' ' < "$d/edited")', want no edit;"
  elif [ "$edited" = no ]; then
    why+=" didn't publish;"
  elif ! diff -q <(printf '%s\n' release edit "$tag" --repo "$repo" --title "Axon $tag" --tag "$tag" --verify-tag \
    --draft=false --prerelease=false) "$d/edited" > /dev/null; then
    why+=" the publish isn't exactly 'release edit $tag --repo $repo --title Axon $tag --tag $tag --verify-tag --draft=false --prerelease=false': '$(tr '\n' ' ' < "$d/edited")';"
  fi
  if [ "$want_revert" = no ]; then
    [ "$reverted" = no ] || why+=" reverted: '$(tr '\n' ' ' < "$d/reverted")', want no revert;"
  elif [ "$reverted" = no ]; then
    why+=" didn't turn it back into a draft;"
  elif ! diff -q <(printf '%s\n' api -X PATCH "repos/$repo/releases/77" -F draft=true \
    -f "name=DO NOT PUBLISH: Axon $tag (unpublished by publish.sh)") "$d/reverted" > /dev/null; then
    why+=" the revert isn't exactly 'api -X PATCH repos/$repo/releases/77 -F draft=true -f name=DO NOT PUBLISH: Axon $tag (unpublished by publish.sh)': '$(tr '\n' ' ' < "$d/reverted")';"
  fi
  cat "$d/out" "$d/err" > "$d/all"
  why+=$(err_problems "$d/all" "$want_err")
  if [ -z "$why" ]; then
    printf 'ok    %-66s rc=%s edit=%s revert=%s tag-asks=%s\n' "publish.sh: $name" "$rc" "$edited" "$reverted" "$calls"
  else
    printf 'FAIL  %-66s%s\n' "publish.sh: $name" "$why"
    sed 's/^/        output: /' "$d/all"
    fails=$((fails + 1))
  fi
}

S=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
S2=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
# A release in the listing, id 77: rel TAG DRAFT TITLE NOTES ASSET... (an asset NAME:STATE isn't "uploaded"). The
# tarball's digest is $DIGEST (JSON; default "sha256:$S"), the other assets' a fixed one.
rel() {
  local tag=$1 draft=$2 title=$3 notes=$4
  shift 4
  jq -cn --arg t "$tag" --argjson d "$draft" --arg n "$title" --arg b "$notes" --argjson dg "${DIGEST:-\"sha256:$S\"}" \
    --args '{id: 77, tag_name: $t, draft: $d, name: $n, body: $b, html_url: ("u/" + $t),
      assets: [$ARGS.positional[] | split(":") | {name: .[0], state: (.[1] // "uploaded"),
        digest: (if .[0] == "axon-x86_64-linux.tar.gz" then $dg else "sha256:" + ("d" * 64) end)}]}' "$@"
}
notes() { echo "Linux x86-64. Install: \`curl ... | sh\`. $*"; }
three=(axon-x86_64-linux.tar.gz axon-x86_64-linux.tar.gz.sha256 install.sh)
CHK=$(checked v0.1.0)
good=$(rel v0.1.0 true "$CHK" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]}")
old=$(rel v0.0.9 false "Axon v0.0.9" "$(notes "SHA-256 of the tarball: $S2. Built from commit $B.")" "${three[@]}")
# good, but with other notes / another title.
with_notes() { rel v0.1.0 true "$CHK" "$(notes "$1")" "${three[@]}"; }
titled() { rel v0.1.0 true "$1" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]}"; }
pub="v0.1.0 published: built from $A, tarball SHA-256 $S; checked again after publishing"
nope='publish: nothing published'
was='v0.1.0 WAS PUBLISHED, but right after'
again='it is a draft again, titled DO NOT PUBLISH'
# A PATH without jq: the stub gh and bash only.
mkdir "$T/nojq"
ln -s "$T/bin/gh" "$T/nojq/gh"
ln -s "$(command -v bash)" "$T/nojq/bash"

# Publishing: only with the draft's SHA-256, the final title set in the publish itself; checked again after.
pcheck "one checked draft, its SHA-256: publishes" "v0.1.0 $S" "$A" "[$good]" 0 yes no 2 "$pub|!$nope|!WAS"
pcheck "the draft on page 2 of the listing: publishes" "v0.1.0 $S" "$A" "[$old][$good]" 0 yes no 2 "$pub"
pcheck "the SHA-256 in capitals: publishes" "v0.1.0 ${S^^}" "$A" "[$good]" 0 yes no 2 "$pub"
pcheck "--check, no SHA-256: prints it, publishes nothing" "--check v0.1.0" "$A" "[$good]" 0 no no 1 \
  "v0.1.0 ok: one draft titled '$CHK'|the tarball's SHA-256 is $S (its notes and GitHub's digest)|publish: scripts/release/publish.sh v0.1.0 $S|--check: nothing published|!$nope|!and yours"
pcheck "--check, its SHA-256: ok" "--check v0.1.0 $S" "$A" "[$good]" 0 no no 1 "GitHub's digest, and yours|!$nope"
pcheck "--check, another SHA-256: refuses" "--check v0.1.0 $S2" "$A" "[$good]" nz no no 0 "not $S2, the one you checked|$nope"
# The SHA-256 argument: required to publish, checked before any API call.
pcheck "publishing without a SHA-256: usage, no gh call" "v0.1.0" "$A" "[$good]" 2 no no none "usage|SHA256"
pcheck "an empty SHA-256: usage, no gh call" "v0.1.0 ''" "$A" "[$good]" 2 no no none "usage"
pcheck "a 63-digit SHA-256: refuses, no gh call" "v0.1.0 ${S:1}" "$A" "[$good]" nz no no none "isn't a SHA-256|$nope"
pcheck "three arguments: usage, no gh call" "v0.1.0 $S $S" "$A" "[$good]" 2 no no none "usage"
# shady's case: checks ran on A's tarball, the tag moved to B, B's run made a new draft. Everything else passes.
newer=$(DIGEST="\"sha256:$S2\"" rel v0.1.0 true "$CHK" "$(notes "SHA-256 of the tarball: $S2. Built from commit $B.")" "${three[@]}")
pcheck "a newer draft than the checked tarball: refuses" "v0.1.0 $S" "$B" "[$newer]" nz no no 0 \
  "the draft's tarball is $S2 (its notes and GitHub's digest), not $S, the one you checked|Run the checks on this draft's tarball|$nope"
# The tarball's digest: GitHub's, which must be the notes'.
pcheck "the tarball re-uploaded (digest isn't the notes'): refuses" "v0.1.0 $S" "$A" \
  "[$(DIGEST="\"sha256:$S2\"" with_notes "SHA-256 of the tarball: $S. Built from commit $A.")]" nz no no 0 \
  "GitHub's digest $S2, but its notes say $S|the tarball was replaced|$nope"
pcheck "--check, the tarball re-uploaded: refuses" "--check v0.1.0" "$A" \
  "[$(DIGEST="\"sha256:$S2\"" with_notes "SHA-256 of the tarball: $S. Built from commit $A.")]" nz no no 0 "but its notes say $S"
pcheck "no digest (null): refuses" "v0.1.0 $S" "$A" \
  "[$(DIGEST=null with_notes "SHA-256 of the tarball: $S. Built from commit $A.")]" nz no no 0 "no SHA-256 digest|('none')"
pcheck "a 63-digit digest: refuses" "v0.1.0 $S" "$A" \
  "[$(DIGEST="\"sha256:${S:1}\"" with_notes "SHA-256 of the tarball: $S. Built from commit $A.")]" nz no no 0 "no SHA-256 digest"
# The notes' two lines, each exactly once.
pcheck "notes without the tarball SHA-256: refuses" "v0.1.0 $S" "$A" "[$(with_notes "Built from commit $A.")]" \
  nz no no 0 "give 0 tarball SHA-256s|keep it, exactly once, when you edit the notes"
pcheck "notes with two tarball SHA-256s: refuses" "v0.1.0 $S" "$A" \
  "[$(with_notes "SHA-256 of the tarball: $S. SHA-256 of the tarball: $S2. Built from commit $A.")]" nz no no 0 "give 2 tarball"
pcheck "notes with a 63-digit tarball SHA-256: refuses" "v0.1.0 $S" "$A" \
  "[$(with_notes "SHA-256 of the tarball: ${S:1}. Built from commit $A.")]" nz no no 0 "which isn't a SHA-256"
pcheck "notes name no commit: refuses" "v0.1.0 $S" "$A" "[$(with_notes "SHA-256 of the tarball: $S.")]" nz no no 0 \
  "name 0 commits|keep it, exactly once, when you edit the notes"
pcheck "notes name two commits: refuses" "v0.1.0 $S" "$A" \
  "[$(with_notes "SHA-256 of the tarball: $S. Built from commit $A. Built from commit $B.")]" nz no no 0 "name 2 commits"
pcheck "notes name a 39-digit commit: refuses" "v0.1.0 $S" "$A" \
  "[$(with_notes "SHA-256 of the tarball: $S. Built from commit ${A:1}.")]" nz no no 0 "isn't a commit"
# The title: only the one the release job sets after its last check.
pcheck "titled 'Axon v0.1.0' (the final title, unpublished): refuses" "v0.1.0 $S" "$A" "[$(titled "Axon v0.1.0")]" \
  nz no no 0 "titled 'Axon v0.1.0', not '$CHK'|$nope"
pcheck "titled DO NOT PUBLISH (unchecked): refuses" "v0.1.0 $S" "$A" \
  "[$(titled "DO NOT PUBLISH: Axon v0.1.0 (unchecked)")]" nz no no 0 \
  "titled 'DO NOT PUBLISH: Axon v0.1.0 (unchecked)', not '$CHK'|its log says what to do|$nope"
pcheck "the checked title plus a space: refuses" "v0.1.0 $S" "$A" "[$(titled "$CHK ")]" nz no no 0 "not '$CHK'"
# The listing.
pcheck "no release for the tag, push access: says none was made" "v0.1.0 $S" "$A" "[$old]" nz no no 0 \
  "no release for v0.1.0 in HaseebKhalid1507/Axon: the release job hasn't made the draft|!push access|$nope"
pcheck "no release for the tag, no push access: says so" "v0.1.0 $S" "$A" "[$old]" nz no no 0 \
  "the account has no push access to HaseebKhalid1507/Axon, and GitHub lists drafts only for accounts with it|$nope" \
  STUB_PUSH=false
pcheck "no release for the tag, access unknown: names both" "v0.1.0 $S" "$A" "[$old]" nz no no 0 \
  "the release job hasn't made the draft, or the account has no push access|$nope" STUB_PUSH=FAIL
pcheck "two drafts for the tag: refuses" "v0.1.0 $S" "$A" "[$good,$good]" nz no no 0 "2 releases for v0.1.0"
pcheck "published already: refuses" "v0.1.0 $S" "$A" \
  "[$(rel v0.1.0 false "Axon v0.1.0" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]}")]" \
  nz no no 0 "published already"
pcheck "install.sh missing: refuses" "v0.1.0 $S" "$A" \
  "[$(rel v0.1.0 true "$CHK" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]:0:2}")]" \
  nz no no 0 "the draft's assets are"
pcheck "an extra asset: refuses" "v0.1.0 $S" "$A" \
  "[$(rel v0.1.0 true "$CHK" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]}" extra.bin)]" \
  nz no no 0 "extra.bin"
pcheck "an asset not uploaded: refuses" "v0.1.0 $S" "$A" \
  "[$(rel v0.1.0 true "$CHK" "$(notes "SHA-256 of the tarball: $S. Built from commit $A.")" "${three[@]:0:2}" install.sh:open)]" \
  nz no no 0 "install.sh (open)"
pcheck "a bad tag: refuses, no gh call" "v0.1.0/../x $S" "$A" "[$good]" nz no no none "isn't a version tag"
pcheck "the listing fails: refuses" "v0.1.0 $S" "$A" "FAIL" nz no no 0 "can't list|$nope"
pcheck "no tag: usage, no gh call" "" "$A" "[$good]" 2 no no none "usage"
pcheck "jq not on PATH: refuses, no gh call" "v0.1.0 $S" "$A" "[$good]" nz no no none "needs jq|$nope" PATH="$T/nojq"
# The tag check runs last, right before the publish.
pcheck "tag moved: refuses, says delete and re-run" "v0.1.0 $S" "$B" "[$good]" nz no no 1 \
  "v0.1.0 names $B now, but the draft was built from $A|gh release delete v0.1.0 --repo HaseebKhalid1507/Axon --yes|re-run it|$nope"
pcheck "tag gone (HTTP 422): refuses, says don't publish" "v0.1.0 $S" "FAIL" "[$good]" nz no no 1 \
  "GitHub gave no commit for v0.1.0|Don't publish|create v0.1.0 at main's tip|$nope"
pcheck "a non-SHA answer: refuses" "v0.1.0 $S" "NOSHA" "[$good]" nz no no 1 "isn't a commit: 'null'|$nope"
pcheck "--check, tag moved: refuses" "--check v0.1.0" "$B" "[$good]" nz no no 1 "v0.1.0 names $B now|$nope"
# After the publish: a miss, or an answer that can't be read, turns it back into a draft and fails.
pcheck "tag moved right after the publish: back to a draft, fails" "v0.1.0 $S" "$A $B" "[$good]" nz yes yes 2 \
  "$was: v0.1.0 names $B now, not $A|$again|gh release delete v0.1.0 --repo HaseebKhalid1507/Axon --yes|git push origin :refs/tags/v0.1.0|!$nope|!published: built"
pcheck "tag gone right after the publish: back to a draft, fails" "v0.1.0 $S" "$A FAIL" "[$good]" nz yes yes 2 \
  "$was: GitHub gave no commit for v0.1.0|$again|retitle it '$CHK' and run publish.sh again|!$nope|!published: built"
pcheck "the re-read fails: back to a draft, fails" "v0.1.0 $S" "$A" "[$good]" nz yes yes 2 \
  "$was: the release (id 77) couldn't be re-read|$again|!published: built" STUB_AFTER=FAIL
pcheck "the re-read shows another title: back to a draft, fails" "v0.1.0 $S" "$A" "[$good]" nz yes yes 2 \
  "$was: the release now has the tag 'v0.1.0' and the title 'x'|$again" \
  STUB_AFTER='{"id":77,"draft":false,"tag_name":"v0.1.0","name":"x"}'
pcheck "the re-read shows another tag: back to a draft, fails" "v0.1.0 $S" "$A" "[$good]" nz yes yes 2 \
  "$was: the release now has the tag 'v0.1.1'|$again" \
  STUB_AFTER='{"id":77,"draft":false,"tag_name":"v0.1.1","name":"Axon v0.1.0"}'
pcheck "the re-read shows draft null: back to a draft, fails" "v0.1.0 $S" "$A" "[$good]" nz yes yes 2 \
  "$was: the release (id 77) couldn't be re-read|$again" \
  STUB_AFTER='{"id":77,"tag_name":"v0.1.0","name":"Axon v0.1.0"}'
pcheck "the revert fails: says to do it by hand, fails" "v0.1.0 $S" "$A $B" "[$good]" nz yes yes 2 \
  "$was|AND IT COULDN'T BE TURNED BACK INTO A DRAFT. Do it now: gh api -X PATCH repos/HaseebKhalid1507/Axon/releases/77 -F draft=true|!$again" \
  REVERT_RC=1
pcheck "the publish fails, still a draft: nothing published, no revert" "v0.1.0 $S" "$A" "[$good]" nz yes no 2 \
  "still a draft after gh release edit (exit code 1)|$nope|!WAS PUBLISHED" EDIT_RC=1

if [ "$live" = 1 ]; then
  # The real endpoint. git/git v2.40.0 is an annotated tag: object d4ca2e31..., commit 73876f48...
  # cli/cli v2.40.0 is a lightweight tag: commit 54d56cab...
  export LIVE_HOME=$HOME
  REPO=git/git check "live: annotated tag, BUILT = its commit: creates" v2.40.0 \
    73876f4861cd3d187a4682290ab75c9dccadbc56 REAL "$none" 0 yes 2 yes ""
  REPO=git/git check "live: annotated tag, BUILT = the tag object: refuses" v2.40.0 \
    d4ca2e3147b409459955613c152220f4db848ee1 REAL "$none" nz no 1 no "names 73876f4861cd3d187a4682290ab75c9dccadbc56 now"
  REPO=cli/cli check "live: lightweight tag, BUILT = its commit: creates" v2.40.0 \
    54d56cab3a0882b43ac794df59924dc3f93bb75c REAL "$none" 0 yes 2 yes ""
  REPO=cli/cli check "live: lightweight tag, another commit: refuses" v2.40.0 $A REAL "$none" nz no 1 no "but this run built $A"
  REPO=HaseebKhalid1507/Axon check "live: a tag that doesn't exist: refuses" v9.9.9 $A REAL "$none" nz no 1 no \
    "can't ask GitHub which commit v9.9.9 names|$rerun"
  # publish.sh --check against real release listings and permissions (read-only: each refuses before the tag check).
  REPO=HaseebKhalid1507/Axon pcheck "live: no release, push access: says none was made" "--check v9.9.9" REAL REAL \
    nz no no 0 "no release for v9.9.9 in HaseebKhalid1507/Axon: the release job hasn't made the draft|$nope" STUB_PUSH=REAL
  REPO=rhysd/actionlint pcheck "live: no release, no push access: says so" "--check v9.9.9" REAL REAL nz no no 0 \
    "the account has no push access to rhysd/actionlint|$nope" STUB_PUSH=REAL
  REPO=rhysd/actionlint pcheck "live: a published release: refuses" "--check v1.7.12" REAL REAL nz no no 0 \
    "the release for v1.7.12 is published already: https://github.com/rhysd/actionlint/releases/tag/v1.7.12|$nope"
fi

echo "test_draft_guards: $n cases, $fails failed"
[ "$fails" = 0 ]
