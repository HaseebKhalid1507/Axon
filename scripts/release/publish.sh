#!/usr/bin/env bash
# Publishes the draft release for a tag, right after checking it (docs/releasing.md, "Checking and publishing the
# draft", step 5).
#
# Usage: scripts/release/publish.sh vX.Y.Z SHA256         check, then publish
#        scripts/release/publish.sh --check vX.Y.Z [SHA256] check only, and print the tarball's SHA-256
# SHA256: the tarball's SHA-256 as written next to the hand checks' results. Publishing needs it, so the script
# publishes only the bytes those checks ran on: a draft made again after them (the tag moved and its new run made
# one, or the tarball was re-uploaded) has another SHA-256 and is refused.
# GH_REPO: OWNER/REPO (default HaseebKhalid1507/Axon). Needs jq, and gh logged in with push access: GitHub lists
# drafts only for it.
#
# It publishes only if all of these hold, checked in this order:
# - exactly one release for the tag, a draft, titled exactly "Axon vX.Y.Z (checked; publish with
#   scripts/release/publish.sh)": the release job sets that title only after its last tag check passed;
# - its notes give one commit ("Built from commit <sha>") and one tarball SHA-256 ("SHA-256 of the tarball: <hex>");
# - its assets are exactly the tarball, its .sha256 and install.sh, all uploaded;
# - the tarball's digest, which GitHub computes when the file is uploaded, is the notes' SHA-256, and so is SHA256;
# - the tag names the notes' commit now (commits/refs/tags/<tag>, which peels an annotated tag). This runs last,
#   right before the publish, so a tag moved during the hours of hand checks is caught.
# Then one request publishes it under its final title: `gh release edit vX.Y.Z --title "Axon vX.Y.Z" --tag vX.Y.Z
# --verify-tag --draft=false --prerelease=false`. A full release, the kind GitHub's `latest` (install.sh) points
# at. In gh 2.102.1 `--verify-tag` checks that the tag exists only when `--tag` is given too; without it,
# publishing a draft whose tag is gone creates the tag at main's tip.
#
# After it: the tag's commit again, and the release re-read by its id. Between the last check and the publish, gh
# takes 1-2 s (its own lookups, then the PATCH), and GitHub has no "publish only if the tag names X": a tag moved
# or deleted in that window goes out wrong. On a miss, or an answer the script can't read, it turns the release
# back into a draft titled DO NOT PUBLISH and fails, so a wrong release is public for seconds. Only a tag ruleset
# closes the window (docs/releasing.md, step 5).
set -euo pipefail

usage() {
  echo "usage: scripts/release/publish.sh vX.Y.Z SHA256 | --check vX.Y.Z [SHA256]" >&2
  echo "  SHA256: the tarball's, as written next to the hand checks' results (--check prints it)" >&2
  exit 2
}
check_only=0
if [ "${1:-}" = --check ]; then check_only=1; shift; fi
case "$check_only $#" in
  "0 2" | "1 1" | "1 2") ;;
  *) usage ;;
esac
tag=$1
sha=${2:-}
[ "$check_only" = 1 ] || [ -n "$sha" ] || usage
sha=${sha,,}
repo=${GH_REPO:-HaseebKhalid1507/Axon}
refuse() {
  echo "publish: $*" >&2
  echo "publish: nothing published" >&2
  exit 1
}

[[ "$tag" =~ ^v[0-9][0-9A-Za-z.+-]*$ ]] || refuse "'$tag' isn't a version tag"
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || refuse "GH_REPO '$repo' isn't OWNER/REPO"
[ -z "$sha" ] || [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || refuse "'$2' isn't a SHA-256 (64 hex digits)"
for tool in gh jq; do
  command -v "$tool" > /dev/null || refuse "needs $tool, which isn't on PATH"
done
checked="Axon $tag (checked; publish with scripts/release/publish.sh)"

# The listing, not `gh release view`: it shows every release for the tag (GitHub allows several drafts for one),
# and with push access it includes drafts. --paginate may print one JSON array per page.
listing=$(gh api --paginate "repos/$repo/releases?per_page=100") || refuse "can't list $repo's releases"
rels=$(jq -c --arg tag "$tag" '.[] | select(.tag_name == $tag)' <<< "$listing") ||
  refuse "can't read $repo's release listing"
count=$(grep -c . <<< "$rels" || true)
if [ "$count" = 0 ]; then
  # Without push access the listing hides drafts, so "none" may only mean "none this account can see".
  push=$(gh api "repos/$repo" --jq .permissions.push 2> /dev/null || true)
  case $push in
    true) refuse "no release for $tag in $repo: the release job hasn't made the draft, or it was deleted" ;;
    false) refuse "no release for $tag that gh's account can see: the account has no push access to $repo, and GitHub lists drafts only for accounts with it (gh auth status)" ;;
    *) refuse "no release for $tag that gh's account can see: the release job hasn't made the draft, or the account has no push access to $repo, which drafts are listed for (gh auth status)" ;;
  esac
fi
[ "$count" = 1 ] || refuse "$count releases for $tag, want one draft"
[ "$(jq -r .draft <<< "$rels")" = true ] || refuse "the release for $tag is published already: $(jq -r .html_url <<< "$rels")"
title=$(jq -r .name <<< "$rels")
[ "$title" = "$checked" ] || refuse "the draft is titled '$title', not '$checked', which the release job sets once its last tag check passes. A DO NOT PUBLISH title: the job didn't confirm the tag after the upload, or didn't finish (cancelled, failed partway); its log says what to do"
id=$(jq -r .id <<< "$rels")
[[ "$id" =~ ^[0-9]+$ ]] || refuse "the draft's id '$id' isn't a number"

# The notes' two lines, which the release job writes. Each exactly once: a second could name another build.
body=$(jq -r '.body // ""' <<< "$rels")
commits=$(grep -oE 'Built from commit [0-9a-f]+' <<< "$body" | cut -d' ' -f4 || true)
n=$(grep -c . <<< "$commits" || true)
[ "$n" = 1 ] || refuse "the draft's notes name $n commits ('Built from commit <sha>'), want one. The release job writes that line: keep it, exactly once, when you edit the notes"
built=$commits
[[ "$built" =~ ^[0-9a-f]{40}$ ]] || refuse "the draft's notes say 'Built from commit $built', which isn't a commit"
shas=$(grep -oE 'SHA-256 of the tarball: [0-9a-f]+' <<< "$body" | cut -d' ' -f5 || true)
n=$(grep -c . <<< "$shas" || true)
[ "$n" = 1 ] || refuse "the draft's notes give $n tarball SHA-256s ('SHA-256 of the tarball: <hex>'), want one. The release job writes that line: keep it, exactly once, when you edit the notes"
noted=$shas
[[ "$noted" =~ ^[0-9a-f]{64}$ ]] || refuse "the draft's notes say 'SHA-256 of the tarball: $noted', which isn't a SHA-256"

assets=$(jq -r '[.assets[] | .name + (if .state == "uploaded" then "" else " (" + (.state | tostring) + ")" end)] | sort | join(", ")' <<< "$rels")
want_assets="axon-x86_64-linux.tar.gz, axon-x86_64-linux.tar.gz.sha256, install.sh"
[ "$assets" = "$want_assets" ] || refuse "the draft's assets are '$assets', want exactly $want_assets (uploaded)"

# The tarball itself: GitHub's digest of the uploaded file, then the SHA-256 the hand checks ran on.
digest=$(jq -r '.assets[] | select(.name == "axon-x86_64-linux.tar.gz") | .digest // "none"' <<< "$rels")
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  refuse "GitHub gives no SHA-256 digest for the draft's tarball ('$digest'), so it can't be tied to the one you checked"
[ "${digest#sha256:}" = "$noted" ] ||
  refuse "the draft's tarball has GitHub's digest ${digest#sha256:}, but its notes say $noted: the tarball was replaced after the release job made the draft. Delete the draft (gh release delete $tag --repo $repo --yes), then re-run the tag run's release job"
if [ -n "$sha" ] && [ "$sha" != "$noted" ]; then
  refuse "the draft's tarball is $noted (its notes and GitHub's digest), not $sha, the one you checked: the draft was made again after your checks (the tag moved, or the tarball was re-uploaded). Run the checks on this draft's tarball, then publish with its SHA-256"
fi

# Last, right before the publish: the tag names the commit the draft was built from.
now=$(gh api "repos/$repo/commits/refs/tags/$tag" --jq .sha) ||
  refuse "GitHub gave no commit for $tag: the tag is gone, or an API error. Don't publish: the draft's target is main, so publishing it with the tag gone would create $tag at main's tip. If the tag is gone, delete the draft (gh release delete $tag --repo $repo --yes)"
[[ "$now" =~ ^[0-9a-f]{40}$ ]] || refuse "GitHub's answer for $tag isn't a commit: '$now'"
[ "$now" = "$built" ] ||
  refuse "$tag names $now now, but the draft was built from $built: the tag moved. Delete the draft (gh release delete $tag --repo $repo --yes; the tag stays). The run that built $now makes a new draft; if its release job already refused, because this draft existed, re-run it. The checks start over on its tarball"

if [ "$check_only" = 1 ]; then
  echo "publish: $tag ok: one draft titled '$checked', its three assets, built from $built, which $tag names"
  echo "publish: the tarball's SHA-256 is $noted (its notes and GitHub's digest${sha:+, and yours}). Run the checks on that tarball, then publish: scripts/release/publish.sh $tag $noted"
  echo "publish: --check: nothing published"
  exit 0
fi
edit_rc=0
gh release edit "$tag" --repo "$repo" --title "Axon $tag" --tag "$tag" --verify-tag --draft=false --prerelease=false ||
  edit_rc=$?

# After the publish: the tag's commit again, and the release by its id.
after_now=$(gh api "repos/$repo/commits/refs/tags/$tag" --jq .sha) || after_now=
after=$(gh api "repos/$repo/releases/$id") || after=
a_draft=$(jq -r .draft <<< "$after" 2> /dev/null || true)
a_tag=$(jq -r .tag_name <<< "$after" 2> /dev/null || true)
a_name=$(jq -r .name <<< "$after" 2> /dev/null || true)
# Still a draft: the edit didn't take, so there is nothing to undo.
[ "$a_draft" != true ] || refuse "the release is still a draft after gh release edit (exit code $edit_rc)"
problem=
if ! [[ "$after_now" =~ ^[0-9a-f]{40}$ ]]; then
  problem="GitHub gave no commit for $tag ('$after_now'): the tag is gone, or an API error"
elif [ "$after_now" != "$built" ]; then
  problem="$tag names $after_now now, not $built, the commit the release was built from"
elif [ "$a_draft" != false ]; then
  problem="the release (id $id) couldn't be re-read"
elif [ "$a_tag" != "$tag" ] || [ "$a_name" != "Axon $tag" ]; then
  problem="the release now has the tag '$a_tag' and the title '$a_name', want $tag and 'Axon $tag'"
fi
if [ -z "$problem" ]; then
  [ "$edit_rc" = 0 ] || echo "publish: gh release edit exited $edit_rc, but the release checks out" >&2
  echo "publish: $tag published: built from $built, tarball SHA-256 $noted; checked again after publishing"
  exit 0
fi

echo "publish: $tag WAS PUBLISHED, but right after: $problem" >&2
if ! gh api -X PATCH "repos/$repo/releases/$id" -F draft=true -f "name=DO NOT PUBLISH: Axon $tag (unpublished by publish.sh)" > /dev/null; then
  echo "publish: AND IT COULDN'T BE TURNED BACK INTO A DRAFT. Do it now: gh api -X PATCH repos/$repo/releases/$id -F draft=true" >&2
  exit 1
fi
echo "publish: it is a draft again, titled DO NOT PUBLISH, after a few seconds in public" >&2
if [[ "$after_now" =~ ^[0-9a-f]{40}$ ]] && [ "$after_now" != "$built" ]; then
  echo "publish: the draft holds $built's tarball: delete it (gh release delete $tag --repo $repo --yes). If nobody pushed $tag to $after_now, GitHub created it there while publishing (the tag was deleted in between): delete that tag too (git push origin :refs/tags/$tag)" >&2
else
  echo "publish: look at it (gh release view $tag --repo $repo). If gh api repos/$repo/commits/refs/tags/$tag --jq .sha prints $built and the draft is otherwise right, retitle it '$checked' and run publish.sh again; if the tag is gone, delete the draft" >&2
fi
exit 1
