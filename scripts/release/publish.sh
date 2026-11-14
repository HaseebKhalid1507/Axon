#!/usr/bin/env bash
# Publishes the draft release for a tag, right after checking it (docs/releasing.md, "Checking and publishing the
# draft", step 5). It publishes only if all of these hold, checked in this order:
# - exactly one release for the tag, a draft, titled exactly "Axon vX.Y.Z". The release job creates it as
#   "DO NOT PUBLISH: Axon vX.Y.Z (unchecked)" and retitles it only after its last tag check passed;
# - its notes name exactly one commit ("Built from commit <sha>");
# - its assets are exactly the tarball, its .sha256 and install.sh, all uploaded;
# - the tag names that commit now (commits/refs/tags/<tag>, which peels an annotated tag). This runs last, right
#   before the publish, so a tag moved during the hours of hand checks is caught.
# Then `gh release edit vX.Y.Z --tag vX.Y.Z --verify-tag --draft=false --prerelease=false`: a full release, the
# kind GitHub's `latest` (install.sh) points at. In gh 2.102.1 `--verify-tag` checks that the tag exists only when
# `--tag` is given too; without it, publishing a draft whose tag is gone creates the tag at main's tip.
#
# Usage: scripts/release/publish.sh [--check] vX.Y.Z
#   --check  run every check and publish nothing (releasing.md runs it before the hand checks too).
# GH_REPO: OWNER/REPO (default HaseebKhalid1507/Axon). Needs gh logged in with push access: drafts are listed only
# for it.
set -euo pipefail

check_only=0
if [ "${1:-}" = --check ]; then check_only=1; shift; fi
[ $# = 1 ] || { echo "usage: scripts/release/publish.sh [--check] vX.Y.Z" >&2; exit 2; }
tag=$1
repo=${GH_REPO:-HaseebKhalid1507/Axon}
refuse() {
  echo "publish: $*" >&2
  echo "publish: nothing published" >&2
  exit 1
}

[[ "$tag" =~ ^v[0-9][0-9A-Za-z.+-]*$ ]] || refuse "'$tag' isn't a version tag"
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || refuse "GH_REPO '$repo' isn't OWNER/REPO"

# The listing, not `gh release view`: it shows every release for the tag (GitHub allows several drafts for one),
# and with push access it includes drafts. --paginate may print one JSON array per page.
listing=$(gh api --paginate "repos/$repo/releases?per_page=100") || refuse "can't list $repo's releases"
rels=$(jq -c --arg tag "$tag" '.[] | select(.tag_name == $tag)' <<< "$listing") ||
  refuse "can't read $repo's release listing"
count=$(grep -c . <<< "$rels" || true)
[ "$count" = 1 ] || refuse "$count releases for $tag, want one draft"
[ "$(jq -r .draft <<< "$rels")" = true ] || refuse "the release for $tag is published already: $(jq -r .html_url <<< "$rels")"
title=$(jq -r .name <<< "$rels")
[ "$title" = "Axon $tag" ] || refuse "the draft is titled '$title', not 'Axon $tag': the release job didn't confirm the tag after the upload, or didn't finish (cancelled, failed partway). Its log says what to do"

commits=$(jq -r '.body // ""' <<< "$rels" | grep -oE 'Built from commit [0-9a-f]+' | cut -d' ' -f4 || true)
n_commits=$(grep -c . <<< "$commits" || true)
[ "$n_commits" = 1 ] || refuse "the draft's notes name $n_commits commits ('Built from commit <sha>'), want one"
built=$commits
[[ "$built" =~ ^[0-9a-f]{40}$ ]] || refuse "the draft's notes say 'Built from commit $built', which isn't a commit"

assets=$(jq -r '[.assets[] | .name + (if .state == "uploaded" then "" else " (" + (.state | tostring) + ")" end)] | sort | join(", ")' <<< "$rels")
want_assets="axon-x86_64-linux.tar.gz, axon-x86_64-linux.tar.gz.sha256, install.sh"
[ "$assets" = "$want_assets" ] || refuse "the draft's assets are '$assets', want exactly $want_assets (uploaded)"

# Last, right before the publish: the tag names the commit the draft was built from.
now=$(gh api "repos/$repo/commits/refs/tags/$tag" --jq .sha) ||
  refuse "GitHub gave no commit for $tag: the tag is gone, or an API error. Don't publish: the draft's target is main, so publishing it with the tag gone would create $tag at main's tip. If the tag is gone, delete the draft (gh release delete $tag --repo $repo --yes)"
[[ "$now" =~ ^[0-9a-f]{40}$ ]] || refuse "GitHub's answer for $tag isn't a commit: '$now'"
[ "$now" = "$built" ] ||
  refuse "$tag names $now now, but the draft was built from $built: the tag moved. Delete the draft (gh release delete $tag --repo $repo --yes; the tag stays). The run that built $now makes a new draft; if its release job already refused, because this draft existed, re-run it. The checks start over on its tarball"

if [ "$check_only" = 1 ]; then
  echo "publish: $tag ok: one draft titled 'Axon $tag', its three assets, built from $built, which $tag names (--check: nothing published)"
  exit 0
fi
gh release edit "$tag" --repo "$repo" --tag "$tag" --verify-tag --draft=false --prerelease=false
echo "publish: $tag published, built from $built"
