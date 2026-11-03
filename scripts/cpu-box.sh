#!/usr/bin/env bash
# Long CPU runs on the rented CPU box (16 cores, AMD EPYC 9655), through git, from any session.
#
#   scripts/cpu-box.sh submit <tag> <request.toml>   push branch runs/<tag>: HEAD, the request as
#                                                    box-request.toml, and the run directories it names
#   scripts/cpu-box.sh status <tag>                  print the result's box-status.toml; exit 1 while pending
#   scripts/cpu-box.sh fetch <tag>                   put the result's files into this worktree's scripts/runs/,
#                                                    with box-status.toml and box-log.txt in scripts/runs/box-<tag>/
#
# Run from the root of your worktree. The worktree is not touched by submit (a temporary index
# builds the commit). The request:
#
#   timeout = 7200                       # seconds per step, at most 43200
#   outputs = ["scripts/runs/<dir>"]     # directories whose files come back (under scripts/runs/)
#   [[steps]]
#   kind = "smilei"                      # Smilei's CPU image, built from the commit's recipe
#   dir = "scripts/runs/<dir>"           # holds run.sh, as for the GPU runner
#   [[steps]]
#   kind = "julia"                       # julia +1.13 <args> in the checkout, no network
#   args = ["--project=scripts", "scripts/crosscode/analyze.jl", "scripts/runs/<dir>"]
#
# Steps run in order and stop at the first failure; the projects named by --project= are
# instantiated first. Files over 20 MB stay on the box (box-status.toml lists them), so analyse
# large dumps in a julia step. The result is one commit on runs/<tag>; nothing comes back if it
# fails the leak scan of CI. The box polls every minute and runs one request at a time, oldest
# first. A run branch is never merged: delete it after fetching (git push origin --delete runs/<tag>).
set -euo pipefail
die() { echo "cpu-box: $*" >&2; exit 64; }
[ $# -ge 2 ] || die "usage: cpu-box.sh submit|status|fetch <tag> [request.toml]"
cmd="$1" tag="$2"
[[ "$tag" =~ ^[A-Za-z0-9._-]{1,80}$ ]] || die "tag: letters, digits, . _ - only"
[ -e .git ] || die "run from the root of your worktree"
case "$cmd" in
  submit)
    [ $# -eq 3 ] && [ -f "$3" ] || die "usage: cpu-box.sh submit <tag> <request.toml>"
    req="$3"
    dirs="$(python3 - "$req" <<'EOF'
import sys, tomllib
r = tomllib.load(open(sys.argv[1], "rb"))
d = set(r.get("outputs", [])) | {s["dir"] for s in r.get("steps", []) if s.get("kind") == "smilei"}
for x in sorted(d):
    if not x.startswith("scripts/runs/") or ".." in x:
        sys.exit(f"cpu-box: not under scripts/runs/: {x}")
    print(x)
EOF
)"
    idx="$(mktemp)"; trap 'rm -f "$idx"' EXIT
    GIT_INDEX_FILE="$idx" git read-tree HEAD
    for d in $dirs; do
      [ -d "$d" ] && GIT_INDEX_FILE="$idx" git add -f -- "$d"
    done
    blob="$(git hash-object -w "$req")"
    GIT_INDEX_FILE="$idx" git update-index --add --cacheinfo "100644,$blob,box-request.toml"
    tree="$(GIT_INDEX_FILE="$idx" git write-tree)"
    commit="$(git commit-tree "$tree" -p HEAD -m "CPU box request $tag")"
    git push -q origin "$commit:refs/heads/runs/$tag"
    echo "pushed runs/$tag at $commit; poll with: scripts/cpu-box.sh status $tag"
    ;;
  status)
    git fetch -q origin "refs/heads/runs/$tag" || die "no branch runs/$tag"
    git show FETCH_HEAD:box-status.toml 2>/dev/null || { echo "pending"; exit 1; }
    ;;
  fetch)
    git fetch -q origin "refs/heads/runs/$tag" || die "no branch runs/$tag"
    git cat-file -e FETCH_HEAD:box-status.toml 2>/dev/null || die "no result on runs/$tag yet"
    if git cat-file -e FETCH_HEAD:scripts/runs 2>/dev/null; then
      git archive FETCH_HEAD scripts/runs | tar -x
    fi
    mkdir -p "scripts/runs/box-$tag"
    git show FETCH_HEAD:box-status.toml > "scripts/runs/box-$tag/box-status.toml"
    git show FETCH_HEAD:box-log.txt > "scripts/runs/box-$tag/box-log.txt"
    cat "scripts/runs/box-$tag/box-status.toml"
    ;;
  *) die "unknown command $cmd (submit, status, fetch)" ;;
esac
