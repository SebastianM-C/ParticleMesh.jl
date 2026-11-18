#!/usr/bin/env bash
# Run outputs on run branches (F3): a report commits the record of a run, and the run's bulk data
# (dumps, large maps, per-run plot sets, animations) lives on a branch runs/<tag> that is never merged.
#
#   scripts/rundata/run-data.sh publish <tag> <dir>...   push branch runs/<tag>: HEAD plus the run
#                                                        directories named; then write published.toml
#                                                        into each of them
#   scripts/rundata/run-data.sh fetch <tag>              put the run directories of runs/<tag> into
#                                                        scripts/runs/; never overwrites a local run
#   scripts/rundata/run-data.sh fetch <published.toml>   the same at the commit that file names, then
#                                                        check every file against its SHA-256
#
# Run from the root of your worktree, with HEAD pushed. The worktree is not touched by publish (a
# temporary index builds the commit). The directories must exist and be named exactly
# (scripts/runs/<name>, no wildcards), and hold no file over 20 MB and nothing that fails CI's leak
# scan: run branches are public and permanent. A tag is used once; publish refuses a tag whose branch
# exists. The CPU box's result branches (scripts/cpu-box.sh) are run branches too, and fetch reads
# them the same way.
#
# published.toml is what a report commits next to the run's manifest: the branch, the commit and,
# for every file of the directory, its path, size and SHA-256. A reader or reviewer gets the data back
# with `fetch <that file>`, which refuses a file whose hash differs.
set -euo pipefail
set -f
LEAK='(ssh:/''/|BEGIN [A-Z ]*PRIVATE KEY|\b([0-9]{1,3}\.){3}[0-9]{1,3}\b)'
REMOTE="${RUN_DATA_REMOTE:-origin}"
die() { echo "run-data: $*" >&2; exit 64; }
[ $# -ge 2 ] || die "usage: run-data.sh publish <tag> <dir>... | fetch <tag> | fetch <published.toml>"
cmd="$1"
[ -e .git ] || die "run from the root of your worktree"
checktag() { [[ "$1" =~ ^[A-Za-z0-9._-]{1,80}$ ]] || die "tag: letters, digits, . _ - only"; }
checkdir() {
  [[ "$1" =~ ^scripts/runs/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || die "name a directory exactly, scripts/runs/<name>: $1"
  case "/$1/" in */../*|*/./*) die "name a directory exactly, scripts/runs/<name>: $1" ;; esac
}

case "$cmd" in
  publish)
    [ $# -ge 3 ] || die "usage: run-data.sh publish <tag> <dir>..."
    tag="$2"; shift 2
    checktag "$tag"
    for d in "$@"; do
      d="${d%/}"
      checkdir "$d"
      [ -d "$d" ] || die "no directory $d"
      [ ! -e "$d/published.toml" ] || die "$d/published.toml exists: this run is published already"
      [ -n "$(find "$d" -type f -print -quit)" ] || die "$d holds no files"
      big="$(find "$d" -type f -size +20M)"; [ -z "$big" ] || die "files over 20 MB: $big"
      ! grep -rIlE "$LEAK" -- "$d" >/dev/null || die "leak-scan pattern in: $(grep -rIlE "$LEAK" -- "$d" | tr '\n' ' ')"
    done
    [ -n "$(git branch -r --contains HEAD 2>/dev/null)" ] || die "HEAD is not on any pushed branch; push it first"
    [ -z "$(git ls-remote "$REMOTE" "refs/heads/runs/$tag")" ] || die "runs/$tag exists; pick a new tag"
    idx="$(mktemp)"; trap 'rm -f "$idx"' EXIT
    GIT_INDEX_FILE="$idx" git read-tree HEAD
    for d in "$@"; do GIT_INDEX_FILE="$idx" git add -f -- "${d%/}"; done
    tree="$(GIT_INDEX_FILE="$idx" git write-tree)"
    commit="$(git commit-tree "$tree" -p HEAD -m "Run data $tag")"
    git push -q "$REMOTE" "$commit:refs/heads/runs/$tag"
    for d in "$@"; do
      d="${d%/}"
      git ls-tree -r -l "$commit" -- "$d" | python3 -c '
import hashlib, sys
branch, commit, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for line in sys.stdin:
    meta, path = line.rstrip("\n").split("\t", 1)
    size = int(meta.split()[3])
    with open(path, "rb") as f:
        rows.append((path, size, hashlib.sha256(f.read()).hexdigest()))
with open(out, "w") as f:
    f.write(f"branch = \"{branch}\"\ncommit = \"{commit}\"\n")
    for p, s, h in rows:
        f.write(f"\n[[files]]\npath = \"{p}\"\nbytes = {s}\nsha256 = \"{h}\"\n")
' "runs/$tag" "$commit" "$d/published.toml"
      echo "$d/published.toml"
    done
    echo "pushed runs/$tag at $commit; commit each published.toml with the run's manifest"
    ;;
  fetch)
    [ $# -eq 2 ] || die "usage: run-data.sh fetch <tag> | fetch <published.toml>"
    if [ -f "$2" ]; then
      spec="$(python3 - "$2" <<'EOF'
import sys, tomllib
p = tomllib.load(open(sys.argv[1], "rb"))
print(p["branch"]); print(p["commit"])
for f in p["files"]:
    print(f["sha256"], f["path"])
EOF
)" || die "cannot read $2"
      branch="$(sed -n 1p <<< "$spec")"; commit="$(sed -n 2p <<< "$spec")"
      [[ "$branch" =~ ^runs/[A-Za-z0-9._-]{1,80}$ ]] || die "branch: $branch"
      [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "commit: $commit"
      git fetch -q "$REMOTE" "refs/heads/$branch" || die "no branch $branch"
      git cat-file -e "$commit^{commit}" 2>/dev/null || die "commit $commit is not on $branch"
      git merge-base --is-ancestor "$commit" FETCH_HEAD || die "commit $commit is not on $branch"
      bad=0
      while read -r sha path; do
        [ -n "$path" ] || continue
        checkdir "$(dirname "$path")"
        if [ -e "$path" ]; then
          echo "kept local $path"
        else
          mkdir -p "$(dirname "$path")"
          git cat-file blob "$commit:$path" > "$path" || die "$path is not in $commit"
        fi
        got="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$path")"
        if [ "$got" != "$sha" ]; then echo "MISMATCH $path" >&2; bad=1; fi
      done <<< "$(sed -n '3,$p' <<< "$spec")"
      [ "$bad" -eq 0 ] || die "files differ from $2"
      echo "every file of $2 matches its SHA-256"
    else
      tag="$2"
      checktag "$tag"
      git fetch -q "$REMOTE" "refs/heads/runs/$tag" || die "no branch runs/$tag"
      dirs="$(git ls-tree --name-only FETCH_HEAD scripts/runs/)"
      [ -n "$dirs" ] || die "runs/$tag holds no run directories"
      while IFS= read -r d; do [ ! -e "$d" ] || die "$d exists already; move it away to fetch again"; done <<< "$dirs"
      git archive FETCH_HEAD scripts/runs | tar -x
      echo "$dirs"
    fi
    ;;
  *) die "unknown command $cmd (publish, fetch)" ;;
esac
