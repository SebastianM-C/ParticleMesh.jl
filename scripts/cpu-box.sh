#!/usr/bin/env bash
# Long CPU runs on the rented CPU box (16 cores, AMD EPYC 9655), through git, from any session.
#
#   scripts/cpu-box.sh submit <tag> <request.toml>   push branch runs/<tag>: HEAD, the request as
#                                                    box-request.toml, and the run directories it names
#   scripts/cpu-box.sh status <tag>                  print the result's box-status.toml; while pending, the
#                                                    box's progress on it (branch box-status), and exit 1
#   scripts/cpu-box.sh fetch <tag>                   put the result into scripts/runs/box-<tag>/ (the run
#                                                    directories, box-status.toml, box-log.txt); never
#                                                    overwrites a local run
#
# Run from the root of your worktree, with HEAD pushed. The worktree is not touched by submit (a
# temporary index builds the commit). The request's directories must exist, be named exactly
# (scripts/runs/<name>, no wildcards), and hold no file over 20 MB and nothing that fails CI's leak
# scan: the request branch is public and permanent. The request:
#
#   timeout = 7200                       # seconds per step, at most 43200
#   category = "validation"              # for the cost ledger: physics, validation, crosscode,
#                                        # benchmark, test or smoke (else "unspecified")
#   outputs = ["scripts/runs/<dir>"]     # directories whose files come back (under scripts/runs/)
#   [[steps]]
#   kind = "smilei"                      # Smilei's CPU image; the request's recipe must equal main's
#   dir = "scripts/runs/<dir>"           # (at one of its last 300 commits); holds run.sh
#   [[steps]]
#   kind = "julia"                       # julia <args> in the checkout, offline, 16 threads
#   args = ["--project=scripts", "scripts/crosscode/analyze.jl", "scripts/runs/<dir>"]
#
# A test suite is a julia step in the workspace: args = ["--project=lib/ParticleMeshCore", "-e",
# "using Pkg; Pkg.test()"] (or --project=. for the umbrella); its output is in box-log.txt and its
# exit code in box-status.toml, and outputs may be empty.
#
# Nothing is installed per request. The box keeps main's scripts/ and workspace environments warm
# (instantiated from main, never from a request); a request whose manifests differ from main's
# runs against them offline, and a package main does not have fails to load. Steps run in order
# and stop at the first failure. The box polls every minute and runs one request at a time,
# oldest first; it holds no credentials, and a relay brings the result back as one commit on
# runs/<tag> about a minute after the end. Files over 20 MB do not come back (box-status.toml
# lists them under dumps_over_20MB; the owner keeps them 30 days), so analyse large dumps in a
# julia step. Nothing comes back if a file fails the leak scan of CI. Run branches are never
# merged and stay, as the record of the run that a report can cite. The box is a spot instance:
# a request lost to an eviction runs again on the next box.
#
# Hands-off batches: give every request of a long campaign the same three fields,
#   batch = "<name>"   batch_total = <number of requests>   branch = "<your pushed branch>"
# and put a CONTINUE.md at the root of that branch saying what to do with the results. When all
# batch_total results are back, a fresh cloud session checks out the branch, fetches the results
# and follows CONTINUE.md, so the submitting session can end once the batch is submitted.
set -euo pipefail
set -f
LEAK='(ssh:/''/|BEGIN [A-Z ]*PRIVATE KEY|\b([0-9]{1,3}\.){3}[0-9]{1,3}\b)'
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
    import re
    if not re.fullmatch(r"scripts/runs/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*", x) or ".." in x.split("/"):
        sys.exit(f"cpu-box: name a directory exactly, scripts/runs/<name>: {x!r}")
    print(x)
EOF
)"
    git cat-file -e HEAD:box-status.toml 2>/dev/null && die "HEAD is a box result commit; submit from your own branch"
    [ -n "$(git branch -r --contains HEAD 2>/dev/null)" ] || die "HEAD is not on any pushed branch; push it first"
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      [ -d "$d" ] || die "no directory $d"
      big="$(find "$d" -type f -size +20M)"; [ -z "$big" ] || die "files over 20 MB in the request: $big"
      ! grep -rIlE "$LEAK" -- "$d" "$req" >/dev/null || die "leak-scan pattern in: $(grep -rIlE "$LEAK" -- "$d" "$req" | tr '\n' ' ')"
    done <<< "$dirs"
    idx="$(mktemp)"; trap 'rm -f "$idx"' EXIT
    GIT_INDEX_FILE="$idx" git read-tree HEAD
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      GIT_INDEX_FILE="$idx" git add -f -- "$d"
    done <<< "$dirs"
    blob="$(git hash-object -w "$req")"
    GIT_INDEX_FILE="$idx" git update-index --add --cacheinfo "100644,$blob,box-request.toml"
    tree="$(GIT_INDEX_FILE="$idx" git write-tree)"
    commit="$(git commit-tree "$tree" -p HEAD -m "CPU box request $tag")"
    git push -q origin "$commit:refs/heads/runs/$tag"
    echo "pushed runs/$tag at $commit; poll with: scripts/cpu-box.sh status $tag"
    ;;
  status)
    git fetch -q origin "refs/heads/runs/$tag" || die "no branch runs/$tag"
    git show FETCH_HEAD:box-status.toml 2>/dev/null && exit 0
    echo "pending"
    if git fetch -q origin refs/heads/box-status 2>/dev/null; then
      git show FETCH_HEAD:status.json | python3 -c '
import json, sys
st, tag = json.load(sys.stdin), sys.argv[1]
c, q, k = st.get("current") or {}, st.get("queue") or [], st.get("keeper") or {}
print(f"box: {st.get('"'"'state'"'"')} (status of {st.get('"'"'updated'"'"', '"'"'?'"'"')}; keeper {k.get('"'"'mode'"'"', '"'"'?'"'"')}, box {k.get('"'"'box'"'"')})")
if c.get("tag") == tag:
    print(f"running: step {c.get('"'"'step'"'"')}/{c.get('"'"'steps'"'"')} ({c.get('"'"'step_kind'"'"')}), {(c.get('"'"'elapsed_s'"'"') or 0) // 60} min in, "
          f"{c.get('"'"'seed_dirs_with_observables'"'"', 0)} seed dirs with observables, {c.get('"'"'output_files'"'"', 0)} output files")
    for l in (st.get("log_tail") or [])[-5:]:
        print("  | " + l)
elif tag in q:
    print(f"queued: position {q.index(tag) + 1} of {len(q)}" + (f", after {c.get('"'"'tag'"'"')} (running)" if c else ""))
else:
    print("not seen by the box yet (it polls every minute; a parked box boots within about 10 minutes)")
' "$tag" || true
    fi
    exit 1
    ;;
  fetch)
    git fetch -q origin "refs/heads/runs/$tag" || die "no branch runs/$tag"
    git cat-file -e FETCH_HEAD:box-status.toml 2>/dev/null || die "no result on runs/$tag yet"
    dest="scripts/runs/box-$tag"
    [ ! -e "$dest" ] || die "$dest exists already; move it away to fetch again"
    mkdir -p "$dest"
    if git cat-file -e FETCH_HEAD:scripts/runs 2>/dev/null; then
      git archive FETCH_HEAD scripts/runs | tar -x -C "$dest" --strip-components=2
    fi
    git show FETCH_HEAD:box-status.toml > "$dest/box-status.toml"
    git show FETCH_HEAD:box-log.txt > "$dest/box-log.txt" 2>/dev/null || rm -f "$dest/box-log.txt"
    cat "$dest/box-status.toml"
    ;;
  *) die "unknown command $cmd (submit, status, fetch)" ;;
esac
