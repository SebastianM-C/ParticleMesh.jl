#!/usr/bin/env bash
# WP-083: the two worktrees of run_gpu.sh, from a ParticleMesh checkout (git steps only).
# Usage: prepare.sh <checkout> <worktree A> <worktree B>
set -eu
C=$1; A=$2; B=$3
cd "$C"
git fetch origin claude/project-thread-xjh4di claude/project-thread-kjjfr2-gpu main
git worktree add --detach "$A" origin/claude/project-thread-xjh4di
git worktree add --detach "$B" origin/claude/project-thread-kjjfr2-gpu
git diff origin/main origin/claude/project-thread-xjh4di -- lib/ParticleMeshElectromagnetic/src > "$B/wp083-sincos.patch"
cd "$B" && git apply wp083-sincos.patch && git diff --stat
