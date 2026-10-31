#!/usr/bin/env bash
# Run a prepared Smilei case outside the factory: a plain podman run on the CPU.
#
#   scripts/crosscode/run-smilei.sh scripts/runs/<tag> [image]
#
# The run directory is mounted read-write at /run-dir, the repository read-only at /worktree;
# no network, no GPU, no capabilities, the caller's uid. The image defaults to the tag derived
# from the recipe hash (scripts/crosscode/containers/README.md).
set -euo pipefail

run_dir=$(realpath "${1:?usage: run-smilei.sh scripts/runs/<tag> [image]}")
repo=$(realpath "$(dirname "$0")/../..")
case "$run_dir" in
  "$repo"/scripts/runs/*) ;;
  *) echo "run directory must be under $repo/scripts/runs/" >&2; exit 2 ;;
esac
test -f "$run_dir/run.sh" || { echo "$run_dir/run.sh missing; run smilei.jl prepare first" >&2; exit 2; }

if [ $# -ge 2 ]; then
  image=$2
else
  cd "$repo/scripts/crosscode/containers"
  hash=$( { echo smilei.Containerfile; find smilei -type f 2>/dev/null; } | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)
  image=particlemesh-smilei:$hash
  cd - > /dev/null
fi

podman image exists "$image" || { echo "image $image is not built (scripts/crosscode/containers/README.md)" >&2; exit 3; }
image_id=$(podman image inspect --format '{{.Id}}' "$image")
echo "# image: $image $image_id" > "$run_dir/image.txt"

set +e
podman run --rm \
  --network=none --cap-drop=all --security-opt=no-new-privileges \
  --userns=keep-id --user "$(id -u):$(id -g)" \
  -v "$run_dir":/run-dir:rw,Z \
  -v "$repo":/worktree:ro,Z \
  -e PM_RUN_DIR=/run-dir -e PM_WORKTREE=/worktree -e PM_GPU=0 \
  "$image" bash /run-dir/run.sh
status=$?
echo "# podman run finished: exit $status" >> "$run_dir/image.txt"
exit $status
