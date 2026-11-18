#!/usr/bin/env bash
# The size budget of reports/ (F3): between two commits, no file added or changed under reports/ may
# exceed 1 MB, and all of them together 5 MB. Bulk run data goes to a run branch instead
# (scripts/rundata/run-data.sh publish). CI runs this on every pull request.
#
#   scripts/rundata/report-budget.sh <base> <head>
#
# Exits 0 within the budget, 1 over it (naming the files), 64 on a usage error.
set -euo pipefail
FILE_MAX=1000000
TOTAL_MAX=5000000
[ $# -eq 2 ] || { echo "usage: report-budget.sh <base> <head>" >&2; exit 64; }
base="$1" head="$2"
total=0 over=0
while IFS=$'\t' read -r meta path; do
  [ -n "$path" ] || continue
  sha="$(awk '{print $4}' <<< "$meta")"
  size="$(git cat-file -s "$sha")"
  total=$((total + size))
  if [ "$size" -gt "$FILE_MAX" ]; then
    echo "over 1 MB: $path ($size bytes)"
    over=1
  fi
done < <(git diff --raw --no-abbrev --no-renames --diff-filter=AMT "$base" "$head" -- reports/)
echo "reports/: $total bytes added or changed (budget $TOTAL_MAX, at most $FILE_MAX per file)"
if [ "$total" -gt "$TOTAL_MAX" ]; then
  echo "over the 5 MB budget for one pull request"
  over=1
fi
if [ "$over" -ne 0 ]; then
  echo "Keep the report's text, manifests, inputs, small sidecars and the figures it shows; publish the"
  echo "rest of the run with scripts/rundata/run-data.sh publish and commit its published.toml (F3)."
  exit 1
fi
