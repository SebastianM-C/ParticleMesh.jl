# Reports

Markdown reports with their figures committed next to them, one directory per report. A report is a record, not prose: it starts with a header block naming the work package, the commit, the run ids and the hashes of its inputs, then setup, results (figures from `scripts/` with their sidecar numbers), verification status, and open issues. The trace builder hashes reports like any other record; the overseer's daily digests live in `trace/digests/`. Benchmark numbers and scientific claims get an independent review before a report is cited outside the repository.
