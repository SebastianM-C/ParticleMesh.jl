# WP-005 T-042: an edit blocked by the path guard

**Question.** The factory's path-guard hook blocked an edit of `trace/wp/WP-005-core-particles.md` (acceptance item T-042) on 2026-10-31, saying that the input mentions a private path or name. The edit contained no path or name that we could identify, so this looks like a false positive. Which part matched, and should the hook allow it?

**The intended change.** The reviewer of WP-004 to WP-008 found that T-042 ("free streaming with `drift!` is exact up to the cell and offset split over N steps") cannot be tested as written: summing a general displacement in Float32 accumulates error of order N·eps. The fix makes the test bitwise only when the displacement is exactly representable in binary, and otherwise bounds the error by a proposed constant times eps(T) times N, recorded in `tolerances.toml`.

**Assumption we proceeded on.** T-042 stays as written in the PR that introduces WP-004 to WP-008 (#39). The implementer of WP-005 applies the change above when writing the test. Nothing else depends on this.

**Affected.** `trace/wp/WP-005-core-particles.md`, PR #39.
