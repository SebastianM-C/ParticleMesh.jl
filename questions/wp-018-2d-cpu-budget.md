# WP-018: a CPU budget for the CI-sized 2D tests

**Question.** What bounds the CPU cost of the CI-sized 2D tests until overview item 5 settles test budgets? A7 asks to measure, then propose; its throughput budget is on the GPU and says nothing about CPU test time.

**Measured** (`reports/wp-018-engine-2d.md`, run `engine2d-ported-t{1,4}-20261101T*`, Intel Xeon Processor @ 2.10GHz on a shared cloud VM, Julia 1.13.3, KernelAbstractions CPU backend; median of 7 runs of 50 steps, spread in brackets):

| engine-only proxy step, 16 × 12 cells | 1 thread | 4 threads |
|---|---|---|
| Float32, Reference, 2 × 3 per cell (1152 particles) | 543 ns [490, 595] | 1246 ns [1015, 1431] |
| Float32, Fast, 2 × 3 per cell | 270 ns [250, 291] | 419 ns [401, 454] |
| Float64, Reference, 2 × 3 per cell | 543 ns [514, 584] | 1135 ns [1021, 1263] |
| Float64, Fast, 2 × 3 per cell | 267 ns [234, 359] | 404 ns [392, 457] |
| Float32, Reference, 8 × 8 per cell (12 288 particles) | 154 ns [142, 173] | 189 ns [145, 299] |

per particle-step (advance, sort, order-2 deposit at (Center, Center), fold, fill, order-2 gather). At the CI size the step is dominated by the fixed cost of about ten kernel launches, which is why 4 threads are slower than 1.

The added `ParticleMeshCore` CPU test wall time of T-260 to T-269: **144 s at 1 thread** (519 s on the base commit, 663 s with this package, one run each, including precompilation; section 6 of the report). Of the added time, about 55 s are the in-process 2D testsets and the rest the two subprocess workers recomputing the N = 1 and 2D digests at 1 and 4 threads. The whole suite takes 607 s at 4 threads.

**Proposed bound.** Until overview item 5 settles test budgets: a CI-sized 2D test of the engine stays at or below 16 × 12 cells with at most 2 × 3 particles per cell when it steps, so one step costs at most about 1.5 µs per particle-step at 4 threads (the upper end of the measured spread), about 2 ms per step at the CI size; and the 2D tests of one package add at most 3 minutes of wall time to its CPU test run at 1 thread (the measured 144 s with room for the ± 30 s run-to-run spread of this VM). Proposals for the owner, not thresholds: no test asserts them.

**Assumption WP-018 proceeded on.** That the CI test size of the package (16 × 12 cells, 2 × 3 particles per cell, a few hundred hand-placed particles otherwise) is the right size for the 2D engine tests, and that a CPU wall-time figure measured on this cloud VM is an acceptable stand-in for the CI runner's until CI timings are collected. The larger-size proxy study is the follow-up issue #135, not part of this proposal.

**Affects.** WP-018 (the 2D engine tests), and the later 2D packages (the flux deposition, the periodic mesh solve, the module's 2D step), whose CI sizes §0.8 item 5 bounds by the validation package's own measurement.
