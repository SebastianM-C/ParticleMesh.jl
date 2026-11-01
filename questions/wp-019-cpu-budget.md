# WP-019: what "cost" means in the CPU budget of the 2D Smilei runs

**Question.** WP-019's budget asks each 2D Smilei run to stay "within the cost of WP-003's 1D Landau reference run" (2.56·10⁷ electrons, 379 s on 8 threads, `reports/crosscode-smilei-1d/README.md`). Is that cost particle-steps or CPU time (wall seconds × threads)?

**Options.**
- Particle-steps: 6.86·10¹⁰ for the 1D reference.
- Thread-seconds: 3 030 for the 1D reference.

**Assumption WP-019 proceeded on.** Particle-steps. The cases were sized before any 2D run, when the cost per particle-step in 2D was not known. Both runs stay below the bound in particle-steps (5.69·10¹⁰ for `landau-oblique-2d`, 1.82·10¹⁰ for `weibel-2d`). In thread-seconds they exceed it: Landau used 3.8 times the reference (716 s on 16 threads, 11 460 thread-seconds) and Weibel 1.2 times (3 640). The report states both measures (`reports/crosscode-smilei-2d/README.md`, section "Cost"). The runs were not repeated smaller: the Landau observable is already noise-limited and the Weibel one resolved.

**Affects.** WP-019 (the budget line and the two runs); later 2D cross-code packages that size runs against the same reference. A budget stated in thread-seconds, or per particle-step with the core type and thread count named, would remove the ambiguity.
