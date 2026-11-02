+++
id = "WP-010"
status = "done"
goals = ["A0", "A2", "A3", "A4", "A5", "A7", "A8", "A9"]
depends_on = []
adds_dependencies = ["KernelAbstractions", "Atomix", "Adapt", "ForwardDiff"]
+++
# WP-010: spike: a 1D periodic electrostatic PIC with a checked gradient

The first runnable result: a 1D electrostatic PIC whose Landau and two-stream runs match theory and whose AD gradient matches finite differences. Outcome: `reports/spike-electrostatic-1d.md`.
