# A7. Acceptance criteria and budgets per work package

Conservation residuals (Gauss's law to roundoff with Esirkepov, energy drift bound over N steps), agreement thresholds against the benchmark set, a throughput budget (nanoseconds per particle-step on the 4080 SUPER, within a stated factor of Smilei on the same card), memory budget (the `FunctionMap` harness doubles state; on 16 GB that caps 2D with P = 3 at roughly 10⁷ particles), and gradient-check tolerances.
