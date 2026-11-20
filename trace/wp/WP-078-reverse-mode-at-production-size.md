+++
id = "WP-078"
status = "done"
goals = ["A0", "A3", "A7", "A8"]
depends_on = ["WP-050", "WP-065", "WP-068", "WP-070", "WP-077"]
adds_dependencies = []
+++
# WP-078: Enzyme's reverse mode on the trapped-beam case at production size

Measure Enzyme's reverse pass against the primal run and forward-mode duals (1 and 4 partials) in time and GPU memory on the λ₀/32 trapped-beam case of WP-065, differentiate that case's objective with respect to its four free parameters in reverse mode and run WP-068's first-bucket optimization at fixed laser energy with those gradients, and measure J and its gradient with the Galerkin plus compensated gather (WP-070) at λ₀/24, /32 and /48. It works when the reverse gradient agrees with the duals' by the chain rule, the optimization reaches WP-068's optimum, and the cost, memory and convergence numbers are in a report. Outcome: `reports/wp-078-reverse-mode-at-production-size.md`; the drift of the gradient with more than 20 steps per reversed segment is #548.
