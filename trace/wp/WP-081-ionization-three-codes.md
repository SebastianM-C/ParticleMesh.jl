+++
id = "WP-081"
status = "active"
goals = ["A5", "A7"]
depends_on = ["WP-053", "WP-059", "WP-064", "WP-065"]
adds_dependencies = []
+++
# WP-081: the ionization-injection case in three codes

WP-064's and WP-065's nitrogen-doped 2D case run in ParticleMesh, Smilei and PIConGPU from one case file, with the dopant's charge states, the born electrons' weight, spectrum and objectives, and the wake compared code by code, the Monte Carlo codes over several seeds and ParticleMesh also with each other code's ADK prefactor, so that a distance splits into the rate convention's share and the rest (A7). It works when the vacuum pulses and the wakes without the dopant agree as in WP-053 and ParticleMesh with a code's rate convention lands within that code's seed spread. Outcome: T-1120 to T-1126 in `scripts/lwfa/test/ionization.jl` and `reports/wp-081-ionization-three-codes.md`.
