+++
id = "WP-063"
status = "done"
goals = ["A1", "A2", "A3", "A4", "A7", "A8", "A10"]
depends_on = ["WP-059", "WP-060"]
adds_dependencies = []
+++
# WP-063: the ionization rate by the level's type, and barrier-suppression ionization

The field ionization takes each level's rate from a method on the level's type, so a level with another rate law plugs in without a change to the step, and the first such level multiplies the tunnelling rate by the barrier-suppression factor of Tong and Lin, smooth in the field and in its numbers. It works when a problem with ADK levels keeps its bits, the new rate is its closed form with its derivatives, a level type defined outside the module runs in one chain with the module's, the level built from an ionization energy agrees with published MCTDHF rates of helium-like ions about the field at which the barrier is gone, and ForwardDiff through a small ionization-injection run agrees with the finite-difference floor of A7 in the suppression. Outcome: T-820 to T-827 in `lib/ParticleMeshElectromagnetic/test/barrier_suppression.jl` and `gpu.jl`, and `reports/wp-063-barrier-suppression.md`.
