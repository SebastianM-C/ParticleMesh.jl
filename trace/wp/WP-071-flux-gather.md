+++
id = "WP-071"
status = "done"
goals = ["A2", "A3", "A7", "A8"]
depends_on = ["WP-042"]
+++
# WP-071: the transpose of the flux deposition

Give `deposit_flux!` its transposed partner in the engine, a gather that reads a field with Esirkepov's weights along each particle's move, for N = 1, 2 and 3, not yet wired into a step. It works when ⟨deposit_flux!(q), E⟩ = ⟨q, gather_flux!(E)⟩ to roundoff in both deposition modes, and when for a vanishing move it reproduces the energy-conserving gather in closed form.
