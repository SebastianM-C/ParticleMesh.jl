+++
id = "WP-042"
status = "proposed"
goals = ["A0", "A2", "A4", "A8"]
depends_on = ["WP-018", "WP-020", "WP-022", "WP-023", "WP-024"]
+++
# WP-042: the periodic 3D3V step, without laser or boundaries

Extend the engine and the module to N = 3 on a periodic box: guard fill and fold, the quiet load, the push, 3D Esirkepov in both deposition modes, the initial field and the diagnostics, generalizing the 2D code over N where that keeps the 1D and 2D bits. It works when Gauss's law holds to roundoff over many steps, the Fast and Reference depositions agree to roundoff with Reference bitwise across reruns and launch shapes, and an oblique Weibel or Landau case in 3D matches linear theory, in Float32, Float64 and `Dual{Float32,3}`.
