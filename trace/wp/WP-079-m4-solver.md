+++
id = "WP-079"
status = "active"
goals = ["A2", "A3", "A4", "A7", "A8"]
depends_on = ["WP-022", "WP-026", "WP-028", "WP-034", "WP-066", "WP-072", "WP-073", "WP-074"]
+++
# WP-079: the M4 field solver

Offer the low-dispersion M4 solver (Lu et al. 2020) next to Yee in 1D, 2D and 3D, with everything Yee works with: the antenna (Gaussian and shaped in 2D, Gaussian in 3D), the PML and the absorbing layer, the moving window and its back-edge PML, deposition, filters, duals and the GPU. It works when its discrete modes follow M4's relation to fourth order with Yee's Courant limit, it keeps its own ∇·B, the antenna launches M4's angular-spectrum beam and its ω₀ plane wave to roundoff, the PML reflects about as little as Yee's, a 3D run invariant along an axis is the 2D run, Gauss's law holds across shifts and through the open faces, and the Reference step is bitwise across reruns and launch shapes, in Float32, Float64 and `Dual{Float32,3}`.
