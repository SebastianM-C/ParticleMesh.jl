# The benchmark configurations, shared by the tests, the determinism worker and scripts/es1d/.
# Landau: kλ_D = 1/2, α = 1/20, 64 cells, 16384 particles per cell (the fit is noise-limited at
# 4096), quadratic shape, Reference mode, 25/ω_p with the fit over t ≤ 20/ω_p.
const LANDAU_KW = (; n = 64, ppc = 16384, tmax_ωp = 25, shape = Quadratic, mode = Reference(), seed = 1)
const LANDAU_FIT_TMAX = 20
# Two-stream: cold beams at ±v₀ = ±c/10, k v₀/ω_p = 3/5 (the most unstable k of the cold theory
# is near 0.61), α = 1/1000, 64 cells, 256 particles per cell per beam, 40/ω_p.
const TWO_STREAM_KW = (; n = 64, ppc = 256, kv0_ωp = 3 // 5, tmax_ωp = 40, shape = Quadratic, mode = Reference(), seed = 1)
