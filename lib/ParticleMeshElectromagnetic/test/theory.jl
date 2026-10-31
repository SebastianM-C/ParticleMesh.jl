# Linear theory of the three 1D cases, computed from the dispersion relations; no value is taken
# from memory. Landau damping and the cold two-stream relation (with the longitudinal mass γ₀³m)
# come from `scripts/crosscode/theory.jl` (module `Theory`, tested by T-160). The cold-fluid
# Weibel relation is derived here, for T-216.
#
# Normalization (A1): c = 1, time in 1/ω₀, length in c/ω₀, density in units of the critical
# density n_c, velocity in c, fields in mₑcω₀/e, charge in e, mass in mₑ. Then the plasma
# frequency of a density n (in n_c) of particles of charge q and mass m is ω_p² = q²n/m (in ω₀²),
# Maxwell's equations read ∂ₜE = ∇×B − J, ∂ₜB = −∇×E, ∇·E = ρ, and the equation of motion is
# dp/dt = q(E + v×B) with p = γmv.
#
# Weibel (filamentation) instability of two cold counter-streaming electron beams. Each beam
# s = ±1 has the lab-frame density n₀/2 and the velocity s·v₀ŷ, γ₀ = (1 − v₀²)^(-1/2); the ions
# are an immobile uniform background of density n₀, so ω_p² = n₀ (q = −1, m = 1). Take a
# perturbation ∝ exp(i(kx − ωt)) with k along x, and fields Ey, Bz (and Ex). There is no y
# dependence, so the convective derivative of the unperturbed flow vanishes.
#
#   Faraday:      −iωBz = −ikEy, so Bz = (k/ω)·Ey.
#   Momentum:     the perturbed momentum of a beam moving along y is δp_x = γ₀m·δv_x (transverse to
#                 the flow) and δp_y = γ₀³m·δv_y (along the flow, δγ = γ₀³v₀δv_y). With
#                 v×B = (v_y·Bz, −vₓ·Bz, 0) and vₓ first order:
#                   −iω·γ₀m·δvₓ,ₛ = q(Ex + s·v₀·Bz),    −iω·γ₀³m·δv_y,ₛ = q·Ey.
#   Continuity:   −iω·δnₛ + ik(n₀/2)·δvₓ,ₛ = 0, so δnₛ = (k/ω)(n₀/2)·δvₓ,ₛ.
#   Gauss:        ik·Ex = q·Σₛ δnₛ. The Bz terms of δvₓ,ₛ cancel in the sum over s, which leaves
#                 Ex·(1 − ω_p²/(γ₀ω²)) = 0: Ex = 0 on the Weibel branch (the plasma oscillation
#                 ω² = ω_p²/γ₀ decouples).
#   Current:      δJ_y = q·Σₛ (s·v₀·δnₛ + (n₀/2)·δv_y,ₛ)
#                      = i(q²n₀/(ωm))·Ey·[v₀²k²/(γ₀ω²) + 1/γ₀³], using Bz = (k/ω)Ey.
#   Ampère (y):   −iω·Ey = −ik·Bz − δJ_y.
#
# Eliminating Ey gives ω² − k² − ω_p²/γ₀³ − ω_p²v₀²k²/(γ₀ω²) = 0, i.e. with X = ω²
#
#     X² − A·X − B = 0,   A = k² + ω_p²/γ₀³,   B = ω_p²v₀²k²/γ₀,
#
# whose root X₋ = (A − √(A² + 4B))/2 is negative for every k > 0: a purely growing mode with the
# field-amplitude growth rate γ = √(−X₋) (the Bz energy grows as e^{2γt}). Checks of the
# formula, asserted in T-216: the root solves the quartic; as k → ∞, γ → ω_p·v₀/√γ₀ (nonrelativistic:
# ω_p·v₀/c); γ grows monotonically with k. No reference is cited for the relation: it is derived
# above and unverified against the literature.

include(joinpath(REPO, "scripts", "crosscode", "theory.jl"))   # module Theory: landau_root, two_stream_cold_growth
using .Theory: landau_root, two_stream_cold_growth

"""
    weibel_cold_growth(k, v0; ωp2 = 1) -> γ

Field-amplitude growth rate of the cold symmetric Weibel instability (derivation above), for
the wavenumber `k` (ω₀/c), the beam speed `v0` (c) and the total beam density as `ωp2 = ω_p²`
(n₀ in n_c for electrons); in units of ω₀.
"""
function weibel_cold_growth(k, v0; ωp2 = 1)
    γ0 = 1 / sqrt(1 - v0^2)
    A = k^2 + ωp2 / γ0^3
    B = ωp2 * v0^2 * k^2 / γ0
    X = (A - sqrt(A^2 + 4B)) / 2
    return X < 0 ? sqrt(-X) : zero(X)
end

"The Weibel dispersion function D(ω) = ω² − k² − ω_p²/γ₀³ − ω_p²v₀²k²/(γ₀ω²); zero at a root."
function weibel_dispersion(ω, k, v0; ωp2 = 1)
    γ0 = 1 / sqrt(1 - v0^2)
    return ω^2 - k^2 - ωp2 / γ0^3 - ωp2 * v0^2 * k^2 / (γ0 * ω^2)
end
