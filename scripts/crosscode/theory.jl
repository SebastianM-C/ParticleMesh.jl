# Linear theory of the cross-code cases, computed from the dispersion relations: the 1D
# electrostatic cases (Landau damping, cold two-stream) and the 2D cases of schema 2 (oblique
# Landau damping, the 2D Weibel instability). Frequencies and rates in units of the electron
# plasma frequency ω_p unless a function says otherwise.
#
# Oblique Landau damping (landau-oblique-2d). For an isotropic Maxwellian the electrostatic
# dispersion relation depends on k only through |k|: the velocity integral along k̂ is the
# same 1D Maxwellian whatever the direction of k. So the root at |k|λ_D is `landau_root`'s,
# with |k| from the mode and the box (`mode_wavevector`).
#
# 2D Weibel (filamentation) instability with k ⊥ v₀ (weibel-2d), cold fluid, derived here.
# Normalization (A1): c = 1, time 1/ω₀, length c/ω₀, density n_c, velocity c, fields mₑcω₀/e,
# charge e, mass mₑ; ω_p² = q²n/m; ∂ₜE = ∇×B − J, ∂ₜB = −∇×E, ∇·E = ρ, dp/dt = q(E + v×B),
# p = γmv. Two cold electron beams s = ±1, each of lab-frame density n₀/2 and velocity
# s·v₀ẑ along the collapsed axis z, γ₀ = (1 − v₀²)^(-1/2); immobile ions of density n₀, so
# ω_p² = n₀. The perturbation is ∝ exp(i(k·r − ωt)) with k = k k̂ in the (x, y) plane; with
# ê = ẑ × k̂ the triple (k̂, ê, ẑ) is right-handed. The fields are E_k (along k̂), E_z and
# B_e (along ê); nothing depends on z, so the convective derivative of the flow vanishes.
#
#   Faraday:      −iω B_e ê = −i k k̂ × (E_z ẑ) = i k E_z ê, so B_e = −(k/ω) E_z.
#   Momentum:     (s v₀ ẑ) × (B_e ê) = −s v₀ B_e k̂. Across the flow the mass is γ₀m, along
#                 it γ₀³m (δγ = γ₀³ v₀ δv_z):
#                   −iω γ₀m δv_k,ₛ = q (E_k − s v₀ B_e),    −iω γ₀³m δv_z,ₛ = q E_z.
#   Continuity:   −iω δnₛ + i k (n₀/2) δv_k,ₛ = 0, so δnₛ = (k/ω)(n₀/2) δv_k,ₛ.
#   Gauss:        i k E_k = q Σₛ δnₛ. The B_e terms cancel in the sum over s, which leaves
#                 E_k (1 − ω_p²/(γ₀ω²)) = 0: E_k = 0 on the Weibel branch.
#   Current:      δJ_z = q Σₛ (s v₀ δnₛ + (n₀/2) δv_z,ₛ)
#                      = i (q²n₀/(ωm)) E_z [v₀²k²/(γ₀ω²) + 1/γ₀³], using B_e = −(k/ω) E_z.
#   Ampère (z):   −iω E_z = (i k k̂ × B_e ê)_z − δJ_z = i k B_e − δJ_z.
#
# Eliminating B_e gives ω² − k² − ω_p²/γ₀³ − ω_p²v₀²k²/(γ₀ω²) = 0; with X = ω²,
#
#     X² − A X − B = 0,   A = k² + ω_p²/γ₀³,   B = ω_p² v₀² k²/γ₀,
#
# whose root X₋ = (A − √(A² + 4B))/2 = −2B/(A + √(A² + 4B)) is negative for every k > 0: a
# purely growing mode with the field-amplitude rate γ = √(−X₋); the magnetic energy grows as
# e^{2γt}. The growing field is B_e, the in-plane component of B perpendicular to k (B·k̂ = 0
# by ∇·B = 0), driven by J_z. A rotation that takes k̂ to x and ẑ to y maps this case onto the
# 1D one of WP-014 (k along x, beams along y, lib/ParticleMeshElectromagnetic/test/theory.jl),
# so the 2D rate at |k| equals the 1D rate at the same |k|, v₀ and ω_p (T-273 checks it). The
# relation is derived here and is unverified against the literature. It is the continuum
# relation: the Yee grid changes k to its discrete form, which the case files do not correct.

module Theory

using SpecialFunctions: erfcx

export plasma_Z, landau_root, two_stream_cold_growth
export mode_wavevector, weibel_cold_growth_2d, weibel_dispersion_2d, yee_energy_ratio

"""
    plasma_Z(ζ)

Plasma dispersion function of Fried and Conte, Z(ζ) = i√π w(ζ), with the Faddeeva function
w(ζ) = exp(-ζ²) erfc(-iζ) = erfcx(-iζ). w is entire, so this is the Landau-continued Z for
every ζ, including Im ζ < 0 (damped modes).
"""
plasma_Z(ζ) = im * sqrt(pi) * erfcx(-im * ζ)

"""
    landau_dispersion(ω, kλ)

Electrostatic dispersion function of a Maxwellian electron plasma with immobile ions,
D(ω, k) = 1 + (kλ_D)⁻² [1 + ζ Z(ζ)], ζ = ω / (√2 k v_th), with ω in ω_p units,
λ_D = v_th/ω_p and v_th = √(T/m) (Fried and Conte, *The Plasma Dispersion Function*, Academic Press, 1961, for Z; the
dispersion relation is the standard kinetic one for Langmuir waves, derived in the report).
"""
function landau_dispersion(ω, kλ)
    ζ = ω / (sqrt(2) * kλ)
    return 1 + (1 + ζ * plasma_Z(ζ)) / kλ^2
end

"""
    landau_root(kλ; guess) -> ω::Complex

The least-damped root of `landau_dispersion` by Newton iteration from the Bohm–Gross
frequency √(1 + 3k²λ²). Real part: the wave frequency; imaginary part: -γ, the damping rate of
the field amplitude (the field energy decays at 2γ).
"""
function landau_root(kλ; guess = complex(sqrt(1 + 3kλ^2), -0.1), maxiter = 100, rtol = 1e-13)
    ω = complex(float(guess))
    for _ in 1:maxiter
        h = 1e-6
        d = (landau_dispersion(ω + h, kλ) - landau_dispersion(ω - h, kλ)) / (2h)
        δ = landau_dispersion(ω, kλ) / d
        ω -= δ
        abs(δ) < rtol * abs(ω) && return ω
    end
    error("landau_root: no convergence for kλ = $kλ")
end

"""
    two_stream_cold_growth(kv0; gamma0 = 1) -> γ

Growth rate of the field amplitude for two symmetric cold electron beams at ±v₀, each with
half the density, immobile ions: 1 = (ω_p²/2γ₀³) [1/(ω - kv₀)² + 1/(ω + kv₀)²]
(the cold-beam relation treated in Birdsall and Langdon, *Plasma Physics via Computer
Simulation*, and derived in the report; the γ₀³ is the longitudinal relativistic mass of a beam with Lorentz factor γ₀, γ₀ = 1 for the
nonrelativistic relation). With a = (kv₀)² and p = 1/γ₀³ the relation is the biquadratic
(ω²)² - (2a + p) ω² + a² - p a = 0, so ω² = a + p/2 ± (1/2)√(p² + 8 a p), and the mode
grows when the smaller root is negative: γ = √(-(a + p/2 - √(p² + 8ap)/2)). Returns 0 for a
stable k. The maximum over k is at a = 3p/8, where γ = √(p/8).
"""
function two_stream_cold_growth(kv0; gamma0 = 1)
    a = kv0^2
    p = 1 / gamma0^3
    x = a + p / 2 - sqrt(p^2 + 8a * p) / 2
    return x < 0 ? sqrt(-x) : zero(x)
end

"""
    mode_wavevector(mode, lengths) -> (; kvec, k, angle)

The wavevector of the integer `mode` (m_x, m_y) in a periodic box of `lengths` (L_x, L_y):
kvec = 2π (m_x/L_x, m_y/L_y) in ω₀/c, its magnitude k and its angle atan(k_y, k_x) to the x axis
in radians. The mode cos(k·r) is periodic in the box by construction.
"""
function mode_wavevector(mode, lengths)
    kvec = Tuple(2π .* Float64.(mode) ./ Float64.(lengths))
    return (; kvec, k = sqrt(sum(abs2, kvec)), angle = atan(kvec[2], kvec[1]))
end

"""
    weibel_cold_growth_2d(k, v0; ωp2 = 1) -> γ

Field-amplitude growth rate of the cold symmetric Weibel instability of two beams at ±v₀ along
the collapsed axis z, for an in-plane wavevector of magnitude `k` (ω₀/c), with `ωp2 = ω_p²` the
total beam density (n_c), in units of ω₀; the root of the relation derived in the file header.
"""
function weibel_cold_growth_2d(k, v0; ωp2 = 1)
    γ0 = 1 / sqrt(1 - v0^2)
    A = k^2 + ωp2 / γ0^3
    B = ωp2 * v0^2 * k^2 / γ0
    X = -2B / (A + sqrt(A^2 + 4B))      # = (A − √(A² + 4B))/2, free of cancellation at large k
    return X < 0 ? sqrt(-X) : zero(X)
end

"The 2D Weibel dispersion function D(ω) = ω² − k² − ω_p²/γ₀³ − ω_p²v₀²k²/(γ₀ω²) of the file header; zero at a root."
function weibel_dispersion_2d(ω, k, v0; ωp2 = 1)
    γ0 = 1 / sqrt(1 - v0^2)
    return ω^2 - k^2 - ωp2 / γ0^3 - ωp2 * v0^2 * k^2 / (γ0 * ω^2)
end

"""
    yee_energy_ratio(kvec, cell_lengths) -> Real

The ratio of the Ex and Ey field energies of a longitudinal wave E = −∇φ, φ ∝ cos(k·r + φ₀),
on the Yee grid: the difference operators give Ex ∝ (2/Δx) sin(k_xΔx/2) and Ey ∝
(2/Δy) sin(k_yΔy/2), so the ratio is (sin(k_xΔx/2)/Δx)² / (sin(k_yΔy/2)/Δy)², which tends to
(k_x/k_y)² as the cells shrink.
"""
yee_energy_ratio(kvec, h) = (sin(kvec[1] * h[1] / 2) / h[1])^2 / (sin(kvec[2] * h[2] / 2) / h[2])^2

end
