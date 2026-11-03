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
#
# Landau damping with a relativistic push (WP-025, post hoc). Smilei and PIConGPU move the
# particles with v = p/γ, so at T = 2.5e-3 mₑc² the resonant electrons (v ≈ ω/k ≈ 3 v_th) are
# slower than their momenta say, and the rate differs from `landau_root`'s by a few per cent,
# by an amount that depends on how the thermal momenta are loaded. `landau_rate_relativistic`
# solves the initial-value problem of one Fourier mode in time instead of continuing a
# dispersion relation: electrons (q = −1, m = 1, n = 1, ω_p = 1), immobile ions, an isotropic
# load f₀(|p|) with ∫ f₀ d³p = 1, and a density perturbation of the load's shape at t = 0.
# Linearizing ∂ₜf + v ∂_z f − E ∂_{p_z} f = 0 with ik Ê = −∫ f̂ d³p gives, for Ê = (iα/k) e,
#
#     e(t) = N(t) + (1/k) ∫₀ᵗ S(t − s) e(s) ds,
#     N(t) = ∫ f₀ cos(k v_z t) d³p,   S(τ) = ∫ f₀'(p) (p_z/p) sin(k v_z τ) d³p,   v = p/γ,
#
# with the angular integrals in closed form (μ = cos θ): ∫ cos(aμ) dμ = 2 sin a / a and
# ∫ μ sin(aμ) dμ = 2 (sin a − a cos a)/a². For the non-relativistic Maxwellian, S(τ) =
# −kτ exp(−k²Tτ²/2) and the asymptotic rate is `landau_root`'s (T-394). The rate is fitted on
# the maxima of e(t)² over a late window, where the higher roots have decayed. A second
# implementation, independent of this one (the physics breaker of WP-025, with a numerical
# angular quadrature), gave the same rates to 1e-5; neither is checked against the literature.
# The grid (TSC form factor, Yee differences) is not included.
#
# The cold-beam initial-value problem (WP-031). The growth rates above are the growing root
# alone; a run starts from a density perturbation, which excites every root, and the
# pre-registered window fit samples the sum. `cold_ivp` writes the linearized cold-fluid
# equations of one Fourier mode e^{ik·r} as dy/dt = M y and solves them exactly, y(t) =
# exp(M t) y(0), so the same fit applied to y(t) gives the rate a run without discretization
# error would report. Beams s with charge q_s, mass m_s, density n_s, velocity v_s and
# γ_s = (1 − v_s²)^(-1/2), immobile ions; the amplitudes are complex.
#
#   Two-stream (k ∥ v, 1D), y = (δn_s, δv_s for each beam, E):
#     ∂ₜδnₛ = −ik vₛ δnₛ − ik nₛ δvₛ,   ∂ₜδvₛ = −ik vₛ δvₛ + qₛ E/(mₛ γₛ³),
#     ∂ₜE = −Σₛ qₛ (vₛ δnₛ + nₛ δvₛ)            (Ampère with ∇×B = 0)
#   Weibel (k ⊥ v, v along z, the frame (k̂, ê, ẑ) of the 2D Weibel derivation above),
#   y = (δnₛ, δv_k,ₛ, δv_z,ₛ for each beam, E_k, E_z, B_e):
#     ∂ₜδnₛ = −ik nₛ δv_k,ₛ,   ∂ₜδv_k,ₛ = qₛ (E_k − vₛ B_e)/(mₛ γₛ),   ∂ₜδv_z,ₛ = qₛ E_z/(mₛ γₛ³),
#     ∂ₜE_k = −Σₛ qₛ nₛ δv_k,ₛ,   ∂ₜE_z = ik B_e − Σₛ qₛ (vₛ δnₛ + nₛ δv_z,ₛ),   ∂ₜB_e = ik E_z.
#
# Gauss's law, ik E(_k) = Σₛ qₛ δnₛ, is a conserved combination of both systems (its time
# derivative vanishes identically), so a state that satisfies it at t = 0 keeps it (T-411).
# The initial state of a case is its density perturbation, δnₛ = αₛ nₛ, with the velocities
# unperturbed, B = 0 and the longitudinal E from Gauss's law: the state Smilei and ParticleMesh
# load, up to the time at which each code takes the loaded momenta (#226) and the grid's
# sampling of the profile, both of which vanish as Δt and Δx go to zero. The eigenvalues of M
# are −iω at the roots of the dispersion relations above, and zero for the static modes.

module Theory

using SpecialFunctions: erfcx

export plasma_Z, landau_root, two_stream_cold_growth
export mode_wavevector, weibel_cold_growth_2d, weibel_dispersion_2d, yee_energy_ratio
export thermal_load, landau_rate_relativistic
export two_stream_ivp_matrix, weibel_ivp_matrix, ivp_solution, cold_ivp, gauss_residual

using LinearAlgebra: LinearAlgebra   # a stdlib, loaded through @stdlib on the default LOAD_PATH (no project entry)

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

"""
    thermal_load(kind, T) -> f₀

The isotropic momentum distribution f₀(|p|) of a thermal load at temperature T (mₑc²),
normalized to ∫ f₀ d³p = 1: `:momentum_maxwellian`, each component of p normal with variance
T (PIConGPU's `unary::Temperature`, and the non-relativistic Maxwellian when pushed with
v = p); `:classical_energy`, the kinetic energy E drawn from the classical Maxwell–Boltzmann
energy distribution 2√(E/π) T^(-3/2) e^(-E/T) and |p| = √((1 + E)² − 1), isotropic
(Smilei's `maxwell-juettner` below T = 0.1, scripts/crosscode/README.md);
`:velocity_maxwellian`, each component of v normal with variance T and u = γv (ParticleMesh's
electromagnetic module in 3V, `problem.jl`), so f₀(p) = f_v(p/γ)/γ⁵ (d³p = γ⁵ d³v).
"""
function thermal_load(kind, T)
    kind === :momentum_maxwellian && return p -> exp(-p^2 / (2T)) / (2π * T)^1.5
    kind === :classical_energy && return function (p)
        γ = sqrt(1 + p^2)
        E = γ - 1
        return 2 / sqrt(π) * T^-1.5 * sqrt(E) * exp(-E / T) * (p / γ) / (4π * p^2)
    end
    kind === :velocity_maxwellian && return function (p)
        γ = sqrt(1 + p^2)
        return exp(-(p / γ)^2 / (2T)) / (2π * T)^1.5 / γ^5
    end
    error("thermal_load: unknown kind $kind")
end

"""
    landau_rate_relativistic(k, T; load = :momentum_maxwellian, relativistic = true,
                             window = (8.0, 22.0), tmax = 25.0, h = 0.0025, np = 3000, pmax)
        -> (; gamma, omega, maxima, t, e)

Damping rate and frequency (ω_p units) of the field amplitude of one Fourier mode k (ω_p/c
units, c = 1) of the linearized Vlasov–Poisson system with the push v = p/γ (`relativistic =
true`) or v = p, for the load `thermal_load(load, T)`: the Volterra equation in the header,
by the trapezoid rule with step h on [0, tmax] and a midpoint rule in |p| with np points up to
pmax (default 14√T, where the load is below 1e-40 of its peak). γ is −½ the slope of a
least-squares line through log e(t)² at its maxima in `window`; ω is π over their mean
spacing, so it is quantized by h (about 3e-4 at the default). At k = 10, T = 2.5e-3 the
momentum grid is converged (np doubled moves γ by 1e-11); halving h moves γ by 2e-6 for the
momentum and velocity Maxwellians and by 1.9e-5 for the classical-energy load, not
monotonically (1.6e-5 at h/4), because the maxima are taken on the time grid without
refinement.
"""
function landau_rate_relativistic(k, T; load = :momentum_maxwellian, relativistic = true,
                                  window = (8.0, 22.0), tmax = 25.0, h = 0.0025, np = 3000,
                                  pmax = 14 * sqrt(T))
    f = thermal_load(load, T)
    dp = pmax / np
    ps = [(i - 0.5) * dp for i in 1:np]
    w = [2π * p^2 * dp for p in ps]
    fp = f.(ps)
    dfp = [(f(p + 1e-6 * pmax) - f(p - 1e-6 * pmax)) / (2e-6 * pmax) for p in ps]
    vel = relativistic ? ps ./ sqrt.(1 .+ ps .^ 2) : ps
    ts = collect(0:h:tmax)
    sinc2(a) = abs(a) < 1e-6 ? 2 - a^2 / 3 : 2 * sin(a) / a
    j1(a) = abs(a) < 1e-4 ? 2a / 3 : 2 * (sin(a) - a * cos(a)) / a^2
    N = [sum(w[i] * fp[i] * sinc2(k * vel[i] * t) for i in 1:np) for t in ts]
    S = [sum(w[i] * dfp[i] * j1(k * vel[i] * t) for i in 1:np) for t in ts]
    n = length(ts)
    e = zeros(n)
    e[1] = N[1]
    for m in 2:n            # trapezoid; the end term vanishes because S(0) = 0
        acc = S[m] * e[1] / 2
        for j in 2:m-1
            acc += S[m - j + 1] * e[j]
        end
        e[m] = N[m] + h / k * acc
    end
    U = e .^ 2
    idx = [i for i in 2:n-1 if U[i-1] < U[i] >= U[i+1] && window[1] <= ts[i] <= window[2]]
    length(idx) >= 3 || error("landau_rate_relativistic: fewer than three maxima in the window")
    x, y = ts[idx], log.(U[idx])
    xm, ym = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
    return (; gamma = -slope / 2, omega = π * (length(x) - 1) / (x[end] - x[1]), maxima = length(x), t = ts, e)
end

"""
    two_stream_ivp_matrix(k, beams) -> Matrix{ComplexF64}

The matrix M of the linearized cold-fluid two-stream system of the file header (k ∥ v), for
`beams`, a list of named tuples (; q, m, n, v); the state is (δn₁, δv₁, δn₂, δv₂, …, E).
"""
function two_stream_ivp_matrix(k, beams)
    nb = length(beams)
    iE = 2nb + 1
    M = zeros(ComplexF64, iE, iE)
    for (j, b) in enumerate(beams)
        n, v = 2j - 1, 2j
        γ = 1 / sqrt(1 - b.v^2)
        M[n, n] = -im * k * b.v
        M[n, v] = -im * k * b.n
        M[v, v] = -im * k * b.v
        M[v, iE] = b.q / (b.m * γ^3)
        M[iE, n] = -b.q * b.v
        M[iE, v] = -b.q * b.n
    end
    return M
end

"""
    weibel_ivp_matrix(k, beams) -> Matrix{ComplexF64}

The matrix M of the linearized cold-fluid Weibel system of the file header (k ⊥ v, the beams
along z), for `beams` (; q, m, n, v), v the z velocity; the state is (δn₁, δv_k,₁, δv_z,₁, …,
E_k, E_z, B_e).
"""
function weibel_ivp_matrix(k, beams)
    nb = length(beams)
    iEk, iEz, iB = 3nb + 1, 3nb + 2, 3nb + 3
    M = zeros(ComplexF64, iB, iB)
    for (j, b) in enumerate(beams)
        n, vk, vz = 3j - 2, 3j - 1, 3j
        γ = 1 / sqrt(1 - b.v^2)
        M[n, vk] = -im * k * b.n
        M[vk, iEk] = b.q / (b.m * γ)
        M[vk, iB] = -b.q * b.v / (b.m * γ)
        M[vz, iEz] = b.q / (b.m * γ^3)
        M[iEk, vk] = -b.q * b.n
        M[iEz, n] = -b.q * b.v
        M[iEz, vz] = -b.q * b.n
    end
    M[iEz, iB] = im * k
    M[iB, iEz] = im * k
    return M
end

"""
    ivp_solution(M, y0, t) -> Matrix{ComplexF64}

The exact solution exp(M tᵢ) y0 of dy/dt = M y at each time of `t`, one column per time (the
matrix exponential, so a defective eigenvalue, such as the repeated static zero modes, needs no
special care).
"""
ivp_solution(M, y0, t) = reduce(hcat, [exp(M * ti) * y0 for ti in t])

"""
    gauss_residual(sol, k, beams; kind) -> Vector

|ik E − Σₛ qₛ δnₛ| at each column of a solution (E the longitudinal field: E for `:two_stream`,
E_k for `:weibel`), relative to max |Σₛ qₛ δnₛ| over beams of the solution; zero for every t
when it is zero at t = 0 (file header).
"""
function gauss_residual(sol, k, beams; kind)
    nb = length(beams)
    per, iE = kind === :two_stream ? (2, 2nb + 1) : (3, 3nb + 1)
    charge = [sum(b.q * sol[per * (j - 1) + 1, c] for (j, b) in enumerate(beams)) for c in axes(sol, 2)]
    scale = maximum(c -> maximum(abs(b.q * sol[per * (j - 1) + 1, c]) for (j, b) in enumerate(beams)), axes(sol, 2))
    return abs.(im * k .* sol[iE, :] .- charge) ./ scale
end

"""
    cold_ivp(case) -> (; kind, k, beams, M, y0, index, energy_scale)

The cold-beam initial-value problem of a cross-code case file (`two-stream`, schema 1, k along
the drift; `weibel-2d`, schema 2, the drift along z and k in the plane): the mobile perturbed
species are the beams (q, m, n, v = the drift along the instability's axis), the immobile ones
the neutralizing background. `y0` is the state of the file header (δnₛ = αₛ nₛ, velocities
unperturbed, B = 0, the longitudinal E from Gauss's law), `index` the entry of the observed
field (E for two-stream, B_e for Weibel), and `energy_scale` the factor that turns |y[index]|²
into the box energy of that field, L/4 in 1D and L_x L_y/4 in 2D (a field Re(A e^{ik·r}) has
∫ ½|f|² dV = |A|² V/4). Refuses a thermal species and a drift off the axis it assumes.
"""
function cold_ivp(case)
    two = get(case, "schema", 1) == 2
    g = case["grid"]
    beams = NamedTuple{(:q, :m, :n, :v, :alpha),NTuple{5,Float64}}[]
    for s in case["species"]
        s["velocity"] == "cold" || error("cold_ivp: species $(s["name"]) is not cold")
        s["mobile"] || continue
        pert = get(s, "perturbation", nothing)
        if two
            d = Float64.(s["drift"])
            d[1] == 0 && d[2] == 0 || error("cold_ivp: species $(s["name"]): the Weibel case drifts along z only")
            v = d[3]
        else
            v = Float64(s["drift"])
        end
        push!(beams, (; q = Float64(s["charge"]), m = Float64(s["mass"]), n = Float64(s["density"]),
                       v, alpha = pert === nothing ? 0.0 : Float64(pert["amplitude"])))
    end
    length(beams) >= 1 || error("cold_ivp: no mobile species")
    if two
        k = mode_wavevector(case["wavevector"]["mode"], g["length"]).k
        M = weibel_ivp_matrix(k, beams)
        nb = length(beams)
        y0 = zeros(ComplexF64, 3nb + 3)
        for (j, b) in enumerate(beams)
            y0[3j - 2] = b.alpha * b.n
        end
        y0[3nb + 1] = sum(b.q * b.alpha * b.n for b in beams) / (im * k)
        return (; kind = :weibel, k, beams, M, y0, index = 3nb + 3, energy_scale = prod(Float64.(g["length"])) / 4)
    else
        modes = unique(Int(s["perturbation"]["mode"]) for s in case["species"] if haskey(s, "perturbation"))
        L = Float64(g["length"])
        k = 2π * only(modes) / L
        M = two_stream_ivp_matrix(k, beams)
        nb = length(beams)
        y0 = zeros(ComplexF64, 2nb + 1)
        for (j, b) in enumerate(beams)
            y0[2j - 1] = b.alpha * b.n
        end
        y0[2nb + 1] = sum(b.q * b.alpha * b.n for b in beams) / (im * k)
        return (; kind = :two_stream, k, beams, M, y0, index = 2nb + 1, energy_scale = L / 4)
    end
end

end
