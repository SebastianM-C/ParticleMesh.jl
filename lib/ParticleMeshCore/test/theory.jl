# Linear theory for the electrostatic benchmarks, computed here from the dispersion relations;
# no value is taken from memory. Shared with scripts/es1d/ (which includes this file).
#
# Landau damping. Linearizing the Vlasov–Poisson system about a Maxwellian
# f₀(v) = n₀ (2π v_th²)^(-1/2) exp(-v²/2v_th²) with immobile ions and Fourier–Laplace
# transforming in (x, t) gives the dispersion relation
#     ε(k, ω) = 1 - (ω_p²/k²) ∫ ∂f₀/∂v / (v - ω/k) dv / n₀ = 0
# with the Landau contour. With ζ = ω/(√2 k v_th) and the plasma dispersion function
#     Z(ζ) = π^(-1/2) ∫ exp(-t²)/(t - ζ) dt  (Landau contour; Fried and Conte, "The Plasma
#     Dispersion Function", Academic Press, 1961),
# the integral gives ε = 1 + (1 + ζ Z(ζ)) / (k λ_D)², λ_D = v_th/ω_p. Z is evaluated through
# the Faddeeva function, Z(ζ) = i√π w(ζ), w(ζ) = erfcx(-iζ), valid in the whole complex plane,
# and Z'(ζ) = -2(1 + ζ Z(ζ)). The least-damped root is found by Newton's method from the
# Bohm–Gross estimate ω² = ω_p² + 3k²v_th².
#
# Cold two-stream. Two cold beams of density n₀/2 each (beam plasma frequency ω_b² = ω_p²/2)
# at velocities ±v₀; linearized cold-fluid equations plus Gauss's law give
#     1 = ω_b²/(ω - k v₀)² + ω_b²/(ω + k v₀)².
# Multiplying out, with X = ω²: X² - 2(k²v₀² + ω_b²) X + k²v₀²(k²v₀² - 2ω_b²) = 0, so
#     X₋ = a - √(a² - b),  a = k²v₀² + ω_b²,  b = k²v₀²(k²v₀² - 2ω_b²),
# and for k v₀ < √2 ω_b = ω_p the root X₋ < 0 is unstable with growth rate γ = √(-X₋).

using SpecialFunctions: erfcx

plasma_Z(ζ) = im * sqrt(π) * erfcx(-im * ζ)

"""
    landau_root(kλD; ω_p = 1) -> ω (complex, in units of ω_p)

Least-damped root of 1 + (1 + ζZ(ζ))/(kλ_D)² = 0, ω = √2 kλ_D ζ ω_p.
"""
function landau_root(kλD::Real; tol = 1e-14, maxiter = 100)
    K2 = kλD^2
    ω = complex(sqrt(1 + 3K2), 0.0)
    ζ = ω / (sqrt(2) * kλD)
    for _ in 1:maxiter
        Z = plasma_Z(ζ)
        D = 1 + (1 + ζ * Z) / K2
        dZ = -2 * (1 + ζ * Z)
        dD = (Z + ζ * dZ) / K2
        Δ = D / dD
        ζ -= Δ
        abs(Δ) < tol * abs(ζ) && break
    end
    return sqrt(2) * kλD * ζ
end

"Growth rate of the symmetric cold two-stream instability, in units of ω_p (zero if stable)."
function two_stream_growth(kv0_ωp::Real)
    K2 = kv0_ωp^2
    ωb2 = 1 / 2
    a = K2 + ωb2
    b = K2 * (K2 - 2ωb2)
    X = a - sqrt(a^2 - b)
    return X < 0 ? sqrt(-X) : 0.0
end
