# Linear theory for the 1D electrostatic cases, computed from the dispersion relations.
# All frequencies and rates in units of the electron plasma frequency ω_p.

module Theory

using SpecialFunctions: erfcx

export plasma_Z, landau_root, two_stream_cold_growth

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

end
