# WP-085: the round beam of a Zernike pupil in PulseShapes, T-1160 to T-1163, against closed forms:
# the polynomials of OSA/ANSI, the round Gaussian beam of a Gaussian pupil, the Airy pattern of a
# uniform one, Maréchal's Strehl ratio, astigmatism's two line foci. em3d_zernike.jl takes the beam
# onto the 3D antenna. Its own helpers only.

using ForwardDiff: ForwardDiff
using SpecialFunctions: besselj1, erf
using ParticleMeshElectromagnetic: PulseShapes
using .PulseShapes: ZernikeBeam, zernike, zernike_index, zernike_radial, radial_node, quadrature_radius, transverse_dimensions,
                    profile, profile_moments, wavefront, parameter_names, parameter_values, with_parameter_values, convert_eltype

"The pupil quadrature of a ZernikeBeam of `nr` radial nodes and `na` angles: (ρ, θ, weight) with Σ weight = π, the disk's area."
function disk_rule(::Type{T}, nr, nθ) where {T}
    pts = Tuple{T,T,T}[]
    for i in 1:nr
        ρ, w = radial_node(nr, i, T)
        for k in 1:nθ
            push!(pts, (ρ, 2 * T(π) * (k - one(T) / 2) / nθ, ρ * w * 2 * T(π) / nθ))
        end
    end
    return pts
end

"The round Gaussian beam of waist w: (1 + iξ/z_R)^(−1)·exp(−r²/(w²(1 + iξ/z_R)))."
round_gaussian(w, ξ, y, z) = (q = 1 + im * 2ξ / w^2; exp(-(y^2 + z^2) / (w^2 * q)) / q)

# --- T-1160 ---------------------------------------------------------------------------------

# The indexing is OSA/ANSI's (j = (n(n + 2) + m)/2), the first fifteen polynomials are the
# textbook ones (Thibos et al. 2002, Table 1, with θ from the pupil's y towards its z), and the
# pupil rule (Gauss–Legendre in ρ, the trapezoid in θ) makes them orthonormal over the disk to
# roundoff up to the order it integrates exactly: 2·nr − 1 in ρ, harmonics below nθ. The nodes
# integrate ρᵏ exactly to k = 2N − 1 in Float32 and Float64, to N·eps(T); the beam is isbits in the A8 matrix and
# stands alone with the module (T-900's copy of the module has it, with the package's bits).
@testset "T-1160 Zernike polynomials are OSA/ANSI's, orthonormal over the pupil quadrature to roundoff; the beam is isbits and stands alone [A7, A8, WP-085]" begin
    nm = [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3), (4, -4), (4, -2), (4, 0), (4, 2), (4, 4)]
    @test [zernike_index(j) for j in 0:14] == nm
    @test all(j -> ((n, m) = zernike_index(j); (n * (n + 2) + m) ÷ 2 == j && abs(m) <= n && iseven(n - m)), 0:200)
    closed = (
        (ρ, θ) -> 1.0,
        (ρ, θ) -> 2ρ * sin(θ), (ρ, θ) -> 2ρ * cos(θ),
        (ρ, θ) -> √6 * ρ^2 * sin(2θ), (ρ, θ) -> √3 * (2ρ^2 - 1), (ρ, θ) -> √6 * ρ^2 * cos(2θ),
        (ρ, θ) -> √8 * ρ^3 * sin(3θ), (ρ, θ) -> √8 * (3ρ^3 - 2ρ) * sin(θ), (ρ, θ) -> √8 * (3ρ^3 - 2ρ) * cos(θ), (ρ, θ) -> √8 * ρ^3 * cos(3θ),
        (ρ, θ) -> √10 * ρ^4 * sin(4θ), (ρ, θ) -> √10 * (4ρ^4 - 3ρ^2) * sin(2θ), (ρ, θ) -> √5 * (6ρ^4 - 6ρ^2 + 1),
        (ρ, θ) -> √10 * (4ρ^4 - 3ρ^2) * cos(2θ), (ρ, θ) -> √10 * ρ^4 * cos(4θ))
    worst = 0.0
    for (ρ, θ) in ((0.0, 0.0), (0.3, 0.4), (0.77, -2.1), (1.0, 2.9), (0.51, 4.0))
        for j in 0:14
            worst = max(worst, abs(zernike(j, ρ, θ) - closed[j + 1](ρ, θ)))
        end
    end
    @info "T-1160 closed forms" worst
    @test worst <= 1e-14
    @test zernike_radial(6, 2, 0.6) ≈ 15 * 0.6^6 - 20 * 0.6^4 + 6 * 0.6^2 rtol = 1e-14    # R₆², a radial order the table above lacks
    # orthonormal over the disk under the rule: j ≤ 44 (n ≤ 8), products of degree 16 in ρ (with the
    # weight's ρ, 17 ≤ 2·9 − 1) and harmonics up to 16 < 40
    for T in (Float32, Float64)
        pts = disk_rule(T, 9, 40)
        @test sum(last, pts) ≈ T(π) rtol = 8eps(T)
        G = [sum(p -> p[3] * zernike(i, p[1], p[2]) * zernike(j, p[1], p[2]), pts) / T(π) for i in 0:44, j in 0:44]
        err = maximum(abs, G .- [T(i == j) for i in 0:44, j in 0:44]) / eps(T)
        @info "T-1160 orthonormality" T err
        record!("T-1160 orthonormality", T, err)
        @test err <= emC("T-1160")
        # the nodes carry Newton's roundoff, about N·eps(T) near ρ = 1, which ρᵏ multiplies by k
        for N in (1, 2, 5, 16, 40)
            e = maximum(k -> abs(sum(i -> ((ρ, w) = radial_node(N, i, T); w * ρ^k), 1:N) * (k + 1) - 1), 0:(2N - 1))
            @test e <= 4N * eps(T)
        end
    end
    for T in TYPES
        b = ZernikeBeam(; na = T(1 // 5), amplitude = SuperGaussianPupil(; radius = T(3 // 4), order = 2), aberrations = T.((1, 2, 3, 4, 5) .// 100))
        @test isbitstype(typeof(b)) && transverse_dimensions(b) == 2
        @test parameter_names(b) == (:na, :radius, :aberration1, :aberration2, :aberration3, :aberration4, :aberration5)
        @test with_parameter_values(b, parameter_values(b)) === b
    end
    m = Module(:Standalone)
    Base.include(m, joinpath(pkgdir(ParticleMeshElectromagnetic), "src", "PulseShapes", "PulseShapes.jl"))
    P = getfield(m, :PulseShapes)
    for T in (Float32, Float64)
        b = ZernikeBeam(; na = T(1 // 4), amplitude = GaussianPupil(; fill = T(1 // 2)), aberrations = T.((0, 0, 3, 0, -2, 1) .// 50),
                        radial = 12, angular = 32)
        b2 = getfield(P, :ZernikeBeam)(; na = b.na, amplitude = getfield(P, :GaussianPupil)(; fill = b.amplitude.fill), aberrations = b.aberrations,
                                       radial = b.radial, angular = b.angular)
        @test parentmodule(typeof(b2)) === P
        for (ξ, y, z) in ((-30.0, 0.0, 0.0), (0.0, 4.5, -2.0), (12.0, -9.0, 7.0))
            @test same_bits(collect(profile(b, T(ξ), T(y), T(z))), collect(getfield(P, :profile)(b2, T(ξ), T(y), T(z))))
        end
    end
    @test_throws ArgumentError ZernikeBeam(; na = 0.2, angular = 30)
    @test_throws ArgumentError ZernikeBeam(; na = 0.2, radial = 0)
end

# --- T-1161 ---------------------------------------------------------------------------------

# The pupil sum is a sum of plane waves of the paraxial equation, so 2i∂U/∂ξ + U_yy + U_zz = 0 at
# every point (ForwardDiff through `profile`, an aberrated flattened pupil), and its second
# derivatives are ForwardDiff's. A Gaussian pupil exp(−ρ²/f²) inside the aperture (f = 1/4: the
# truncation at ρ = 1 is exp(−16) ≈ 1.1·10⁻⁷) focuses to the round Gaussian beam of waist 2/(na·f);
# defocus Z₄ moves that focus by 8√3π·c/na² with the piston phase −2√3π·c, and tilt Z₂ (Z₁) moves
# it by 4π·c/na along y (z). A uniform pupil focuses to the Airy pattern 2J₁(na·r)/(na·r) in every
# direction, with its zeros at the roots of J₁ (3.8317…, 7.0156…, here found again by bisection on
# SpecialFunctions' J₁), and on the axis to (1 − e^{−iu})/(iu), u = na²ξ/2, the axial zeros at
# ξ = 4πm/na². Each to roundoff (Gauss–Legendre is exponentially accurate on these analytic
# integrands) or to the truncation's 10⁻⁷.
@testset "T-1161 ZernikeBeam solves the paraxial equation, focuses a Gaussian pupil to the round Gaussian beam and a uniform one to the Airy pattern with its zeros; defocus and tilt move the focus [A2, A7, WP-085]" begin
    ab = ZernikeBeam(; na = 0.3, amplitude = SuperGaussianPupil(; radius = 0.8, order = 3), aberrations = (0.02, -0.01, 0.05, 0.1, -0.04, 0.03, 0.06, 0, 0, 0, 0, 0.05),
                     radial = 24, angular = 64)
    worst_pde = worst_d = 0.0
    for (ξ, y, z) in ((0.0, 0.0, 0.0), (-20.0, 3.0, -4.0), (35.0, -8.0, 2.5), (5.0, 12.0, 10.0))
        U(x, a, b) = profile(ab, x, a, b)[1]
        cd(f) = complex(f(real), f(imag))
        dξ = cd(p -> ForwardDiff.derivative(x -> p(U(x, y, z)), ξ))
        dyy = cd(p -> ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> p(U(ξ, b, z)), a), y))
        dzz = cd(p -> ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> p(U(ξ, y, b)), a), z))
        dyz = cd(p -> ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> p(U(ξ, b, a)), y), z))
        u, uyy, uzz, uyz = profile(ab, ξ, y, z)
        worst_pde = max(worst_pde, abs(2im * dξ + dyy + dzz) / ab.na^2)
        worst_d = max(worst_d, abs(uyy - dyy), abs(uzz - dzz), abs(uyz - dyz)) / ab.na^2
    end
    @info "T-1161 paraxial" worst_pde worst_d
    @test worst_pde <= 1e-13
    @test worst_d <= 1e-13
    na, f = 0.3, 0.25
    w = 2 / (na * f)
    zR = w^2 / 2
    pts = ((0.0, 0.0, 0.0), (0.0, 10.0, 5.0), (zR, 20.0, -3.0), (-0.5zR, 4.0, 30.0), (2zR, -40.0, 25.0))
    gp(; ab = ()) = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = f), aberrations = ab, radial = 24, angular = 64)
    c = 0.03
    errs = (gauss = 0.0, defocus = 0.0, tilt_y = 0.0, tilt_z = 0.0)
    for (ξ, y, z) in pts
        errs = (gauss = max(errs.gauss, abs(profile(gp(), ξ, y, z)[1] - round_gaussian(w, ξ, y, z))),
                defocus = max(errs.defocus, abs(profile(gp(; ab = (0, 0, 0, c)), ξ, y, z)[1] -
                                                cis(-2√3 * π * c) * round_gaussian(w, ξ - 8√3 * π * c / na^2, y, z))),
                tilt_y = max(errs.tilt_y, abs(profile(gp(; ab = (0, c)), ξ, y, z)[1] - round_gaussian(w, ξ, y - 4π * c / na, z))),
                tilt_z = max(errs.tilt_z, abs(profile(gp(; ab = (c,)), ξ, y, z)[1] - round_gaussian(w, ξ, y, z - 4π * c / na))))
    end
    @info "T-1161 Gaussian pupil" errs
    @test all(e -> e <= 3e-7, values(errs))
    ub = ZernikeBeam(; na = 0.2, radial = 24, angular = 64)
    airy = 0.0
    for r in (0.5, 3.0, 11.0, 27.0, 44.0), φ in (0.0, 0.3, 1.1)
        x = ub.na * r
        airy = max(airy, abs(profile(ub, 0.0, r * cos(φ), r * sin(φ))[1] - 2besselj1(x) / x))
    end
    axial = maximum(ξ -> (u = ub.na^2 * ξ / 2; abs(profile(ub, ξ, 0.0, 0.0)[1] - (1 - cis(-u)) / (im * u))), (-300.0, -40.0, 7.0, 150.0, 314.0))
    @info "T-1161 Airy" airy axial
    @test airy <= 1e-13
    @test axial <= 1e-13
    # quadrature_radius: the Airy pattern holds out to it, and four times further out the angular
    # rule's alias J_K(na·r)·cos(K·φ) (K samples) is of order one (radial nodes to spare)
    ua = ZernikeBeam(; na = 0.2, radial = 40, angular = 16)
    ra = quadrature_radius(ua)
    @test ra == 16 / (4 * 0.2)
    alias(r) = maximum(φ -> abs(profile(ua, 0.0, r * cos(φ), r * sin(φ))[1] - 2besselj1(ua.na * r) / (ua.na * r)), range(0, π / 8; length = 9))
    @info "T-1161 quadrature radius" ra inside = alias(ra) outside = alias(4ra)
    @test alias(ra) <= 1e-6
    @test alias(4ra) >= 1e-2
    function root(f, a, b)                       # bisection to the last bit
        fa = f(a)
        for _ in 1:200
            m = (a + b) / 2
            (fm = f(m)) == 0 && return m
            sign(fm) == sign(fa) ? (a = m; fa = fm) : (b = m)
        end
        return (a + b) / 2
    end
    j1 = (root(besselj1, 3.0, 4.5), root(besselj1, 6.5, 7.5))
    @test j1[1] ≈ 3.8317059702075125 rtol = 1e-15
    @test j1[2] ≈ 7.015586669815619 rtol = 1e-15
    for (k, z0) in enumerate(j1)
        r = root(r -> real(profile(ub, 0.0, r, 0.0)[1]), (z0 - 0.3) / ub.na, (z0 + 0.3) / ub.na)
        @info "T-1161 Airy zero" k r z0 / ub.na
        @test r ≈ z0 / ub.na rtol = 1e-12
    end
    @test abs(profile(ub, 4π / ub.na^2, 0.0, 0.0)[1]) <= 1e-13           # the first axial zero
end

# --- T-1162 ---------------------------------------------------------------------------------

# Maréchal: a small wavefront error of rms σ lowers the focus of a uniform pupil to 1 − (2πσ)², up
# to a remainder of order (2πσ)⁴, for each of the modes 3 to 14 (tilts move the peak, so they are
# left out) and for a mix; exp(−(2πσ)²) to the same order. A phase-only wavefront keeps the energy:
# profile_moments does not see it, and its power is the beam's ∫∫|U|² dy dz and ⟨k_y²⟩ its
# −∫∫ Re(U*·U_yy)/∫∫|U|², by a direct quadrature over the plane at the focus and half a Rayleigh
# length before it (steps of 2.5 across a beam of waist 33 over ±150, where the Gaussian's tails
# are below exp(−30); the pupil's cut edge, exp(−11) in amplitude, leaves 10⁻⁸ of the power in
# slowly decaying tails beyond), aberrated or not; ⟨k_z²⟩ is ⟨k_y²⟩.
@testset "T-1162 Maréchal: a wavefront of rms σ lowers the focus to 1 − (2πσ)² up to (2πσ)⁴; a phase-only wavefront keeps the beam's power, which is ∫∫|U|² [A2, A7, WP-085]" begin
    na = 0.2
    σ = 0.02
    worst = worst_exp = 0.0
    for j in 3:14
        ab = ntuple(i -> i == j ? σ : 0.0, j)
        S = abs2(profile(ZernikeBeam(; na, aberrations = ab, radial = 16, angular = 64), 0.0, 0.0, 0.0)[1])
        worst = max(worst, abs(S - (1 - (2π * σ)^2)) / (2π * σ)^4)
        worst_exp = max(worst_exp, abs(S - exp(-(2π * σ)^2)) / (2π * σ)^4)
    end
    mix = (0, 0, 0.012, 0.0, 0.009, -0.01, 0.006, 0.0, 0.004, -0.008, 0.0, 0.011)
    σm = sqrt(sum(abs2, mix))
    Sm = abs2(profile(ZernikeBeam(; na, aberrations = mix, radial = 16, angular = 64), 0.0, 0.0, 0.0)[1])
    worst = max(worst, abs(Sm - (1 - (2π * σm)^2)) / (2π * σm)^4)
    @info "T-1162 Maréchal" worst worst_exp mix_S = Sm
    @test worst <= 1
    @test worst_exp <= 1
    @test Sm < 1 - (2π * σm)^2 / 2
    # the power by a direct quadrature over the plane
    fill_ = 0.3
    w = 2 / (na * fill_)
    zR = w^2 / 2
    plain = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = fill_), radial = 48, angular = 200)
    aber = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = fill_), aberrations = (0, 0, 0.2, 0.1, -0.15, 0.1, 0.12, -0.1, 0, 0, 0, 0.1),
                       radial = 48, angular = 200)
    @test profile_moments(plain) === profile_moments(aber)
    P, ky2 = profile_moments(plain)
    # the Gaussian pupil cut at ρ = 1: πw²/2 and 1/w² up to its tails, exp(−1/f²) = 1.5·10⁻⁵
    e1, g = exp(-1 / fill_^2), 2 / fill_^2
    @test P ≈ π * w^2 / 2 * (1 - e1^2) / (1 - e1)^2 rtol = 1e-12
    @test ky2 ≈ na^2 / 2 * (1 - (1 + g) * exp(-g)) / (g * (1 - exp(-g))) rtol = 1e-12
    @test abs(P / (π * w^2 / 2) - 1) < 3 * e1
    worst_p = worst_k = worst_kz = 0.0
    h = 2.5
    for b in (plain, aber), ξ in (0.0, -zR / 2)
        p = py = pz = 0.0
        for y in -150:h:150, z in -150:h:150
            u, uyy, uzz, _ = profile(b, ξ, y, z)
            p += abs2(u)
            py -= real(conj(u) * uyy)
            pz -= real(conj(u) * uzz)
        end
        worst_p = max(worst_p, abs(p * h^2 / P - 1))
        worst_k = max(worst_k, abs(py / p / ky2 - 1))
        worst_kz = max(worst_kz, abs(pz / p / ky2 - 1))
    end
    @info "T-1162 power" worst_p worst_k worst_kz
    @test worst_p <= 1e-7                             # the tails of the pupil's edge, exp(−11) in amplitude, outside ±150
    @test worst_k <= 1e-6
    @test worst_kz <= 1e-6
end

# --- T-1163 ---------------------------------------------------------------------------------

# Astigmatism: with a Gaussian pupil the wavefront c·Z₅ = c·√6(ρ_y² − ρ_z²) splits the beam into
# two slab beams, U(ξ, y, z) = g(ξ − ξ_a, y)·g(ξ + ξ_a, z) with ξ_a = 4√6π·c/na² and g PulseShapes'
# 2D GaussianBeam of waist 2/(na·f), so the beam is a line along z at ξ = ξ_a (narrowest in y) and
# a line along y at ξ = −ξ_a, the two line foci; Z₃ = √6·ρ²·sin 2θ is the same along the
# diagonals. The rms widths of |U|² along y and z, minimized over ξ, find the line foci at ±ξ_a.
@testset "T-1163 astigmatism focuses a Gaussian pupil to two line foci at ±4√6π·c/na², the product of two slab beams; oblique astigmatism along the diagonals [A2, A7, WP-085]" begin
    na, f, c = 0.2, 0.25, 0.05
    w = 2 / (na * f)
    ξa = 4√6 * π * c / na^2
    g = GaussianBeam(; waist = w)
    g1(ξ, y) = profile(g, ξ, y)[1]
    b5 = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = f), aberrations = (0, 0, 0, 0, c), radial = 24, angular = 64)
    b3 = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = f), aberrations = (0, 0, c), radial = 24, angular = 64)
    e5 = e3 = 0.0
    for ξ in (-2ξa, -ξa, 0.0, 0.4ξa, ξa, 3ξa), (y, z) in ((0.0, 0.0), (12.0, -5.0), (-30.0, 20.0), (4.0, 45.0))
        e5 = max(e5, abs(profile(b5, ξ, y, z)[1] - g1(ξ - ξa, y) * g1(ξ + ξa, z)))
        e3 = max(e3, abs(profile(b3, ξ, y, z)[1] - g1(ξ - ξa, (y + z) / √2) * g1(ξ + ξa, (y - z) / √2)))
    end
    @info "T-1163 product" ξa e5 e3
    @test e5 <= 3e-7 && e3 <= 3e-7
    # the line foci of a strong astigmatism (ξ_a ≈ 1.15 z_R): the rms widths of |U|² along y and z
    # through the axis, minimized over ξ, at ±ξ_a, where the beam is a line
    na, c = 0.5, 1.2
    w = 2 / (na * f)
    zR = w^2 / 2
    ξa = 4√6 * π * c / na^2
    bs = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = f), aberrations = (0, 0, 0, 0, c), radial = 64, angular = 240)
    # its quadrature by convergence (quadrature_radius is pessimistic for a Gaussian pupil)
    bf = ZernikeBeam(; na, amplitude = GaussianPupil(; fill = f), aberrations = (0, 0, 0, 0, c), radial = 96, angular = 320)
    conv = maximum(((ξ, y, z),) -> abs(profile(bs, ξ, y, z)[1] - profile(bf, ξ, y, z)[1]),
                   ((ξa, 120.0, 0.0), (-ξa, 0.0, 120.0), (ξa, 0.0, -60.0), (0.0, 85.0, 0.0), (2ξa, -120.0, 0.0)))
    @info "T-1163 strong astigmatism quadrature" conv
    @test conv <= 1e-9
    function width(ξ, axis)
        s = s2 = 0.0
        for x in -120:1.0:120
            u = axis == 2 ? profile(bs, ξ, x, 0.0)[1] : profile(bs, ξ, 0.0, x)[1]
            s += abs2(u)
            s2 += abs2(u) * x^2
        end
        return sqrt(s2 / s)
    end
    function argmin_golden(f, a, b)
        φ = (√5 - 1) / 2
        x1, x2 = b - φ * (b - a), a + φ * (b - a)
        f1, f2 = f(x1), f(x2)
        for _ in 1:40
            if f1 < f2
                b, x2, f2 = x2, x1, f1
                x1 = b - φ * (b - a)
                f1 = f(x1)
            else
                a, x1, f1 = x1, x2, f2
                x2 = a + φ * (b - a)
                f2 = f(x2)
            end
        end
        return (a + b) / 2
    end
    fy = argmin_golden(ξ -> width(ξ, 2), 0.0, 2ξa)
    fz = argmin_golden(ξ -> width(ξ, 3), -2ξa, 0.0)
    wy, wz = width(fy, 2), width(fy, 3)
    @info "T-1163 line foci" ξa zR fy fz wy wz
    @test fy ≈ ξa rtol = 1e-4
    @test fz ≈ -ξa rtol = 1e-4
    @test wy ≈ w / 2 rtol = 1e-5                              # the rms width of exp(−2y²/w²) is w/2
    @test wz ≈ w / 2 * sqrt(1 + (2ξa / zR)^2) rtol = 1e-5      # the slab beam along z, 2ξ_a past its focus
    @test wz > 2 * wy                                         # a line along z
end
