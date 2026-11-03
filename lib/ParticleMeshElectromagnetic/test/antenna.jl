# WP-026: the laser antenna, T-400 to T-409. The physics runs (T-401 to T-405) launch a Gaussian
# beam of waist 6 c/ω₀ (about one wavelength, so that the effects of focusing are large against
# their tolerances) in vacuum, Float64, 16 cells per wavelength along x and 8 along y, and read it
# on probe columns from the antenna plane to two Rayleigh lengths past the focus.

using ForwardDiff: ForwardDiff
using ParticleMeshElectromagnetic: Antenna, laser_field, antenna_dy, RHO_STAGGER_2D

const ANTENNA_PROBES = (10, 20, 28, 40, 50, 64)

"""
The main vacuum run of T-401 to T-405 for a polarization (computed once per polarization): the
plane at x = 10 of a box 300 long, the pulse launched by t = 96, snapshots every 4 from t = 100 to
140. At t ≤ 140 the pulse lies within x ≤ 160 and the backward leak, wrapped round the periodic box,
beyond x = 170, so the window of columns from the plane to x = 160 holds the pulse and its
complement the leak (`forward_window`).
"""
const ANTENNA_RUNS = Dict{Float64,Any}()
function antenna_main(pol)
    get!(ANTENNA_RUNS, pol) do
        prob = antenna_problem(Float64; polarization = pol)
        r = antenna_run(prob; probes = ANTENNA_PROBES, snaps = round(Int, 100 / prob.dt):round(Int, 4 / prob.dt):round(Int, 140 / prob.dt))
        (; r..., window = forward_window(r, 160))
    end
end
const ANTENNA_POLS = (0.0, π / 2, 0.6)
"The order (k_y/k)⁴ of the antenna's B against E for a beam of waist w0 (ω₀ = 1)."
k4(w0) = inv(Float64(w0))^4
"The cell size along x of a run."
a_dx(r) = cellsize(r.st.grid)[1]
"The scale of T-402: the relative L² error (3/16)·√105/w0⁴ of the antenna's B (see T-402)."
angular_scale(w0) = 3 / 16 * sqrt(105) * k4(w0)

# --- T-400 ----------------------------------------------------------------------------------

# The incident field is the paraxial Gaussian beam in two dimensions: with U = e(phase = 0) +
# i·e(phase = −π/2) at s = 0 (the complex amplitude, carrier removed), 2i·∂U/∂ξ + ∂²U/∂y² = 0, |U|
# and arg U on the axis are (1 + ξ²/z_R²)^(−1/4)·a0·exp(−y²/w²) and −½atan(ξ/z_R), and at the focus
# on the axis e = a0·exp(−s²/τ²)·cos(s + chirp·s² − phase). Its B is E ∓ ½∂²E/∂y² (bp, bs), and
# the flux ∫∫ E·B dy dt of a host quadrature matches laser_energy (the trapezoid rule is
# spectrally accurate for these Gaussians).
@testset "T-400 the antenna's incident field is the 2D paraxial Gaussian beam with its B, and laser_energy its flux [A2, A3, WP-026]" begin
    l = GaussianLaser(; a0 = 0.7, waist = 5.0, duration = 6.0, delay = 30.0, focus = 3.0, position = 0.0, center = 0.0,
                      chirp = 0.004, polarization = 0.3, phase = 0.2)
    U(lv, ξ, y) = complex(laser_field(lv, ξ, y, 0.0)[1], laser_field(GaussianLaser(; a0 = lv.a0, waist = lv.waist,
                          duration = lv.duration, delay = lv.delay, focus = lv.focus, position = lv.position, center = lv.center,
                          chirp = lv.chirp, polarization = lv.polarization, phase = lv.phase - π / 2), ξ, y, 0.0)[1])
    l0 = GaussianLaser(; a0 = l.a0, waist = l.waist, duration = l.duration, delay = l.delay, focus = l.focus, position = 0.0,
                       center = 0.0, chirp = l.chirp, polarization = l.polarization, phase = 0.0)
    zR = l.waist^2 / 2
    worst_pde = worst_mod = worst_b = 0.0
    for ξ in (-2zR, -0.7zR, 0.0, 0.4zR, 1.5zR), y in (0.0, 1.3, -4.0, 7.5)
        dξ = ForwardDiff.derivative(x -> real(U(l0, x, y)), ξ) + im * ForwardDiff.derivative(x -> imag(U(l0, x, y)), ξ)
        d2 = ForwardDiff.derivative(yy -> ForwardDiff.derivative(z -> real(U(l0, ξ, z)), yy), y) +
             im * ForwardDiff.derivative(yy -> ForwardDiff.derivative(z -> imag(U(l0, ξ, z)), yy), y)
        u = U(l0, ξ, y)
        worst_pde = max(worst_pde, abs(2im * dξ + d2) / (abs(u) / l.waist^2))
        w = l.waist * sqrt(1 + (ξ / zR)^2)
        worst_mod = max(worst_mod, abs(abs(u) - l.a0 * (1 + (ξ / zR)^2)^(-1 / 4) * exp(-y^2 / w^2)) / l.a0)
        y == 0 && (worst_mod = max(worst_mod, abs(angle(u) + atan(ξ / zR) / 2)))
        for s in (-7.0, -0.3, 0.0, 2.2)
            e, bp, bs = laser_field(l, ξ, y, s)
            e2 = ForwardDiff.derivative(yy -> ForwardDiff.derivative(z -> laser_field(l, ξ, z, s)[1], yy), y)
            worst_b = max(worst_b, abs(bp - (e - e2 / 2)) / l.a0, abs(bs - (e + e2 / 2)) / l.a0)
        end
    end
    @test worst_pde <= 100 * eps()
    @test worst_mod <= 100 * eps()
    @test worst_b <= 100 * eps()
    for s in (-5.0, 0.0, 1.7)
        @test laser_field(l, 0.0, 0.0, s)[1] ≈ l.a0 * exp(-(s / l.duration)^2) * cos(s + l.chirp * s^2 - l.phase) rtol = 1e-14
    end
    # The flux of the incident field through the plane, polarization by polarization.
    # (duration 8, so that the exp(−duration²/2) term is below roundoff; steps of a quarter, far
    # below the Gaussians' widths and the Nyquist step π/2 of the products' 2ω₀ term)
    hy, ht = 0.25, 0.25
    for θ in (0.0, π / 2, 0.6)
        lθ = GaussianLaser(; a0 = l.a0, waist = l.waist, duration = 8.0, delay = l.delay, focus = l.focus, position = 0.0,
                           center = 0.0, chirp = 0.0, polarization = θ, phase = l.phase)
        Φ = 0.0
        for y in -40:hy:40, s in -56:ht:56
            e, bp, bs = laser_field(lθ, -zR, y, s)
            Φ += e * (bp * cos(θ)^2 + bs * sin(θ)^2)
        end
        Φ *= hy * ht
        @test Φ / laser_energy(lθ) - 1 ≈ 0 atol = 1e-12
        # and the polarization term is resolved: the plain π·a0²·w0·τ/4 is off by cos(2θ)/(2w0²)
        θ == 0 && @test abs(Φ / (π * l.a0^2 * l.waist * lθ.duration / 4) - 1 - 1 / (2 * l.waist^2)) <= 1e-12
    end
end

# --- T-401 ----------------------------------------------------------------------------------

# One way: behind the plane only what the incident field fails to satisfy of the discrete
# equations leaves, a measured fraction of the pulse energy, and no antenna is the plain step bit
# for bit. The leak is measured where it is: in a periodic box it wraps round and comes back from
# the far side, so each run is long enough that at the time of the measurement the pulse and the
# wrapped leak lie in separate columns (forward_window). The antenna is exact at ω₀ only (T-402's
# plane wave): the leak comes from the rest of the pulse's band (about 1/duration wide) and from
# the beam's modes near grazing incidence, where B/E = 1/cos α leaves its expansion. Small runs in
# the A8 matrix (waist 4, duration 5) with the laser's a0, focus and chirp seeded as the three
# partials of the dual: the fields are linear in a0, so a0·∂E/∂a0 = E to roundoff.
@testset "T-401 the antenna launches one way, is linear in a0, and without antennas the step is the plain step bit for bit [A2, A3, A8, WP-026]" begin
    C, Cl = emC("T-401"), emC("T-401", "C_linear")
    for T in TYPES
        D = T === D3
        a0 = D ? mkdual(1, 1, 0, 0) : one(T)
        focus = D ? mkdual(14, 0, 1, 0) : T(14)
        chirp = D ? mkdual(1 // 100, 0, 0, 1) : T(1 // 100)
        # at t = 66 the pulse lies within x ≤ 74, the wrapped leak beyond x = 8 + 150 − 66 = 92
        prob = antenna_problem(T; length = (150, 50), waist = 4, duration = 5, position = 8, focus, chirp, a0, tend = 66)
        r = antenna_run(prob)
        fwd, back = forward_window(r, 80)
        f = map(Array, r.st.fields)
        frac = Float64(value_of(plain_energy(f, r.st.grid; cols = back) / conserved_energy(f, r.st.grid, prob.dt; cols = fwd)))
        record!("T-401", T, frac / C)
        @test frac <= C
        if D
            # a0 = 1: the a0 partial of every field equals the field
            worst = 0.0
            for c in eb(r.st.fields)
                v = Array(c)
                m = maximum(x -> abs(value_of(x)), v)
                d = maximum(x -> abs(ForwardDiff.partials(x)[1] - value_of(x)), v)
                worst = max(worst, m == 0 ? (d == 0 ? 0.0 : Inf) : d / (eps(Float32) * m * sqrt(prob.nsteps)))
            end
            record!("T-401 linear", T, worst)
            @test worst <= Cl
            # the focus and chirp partials are nonzero: they enter through the antenna
            @test any(x -> !iszero(ForwardDiff.partials(x)[2]), Array(r.st.fields.Ey))
            @test any(x -> !iszero(ForwardDiff.partials(x)[3]), Array(r.st.fields.Ey))
        end
    end
    # The physics runs' beam (waist 6, duration 8) leaks less, at t = 140.
    Cm = emC("T-401", "C_main")
    for pol in ANTENNA_POLS
        r = antenna_main(pol)
        f = r.shots[end].f
        frac = plain_energy(f, r.st.grid; cols = r.window[2]) / conserved_energy(f, r.st.grid, r.prob.dt; cols = r.window[1])
        record!("T-401 main", Float64, frac / Cm)
        @test frac <= Cm
    end
    # No antennas: the plain step, bit for bit, on random fields.
    for T in TYPES
        grid = PeriodicGrid((12, 10), (T(1 // 2), T(2 // 3)), 3)
        f1 = em_fields(CPU(), T, grid)
        rng = Xoshiro(26)
        for c in values(f1)
            interior_view(c, grid) .= T.(rand(rng, size(interior_view(c, grid))...) .- 1 // 2)
            fill_guards!(c, grid)
        end
        f2 = map(copy, f1)
        dt = T(3 // 10)
        for k in 1:5
            solve_fields!(Yee(), f1, grid, dt)
            solve_fields!(Yee(), f2, grid, dt, (), (k - 1) * dt)
        end
        @test all(same_bits(getfield(f1, k), getfield(f2, k)) for k in keys(f1))
    end
end

# --- T-402 ----------------------------------------------------------------------------------

# The ω₀ = 1 component of the launched beam on every probe column equals the Yee angular-spectrum
# propagation of the incident field's ω₀ component at the plane (yee_kx, evanescent modes
# included), so the antenna launches a free beam of the scheme and nothing else reaches the
# probes. The one approximation at ω₀ is the antenna's B: E ∓ ½∂²E/∂y² is the expansion of the
# plane-wave ratio 1/cos α (cos α) to order k_y²; its next term, (3/8)·k_y⁴ (1/8 for the other
# polarization), halved in the forward wave, gives a relative L² error of (3/16)·√⟨k_y⁸⟩ =
# (3/16)·√105/w0⁴ for the Gaussian spectrum of 1/e field radius w0 in k (⟨k_y²⟩ = 1/w0²), the scale
# of the test at this waist. At waists of 24 and more a floor of order Δ²/w0² takes over
# (questions/wp-026-antenna.md §4). The run must last until the pulse's slow components have
# passed the far probe (its transform is over the whole run).
@testset "T-402 the launched beam is the Yee angular-spectrum propagation of the incident field, to the antenna's order (k_y/k)⁴ [A2, A7, WP-026]" begin
    C = emC("T-402")
    for pol in ANTENNA_POLS
        r = antenna_main(pol)
        for comp in (pol == π / 2 ? (:Ez,) : pol == 0 ? (:Ey,) : (:Ey, :Ez))
            errs = [x.err for x in angular_error(r; component = comp)]
            record!("T-402 $(comp)", Float64, maximum(errs) / angular_scale(6))
            @test maximum(errs) <= C * angular_scale(6)
        end
    end
    # A plane wave (a waist far beyond the box, which then holds the one mode k_y = 0): the
    # antenna's E and B at ω₀ are those of the scheme's discrete plane wave, B at x_a − Δx/2 being E
    # at x_a a time k·Δx/2 later (k the Yee wavenumber of ω₀), so its ω₀ component is launched
    # exactly, to roundoff, in the A8 matrix. With the continuum's Δx/2 in place of k·Δx/2 it would
    # be off by (k − 1)·Δx/4 = 1.8·10⁻⁴.
    Cp = emC("T-402", "C_plane")
    for T in TYPES
        prob = antenna_problem(T; length = (110, 4.8), waist = 1e8, duration = 5, position = 8, focus = 14, tend = 90)
        r = antenna_run(prob; probes = (8, 20, 30))
        err = maximum(x -> x.err, angular_error(r)) / (base_eps(T) * sqrt(prob.nsteps))
        record!("T-402 plane", T, err)
        @test err <= Cp
    end
    # The test sees the dispersion: the same propagation with the continuum's k_x = √(1 − k_y²) misses by far more.
    r = antenna_main(0.0)
    a = run_antenna(r)
    S = dft(incident_dtft(r).Ey)
    ky = ky_modes(length(S), a.dx[2])
    x = (r.ip[end] - 1) * a.dx[1]
    cont = idft(S .* [cis(sqrt(complex(1 - k^2)) * (x - a.xa)) for k in ky])
    @test sqrt(sum(abs2, r.A[end].Ey .- cont) / sum(abs2, cont)) > 20 * C * angular_scale(6)
end

# --- T-403 ----------------------------------------------------------------------------------

# Waist and Gouy phase. What the scheme predicts is the beam of T-402's angular-spectrum
# propagation (angular_profile): its 1/e radius (the second moment of |Â|²) and its phase on the
# axis, against which the measured ones are held to T-402's scale. The closed form is a sanity
# check: the paraxial Gaussian beam of the scheme (paraxial_beam: the antenna's profile is a beam
# of q_a = (x_a − x_f) − i·z_R at ω₀ = 1, which the scheme diffracts with keff = sin(kΔx)/Δx). The
# non-paraxial terms widen the second moment by a fraction up to about ⟨k_y⁴⟩/⟨k_y²⟩ = 3/w0² (5.9 %
# at waist 6 two Rayleigh lengths out, partly offset by the transverse dispersion at coarse Δy), so
# the waist is held to 3/w0² and the Gouy phase to 1/w0²; at waist 6 that does not tell keff from
# the continuum's k = 1, which T-402 does.
@testset "T-403 the launched beam keeps the waist and the 2D Gouy phase the scheme predicts [A2, A7, WP-026]" begin
    Ca, Cw, Cg = emC("T-403", "C_angular"), emC("T-403", "C_waist"), emC("T-403", "C_gouy")
    for pol in ANTENNA_POLS
        r = antenna_main(pol)
        comp = pol == π / 2 ? :Ez : :Ey
        bp = beam_profile(r; component = comp)
        ap = angular_profile(r; component = comp)
        φ0, φ0a = bp[1].phase, ap[1].phase
        worst_a = worst_w = worst_g = 0.0
        for (b, p) in zip(bp, ap)
            pb = paraxial_beam(r, b.x)
            worst_a = max(worst_a, abs(b.w / p.w - 1), abs(rem2pi((b.phase - φ0) - (p.phase - φ0a), RoundNearest)))
            worst_w = max(worst_w, abs(b.w / pb.w - 1))
            worst_g = max(worst_g, abs(rem2pi(b.phase - φ0 - pb.gouy, RoundNearest)))
        end
        record!("T-403 angular", Float64, worst_a / angular_scale(6))
        record!("T-403 waist", Float64, worst_w / (3 / 36))
        record!("T-403 gouy", Float64, worst_g * 36)
        @test worst_a <= Ca * angular_scale(6)
        @test worst_w <= Cw * 3 / 36
        @test worst_g <= Cg / 36
        # the beam's axis is the laser's, on every probe and for each component, in the row
        # coordinates of the staggering (Ey at (j − ½)Δy, Ez at (j − 1)Δy), not the antenna's
        for c in (pol == 0 ? (:Ey,) : pol == π / 2 ? (:Ez,) : (:Ey, :Ez))
            ya = maximum(b -> abs(b.y), beam_profile(r; component = c)) / Float64(cellsize(r.st.grid)[2])
            record!("T-403 axis", Float64, ya)
            @test ya <= emC("T-403", "C_axis")
        end
        # the probes resolve the beam: the waist near the focus and the Gouy phase over two Rayleigh lengths
        @test minimum(b -> b.w, bp) ≈ 6 rtol = 1e-2
        @test abs(rem2pi(bp[end].phase - φ0, RoundNearest)) > 0.9
    end
end

# --- T-404 ----------------------------------------------------------------------------------

# The group velocity: once the pulse has left the antenna, the centroid of its energy moves at the
# energy-weighted mean of the scheme's group velocity v_x = ∂ω/∂k_x over the pulse's modes
# (spectral_group_velocity of a snapshot, the exact discrete relation), measured by a straight-line
# fit over the eleven snapshots from t = 100 to 140, both over the pulse's window. The on-axis
# plane-wave value cos(kΔx/2)/cos(Δt/2) misses the beam's spread of directions, about 1/(2w0²) =
# 1.4 %, which the test resolves.
@testset "T-404 the pulse's energy travels at the Yee group velocity of its modes [A2, A7, WP-026]" begin
    C = emC("T-404")
    for pol in ANTENNA_POLS
        r = antenna_main(pol)
        grid = r.st.grid
        cols = r.window[1]
        @test length(r.shots) == 11
        ts = [s.t for s in r.shots]
        X = [energy_centroid(s.f, grid; cols) for s in r.shots]
        M = [ones(length(ts)) ts]
        c = M \ X
        vs = [spectral_group_velocity(s.f, grid, r.prob.dt; cols) for s in r.shots[[1, end]]]
        rel = abs(c[2] / vs[end] - 1)
        record!("T-404", Float64, rel / C)
        @test rel <= C
        @test abs(vs[1] / vs[end] - 1) <= C                # the spectrum does not change in vacuum
        @test maximum(abs.(M * c - X)) <= 1e-3 * a_dx(r)   # a straight line, to a thousandth of a cell
        vg1 = yee_axis(cellsize(grid)[1], r.prob.dt).vg
        @test abs(c[2] / vg1 - 1) > 1e4 * C
    end
end

# --- T-405 ----------------------------------------------------------------------------------

# The energy the pulse carries (the scheme's conserved energy of T-323 over the pulse's window at
# t = 140, once it has left the antenna) equals the discrete Poynting flux of the incident field
# through the plane (incident_flux), up to the antenna's order (k_y/k)⁴. That flux tends to
# laser_energy (the continuum's ∫∫ E·B) at second order in the cells and the step: the staggered
# product of E at integer times on the plane and B at half times half a cell behind is about
# laser_energy·cos(Δt/2)·cos(kΔx/2), with further terms of the same order from the band and the
# beam's diffraction over the half cell (scripts/em2d/antenna.jl measures the convergence).
@testset "T-405 the energy the pulse carries matches the Poynting flux through the antenna [A2, A7, WP-026]" begin
    C, Cf = emC("T-405"), emC("T-405", "C_flux")
    for pol in ANTENNA_POLS
        r = antenna_main(pol)
        a = run_antenna(r)
        W = conserved_energy(r.shots[end].f, r.st.grid, r.prob.dt; cols = r.window[1])
        Φ = incident_flux(r)
        rel = abs(W / Φ - 1)
        record!("T-405", Float64, rel / k4(6))
        @test rel <= C * k4(6)
        relf = abs(Φ / laser_energy(a.laser) - 1) / (r.prob.dt^2 + a.dx[1]^2)
        record!("T-405 flux", Float64, relf / Cf)
        @test relf <= Cf
    end
end

# --- T-406 ----------------------------------------------------------------------------------

"""
    antenna_divergences(st, nk, dt) -> (; gauss, gauss_col, divb, divb_col, scale, bscale)

The Gauss residual `D⁻·E − (ρ − mean ρ)` (ρ deposited now in the state's mode) at every interior
node and ∇·B at every cell centre, split into the maximum away from the antenna's columns
(`gauss` off the E column `ia`, `divb` off the cell column `ia − 1`) and the maximum on them after
subtracting the surface charges the antenna's corrections have accumulated over the `nk`
steps of `dt`: per step, `(Δt/Δx)·cos θ·D⁻_y bp` on the E column (the B of the plane behind at the half
time) and `−(Δt/2Δx)·sin θ·D⁺_y e` on the B column, twice (the incident E at the step's two ends).
"""
function antenna_divergences(st, nk, dt)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    a = only(st.aux.antennas)
    l = a.laser
    ρd = field(CPU(), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, RHO_STAGGER_2D, grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    ρ, Ex, Ey, Bx, By = Array(ρd), Array(st.fields.Ex), Array(st.fields.Ey), Array(st.fields.Bx), Array(st.fields.By)
    ρ̄ = sum(ρ[I1, I2]) / prod(n)
    sθ, cθ = sincos(l.polarization)
    # the antenna's accumulated divergences per row
    qe, qb = zeros(T, n[2]), zeros(T, n[2])
    for k in 1:nk
        t = (k - 1) * dt
        bpv = [laser_field(l, a.xa - a.hx - l.focus, antenna_dy(a, Int32(g + j), one(T) / 2), t + dt / 2 + a.ht - l.delay)[2]
               for j in 1:n[2]]
        for j in 1:n[2]
            qe[j] += dt / a.dx[1] * (cθ * bpv[j] - cθ * bpv[mod1(j - 1, n[2])]) / a.dx[2]
        end
        for tt in (t, t + dt)
            ev = [laser_field(l, a.xa - l.focus, antenna_dy(a, Int32(g + j), zero(T)), tt - l.delay)[1] for j in 1:n[2]]
            for j in 1:n[2]
                qb[j] -= dt / 2 / a.dx[1] * (sθ * ev[mod1(j + 1, n[2])] - sθ * ev[j]) / a.dx[2]
            end
        end
    end
    ia = Int(a.ia)
    gauss = gauss_col = divb = divb_col = zero(Float64)
    for i in 1:n[1], j in 1:n[2]
        I, J = g + i, g + j
        r = (Ex[I, J] - Ex[I - 1, J]) / h[1] + (Ey[I, J] - Ey[I, J - 1]) / h[2] - (ρ[I, J] - ρ̄)
        b = (Bx[I + 1, J] - Bx[I, J]) / h[1] + (By[I, J + 1] - By[I, J]) / h[2]
        if i == ia
            gauss_col = max(gauss_col, Float64(absmax(r - qe[j])))
        else
            gauss = max(gauss, Float64(absmax(r)))
        end
        if i == ia - 1
            divb_col = max(divb_col, Float64(absmax(b - qb[j])))
        else
            divb = max(divb, Float64(absmax(b)))
        end
    end
    hv = map(value_of, h)
    Em = Float64(maximum(absmax, Ey[I1, I2]) + maximum(absmax, Ex[I1, I2]))
    Bm = Float64(maximum(absmax, Bx[I1, I2]) + maximum(absmax, By[I1, I2]))
    return (; gauss, gauss_col, divb, divb_col, scale = sum(n) * Float64(maximum(absmax, ρ[I1, I2])) + (2 / hv[1] + 2 / hv[2]) * Em,
            bscale = (2 / hv[1] + 2 / hv[2]) * Bm)
end

# Gauss's law and ∇·B with a plasma crossing the antenna plane: the module's step keeps both to
# roundoff away from the antenna's columns (T-360), and on them the residual is the surface
# charge the antenna's sheets have accumulated, computed on the host from the same incident field,
# to roundoff; the polarization 0.6 drives both sheets. Both deposition modes, the A8 matrix.
@testset "T-406 with the antenna Gauss's law and ∇·B hold to roundoff off its columns and on them up to its surface charges [A2, A4, A8, WP-026]" begin
    C = emC("T-406")
    for T in TYPES, mode in (Reference(), Fast())
        sp = (SpeciesDescription{T}(:electron, -one(T), one(T), T(1 // 20), zero(T), (1, 0), (2, 2), (T(3 // 10), T(1 // 10), zero(T)),
                                    (T(1 // 20), T(1 // 20), T(1 // 20)), 1),)
        prob = antenna_problem(T; length = (24, 16), waist = 3, duration = 3, position = 5, focus = 10, polarization = 6 // 10,
                               a0 = 1 // 2, tend = 30, species = sp)
        st = EMState(prob, EMNumerics(; mode, current = Esirkepov()))
        initialize!(st, prob)
        p = params(prob)
        worst = worst_col = worst_b = worst_bcol = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 10 == 0 || k == prob.nsteps || continue
            d = antenna_divergences(st, k, prob.dt)
            s = base_eps(T) * sqrt(k + 1)
            worst = max(worst, d.gauss / (s * d.scale))
            worst_col = max(worst_col, d.gauss_col / (s * d.scale))
            worst_b = max(worst_b, d.divb / (s * d.bscale))
            worst_bcol = max(worst_bcol, d.divb_col / (s * d.bscale))
        end
        record!("T-406", T, max(worst, worst_col, worst_b, worst_bcol))
        @test worst <= C
        @test worst_col <= C
        @test worst_b <= C
        @test worst_bcol <= C
    end
end

# --- T-407 ----------------------------------------------------------------------------------

# A3: laser parameters enter through the antenna's source, focus and chirp through its phase;
# the gradient of an observable of the launched beam (the intensity of its ω₀ component on the
# axis of a probe past the focus) against focus and chirp, by ForwardDiff through the whole run,
# agrees with the finite-difference floor of A7 (fd_scan: Float64, Reference, one seed).
function antenna_observable(x, mode, seed; which)
    T = typeof(x)
    kw = which === :focus ? (; focus = x, chirp = 1 // 100) : (; focus = 12, chirp = x)
    prob = antenna_problem(T; length = (40, 30), waist = 3, duration = 3, position = 5, tend = 51, kw...)
    r = antenna_run(prob; mode, probes = (20,))
    A = r.A[1].Ey
    j0 = length(A) ÷ 2
    return abs2(A[j0])
end

@testset "T-407 laser parameters enter through the antenna: gradients in focus and chirp agree with the finite-difference floor [A3, A7, WP-026]" begin
    C = emC("T-407")
    hs = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    for (which, x0) in ((:focus, 12.0), (:chirp, 1e-2))
        f = (x, mode, seed) -> antenna_observable(x, mode, seed; which)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
        hsx = which === :chirp ? hs ./ 100 : hs
        s = fd_scan(f, x0, hsx; seed = 1, ad)
        record!("T-407 $which", Float64, s.rel[s.istar] / s.floor_rel)
        @test abs(ad) > 0
        @test s.rel[s.istar] <= C * max(s.floor_rel, 1e-10)
    end
end

# --- T-408 ----------------------------------------------------------------------------------

@testset "T-408 the antenna refuses a plane at the periodic seam and lasers at N = 1 [A2, WP-026]" begin
    l(x) = GaussianLaser(; a0 = 1, waist = 3, duration = 3, delay = 18, focus = 5, position = x, center = 4)
    grid = PeriodicGrid((40, 16), (0.5, 0.5), 3)
    @test_throws ArgumentError Antenna(l(1.0), grid, 0.3)        # column 3: within the guards' reach of the seam
    @test_throws ArgumentError Antenna(l(1.5), grid, 0.3)        # column 4 = g + 1: its B column g would be a stale guard image
    @test_throws ArgumentError Antenna(l(18.6), grid, 0.3)       # column 38 > 40 − 3
    @test Antenna(l(2.0), grid, 0.3).ia == 5
    @test Antenna(l(18.0), grid, 0.3).ia == 37
    sp1 = (SpeciesDescription{Float64}(:e, -1.0, 1.0, 1.0, 0.0, 1, 2, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1),)
    @test_throws ArgumentError EMProblem("x", 6.0, 12, 0.1, 4, sp1, 1, 1.0; lasers = (l(2.0),))
    @test_throws ArgumentError EMProblem("x", (6.0, 6.0), (12, 12), 0.1, 4, (), 1, (1, 0); lasers = (1.0,))
end

# --- T-409 ----------------------------------------------------------------------------------

# The beam's axis may sit anywhere along the periodic y: a run with the axis m rows above the
# box's middle (up to three rows from the seam, and across it) is the centred run shifted by m
# rows, to roundoff, in every component. The guard rows take the bits of their interior images.
@testset "T-409 the antenna's beam is periodic in y: its axis may sit at the seam [A2, A8, WP-026]" begin
    C = emC("T-409")
    kw = (; length = (24, 16), waist = 3, duration = 3, position = 5, focus = 10, polarization = 6 // 10, tend = 30)
    for T in (Float32, Float64)
        p0 = antenna_problem(T; kw...)
        n2 = p0.cells[2]
        r0 = antenna_run(p0)
        g, n = guards(r0.st.grid), ncells(r0.st.grid)
        I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
        for m in (7, n2 ÷ 2 + 1, n2 - 3)
            r1 = antenna_run(antenna_problem(T; kw..., center = n2 / 2 + m))
            worst = 0.0
            for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
                A = Array(getfield(r0.st.fields, c))[I1, I2]
                B = Array(getfield(r1.st.fields, c))[I1, I2]
                worst = max(worst, maximum(abs.(circshift(A, (0, m)) .- B)) / maximum(abs.(A)))
            end
            worst /= base_eps(T) * sqrt(p0.nsteps + 1)
            record!("T-409", T, worst)
            @test worst <= C
            a = only(r1.st.aux.antennas)
            for s in (zero(T), one(T) / 2)
                @test all(antenna_dy(a, Int32(j), s) === antenna_dy(a, Int32(j + n2), s) for j in 1:g)
                @test all(antenna_dy(a, Int32(j), s) === antenna_dy(a, Int32(j - n2), s) for j in (n2 + g + 1):(n2 + 2g))
            end
        end
    end
end
