# WP-085: the Zernike pupil beam on the 3D antenna, T-1164 to T-1167 on the CPU; T-1168 (the table
# and the Gaussian limit on CUDA) and T-1169 (Enzyme's reverse mode through the table's kernels and
# the antenna's rows against ForwardDiff) run their bodies from gpu.jl. The table of the plane
# against the direct pupil sum, the Gaussian pupil against the 3D GaussianLaser, the launched beam
# against the Yee propagation of its incident field, the energy the plane drives, gradients. After
# em3d_antenna.jl (its helpers: plane_d, EY_AT, EZ_AT, dft2, idft2, yee_kx_3d, complex_host,
# conserved_energy_3d, forward_window_3d) and zernike.jl.

using ForwardDiff: ForwardDiff
using SpecialFunctions: erf
using Enzyme: Enzyme
using ParticleMeshElectromagnetic: Antenna, AntennaPlane, PlaneTable, tabulate, launched_energy, laser_field, antenna_b_rows,
                                   antenna_e_rows, antenna_d, antipode_weight, time_factor, check_params
using ParticleMeshElectromagnetic.PulseShapes: ZernikeBeam, profile, profile_moments, quadrature_radius, pupil_sample, radial_node,
                                               GaussianEnvelope, GaussianPupil, SuperGaussianPupil, GaussianBeam

"The Gaussian pupil whose focus is the round Gaussian beam of waist `w` (fill 1/4: truncated at exp(−16)), resolved on a box of half diagonal `r`."
gaussian_zernike(w; r = 18, aberrations = ()) = (na = 8 / w;
    ZernikeBeam(; na, amplitude = GaussianPupil(; fill = 1 // 4), aberrations, radial = ceil(Int, 1.1na * r), angular = 4 * ceil(Int, 1.1na * r)))

"The aberrated flattened pupil of the runs: na 1/2, a super-Gaussian near field, astigmatism, coma and spherical aberration."
flat_zernike(; aberrations = (0, 0, 0.03, 0, 0.05, 0, 0.04, 0, 0, 0, 0, 0.05), radial = 16, angular = 48) =
    ZernikeBeam(; na = 0.5, amplitude = SuperGaussianPupil(; radius = 0.6, order = 2), aberrations, radial, angular)

"""
    zernike_problem(T; beam, ncx = 12, nct = 6, length = (60, 24, 24), duration = 5, position = 6, focus = 14, polarization = 0,
                    chirp = 0, a0 = 1, tend = 60, center = middle, boundaries = periodic) -> EMProblem{3}

[`antenna3d_problem`](@ref) with a 3D [`ShapedLaser`](@ref) of the ZernikeBeam `beam` and a
GaussianEnvelope in place of the GaussianLaser; the beam's numbers converted to `T`.
"""
function zernike_problem(::Type{T}; beam, ncx = 12, nct = 6, length = (60, 24, 24), duration = 5, position = 6, focus = 14, polarization = 0,
                         chirp = 0, a0 = 1, tend = 60, center = nothing, boundaries = (:periodic, :periodic, :periodic), species = ()) where {T}
    ny, nz = nct isa Tuple ? nct : (nct, nct)
    h = (2π / ncx, 2π / ny, 2π / nz)
    n = ntuple(d -> round(Int, length[d] / h[d]), 3)
    L = ntuple(d -> T(n[d] * h[d]), 3)
    dt = T(95 // 100) / sqrt(sum(d -> 1 / (L[d] / n[d])^2, 1:3))
    c = center === nothing ? (n[2] * h[2] / 2, n[3] * h[3] / 2) : center
    laser = ShapedLaser(; a0 = T(a0), delay = 6 * T(duration), focus = T(focus), position = Float64(position), center = c,
                        polarization = T(polarization), profile = PulseShapes.convert_eltype(T, beam),
                        envelope = GaussianEnvelope(; duration = T(duration), chirp = T(chirp)))
    return EMProblem("zernike-3d", L, n, dt, round(Int, tend / value_of(dt)), species, 1, (1, 0, 0); lasers = (laser,), boundaries)
end

"""
    zernike_run(prob; mode = Reference(), probes = (), be = CPU(), workgroupsize = nothing) -> (; st, prob, p, A, ip)

[`antenna3d_run`](@ref) with the parameters tabulated on the state (`p`): `A[q]` the discrete-time
Fourier transforms at ω = 1 of Ey and Ez on the probe plane nearest to `probes[q]`.
"""
function zernike_run(prob; mode = Reference(), probes = (), be = CPU(), workgroupsize = nothing)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current = Esirkepov()); backend = be)
    p = tabulate(st, params(prob))
    initialize!(st, p)
    (; grid, fields) = st
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    T = eltype(st.ρ)
    dt = prob.dt
    ip = [round(Int, x / value_of(h[1])) + 1 for x in probes]
    J, K = (g + 1):(g + n[2]), (g + 1):(g + n[3])
    A = [(Ey = zeros(Complex{T}, n[2], n[3]), Ez = zeros(Complex{T}, n[2], n[3])) for _ in ip]
    function accumulate!(k)
        w = cis(k * dt) * dt
        Ey, Ez = Array(fields.Ey), Array(fields.Ez)
        for (q, i) in enumerate(ip)
            A[q].Ey .+= w .* view(Ey, g + i, J, K)
            A[q].Ez .+= w .* view(Ez, g + i, J, K)
        end
    end
    accumulate!(0)
    for k in 1:prob.nsteps
        step!(st, p, k)
        isempty(ip) || accumulate!(k)
    end
    return (; st, prob, p, A, ip)
end

"The host copy of a tabulated laser's table, (n_y, n_z, 12)."
host_table(l) = Array(l.profile.values)

"""
    table_dtft(r) -> (; Ey, Ez)

The discrete-time Fourier transform at ω = 1 of the incident Ey and Ez at the plane over the run's
times, from the table: Re[U·g(t)]'s transform is (U·Σ g w + U*·Σ g* w)/2, w = cis(mΔt)·Δt.
"""
function table_dtft(r)
    l = only(r.p.lasers)
    V = host_table(l)
    dt = r.prob.dt
    sθ, cθ = sincos(l.polarization)
    a = b = zero(Complex{eltype(V)})
    for m in 0:r.prob.nsteps
        gm = time_factor(l, m * dt - l.delay)
        w = cis(m * dt) * dt
        a += gm * w / 2
        b += conj(gm) * w / 2
    end
    U1 = complex.(V[:, :, 1], V[:, :, 2])
    U2 = complex.(V[:, :, 7], V[:, :, 8])
    return (; Ey = (U1 .* a .+ conj.(U1) .* b) .* cθ, Ez = (U2 .* a .+ conj.(U2) .* b) .* sθ)
end

"`angular_error_3d` with the incident transform from the table ([`table_dtft`](@ref))."
function table_angular_error(r; component = :Ey, against = component)
    a = Antenna(only(r.p.lasers), only(antenna_planes(r.st)))
    h = Float64.(value_of.(a.dx))
    dt = Float64(value_of(r.prob.dt))
    inc = map(complex_host, table_dtft(r))
    S, Sref = dft2(getfield(inc, component)), dft2(getfield(inc, against))
    ky, kz = ky_modes(size(S, 1), h[2]), ky_modes(size(S, 2), h[3])
    xa = Float64(value_of(a.xa))
    out = []
    for (q, i) in enumerate(r.ip)
        x = (i - 1) * h[1]
        x < xa - h[1] / 2 && continue
        P = [cis(yee_kx_3d(1.0, ky[p], kz[s], h[1], h[2], h[3], dt) * (x - xa)) for p in eachindex(ky), s in eachindex(kz)]
        pred, ref = idft2(S .* P), idft2(Sref .* P)
        meas = complex_host(getfield(r.A[q], component))
        push!(out, (; x, err = sqrt(sum(abs2, meas .- pred) / sum(abs2, ref))))
    end
    return out
end

"⟨k_⊥⁴⟩ of a ZernikeBeam's angular spectrum, Σ w a² κ⁴/Σ w a², the scale of the antenna's error."
function k4_moment(b)
    s2 = k4 = 0.0
    for i in 1:b.radial
        ρ, w = radial_node(b.radial, i, Float64)
        a = PulseShapes.pupil_amplitude(b.amplitude, ρ)
        s2 += ρ * w * a^2
        k4 += ρ * w * a^2 * (b.na * ρ)^4
    end
    return k4 / s2
end

# --- T-1164 ---------------------------------------------------------------------------------

"""
    table_against_direct(T, beam; be) -> ratio

The largest difference between the table that `tabulate` builds on `be` and the direct pupil sum
(`profile` on the host) at every staggered place of a 3D plane of 23 × 21 cells (so that rows lie
on the antipode of an axis near the seams, where the table holds the mean of the two images),
all twelve numbers,
relative to eps(T)·max|U|·√(samples); for duals the partials as well.
"""
function table_against_direct(::Type{T}, beam; be = CPU()) where {T}
    grid = PeriodicGrid((40, 23, 21), (T(2π / 12), T(2π / 6), T(2π / 5)), 3)
    plane = AntennaPlane(6.0, grid, T(1 // 10))
    c = (T(2 * 2π / 6), T(3 * 2π / 5))
    b = PulseShapes.convert_eltype(T, beam)
    l = ShapedLaser(; a0 = one(T), delay = T(30), focus = T(14), position = 6.0, center = c, profile = b,
                    envelope = GaussianEnvelope(; duration = T(5)), polarization = T(3 // 5))
    t = tabulate(l, plane, be)
    V = Array(t.profile.values)
    a = Antenna(t, plane)
    ξe, ξb = plane.xa - t.focus, plane.xa - plane.hx - t.focus
    worst = 0.0
    seams = 0
    nq = Int(b.radial) * Int(b.angular)
    comps(x) = x isa ForwardDiff.Dual ? (value_of(x), ForwardDiff.partials(x)...) : (x,)
    for j in 1:Int(grid.n[2]), k in 1:Int(grid.n[3])
        for (p, at, sgn) in ((1, EY_AT, 1), (2, EZ_AT, -1))
            dy, dz = plane_d(a, j, k, at)
            # the nearest image, or on the antipode (within Δ/1000) the mean of the two
            images(d, axis) = abs(2 * abs(d) - plane.len[axis]) * 1000 < plane.dx[axis] ? (d, d - sign(d) * plane.len[axis]) : (d,)
            ys, zs = images(dy, 2), images(dz, 3)
            seams += (length(ys) > 1) + (length(zs) > 1)
            mean(f) = sum(f(y, z) for y in ys, z in zs) / (length(ys) * length(zs))
            u = mean((y, z) -> profile(b, ξe, y, z)[1])
            um = mean((y, z) -> (q = profile(b, ξb, y, z); q[1] - sgn * (q[2] - q[3]) / 2))
            uyz = mean((y, z) -> profile(b, ξb, y, z)[4])
            ref = (real(u), imag(u), real(um), imag(um), real(uyz), imag(uyz))
            for m in 1:6
                worst = max(worst, maximum(abs.(comps(V[j, k, 6(p - 1) + m]) .- comps(ref[m]))))
            end
        end
    end
    seams > 0 || error("no place on the antipode: the test does not reach the seam rule")
    return worst / (base_eps(T) * sqrt(nq))
end

# The Gaussian limit on the plane: the rows the antenna adds for a tabulated Zernike laser of the
# Gaussian pupil against those of the GaussianLaser of the same waist, at two times on the pulse.
function gaussian_rows_difference(::Type{T}; be = CPU()) where {T}
    grid = PeriodicGrid((40, 23, 23), (T(2π / 12), T(2π / 6), T(2π / 6)), 3)
    plane = AntennaPlane(6.0, grid, T(1 // 10))
    c = (T(11 * 2π / 6), T(12 * 2π / 6))
    kw = (; a0 = T(13 // 10), delay = T(30), focus = T(14), position = 6.0, center = c, polarization = T(3 // 5), phase = T(1 // 5))
    gl = Antenna(GaussianLaser(; kw..., waist = T(4), duration = T(5), chirp = T(1 // 100)), plane)
    zl = Antenna(tabulate(ShapedLaser(; kw..., profile = PulseShapes.convert_eltype(T, gaussian_zernike(4)),
                                      envelope = GaussianEnvelope(; duration = T(5), chirp = T(1 // 100))), plane, be), plane)
    zl = Antenna(ParticleMeshElectromagnetic.Adapt.adapt(Array, zl.laser), plane)
    worst = 0.0
    n = Int(grid.n[2]) + 2 * Int(plane.g)
    for t in (T(26), T(33)), j in Int32(1):Int32(n), k in Int32(1):Int32(n)
        gb = time_factor(zl.laser, t - zl.laser.delay)
        ge = time_factor(zl.laser, t + plane.ht - zl.laser.delay)
        d = (antenna_b_rows(zl, j, k, gb, T) .- antenna_b_rows(gl, j, k, t, T)..., antenna_e_rows(zl, j, k, ge, T) .- antenna_e_rows(gl, j, k, t, T)...)
        worst = max(worst, maximum(x -> abs(value_of(x)), d))
    end
    return worst / kw.a0
end

# Two runs of the same problem, the Zernike laser of the Gaussian pupil and the GaussianLaser of its
# waist: every field relative to its largest value, and with duals the partials in a0 and focus (the
# third, a spherical aberration's, has no counterpart and must reach the fields).
function gaussian_run_difference(::Type{T}; be = CPU()) where {T}
    D = T === D3
    a0 = D ? mkdual(1, 1, 0, 0) : one(T)
    focus = D ? mkdual(14, 0, 1, 0) : T(14)
    z12 = D ? mkdual(0, 0, 0, 1) : zero(T)
    kw = (; length = (40, 24, 24), position = 6, focus, a0, polarization = 3 // 5, tend = 40, chirp = 1 // 100)
    pg = antenna3d_problem(T; kw..., waist = 4)
    pz = zernike_problem(T; kw..., beam = gaussian_zernike(4; aberrations = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, z12)))
    rg = antenna3d_run(pg; be)
    rz = zernike_run(pz; be)
    fg, fz = map(Array, rg.st.fields), map(Array, rz.st.fields)
    worst = worst_p = 0.0
    reach = false
    for c in keys(fg)
        a, b = getfield(fz, c), getfield(fg, c)
        m = maximum(x -> abs(value_of(x)), b)
        m == 0 && continue
        worst = max(worst, maximum(abs.(value_of.(a) .- value_of.(b))) / m)
        if D
            for q in 1:2
                mp = maximum(x -> abs(ForwardDiff.partials(x)[q]), b)
                worst_p = max(worst_p, maximum(abs.(ForwardDiff.partials.(a, q) .- ForwardDiff.partials.(b, q))) / mp)
            end
            reach |= any(x -> !iszero(ForwardDiff.partials(x)[3]), a)
        end
    end
    return (; worst, worst_p, reach, nsteps = pz.nsteps)
end

function body_T1164(be)
    C, Cr, Cn = emC("T-1164"), emC("T-1164", "C_rows"), emC("T-1164", "C_run")
    for T in TYPES
        r = record!("T-1164 table", T, table_against_direct(T, flat_zernike(; radial = 12, angular = 48); be), be)
        @info "T-1164 table against the pupil sum" T r
        @test r <= C
    end
    for T in (Float32, Float64)
        d = record!("T-1164 rows", T, gaussian_rows_difference(T; be) / max(1e-6, 64 * base_eps(T)), be)
        @info "T-1164 Gaussian rows" T d
        @test d <= Cr
    end
    for T in TYPES
        d = gaussian_run_difference(T; be)
        r = record!("T-1164 run", T, max(d.worst, d.worst_p) / max(1e-6, d.nsteps * base_eps(T)), be)
        @info "T-1164 Gaussian run" T d
        @test r <= Cn
        T === D3 && @test d.reach
    end
end

# The table is the direct pupil sum at every staggered place (on the antipode rows the mean of the
# two images, which an aberrated beam makes differ: CUDA's Float32 once took the other image there),
# in the A8 matrix with an aberration's partial; a Gaussian pupil filled to 1/4 of the
# aperture (truncated at exp(−16) ≈ 10⁻⁷) gives the GaussianLaser of its waist: the rows the antenna
# adds for both corrections at an oblique polarization, and the fields of a whole run, to the
# truncation's 10⁻⁷ (or Float32's roundoff), the duals' partials in a0 and focus included.
@testset "T-1164 the plane's table is the pupil sum; a Gaussian pupil's table and run are the 3D GaussianLaser's to the truncation, in the A8 matrix [A2, A3, A8, WP-085]" begin
    body_T1164(CPU())
end

# --- T-1165 ---------------------------------------------------------------------------------

# The launched beam of an aberrated, flattened pupil (astigmatism, coma, spherical; na 1/2) is the
# Yee angular-spectrum propagation of its incident ω₀ transform at the plane (from the table) on
# probe planes past the focus, to the antenna's order k_⊥⁴ (scaled by the beam's ⟨k_⊥⁴⟩, as T-1022
# by 1/w0⁴), and the component the incident field lacks stays at that order. The energy: the
# pulse's conserved energy once past the plane is what the plane drives (launched_energy, from the
# table), to the antenna's order and the flux's Δ² (T-1023); a phase-only wavefront keeps it, so
# the aberrated run carries the unaberrated run's energy in the ratio of their launched energies,
# and a beam inside the box launches laser_energy. On the table alone: an astigmatic Gaussian beam
# cut by the box launches the fraction erf(L_y/(√2·w_y))·erf(L_z/(√2·w_z)) of laser_energy.
@testset "T-1165 the launched Zernike beam is the Yee propagation of its incident field to the antenna's order and carries the energy the plane drives; a phase-only wavefront keeps it [A2, A7, WP-085]" begin
    C, Cx = emC("T-1165"), emC("T-1165", "C_cross")
    beam = flat_zernike()
    k4 = k4_moment(beam)
    for θ in (0.0, 0.6)
        prob = zernike_problem(Float64; beam, duration = 8, position = 6, focus = 16, polarization = θ, tend = 130, length = (140, 30, 30))
        r = zernike_run(prob; probes = (6, 12, 20, 30))
        e = table_angular_error(r; component = :Ey)
        x = θ == 0 ? table_angular_error(r; component = :Ez, against = :Ey) : [(; err = 0.0)]
        worst, worst_x = maximum(p -> p.err, e) / k4, maximum(p -> p.err, x) / k4
        @info "T-1165 angular" θ k4 errors = join([round(p.err; sigdigits = 3) for p in e], " ") cross = join([round(p.err; sigdigits = 3) for p in x], " ")
        record!("T-1165 θ=$θ", Float64, worst)
        @test worst <= C
        θ == 0 && (record!("T-1165 cross", Float64, worst_x); @test worst_x <= Cx)
    end
    # the energy
    Ce, Ck, Cks = emC("T-1165", "C_energy"), emC("T-1165", "C_kept"), emC("T-1165", "C_kept_strong")
    W, Lw = Dict{Symbol,Float64}(), Dict{Symbol,Float64}()
    Δ2 = 0.0
    for (name, b) in ((:plain, flat_zernike(; aberrations = ())), (:aberrated, beam), (:strong, flat_zernike(; aberrations = (0, 0, 0.25, 0, 0.3, 0, 0.2))))
        prob = zernike_problem(Float64; beam = b, position = 6, focus = 12, polarization = 0.6, tend = 60, length = (100, 30, 30))
        r = zernike_run(prob)
        fwd, _ = forward_window_3d(r, 58)
        W[name] = conserved_energy_3d(map(Array, r.st.fields), r.st.grid, prob.dt; cols = fwd)
        l = only(r.p.lasers)
        Lw[name] = launched_energy(l)
        Δ2 = prob.dt^2 + (prob.length[1] / prob.cells[1])^2
        rel = abs(W[name] / Lw[name] - 1) / Δ2
        @info "T-1165 energy" name W = W[name] launched = Lw[name] laser = laser_energy(l) rel
        record!("T-1165 energy $name", Float64, rel)
        @test laser_energy(l) == laser_energy(only(params(prob).lasers))       # the table does not change it
        @test rel <= Ce
    end
    @test laser_energy(only(zernike_problem(Float64; beam = flat_zernike(; aberrations = ())).lasers)) ==
          laser_energy(only(zernike_problem(Float64; beam).lasers))
    @test Lw[:strong] < (1 - 1e-3) * Lw[:plain]                                # its wings leave the cross-section
    for (name, c) in ((:aberrated, Ck), (:strong, Cks))
        kept = abs(W[name] / W[:plain] - Lw[name] / Lw[:plain]) / Δ2
        @info "T-1165 kept" name kept
        record!("T-1165 kept $name", Float64, kept)
        @test kept <= c
    end
    # the fraction of an astigmatic Gaussian beam that the cross-section holds
    grid = PeriodicGrid((40, 24, 20), (2π / 12, 2π / 6, 2π / 6), 3)
    plane = AntennaPlane(6.0, grid, 0.1)
    w0, c5 = 8.0, 0.02
    b = gaussian_zernike(w0; r = 22, aberrations = (0, 0, 0, 0, c5))
    ξa = 4√6 * π * c5 / b.na^2
    zR = w0^2 / 2
    for θ in (0.0, 0.6)
        # the axis near a corner, so that the nearest image wraps in both directions
        l = ShapedLaser(; a0 = 1.0, delay = 30.0, focus = 20.0, position = 6.0, center = (3.0, 17.0), profile = b,
                        envelope = GaussianEnvelope(; duration = 5.0), polarization = θ)
        lt = tabulate(l, plane, CPU())
        ξ = plane.xa - l.focus
        wy, wz = w0 * sqrt(1 + ((ξ - ξa) / zR)^2), w0 * sqrt(1 + ((ξ + ξa) / zR)^2)
        frac = erf(plane.len[2] / (√2 * wy)) * erf(plane.len[3] / (√2 * wz))
        got = launched_energy(lt) / laser_energy(lt)
        @info "T-1165 cut beam" θ frac got
        @test frac < 0.99
        @test got ≈ frac rtol = 1e-3                # the plane's places sample the cut edges: measured ≤ 3·10⁻⁴
    end
end

# --- T-1166 ---------------------------------------------------------------------------------

# A3: the beam's numbers enter through the table of the plane. The on-axis intensity of the
# launched ω₀ component on a probe plane past the focus, by ForwardDiff through tabulate and the
# whole run, against the finite-difference floor of A7 (fd_scan: Float64, Reference), in a spherical
# aberration, a coma (odd under y → −y, taken from 0.05 where its derivative is not zero by
# symmetry), the numerical aperture and the focus.
function zernike_observable(x, mode, seed; which)
    T = typeof(x)
    ab = Any[0, 0, 0.02, 0, 0.04, 0, 0.05, 0, 0, 0, 0, 0.05]
    na, focus = 0.5, 12.0
    which === :spherical && (ab[12] = x)
    which === :coma && (ab[7] = x)
    which === :na && (na = x)
    which === :focus && (focus = x)
    beam = ZernikeBeam(; na = T(na), amplitude = SuperGaussianPupil(; radius = T(0.6), order = 2), aberrations = Tuple(T.(ab)),
                       radial = 12, angular = 32)
    prob = zernike_problem(T; beam, ncx = 10, nct = 5, length = (30, 18, 18), duration = 3, position = 5, focus = T(focus), tend = 40)
    r = zernike_run(prob; mode, probes = (20,))
    A = r.A[1].Ey
    return abs2(A[size(A, 1) ÷ 2, size(A, 2) ÷ 2 + 1])
end

@testset "T-1166 the beam's numbers enter through the table: gradients in a spherical aberration, a coma, the numerical aperture and the focus agree with the finite-difference floor [A3, A7, WP-085]" begin
    C = emC("T-1166")
    hs = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
    for (which, x0, scale) in ((:spherical, 0.05, 1.0), (:coma, 0.05, 1.0), (:na, 0.5, 1.0), (:focus, 12.0, 10.0))
        f = (x, mode, seed) -> zernike_observable(x, mode, seed; which)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
        s = fd_scan(f, x0, hs .* scale; seed = 1, ad)
        @info "T-1166" which ad rel = s.rel[s.istar] floor = s.floor_rel
        record!("T-1166 $which", Float64, s.rel[s.istar] / max(s.floor_rel, 1e-10))
        @test abs(ad) > 0
        @test s.rel[s.istar] <= C * max(s.floor_rel, 1e-10)
    end
end

"""
    zernike_digest(; be = CPU(), workgroupsize = nothing) -> UInt64

T-1167 and T-1168: a Reference run of an aberrated Zernike laser polarized at 0.6 rad into a
drifting plasma, the digest of its table and of every evolved array.
"""
function zernike_digest(; be = CPU(), workgroupsize = nothing)
    T = Float64
    tenth = one(T) / 10
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), tenth, zero(T), (1, 0, 0), (1, 2, 2), (3tenth, tenth, -tenth),
                                (tenth / 2, tenth / 2, tenth / 2), 1),)
    prob = zernike_problem(T; beam = flat_zernike(; radial = 12, angular = 32), ncx = 8, nct = 6, length = (16, 12, 12), duration = 2,
                           position = 4, focus = 8, polarization = 3 // 5, a0 = 1 // 2, tend = 15, species = sp)
    s = EMState(prob, EMNumerics(; workgroupsize, current = Esirkepov()); backend = be)
    q = tabulate(s, params(prob))
    initialize!(s, q)
    run!(s, q, prob.nsteps)
    h = hash(bits(host_table(only(q.lasers))))
    for a in evolved(s)
        h = hash(bits(Array(a)), h)
    end
    return h
end

# --- T-1167 ---------------------------------------------------------------------------------

# The checks: a ShapedLaser's center has as many coordinates as its profile has transverse axes;
# the 3D problem takes a ZernikeBeam laser; the step refuses it untabulated, with a table for
# another focus or plane, with the M4 solver, or with a quadrature that does not reach the box's
# corners; initialize!(state, problem) tabulates. Replacing the beam's numbers (with_tunables)
# drops the table, and tabulating again gives the same bits, as a second tabulate of a tabulated
# laser does; the Reference step is bitwise across reruns and launch shapes, and the table across
# launch shapes; the table crosses to a device array with Adapt.
@testset "T-1167 a 3D ShapedLaser is tabulated before it steps, its table is refused when stale, and the step is bitwise across reruns and launch shapes [A3, A4, A10, WP-085]" begin
    T = Float64
    beam = flat_zernike(; radial = 12, angular = 48)
    ok = zernike_problem(T; beam, length = (30, 24, 24), tend = 20)
    l = only(ok.lasers)
    @test l isa ShapedLaser && ParticleMeshElectromagnetic.transverse_axes(l) == 2
    env = GaussianEnvelope(; duration = 3)
    @test_throws ArgumentError ShapedLaser(; a0 = 1, delay = 18, focus = 5, position = 6, center = 4.0, profile = beam, envelope = env)
    @test_throws ArgumentError ShapedLaser(; a0 = 1, delay = 18, focus = 5, position = 6, center = (4.0, 5.0), profile = GaussianBeam(; waist = 3),
                                           envelope = env)
    flat = ShapedLaser(; a0 = 1, delay = 18, focus = 5, position = 6, center = 4.0, profile = GaussianBeam(; waist = 3), envelope = env)
    @test_throws ArgumentError EMProblem("x", ok.length, ok.cells, ok.dt, 2, (), 1, (1, 0, 0); lasers = (flat,))
    @test_throws ArgumentError EMProblem("x", ok.length[1:2], ok.cells[1:2], ok.dt, 2, (), 1, (1, 0); lasers = (l,))
    st = EMState(ok, EMNumerics())
    p = params(ok)
    @test_throws ArgumentError initialize!(st, p)                        # untabulated
    pt = tabulate(st, p)
    lt = only(pt.lasers)
    @test lt.profile isa PlaneTable && size(lt.profile.values) == (ok.cells[2], ok.cells[3], 12)
    @test check_params(st, pt) === pt
    @test same_bits(host_table(only(tabulate(st, pt).lasers)), host_table(lt))      # tabulated again from its beam
    # a table for another focus or plane, the M4 solver, a short quadrature
    moved = EMParams(pt; lasers = (ShapedLaser(lt.a0, lt.delay, lt.focus + 1, lt.polarization, lt.phase, lt.profile, lt.envelope, lt.position,
                                               lt.center),))
    @test_throws ArgumentError check_params(st, moved)
    po = zernike_problem(T; beam, length = (30, 24, 24), tend = 20, nct = (6, 5), center = l.center)
    other = EMState(po, EMNumerics())
    @test_throws ArgumentError check_params(other, EMParams(params(po); lasers = pt.lasers))     # the table of another plane
    @test_throws ArgumentError initialize!(EMState(ok, EMNumerics(; solver = M4())), ok)
    @test_throws ArgumentError tabulate(st, params(zernike_problem(T; beam = flat_zernike(; radial = 12, angular = 16), length = (30, 24, 24), tend = 20)))
    # initialize!(state, problem) tabulates
    st2 = EMState(ok, EMNumerics())
    @test initialize!(st2, ok) === st2
    # the tunables: names, a round trip that drops the table, and the same bits once tabulated again
    names = tunable_names(pt)
    @test names[1:7] == [:laser1_a0, :laser1_delay, :laser1_focus, :laser1_polarization, :laser1_phase, :laser1_na, :laser1_radius]
    @test :laser1_aberration12 in names && :laser1_duration in names
    back = with_tunables(pt, tunables(pt))
    @test only(back.lasers).profile isa ZernikeBeam
    @test_throws ArgumentError check_params(st, back)
    @test same_bits(host_table(only(tabulate(st, back).lasers)), host_table(lt))
    # bitwise: reruns, launch shapes; the table across launch shapes
    d = zernike_digest()
    @test zernike_digest() == d
    for wgs in (32, 256)
        @test zernike_digest(; workgroupsize = wgs) == d
    end
    # Adapt: the table crosses to another array type with the laser and the antenna
    a = Antenna(lt, only(antenna_planes(st)))
    ad = ParticleMeshElectromagnetic.Adapt.adapt(Array, a)
    @test ad isa Antenna && ad.laser.profile.values == lt.profile.values && ad.laser.profile.beam === lt.profile.beam
end

# --- T-1169 (run by gpu.jl) -----------------------------------------------------------------

# The beam's numbers from a device array, as T-1140's laser (Enzyme's reverse pass of a
# KernelAbstractions kernel on the GPU takes no active scalar argument): na, the Gaussian pupil's
# fill and M aberrations, then the engine's own pupil sample.
@kernel function pupil_from_array!(P, @Const(c), radial, angular, ::Val{M}) where {M}
    q = Int32(@index(Global, Linear))
    T = eltype(P)
    @inbounds b = ZernikeBeam{T,GaussianPupil{T},M}(c[1], GaussianPupil{T}(c[2]), ntuple(m -> c[2 + m], Val(M)), radial, angular)
    i = (q - Int32(1)) ÷ angular + Int32(1)
    k = (q - Int32(1)) % angular + Int32(1)
    ρ, wρ = radial_node(radial, i, T)
    o = pupil_sample(b, ρ, wρ, k)
    @inbounds ntuple(m -> (P[q, m] = o[m]; nothing), Val(5))
end

# The four values the antenna's kernels add at the q-th (row, layer) of the plane, guards included,
# for the laser `l0` with its profile the table `V`.
@kernel function zernike_rows_kernel!(out, @Const(V), l0, plane, gb, ge, nj)
    q = Int32(@index(Global, Linear))
    T = eltype(out)
    j, k = (q - Int32(1)) % nj + Int32(1), (q - Int32(1)) ÷ nj + Int32(1)
    t = PlaneTable(l0.profile, T(l0.focus), plane, l0.center, V)
    l = ShapedLaser(l0.a0, l0.delay, l0.focus, l0.polarization, l0.phase, t, l0.envelope, l0.position, l0.center)
    a = Antenna(l, plane)
    o = (antenna_b_rows(a, j, k, gb, T)..., antenna_e_rows(a, j, k, ge, T)...)
    @inbounds ntuple(m -> (out[m, q] = o[m]; nothing), Val(4))
end

"The pupil, the engine's table kernel and the rows: the function Enzyme reverses."
function zernike_rows!(out, c, P, V, l0, plane, gb, ge, nj, vm)
    be = get_backend(out)
    b = l0.profile
    pupil_from_array!(be)(P, c, b.radial, b.angular, vm; ndrange = size(P, 1))
    ParticleMeshElectromagnetic.zernike_table_kernel!(be)(V, P, plane, l0.center, plane.xa - l0.focus, plane.xa - plane.hx - l0.focus,
                                                         Int32(size(P, 1)); ndrange = (size(V, 1), size(V, 2)))
    zernike_rows_kernel!(be)(out, V, l0, plane, gb, ge, nj; ndrange = size(out, 2))
    return nothing
end

function t1169_case(::Type{T}) where {T}
    grid = PeriodicGrid((16, 12, 10), (T(1 // 5), T(3 // 4), T(3 // 4)), 3)
    plane = AntennaPlane(T(2), grid, T(1 // 10))
    c = T[1 // 4, 1 // 2, 3 // 100, -2 // 100, 5 // 100, 4 // 100, -6 // 100, 2 // 100, 7 // 100, -3 // 100, 1 // 100, 4 // 100, -5 // 100,
          6 // 100]
    M = length(c) - 2
    beam = ZernikeBeam(; na = c[1], amplitude = GaussianPupil(; fill = c[2]), aberrations = Tuple(c[3:end]), radial = 8, angular = 24)
    l0 = ShapedLaser(; a0 = T(13 // 10), delay = T(48), focus = T(28), position = T(2), center = (T(9 // 2), T(15 // 4)), profile = beam,
                     envelope = GaussianEnvelope(; duration = T(8), chirp = T(1 // 50)), polarization = T(3 // 10), phase = T(1 // 5))
    t = T(41)
    gb, ge = time_factor(l0, t - l0.delay), time_factor(l0, t + plane.ht - l0.delay)
    nj, nk = Int(grid.n[2]) + 2 * guards(grid), Int(grid.n[3]) + 2 * guards(grid)
    return (; c, M, l0, plane, gb, ge, nj = Int32(nj), nrows = nj * nk, nq = Int(beam.radial) * Int(beam.angular),
            nv = (Int(grid.n[2]), Int(grid.n[3])))
end

"""
    t1169_check(be, T) -> ratio

Enzyme's reverse mode through [`zernike_rows!`](@ref) on `be`, seeded with a random adjoint w̄ of
the four values of every row, against ForwardDiff through the same kernels on the CPU: the adjoint
of the beam's numbers c̄ = Σ_q J_qᵀ w̄_q. The ratio is the largest |reverse − forward| /
(eps(T)·√samples·Σ_q |J_q|ᵀ|w̄_q|) over the numbers.
"""
function t1169_check(be, ::Type{T}) where {T}
    k = t1169_case(T)
    w̄ = randn(Xoshiro(1169), T, 4, k.nrows)
    dev(a) = (b = KernelAbstractions.allocate(be, eltype(a), size(a)); copyto!(b, a); b)
    zer(dims...) = dev(zeros(T, dims...))
    out, cd, c̄ = zer(4, k.nrows), dev(k.c), zer(length(k.c))
    P, V = zer(k.nq, 5), zer(k.nv..., 12)
    Enzyme.autodiff(Enzyme.Reverse, zernike_rows!, Enzyme.Const, Enzyme.Duplicated(out, dev(w̄)), Enzyme.Duplicated(cd, c̄),
                    Enzyme.Duplicated(P, zer(k.nq, 5)), Enzyme.Duplicated(V, zer(k.nv..., 12)), Enzyme.Const(k.l0), Enzyme.Const(k.plane),
                    Enzyme.Const(k.gb), Enzyme.Const(k.ge), Enzyme.Const(k.nj), Enzyme.Const(Val(k.M)))
    KernelAbstractions.synchronize(be)
    function f(x)
        D = eltype(x)
        o, Pd, Vd = zeros(D, 4, k.nrows), zeros(D, k.nq, 5), zeros(D, k.nv..., 12)
        zernike_rows!(o, x, Pd, Vd, k.l0, k.plane, k.gb, k.ge, k.nj, Val(k.M))
        return vec(o)
    end
    J = ForwardDiff.jacobian(f, k.c)
    g = J' * vec(Float64.(w̄))
    s = abs.(J)' * vec(abs.(Float64.(w̄)))
    return maximum(abs.(Float64.(Array(c̄)) .- g) ./ (eps(T) * sqrt(k.nq) .* max.(s, floatmin(Float64))))
end

function body_T1169(be)
    C = emC("T-1169")
    for T in (Float32, Float64)
        r = record!("T-1169", T, t1169_check(be, T), be)
        @info "T-1169 Enzyme reverse against ForwardDiff" T ratio = r
        @test r <= C
    end
end
