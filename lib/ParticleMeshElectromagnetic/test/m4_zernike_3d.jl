# WP-093: the M4 solver with the 3D shaped antenna, T-1266 to T-1268 on the CPU; T-1269 reruns
# the bodies on CUDA, in gpu.jl. M4's B corrections read the incident E on the columns beside
# the plane as well (m4_antenna.jl), which a tabulated 3D ShapedLaser has in eight more numbers
# of its table. After em3d_antenna.jl and em3d_zernike.jl (their helpers: plane_d, EY_AT, EZ_AT,
# flat_zernike, gaussian_zernike, zernike_problem, zernike_run, host_table, table_angular_error,
# k4_moment, gaussian_run_difference, zernike_digest) and m4_cases.jl (tfsf_operator_error,
# incident_column_error, m4_kx, conserved_energy_m4_3d).

using ForwardDiff: ForwardDiff
using ParticleMeshElectromagnetic: Antenna, AntennaPlane, PlaneTable, tabulate, check_params, launched_energy
using ParticleMeshElectromagnetic.PulseShapes: GaussianEnvelope, profile

# --- T-1266 ---------------------------------------------------------------------------------

"""
    m4_columns_against_direct(T, beam; be) -> (; ratio, same)

`table_against_direct` for the eight numbers that `tabulate` adds on an M4 plane (23 × 21 cells,
rows on the antipode of an axis): `U` at ξ = x_a ∓ Δx − focus at the places of Ey and of Ez,
against the direct pupil sum (`profile` on the host; on the antipode the mean of the two
images), `ratio` the largest difference relative to eps(T)·√(samples), for duals the partials
as well. `same`: the table has twenty numbers and its first twelve are the bits of the Yee
plane's twelve.
"""
function m4_columns_against_direct(::Type{T}, beam; be = CPU()) where {T}
    grid = PeriodicGrid((40, 23, 21), (T(2π / 12), T(2π / 6), T(2π / 5)), 3)
    yee, plane = AntennaPlane(6.0, grid, T(1 // 10)), AntennaPlane(6.0, grid, T(1 // 10), M4())
    c = (T(2 * 2π / 6), T(3 * 2π / 5))
    b = PulseShapes.convert_eltype(T, beam)
    l = ShapedLaser(; a0 = one(T), delay = T(30), focus = T(14), position = 6.0, center = c, profile = b,
                    envelope = GaussianEnvelope(; duration = T(5)), polarization = T(3 // 5))
    t = tabulate(l, plane, be)
    V, Vy = Array(t.profile.values), Array(tabulate(l, yee, be).profile.values)
    a = Antenna(t, plane)
    ξ = (plane.xa - plane.dx[1] - t.focus, plane.xa + plane.dx[1] - t.focus)
    worst = 0.0
    seams = 0
    comps(x) = x isa ForwardDiff.Dual ? (value_of(x), ForwardDiff.partials(x)...) : (x,)
    for j in 1:Int(grid.n[2]), k in 1:Int(grid.n[3]), (p, at) in ((1, EY_AT), (2, EZ_AT))
        dy, dz = plane_d(a, j, k, at)
        images(d, axis) = abs(2 * abs(d) - plane.len[axis]) * 1000 < plane.dx[axis] ? (d, d - sign(d) * plane.len[axis]) : (d,)
        ys, zs = images(dy, 2), images(dz, 3)
        seams += (length(ys) > 1) + (length(zs) > 1)
        for q in 1:2
            u = sum(profile(b, ξ[q], y, z)[1] for y in ys, z in zs) / (length(ys) * length(zs))
            o = 12 + 4 * (p - 1) + 2 * (q - 1)
            worst = max(worst, maximum(abs.(comps(V[j, k, o + 1]) .- comps(real(u)))), maximum(abs.(comps(V[j, k, o + 2]) .- comps(imag(u)))))
        end
    end
    seams > 0 || error("no place on the antipode: the test does not reach the seam rule")
    same = size(V, 3) == 20 && size(Vy, 3) == 12 && same_bits(V[:, :, 1:12], Vy)
    return (; ratio = worst / (base_eps(T) * sqrt(Int(b.radial) * Int(b.angular))), same)
end

"""
    m4_zernike_plane(T; be, θ = 3 // 5) -> (; st, prob, laser, t)

An M4 state of 11 × 10 cells across (y and z cells differ, the axis off the cells' centres and off
the antipodes, so that no row takes the mean of two images) with an aberrated Zernike laser
polarized at `θ`, its laser tabulated on `be` and brought back to the host, and the time of the
pulse's peak at the plane.
"""
function m4_zernike_plane(::Type{T}; be = CPU(), θ = 3 // 5) where {T}
    prob = zernike_problem(T; beam = flat_zernike(; radial = 12, angular = 48), length = (20, 12, 10), nct = (6, 5), duration = 3, position = 6,
                           polarization = θ, tend = 1, center = (6.5, 4.0))
    st = EMState(prob, EMNumerics(; solver = M4()); backend = be)
    laser = ParticleMeshElectromagnetic.Adapt.adapt(Array, only(tabulate(st, params(prob)).lasers))
    return (; st = EMState(prob, EMNumerics(; solver = M4())), prob, laser, t = value_of(only(prob.lasers).delay))
end

# The table of an M4 plane: its eight numbers more are the pupil sum on the columns ia − 1 and
# ia + 1 at the places of Ey and Ez (on antipode rows the mean of the two images), in the A8 matrix
# with an aberration's partial, and the first twelve are the Yee plane's bits. The incident E that
# M4's B corrections read from it on the three columns (`incident_e`) is the beam's own field
# there, written out from the untabulated laser (`laser_field`, the pupil sum at the point): at
# x_a + mΔx, a time m·kΔx later (k M4's wavenumber of ω₀), Ey at its place and Ez at its place,
# polarized by cos θ and sin θ, over every array row and layer.
function body_T1266(be)
    C, Ci = emC("T-1266"), emC("T-1266", "C_incident")
    for T in TYPES
        r = m4_columns_against_direct(T, flat_zernike(; radial = 12, angular = 48); be)
        record!("T-1266 columns", T, r.ratio, be)
        @test r.ratio <= C
        @test r.same
        s = m4_zernike_plane(T; be)
        nq = 12 * 48
        e = incident_column_error(s.st, s.prob, s.t; laser = s.laser, reference = only(s.prob.lasers)) / (base_eps(T) * sqrt(nq))
        record!("T-1266 incident columns", T, e, be)
        @test e <= Ci
    end
end

@testset "T-1266 an M4 plane's table has the pupil sum on the columns beside the plane, its first twelve numbers are the Yee plane's bits, and the incident E read from it is the beam's on all three columns [A2, A3, A8, WP-093]" begin
    body_T1266(CPU())
end

# --- T-1267 ---------------------------------------------------------------------------------

# T-1164's Gaussian limit with M4, on `be`: the Zernike laser of a Gaussian pupil and the
# GaussianLaser of its waist launch the same M4 run, every field to the pupil's truncation (10⁻⁷,
# or Float32's roundoff), the duals' partials in a0 and the focus included, and the partial of a
# spherical aberration reaches the fields. The GaussianLaser's M4 antenna is T-1094's and
# T-1095's, so this pins the table's columns, their times and their places in a whole run.
function body_T1267(be)
    Cn = emC("T-1164", "C_run")
    for T in TYPES
        d = gaussian_run_difference(T; be, solver = M4())
        r = record!("T-1267 run", T, max(d.worst, d.worst_p) / max(1e-6, d.nsteps * base_eps(T)), be)
        @info "T-1267 Gaussian run with M4" T d
        @test r <= Cn
        T === D3 && @test d.reach
    end
end

# The launched pulse. The B corrections of a tabulated laser are the M4 half step's TF/SF split to
# roundoff (T-1095's operator test: every column, row, layer, stagger and region flag, at θ = 3/5
# on y ≠ z cells), in the A8 matrix. The Gaussian limit (body_T1267). And the launched beam of the
# aberrated, flattened pupil of T-1165 (na 1/2) is M4's angular-spectrum propagation of its
# incident ω₀ transform on probe planes past the focus, to the antenna's order ⟨k_⊥⁴⟩, in both
# components of a beam polarized at 0.6 rad, where Yee's propagation of the same incident field
# misses it by several times that. It launches one way: the energy behind the plane within
# T-1021's fraction of the energy ahead (T-1095's box, where the wrapped backward wave lies outside
# the forward window), in the A8 matrix, and the fields are linear in a0 (a0·∂E/∂a0 = E). And the
# pulse carries what the plane drives: M4's conserved energy of the launched pulse is
# launched_energy (from the table) to the antenna's order and the flux's Δ² (T-1165's check, whose
# bound it takes; a cross-section of 48 holds the beam).
@testset "T-1267 the 3D shaped beam launched with M4 is M4's propagation of its incident field, its B corrections the M4 split to roundoff, and a Gaussian pupil launches the GaussianLaser's M4 run [A2, A3, A7, A8, WP-093]" begin
    for T in TYPES
        s = m4_zernike_plane(T)
        r = tfsf_operator_error(s.st, s.prob, s.t; laser = s.laser)
        worst = max(r.scattered, r.total) / base_eps(T)
        record!("T-1267 operator", T, worst)
        @test worst <= emC("T-1086", "C_operator")
    end
    body_T1267(CPU())
    C, Cl = emC("T-1165"), emC("T-1021")
    beam = flat_zernike(; radial = 24, angular = 160)
    k4 = k4_moment(beam)
    prob = zernike_problem(Float64; beam, duration = 8, position = 6, focus = 16, polarization = 0.6, tend = 130, length = (140, 48, 48))
    r = zernike_run(prob; probes = (6, 12, 20, 30), solver = M4())
    for comp in (:Ey, :Ez)
        e = table_angular_error(r; component = comp, kx = m4_kx)
        yee = table_angular_error(r; component = comp)
        worst, naive = maximum(p -> p.err, e) / k4, maximum(p -> p.err, yee) / k4
        @info "T-1267 angular" comp k4 errors = join([round(p.err; sigdigits = 3) for p in e], " ") worst naive
        record!("T-1267 $comp", Float64, worst)
        @test worst <= C
        @test naive > 3 * worst
    end
    let prob = zernike_problem(Float64; beam, position = 6, focus = 12, polarization = 0.6, tend = 60, length = (100, 48, 48))
        r = zernike_run(prob; solver = M4())
        fwd, _ = forward_window_3d(r, 58)
        W = conserved_energy_m4_3d(map(Array, r.st.fields), r.st.grid, prob.dt; cols = fwd)
        L = launched_energy(only(r.p.lasers))
        rel = abs(W / L - 1) / (prob.dt^2 + (prob.length[1] / prob.cells[1])^2)
        @info "T-1267 energy" W launched = L rel
        record!("T-1267 energy", Float64, rel)
        @test rel <= emC("T-1165", "C_energy")
    end
    for T in TYPES
        D = T === D3
        a0 = D ? mkdual(1, 1, 0, 0) : one(T)
        focus = D ? mkdual(14, 0, 1, 0) : T(14)
        prob = zernike_problem(T; beam = flat_zernike(; radial = 12, angular = 48), length = (100, 24, 24), position = 6, focus, a0,
                               polarization = 3 // 5, tend = 60)
        r = zernike_run(prob; solver = M4())
        fwd, back = forward_window_3d(r, 58)
        f = map(Array, r.st.fields)
        frac = Float64(value_of(plain_energy_3d(f, r.st.grid; cols = back) / plain_energy_3d(f, r.st.grid; cols = fwd)))
        @info "T-1267 behind the plane" T frac
        record!("T-1267 leak", T, frac)
        @test frac <= Cl
        if D
            worst = 0.0
            for c in (f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz)
                m = maximum(x -> abs(value_of(x)), c)
                d = maximum(x -> abs(ForwardDiff.partials(x)[1] - value_of(x)), c)
                worst = max(worst, m == 0 ? (d == 0 ? 0.0 : Inf) : d / (eps(Float32) * m * sqrt(prob.nsteps)))
            end
            record!("T-1267 linear", T, worst)
            @test worst <= emC("T-1021", "C_linear")
            @test any(x -> !iszero(ForwardDiff.partials(x)[2]), f.Ey)
        end
    end
end

# --- T-1268 ---------------------------------------------------------------------------------

# The checks with M4: an M4 state takes a tabulated 3D ShapedLaser, whose table has twenty numbers
# per row and layer, the first twelve the Yee state's; a table is the table of its plane, so the
# Yee state's is refused by the M4 state and the M4 state's by the Yee state; an untabulated laser
# is refused and initialize!(state, problem) tabulates. The Reference M4 step with the table is
# bitwise across reruns and launch shapes, and is not the Yee run.
@testset "T-1268 an M4 state takes the tabulated 3D ShapedLaser and refuses the Yee plane's table, and its Reference step is bitwise across reruns and launch shapes [A3, A4, A10, WP-093]" begin
    T = Float64
    ok = zernike_problem(T; beam = flat_zernike(; radial = 12, angular = 48), length = (30, 24, 24), tend = 20)
    sy, sm = EMState(ok, EMNumerics()), EMState(ok, EMNumerics(; solver = M4()))
    py, pm = tabulate(sy, params(ok)), tabulate(sm, params(ok))
    ty, tm = host_table(only(py.lasers)), host_table(only(pm.lasers))
    @test size(ty) == (ok.cells[2], ok.cells[3], 12) && size(tm) == (ok.cells[2], ok.cells[3], 20)
    @test same_bits(tm[:, :, 1:12], ty)
    @test any(!iszero, tm[:, :, 13:20])
    @test check_params(sm, pm) === pm && check_params(sy, py) === py
    @test_throws "another plane, center or focus" check_params(sm, py)
    @test_throws "another plane, center or focus" check_params(sy, pm)
    @test_throws "call `params = tabulate(state, params)`" initialize!(sm, params(ok))
    @test same_bits(host_table(only(tabulate(sm, pm).lasers)), tm)             # tabulated again from its beam
    s2 = EMState(ok, EMNumerics(; solver = M4()))
    @test initialize!(s2, ok) === s2
    d = zernike_digest(; solver = M4())
    @test zernike_digest(; solver = M4()) == d
    for wgs in (32, 256)
        @test zernike_digest(; solver = M4(), workgroupsize = wgs) == d
    end
    @test d != zernike_digest()
end
