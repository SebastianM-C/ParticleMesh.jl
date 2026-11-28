# Tests of the 3D half of WP-087 (T-1230 to T-1237), included by runtests.jl and runnable alone:
#   julia +1.13 --project=scripts -t 4 scripts/optim/test/pulse3d.jl
# The runs take a small 3D box on the CPU (`SMALL3`), about 4 minutes each at four threads.
using Test
using ForwardDiff: ForwardDiff
include(joinpath(@__DIR__, "..", "pulse3d_opt.jl"))
using ParticleMeshElectromagnetic: PlaneTable, antenna_planes
using ParticleMeshElectromagnetic.PulseShapes: zernike_index, radial_node, pupil_sample

"""
The 3D half's sizes for the CPU tests: λ₀/8, a cross-section of 48 cells, one plasma electron per
cell, read at t = 190, in Float64, with a pupil sum of 24 × 96 samples (enough for this box's half
diagonal at the tests' aberrations, up to 0.05 waves per mode). A run takes about 4 minutes on four threads. Most of the
beam's cross-section lies in the absorbers of this box, so it is a test of the derivatives, not of
the physics: the plasma's electrons in the first bucket with u_x from 0.2 to 1 carry J.
"""
const SMALL3 = (; ppl = 8, rows = 48, ppc = (1, 1, 1), t_end = 190.0, float = Float64, quadrature = (24, 96))
const SMALL3_READ = (; rise = (0.2, 1.0), fade = BCASE.fade)

"WP-087's 3D objective on the small box of `SMALL3`, with the weight `SMALL3_READ`; `phase` the laser's carrier phase."
pulse3_small(x; phase = 0.0, kw...) = pulse3_objective(x; SMALL3..., phase, SMALL3_READ..., kw...)

"Duals of the fourteen parameters at `x`, the partials of the directions `dirs` (unit vectors in the order given)."
duals3(x, dirs = 1:length(x)) = [ForwardDiff.Dual{Nothing}(Float64(x[i]), ntuple(j -> Float64(dirs[j] == i), length(dirs))) for i in eachindex(x)]

"The corner of the box with the least share launched inside the absorbers of 82 sign patterns of the modes at focus 64 and 110 (`questions/wp-087-pulse-shape-study.md` §8)."
pulse3_corner() = [110.0; PULSE3_MODE_BOUND .* [-1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]; 12.0; -30.0]

"The ±1 signs of the eleven modes under the mirror y → −y (θ → π − θ): (−1)^m for cos mθ (m ≥ 0), −(−1)^|m| for sin |m|θ (m < 0)."
mirror_y_signs() = [(m = zernike_index(j)[2]; m >= 0 ? (-1)^m : -(-1)^abs(m)) for j in PCASE3.modes]

"The modes odd under the mirror z → −z (θ → −θ): sin |m|θ, m < 0."
odd_z_modes() = [zernike_index(j)[2] < 0 for j in PCASE3.modes]

"The share of the power of the profile of `l` (a 3D ShapedLaser) on the plane's Ey places of the rows `J` and layers `K`, by the direct pupil sum."
function direct_share(l, plane, J, K)
    b = PulseShapes.convert_eltype(Float64, l.profile.beam)
    ξ = Float64(l.position) - Float64(l.focus)
    hy, hz = Float64(plane.dx[2]), Float64(plane.dx[3])
    yc, zc = Float64.(l.center)
    s = sum(abs2(first(PulseShapes.profile(b, ξ, (j - 0.5) * hy - yc, (k - 1) * hz - zc))) for j in J, k in K)
    return s * hy * hz / first(PulseShapes.profile_moments(b))
end

@testset "T-1230 the 3D pulse's shape: a round flat-top focus as wide as the base Gaussian's, eleven Zernike modes, a pupil sum that resolves the box and a spectral phase that fits the window [A3, WP-087]" begin
    @test PCASE3.modes == (3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14) && length(PCASE3.names) == pulse3_nparams() == 14
    @test PCASE3.names[[1, 2, 13, 14]] == ("focus", "z3", "gdd", "tod")
    @test all(PULSE3_BOX.lo .< collect(PCASE3.x0) .< PULSE3_BOX.hi) && PULSE3_BOX.scale == collect(PCASE3.scale)
    @test all(maximum(zernike_index(j)[1] for j in PCASE3.modes) .== 4) && !any(in(PCASE3.modes), (1, 2, 4))
    # the unaberrated focus has the round base Gaussian's intensity FWHM, and the focus scales as 1/na
    @test focal_fwhm3(PCASE3.na) ≈ TCASE.x0[2] * sqrt(2 * log(2)) rtol = 1e-5
    @test focal_fwhm3(PCASE3.na; radial = PCASE3.radial, angular = PCASE3.angular) ≈ focal_fwhm3(PCASE3.na) rtol = 1e-9
    @test focal_fwhm3(2 * PCASE3.na) ≈ focal_fwhm3(PCASE3.na) / 2 rtol = 1e-9
    # the modes: j = 1, 2, 4 stay zero, the others are the free numbers in order; the spectral phase is the envelope's
    x = [90.0; collect(range(-0.09, 0.08; length = 11)); 3.0; -7.0]
    sh = pulse3_shape(x)
    ab = sh.profile.aberrations
    @test length(ab) == 14 && ab[1] == ab[2] == ab[4] == 0 && [ab[j] for j in PCASE3.modes] == x[2:12]
    @test (sh.envelope.duration, sh.envelope.gdd, sh.envelope.tod, sh.envelope.fod) === (6.0, 3.0, -7.0, 0.0)
    @test (sh.profile.radial, sh.profile.angular) == (PCASE3.radial, PCASE3.angular) && pulse3_shape(x; quadrature = (16, 64)).profile.radial == 16
    @test_throws ArgumentError pulse3_shape(x[1:13])
    # the pupil sum resolves the whole cross-section of the case's box at every point of the box (the bound is on |c|)
    r = BCASE.rows * π / 4 * sqrt(2) / 2
    @test PulseShapes.quadrature_radius(pulse3_shape(PULSE3_BOX.hi).profile) > 1.1 * r
    # the Strehl ratio: 1 unaberrated; for spherical aberration alone the on-axis value of the pupil sum, |Σ w a e^{2πiW}|²/(Σ w a)²
    @test strehl3(pulse3_shape(collect(PCASE3.x0))) ≈ 1 rtol = 1e-10
    xs = [80.0; [j == 12 ? 0.06 : 0.0 for j in PCASE3.modes]; 0.0; 0.0]
    b = pulse3_shape(xs).profile
    num, den = 0.0im, 0.0
    for i in 1:b.radial, k in 1:b.angular
        ρ, wρ = radial_node(b.radial, i, Float64)
        pr, pim, _, _, wa = pupil_sample(b, ρ, wρ, k)
        num += complex(pr, pim)
        den += wa
    end
    @test strehl3(pulse3_shape(xs)) ≈ abs2(num / den) rtol = 1e-6
    @test strehl3(pulse3_shape(xs)) < 0.9 && strehl3(pulse3_shape(x)) < strehl3(pulse3_shape(xs))
    xd = [ForwardDiff.Dual{Nothing}(Float32(v), ntuple(j -> Float32(j == i), 14)) for (i, v) in enumerate(x)]
    shd = pulse3_shape(xd)
    @test shd.profile.na isa ForwardDiff.Dual{Nothing,Float32,14} && shd.envelope.tod === xd[14] && shd.profile.aberrations[14] === xd[12]
    # every corner of the spectral box starts after t = 0 and ends before the window passes the antenna; twice the GDD does not
    for gdd in (-12.0, 12.0), tod in (-30.0, 30.0)
        @test check_pulse3_time(pulse3_shape([80.0; zeros(11); gdd; tod])) === nothing
    end
    @test_throws ArgumentError pulse3_problem([80.0; zeros(11); 30.0; 0.0]; ppl = 8)
end

@testset "T-1231 the 3D energy is the budget at every point with no derivative in any parameter, and the share the antenna drives is recorded [A3, WP-087]" begin
    E = budget3(PCASE3.a0_energy)
    a0 = pulse3_a0(pulse3_shape(collect(PCASE3.x0)))
    @test a0 ≈ 1.9434 rtol = 1e-4                                       # the round flat-top of the Gaussian's FWHM: 2.3 % above the Gaussian's 1.9
    for x in [collect(PCASE3.x0), PULSE3_BOX.lo, PULSE3_BOX.hi, pulse3_corner(), random_start3(1), random_start3(2)]
        l = pulse3_problem(x; ppl = 8, float = Float64).lasers[1]
        @test l isa ShapedLaser && l.profile isa ZernikeBeam
        @test laser_energy(l) ≈ E rtol = 8eps()
        @test l.a0 == a0
    end
    xd = duals3(random_start3(3))
    ld = pulse3_problem(xd; ppl = 8, float = Float64).lasers[1]
    @test all(iszero, ForwardDiff.partials(laser_energy(ld))) && all(iszero, ForwardDiff.partials(ld.a0))
    @test ForwardDiff.partials(ld.focus) == ForwardDiff.partials(xd[1])
    # the table's share on the case's 128 × 128 plane against the direct pupil sum, unaberrated and at the worst corner tried
    prob = pulse3_problem(collect(PCASE3.x0); ppl = 8, float = Float64, capacity = 1000)
    st = EMState(prob, trapped3d_numerics(8, 1); record = (), backend = CPU())
    share(x) = (p = tabulate(st, params(pulse3_problem(x; ppl = 8, float = Float64, capacity = 1000))); (launched_share3(p.lasers[1]), p.lasers[1]))
    plane = only(antenna_planes(st))
    k, n = BCASE.pml_y, BCASE.rows
    for x in (collect(PCASE3.x0), pulse3_corner())
        s, l = share(x)
        @test s.interior ≈ direct_share(l, plane, (k + 1):(n - k), (k + 1):(n - k)) rtol = 1e-6
        @test s.box ≈ launched_energy(l) / laser_energy(l) rtol = 1e-12
    end
    s0 = share(collect(PCASE3.x0))[1]
    @test 0.99 < s0.interior < s0.box < 1
    @test 0.8 < share(pulse3_corner())[1].interior < 0.85               # the worst corner tried loses 17 %
    @test all(share(random_start3(k))[1].interior > 0.98 for k in 1:4)
end

@testset "T-1232 the 3D shaped problem is WP-084's case with the ShapedLaser in place of the Gaussian, and the shape's duals reach the table [WP-084, WP-085, WP-087]" begin
    x = random_start3(2)
    s = pulse3_problem(x; ppl = 8, float = Float64)
    g = trapped3d_problem(PCASE3.duration, TCASE.x0[2], x[1], PCASE3.delay, PCASE3.density; ppl = 8, float = Float64)
    @test s.cells == g.cells && s.length == g.length && s.dt == g.dt && s.nsteps == g.nsteps
    @test s.species == g.species && s.ionization == g.ionization && s.window == g.window && s.profile == g.profile
    ls, lg = s.lasers[1], g.lasers[1]
    @test ls.position == lg.position && ls.center == lg.center && ls.focus == lg.focus && ls.delay == lg.delay
    @test ls.phase == 0 && pulse3_problem(x; ppl = 8, float = Float64, phase = π).lasers[1].phase == Float64(π)
    @test_throws ArgumentError trapped3d_problem(6.0, 10.0, 80.0, 54.0, 0.072; shape = pulse3_shape(x))      # a shape needs its a0
    # Float32 by default, as the 3D case; duals of Float32 stay Float32 duals through the table
    @test eltype(params(pulse3_problem(x; ppl = 8))) === Float32
    xd32 = [ForwardDiff.Dual{Nothing}(Float32(v), ntuple(j -> Float32(j == i), 3)) for (i, v) in enumerate(x)]
    @test eltype(params(pulse3_problem(xd32; ppl = 8))) === ForwardDiff.Dual{Nothing,Float32,3}      # the energy and a₀ do not promote the duals to Float64
    @test pulse3_a0(pulse3_shape(xd32)) isa ForwardDiff.Dual{Nothing,Float32,3}
    st, p, _ = pulse3_start([ForwardDiff.Dual{Nothing}(Float32(v), ntuple(j -> Float32((i, j) in ((5, 1), (13, 2))), 2)) for (i, v) in enumerate(x)];
                            merge(SMALL3, (; float = Float32))..., capacity = 1000)
    t = p.lasers[1].profile
    @test t isa PlaneTable && eltype(t.values) === ForwardDiff.Dual{Nothing,Float32,2}
    @test any(v -> !iszero(ForwardDiff.partials(v)[1]), t.values) && all(v -> iszero(ForwardDiff.partials(v)[2]), t.values)   # a mode moves the table, the GDD does not
end

@testset "T-1233 ForwardDiff through the 3D run gives the objective's gradient: duals in chunks against central differences [A3, A7, WP-087]" begin
    # Three directions, each a mode or a temporal knob that reaches the run only through the table
    # or the envelope: an even mode (z5, astigmatism), a mode odd in y (z8, coma), the GDD; and the
    # projection of the whole duals' gradient on a random direction.
    x0 = random_start3(1)
    dirs = (3, 6, 13)
    Jd = pulse3_small(duals3(x0, dirs))
    J = pulse3_small(x0)
    @test ForwardDiff.value(Jd) ≈ J rtol = 64eps()
    @test J > 0.05
    ad = ForwardDiff.partials(Jd)
    for (q, i) in enumerate(dirs)
        h = PULSE3_BOX.scale[i] * 2.0^-16
        xp, xm = copy(x0), copy(x0)
        xp[i] += h; xm[i] -= h
        fd = (pulse3_small(xp) - pulse3_small(xm)) / 2h
        @test ad[q] ≈ fd rtol = 1.0e-5
        @test abs(ad[q]) > 1.0e-3 * J / PULSE3_BOX.scale[i]                # the direction moves J
    end
end

@testset "T-1234 the 3D box's mirrors: modes odd in z have no gradient at the unaberrated pulse, and y → −y pairs the modes with the opposite carrier [A7, WP-087]" begin
    # E along y. z → −z keeps the polarization and the box, so J is even in each mode odd in z
    # (sin mθ), and their partials vanish at the unaberrated pulse to roundoff of the run. y → −y
    # flips E_y, the carrier phase moved by π: J(c, φ) = J(M c, φ + π), M the modes' signs.
    @test odd_z_modes() == [true, false, true, true, false, false, true, true, false, false, false]   # j = 3, 6, 7, 10, 11
    @test count(==(-1), mirror_y_signs()) == 5
    x0 = collect(PCASE3.x0)
    g = ForwardDiff.partials(pulse3_small(duals3(x0, (2, 3, 5, 6)))) .* PULSE3_BOX.scale[[2, 3, 5, 6]]   # z3 and z7 odd in z, z5 and z8 not
    @test abs(g[1]) < 1e-6 * maximum(abs, g) && abs(g[3]) < 1e-6 * maximum(abs, g)
    @test abs(g[2]) > 1e-3 * maximum(abs, g) && abs(g[4]) > 1e-3 * maximum(abs, g)
    x = [80.0; 0.04 .* [(-1)^k for k in 1:11] .+ 0.01 .* (1:11) ./ 11; 3.0; 8.0]
    xm = [x[1]; mirror_y_signs() .* x[2:12]; x[13:14]]
    Jp, Jmπ, Jm = pulse3_small(x), pulse3_small(xm; phase = Float64(π)), pulse3_small(xm)
    @test Jmπ ≈ Jp rtol = 1e-9
    @test abs(Jm - Jp) > 1e-3 * Jp
end

@testset "T-1235 the 3D random starts, coordinates, chunks and the store's L-BFGS-B on the 3D box [WP-068, WP-087]" begin
    @test random_start3(3) == random_start3(3) && random_start3(3; seed = 874) != random_start3(3)
    starts = pulse3_starts(PULSE3_OPT_DEFAULTS.starts)
    @test keys(starts) == (:r1, :r2, :r3, :r4)
    w = PULSE3_BOX.hi .- PULSE3_BOX.lo
    for s in values(starts)
        o = [1; 13; 14]
        @test all(PULSE3_BOX.lo[o] .+ 0.1 .* w[o] .<= s[o] .<= PULSE3_BOX.hi[o] .- 0.1 .* w[o])
        @test all(m -> 0 < abs(m) <= 0.05, s[2:12])                       # every mode starts away from zero, within the spread
    end
    @test length(unique(values(starts))) == 4
    @test pulse3_starts("r2"; spread = 0.02).r2[2:12] ≈ 0.4 .* starts.r2[2:12] rtol = 1e-15
    @test round.(starts.r1[[1, 2, 13, 14]]; digits = 4) == round.(random_start3(1)[[1, 2, 13, 14]]; digits = 4)
    @test_throws ErrorException pulse3_starts("r1,base")
    c = Pulse3Coords()
    @test c.lo == PULSE3_BOX.lo ./ PULSE3_BOX.scale
    @test free_of(c, z_of(c, starts.r2)) ≈ starts.r2 rtol = 1e-15
    @test grad_z(c, z_of(c, starts.r2), collect(1.0:14.0)) ≈ collect(1.0:14.0) .* PULSE3_BOX.scale rtol = 1e-15
    @test gradient_chunks(14, 4) == [1:4, 5:8, 9:12, 13:14] && gradient_chunks(14, 14) == [1:14] && gradient_chunks(3, 1) == [1:1, 2:2, 3:3]
    # WP-068's store and optimize_start on these coordinates: a concave quadratic with its maximum inside the box
    m = [92.0; collect(range(-0.06, 0.07; length = 11)); -4.0; 10.0]
    Jq(x) = 1.0 - sum(((x .- m) ./ PULSE3_BOX.scale) .^ 2)
    ran = Ref(0)
    store = PointStore(joinpath(mktempdir(), "points"); run = x -> (ran[] += 1; Dict{String,Any}("J" => Jq(x), "gradient" => -2 .* (x .- m) ./ PULSE3_BOX.scale .^ 2, "a0" => 1.94)))
    dir = mktempdir()
    r = optimize_start(store, starts.r3, dir; maxiters = 80, pgtol = 1e-10, coords = c)
    @test r["free"] ≈ m rtol = 1e-6
    @test isfile(joinpath(dir, "result.json")) && ran[] == store.made
    # the 3D summary's sums are the 2D summary's times Δz, and J is trapped_energy3's
    st = pulse3_state(random_start3(1); SMALL3..., t_end = 120.0)
    s3, s2 = trapped_summary3(st; SMALL3_READ...), trapped_summary(st; SMALL3_READ...)
    hz = Float64(cellsize(st.grid)[3])
    @test s3.J == hz * s2.J && s3.dark_weight == hz * s2.dark_weight && s3.particles == s2.particles && s3.ux_max_plasma == s2.ux_max_plasma
    @test s3.J ≈ trapped_energy3(st; SMALL3_READ...) rtol = 1e-12
end

@testset "T-1236 a 3D point's record: J and the gradient in chunks of duals, the same partials whatever the chunk, and the shape's numbers [WP-087]" begin
    # the record of a run point at t = 150 with chunks of 2 over three directions only (the
    # others' partials are the same computation), against one run with duals in the three
    x = random_start3(2)
    kw = (; SMALL3..., t_end = 150.0, read = SMALL3_READ)
    rec = pulse3_run_point(x; kw..., grad = true, chunk = 2, directions = [1, 5, 13])
    Jd = pulse3_objective(duals3(x, (1, 5, 13)); SMALL3..., t_end = 150.0, SMALL3_READ...)
    @test rec["J"] == ForwardDiff.value(Jd) && length(rec["J_chunks"]) == 2
    @test rec["J_chunks"][2] ≈ rec["J"] rtol = 64eps()
    @test rec["gradient"][[1, 5, 13]] ≈ collect(ForwardDiff.partials(Jd)) rtol = 64eps()
    @test all(iszero, rec["gradient"][setdiff(1:14, [1, 5, 13])]) && rec["directions"] == [1, 5, 13]
    @test rec["J"] > 0
    @test rec["J"] ≈ pulse3_run_point(x; kw..., grad = false)["J"] rtol = 64eps()
    @test rec["a0"] == pulse3_a0(pulse3_shape(x)) && 0 < rec["strehl"] < 1 && rec["z"] == x ./ PULSE3_BOX.scale
    @test 0 < rec["launched_interior"] < rec["launched_box"] < 1 && rec["envelope_duration"] > PCASE3.duration
    @test rec["cells"] == [112, 48, 48] && rec["chunk"] == 2 && length(rec["seconds_runs"]) == 2
    @test rec["state_eltype"] == string(ForwardDiff.Dual{Nothing,Float64,2})        # the run's own element type, the partials of a chunk
end

@testset "T-1237 the 3D check's directions and central differences from records [WP-087]" begin
    o = Dict{String,Any}("steps" => "5,7", "directions" => 3)
    x = random_start3(4)
    g = collect(range(-1.0, 2.0; length = 14))
    jobs = pulse3_check_jobs(["best" => (x, g)], o)
    @test length(jobs) == 3 * 2 * 2
    c = Pulse3Coords()
    gz = grad_z(c, z_of(c, x), g)
    j1 = Dict(jobs)["best-d1-s5-p"]
    @test j1[:d] ≈ gz ./ sqrt(sum(abs2, gz)) rtol = 1e-15
    @test j1[:projection] ≈ sqrt(sum(abs2, gz)) rtol = 1e-14
    @test j1[:x] ≈ free_of(c, z_of(c, x) .+ 2.0^-5 .* j1[:d]) rtol = 1e-15
    @test Dict(jobs)["best-d2-s7-m"][:d] == random_direction(1)
    @test sum(abs2, random_direction(2)) ≈ 1 rtol = 1e-14
    # a linear J along the directions: the central difference is the projection
    cdir = mktempdir()
    for (id, j) in jobs
        zj = z_of(c, j[:x])
        Jv = 1.0 + sum(gz .* (zj .- z_of(c, x))) * (j[:direction] == 3 ? 1.1 : 1.0)
        open(io -> JSON.print(io, Dict("J" => Jv, "point" => j[:point], "direction" => j[:direction], "step" => j[:step], "sign" => j[:sign], "projection" => j[:projection])),
             joinpath(cdir, id * ".json"), "w")
    end
    rows = pulse3_check_rows(cdir, "best")
    @test length(rows) == 6 && [r["direction"] for r in rows] == [1, 1, 2, 2, 3, 3]
    @test all(r -> r["relative"] < 1e-9, rows[1:4]) && all(r -> isapprox(r["relative"], 0.1; rtol = 1e-6), rows[5:6])
    @test isempty(pulse3_check_rows(cdir, "reference"))
end
