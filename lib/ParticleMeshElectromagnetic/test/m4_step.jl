# WP-079: the M4 solver in the 2D step, T-1085 to T-1093 (the 3D step is m4_3d.jl, T-1094 to
# T-1097; T-1098 and T-1099 are their reruns on CUDA, in gpu.jl). M4 changes only Faraday's law
# (m4.jl), so what the step adds around the solver has to follow its B difference: the antenna's
# B corrections (m4_antenna.jl, the total-field/scattered-field rule on M4's wider stencil, the
# incident wave with M4's wavenumber and B/E ratio), the PML's B corrections (M4's difference along
# the layer axis), and the moving window's and the antenna's reach (two columns). Each test runs
# the Yee test of the same property with `solver = M4()`, against M4's own discrete theory
# (`m4_kx`, m4_cases.jl) where Yee's test used Yee's, and shows where it can that the M4
# treatment matters: the same run with the Yee treatment of the antenna or the PML (the "naive"
# runs) misses by orders of magnitude. After antenna.jl (`angular_scale`, `ANTENNA_PROBES`),
# shaped_antenna.jl (`shaped_problem`, `pupil_angular_scale`) and pml_back.jl.

using ForwardDiff: ForwardDiff
using ParticleMeshElectromagnetic: PMLFields, YeeStencil, faraday_stencil, forward_difference, m4_wavenumber, AntennaPlane, shift_antenna

"The main vacuum beam of antenna.jl (waist 6, duration 8) per polarization, with M4; a snapshot at t = 140."
const M4_RUNS = Dict{Float64,Any}()
m4_main(pol) = get!(M4_RUNS, pol) do
    prob = antenna_problem(Float64; polarization = pol)
    r = antenna_run(prob; probes = ANTENNA_PROBES, snaps = (round(Int, 140 / prob.dt),), solver = M4())
    (; r..., window = forward_window(r, 160))
end

"The fraction of the energy behind the plane in the last snapshot of an antenna run (T-401's leak)."
function leak_fraction(r, xmax)
    fwd, back = forward_window(r, xmax)
    f = r.shots[end].f
    return Float64(value_of(plain_energy(f, r.st.grid; cols = back) / conserved_energy_m4(f, r.st.grid, r.prob.dt; cols = fwd)))
end

"""
    conserved_energy_m4(f, grid, dt; cols) -> W

`conserved_energy` of M4: ½·Σ(Eⁿ)² + ½·Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² over the interior columns `cols`, the half
steps of B by M4's difference (the energy M4's leapfrog keeps, as Yee's does with its own).
"""
function conserved_energy_m4(f, grid, dt; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    st = faraday_stencil(M4(), grid, dt)
    fp, fm = map(copy, f), map(copy, f)
    m4_b_half!(fp, grid, dt / 2, st)
    m4_b_half!(fm, grid, -dt / 2, st)
    I = (g .+ cols, (g + 1):(g + n[2]))
    W = sum(c -> sum(abs2, c[I...]), (f.Ex, f.Ey, f.Ez))
    W += sum(k -> sum(getfield(fp, k)[I...] .* getfield(fm, k)[I...]), (:Bx, :By, :Bz))
    return W * prod(h) / 2
end

"""
    m4_divergence(st, ranges) -> max |Σ_d D̃_d B_d|

The largest divergence that M4 keeps (`m4_div`, its own difference) of the state's B over the cell
centres of the interior index `ranges` (one range per axis), on the host.
"""
function m4_divergence(st::EMState{N}, ranges) where {N}
    stc = faraday_stencil(M4(), st.grid, st.dt)
    B = map(Array, (st.fields.Bx, st.fields.By, st.fields.Bz))
    g, h = guards(st.grid), cellsize(st.grid)
    I = CartesianIndices(map(r -> g .+ r, ranges))
    return maximum(j -> absmax(sum(d -> forward_difference(stc, B[d], j, Val(d), Val(N)) / h[d], 1:N)), I)
end

"The scale of an M4 divergence check: Σ_d 2/Δ_d·max|B| over the state's interior."
function m4_div_scale(st::EMState{N}) where {N}
    g, n, h = guards(st.grid), ncells(st.grid), cellsize(st.grid)
    I = ntuple(d -> (g + 1):(g + n[d]), N)
    return sum(d -> 2 / Float64(value_of(h[d])), 1:N) * maximum(c -> Float64(maximum(absmax, Array(getfield(st.fields, c))[I...])), (:Bx, :By, :Bz))
end

# --- T-1085 ---------------------------------------------------------------------------------

# T-402 with M4: the ω₀ component of the launched beam on every probe column is M4's angular-
# spectrum propagation of the incident field's ω₀ component (`m4_kx`, evanescent modes included),
# to the antenna's order (k_y/k)⁴ of T-402, for the three polarizations; Yee's propagation misses
# it by M4's smaller dispersion. A plane wave (waist far beyond the box): at ω₀ the antenna's E and
# B are those of M4's discrete plane wave (E at x_a + mΔx a time m·kΔx later, k M4's wavenumber of
# ω₀, and B/E M4's ratio ρ), so it is launched exactly, to roundoff, in the A8 matrix; the same run
# with the Yee antenna (Yee's two corrections, wavenumber and ratio) misses by orders of magnitude.
# Roundoff holds at this resolution (16 × 8 cells per wavelength); at 8 × 4 and half the Courant
# number the same measurement gives up to 7·10⁻¹² relative, and Yee's antenna in its own run up to
# 10⁻⁸ (the PR's physics breaker; the cause of that floor, in both schemes, is not established).
function body_T1085(be)
    C, Cp, Cn = emC("T-1085"), emC("T-1085", "C_plane"), emC("T-1085", "C_naive")
    if be isa CPU
        for pol in ANTENNA_POLS
            r = m4_main(pol)
            for comp in (pol == π / 2 ? (:Ez,) : pol == 0 ? (:Ey,) : (:Ey, :Ez))
                err = maximum(x -> x.err, angular_error(r; component = comp, kx = m4_kx))
                yee = maximum(x -> x.err, angular_error(r; component = comp))
                @info "T-1085" pol comp err yee scale = angular_scale(6)
                record!("T-1085 $(comp)", Float64, err / angular_scale(6))
                @test err <= C * angular_scale(6)
                @test yee > 10 * err
            end
        end
    end
    for T in TYPES
        prob = antenna_problem(T; length = (110, 4.8), waist = 1e8, duration = 5, position = 8, focus = 14, tend = 90)
        r = antenna_run(prob; probes = (8, 20, 30), solver = M4(), be)
        err = maximum(x -> x.err, angular_error(r; kx = m4_kx)) / (base_eps(T) * sqrt(prob.nsteps))
        record!("T-1085 plane", T, err, be)
        @test err <= Cp
        if T === Float64
            rn = antenna_run(prob; probes = (8, 20, 30), solver = M4(), antenna_solver = Yee(), be)
            naive = maximum(x -> x.err, angular_error(rn; kx = m4_kx)) / (base_eps(T) * sqrt(prob.nsteps))
            @info "T-1085 plane wave" err naive
            @test naive > Cn
        end
    end
end

@testset "T-1085 the launched M4 beam is M4's angular-spectrum propagation of the incident field to the antenna's order (k_y/k)⁴, and its ω₀ plane wave is exact [A2, A7, A8, WP-079]" begin
    body_T1085(CPU())
end

# --- T-1086 ---------------------------------------------------------------------------------

# T-401 with M4: the antenna launches one way, the energy behind the plane at T-401's measured
# fraction for the small beam of the A8 matrix (a0, focus and chirp seeded as the dual's partials;
# the fields are linear in a0, a0·∂E/∂a0 = E to roundoff) and for the main beam. A narrow band beam
# (waist 20, duration 25) leaks far less, and the Yee antenna in the same M4 run leaks a hundred
# times more and beyond: its two corrections leave B points of the scattered region reading
# total-field E through M4's wider stencil. And the B corrections as an operator
# (`tfsf_operator_error`, m4_cases.jl): on the antenna's own incident E, one M4 half step of the
# stored field plus the corrections leaves the scattered region's B zero and gives the total
# region the half step of the whole incident field, to roundoff, for both lasers and three
# polarizations in the A8 matrix; with Yee's corrections the scattered B is 4.6 % of the field's.
# The operator test takes the incident E from the antenna's own `incident_e`, so the columns it
# reads are checked apart (`incident_column_error`): on ia − 1, ia and ia + 1 they are the beam
# written out from its definition, a distance mΔx from the plane and a time m·kΔx later.
# (Under M4, Bx also reads across the plane: the β terms of D̃_y Ez average Ez over the columns
# beside, so Bx on ia − 1 and ia is corrected too; Yee needs no Bx correction.)
function narrowband_leak(pol; solver = M4(), antenna_solver = solver)
    # the snapshot at t = 160, the envelope's peak crossing the plane at t = 150: what leaks backward
    # from t = 132 on lies behind x = 180 after wrapping round the periodic box, and the pulse ahead
    prob = antenna_problem(Float64; length = (200, 160), waist = 20, duration = 25, position = 8, focus = 30, polarization = pol, tend = 160)
    r = antenna_run(prob; snaps = (prob.nsteps,), solver, antenna_solver)
    return leak_fraction(r, 180)
end

@testset "T-1086 the M4 antenna launches one way and is linear in a0; the Yee antenna in an M4 run leaks [A2, A3, A8, WP-079]" begin
    C, Cl, Cm, Cb, Cn = emC("T-401"), emC("T-401", "C_linear"), emC("T-401", "C_main"), emC("T-1086"), emC("T-1086", "C_naive")
    for T in TYPES
        D = T === D3
        a0 = D ? mkdual(1, 1, 0, 0) : one(T)
        focus = D ? mkdual(14, 0, 1, 0) : T(14)
        chirp = D ? mkdual(1 // 100, 0, 0, 1) : T(1 // 100)
        prob = antenna_problem(T; length = (150, 50), waist = 4, duration = 5, position = 8, focus, chirp, a0, tend = 66)
        r = antenna_run(prob; snaps = (prob.nsteps,), solver = M4())
        frac = leak_fraction(r, 80)
        record!("T-1086 small", T, frac / C)
        @test frac <= C
        if D
            worst = 0.0
            for c in eb(r.st.fields)
                v = Array(c)
                m = maximum(x -> abs(value_of(x)), v)
                d = maximum(x -> abs(ForwardDiff.partials(x)[1] - value_of(x)), v)
                worst = max(worst, m == 0 ? (d == 0 ? 0.0 : Inf) : d / (eps(Float32) * m * sqrt(prob.nsteps)))
            end
            record!("T-1086 linear", T, worst)
            @test worst <= Cl
            @test any(x -> !iszero(ForwardDiff.partials(x)[2]), Array(r.st.fields.Ey))
            @test any(x -> !iszero(ForwardDiff.partials(x)[3]), Array(r.st.fields.Ey))
        end
    end
    Co, Ci = emC("T-1086", "C_operator"), emC("T-1086", "C_incident")
    for T in TYPES
        cases = Any[(pol, antenna_problem(T; length = (40, 30), waist = 3, duration = 3, position = 8, polarization = pol, tend = 1)) for pol in (0, 3 // 5, π / 2)]
        push!(cases, (:shaped, shaped_problem(T; length = (60, 40), tend = 1)))
        for (label, prob) in cases
            st = EMState(prob, EMNumerics(; solver = M4()))
            t = value_of(prob.lasers[1].delay)
            r = tfsf_operator_error(st, prob, t)
            worst = max(r.scattered, r.total) / base_eps(T)
            record!("T-1086 operator", T, worst)
            @test worst <= Co
            inc = incident_column_error(st, prob, t) / base_eps(T)
            record!("T-1086 incident columns", T, inc)
            @test inc <= Ci
            if T === Float64 && label === 0
                n = tfsf_operator_error(st, prob, t; source_solver = Yee())
                @info "T-1086 operator" r n
                @test n.scattered > 1e-2
            end
        end
    end
    for pol in ANTENNA_POLS
        r = m4_main(pol)
        frac = leak_fraction(r, 160)
        record!("T-1086 main", Float64, frac / Cm)
        @test frac <= Cm
    end
    for pol in (0.0, π / 2)
        m4, naive = narrowband_leak(pol), narrowband_leak(pol; antenna_solver = Yee())
        @info "T-1086 narrow band" pol m4 naive
        record!("T-1086 narrow band θ=$(round(pol; digits = 2))", Float64, m4)
        @test m4 <= Cb
        @test naive > Cn * m4
    end
end

# --- T-1087 ---------------------------------------------------------------------------------

# T-906 with M4: the shaped beam (an aberrated flattened pupil, a pulse stretched by GDD and TOD,
# polarized at 0.6 rad) is M4's free beam of its incident field, to the antenna's order of the
# pupil's spectrum, and launches one way, in the A8 matrix. The shaped antenna's B source evaluates
# the incident wave on three columns, from the time factors of three times. On every probe past
# the plane the error is T-906's; on the plane's own column, where the corrections set the field,
# Ey's is 3.4 times the scale (Yee's 1.35): the antenna's B carries M4's B/E ratio of the axis, ρ,
# and the paraxial k_y² term of Yee's, while M4's ratio also varies with the angle through δ_x and
# β_xy, a term of order Δt²k_y² that the pupil's wide spectrum (NA 0.3) shows and the main beam
# (waist 6) does not. It is the antenna's own field at the plane, not in the launched beam: at
# half the time step the plane column's excess over the next probe falls 4.5 times (measured in
# the PR that added this test), and the probes past the plane stay at T-906's.
function body_T1087(be)
    C, Cl = emC("T-906"), emC("T-906", "C_leak")
    for T in TYPES
        prob = shaped_problem(T)
        _, w = envelope_extent(prob.lasers[1].envelope)
        d = 50
        k = round(Int, (value_of(prob.lasers[1].delay) + d) / value_of(prob.dt))
        r = antenna_run(prob; probes = (10, 25, 40, 60, 80), snaps = (k,), be, solver = M4())
        sc = pupil_angular_scale(prob.lasers[1].profile)
        for comp in (:Ey, :Ez)
            errs = angular_error(r; component = comp, kx = m4_kx)
            past = maximum(x -> x.err, errs[2:end])
            at = errs[1].err
            @test errs[1].x ≈ Float64(value_of(run_antenna(r).xa))      # the first probe is the plane's column
            record!("T-1087 $comp", T, past / sc, be)
            record!("T-1087 $comp at the plane", T, at / sc, be)
            @test past <= C * sc
            @test at <= emC("T-1087", "C_at_plane") * sc
        end
        frac = leak_fraction(r, value_of(10 + d + 6 * w))
        record!("T-1087 leak", T, frac, be)
        @test frac <= Cl
    end
end

@testset "T-1087 the shaped beam launched with M4 is M4's angular-spectrum propagation of its incident field, and launches one way, in the A8 matrix [A2, A7, A8, WP-079]" begin
    body_T1087(CPU())
end

# --- T-1088 ---------------------------------------------------------------------------------

# The PML with M4: in its B corrections the difference along the layer axis is M4's, so the
# layer's stretched B update is M4's own (its E corrections are Yee's, as Ampère's law). The
# slab packets of T-540 (both polarizations, 8 and 16 cells, 0 to 0.8 rad, PIConGPU's profile)
# reflect no more than Yee's packets do at the same geometry (their discrete theory, T-540); the
# same layers with Yee's difference in the B corrections, in the M4 run, reflect 7 to 600 times
# more. (Past the test's angles the M4 layer reflects slightly more than Yee's: 1.03 to 1.22 times
# at 1.1 to 1.2 rad, at reflections of 10⁻⁸ to 10⁻¹⁴, measured by the PR's physics breaker; the
# tolerance of 1.5 covers that.) A zero profile is the periodic M4 step bit for bit (TYPES), and random fields in an open
# box with the PML stay bounded over a long run.
function m4_slab_run(geo, pol; absorber, layer_stencil = nothing, T = Float64)
    prob, num = slab_problem(T, geo; absorber, solver = M4())
    st = EMState(prob, num)
    if layer_stencil !== nothing
        st = with_modifiers(st, map(m -> m isa PMLFields{2} ?
                                         PMLFields{2}(m.node, m.center, m.ψ, m.scratch, m.cells, m.g, m.n, m.dt, m.sigma_max, layer_stencil) : m,
                                    st.modifiers))
    end
    initialize!(st, prob)
    slab_init!(st, geo, pol; solver = M4())
    rows = slab_rows(st, geo)
    W0 = slab_energy(st, rows.interior)
    run!(st, params(prob), prob.nsteps)
    return (; R = slab_energy(st, rows.upper) / W0, Tr = slab_energy(st, rows.lower) / W0, st)
end

function m4_pml_long_run(; nsteps = 4000, n = (16, 40), every = 100)
    L = (2 * π, 2 * π * n[2] / n[1])
    dt = 0.95 / sqrt(sum(d -> (n[d] / L[d])^2, 1:2))
    prob = EMProblem("long", L, n, dt, nsteps, (), 1, (1, 1); boundaries = (:periodic, :open))
    st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = PML(; cells = 8, kappa_max = 3, alpha_max = 0.1), solver = M4()))
    initialize!(st, prob)
    rng = Xoshiro(3)
    foreach(c -> (a = getfield(st.fields, c); a .= randn(rng, size(a)); fill_guards!(a, st.grid)), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    g = guards(st.grid)
    W() = sum(c -> sum(abs2, view(getfield(st.fields, c), (g + 1):(g + n[1]), (g + 1):(g + n[2]))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz)) / 2
    W0 = W()
    p = params(prob)
    out = Float64[]
    for k in 1:nsteps
        step!(st, p, k)
        k % every == 0 && push!(out, W() / W0)
    end
    return out
end

@testset "T-1088 the PML with M4 reflects as little as Yee's, and with Yee's difference far more; a zero profile is the periodic M4 step bit for bit; bounded [A2, A4, A7, A8, WP-079]" begin
    C, Cn = emC("T-1088"), emC("T-1088", "C_naive")
    worst = 0.0
    for (cells, θ, width) in ((8, 0.0, 3), (16, 0.0, 3), (8, 0.4, 3), (16, 0.8, 4)), pol in (:TE, :TM)
        geo = slab_geometry(; theta = θ, cells, width)
        pml = slab_pml(geo)
        yee = pml_prediction(geo, pml, pol).R
        m4 = m4_slab_run(geo, pol; absorber = pml)
        naive = m4_slab_run(geo, pol; absorber = pml, layer_stencil = YeeStencil())
        @info "T-1088" cells θ pol R_yee = yee R_m4 = m4.R R_naive = naive.R Tr_m4 = m4.Tr
        worst = max(worst, m4.R / yee)
        @test m4.R <= C * yee
        @test naive.R > Cn * m4.R
    end
    record!("T-1088", Float64, worst)
    for T in TYPES
        geo = slab_geometry(; theta = 0.0, cells = 8)
        prob, num = slab_problem(T, geo; absorber = PML(; cells = 8, sigma_max = 0, kappa_max = 1, alpha_max = 0), solver = M4())
        pp = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, (), 1, prob.kmode)
        so = EMState(prob, num)
        sp = EMState(pp, EMNumerics(; current = NoCurrent(), solver = M4()))
        for s in (so, sp)
            initialize!(s, s === so ? prob : pp)
            slab_init!(s, geo, :TE; solver = M4())
        end
        run!(so, params(prob), prob.nsteps)
        run!(sp, params(pp), pp.nsteps)
        @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), keys(so.fields))
    end
    w = m4_pml_long_run()
    q = length(w) ÷ 4
    @info "T-1088 long run" max = maximum(w) last = w[end]
    @test maximum(w) <= emC("T-1088", "C_long")
    @test sum(w[(3q + 1):end]) <= sum(w[(q + 1):(2q)])
end

# --- T-1089 ---------------------------------------------------------------------------------

# Gauss's law and M4's ∇·B with particles leaving through the PML, all three types, both modes
# (T-543 with M4): Ampère's law is Yee's, so D⁻·E = ρ − ρ̄₀ off the layer to roundoff; M4's
# divergence Σ_d D̃_d B_d is kept on the centres whose stencil reads no layer row (D̃_y reaches the
# rows j − 1 to j + 2, the β terms of D̃_x one row across), and is far from roundoff in the layer,
# which shows the check sees it. And in 1D (T-212's Landau case with M4, whose x difference is
# D⁺ + δ_xΔ³): Gauss's law to T-212's roundoff.
@testset "T-1089 Gauss's law and M4's ∇·B to roundoff off the PML with particles leaving, all three types, both modes, and Gauss's law with M4 in 1D [A2, A7, A8, WP-079]" begin
    C, CB = emC("T-451"), emC("T-1089", "C_divB")
    for T in TYPES, mode in (Reference(), Fast())
        prob, _ = open_plasma_2d(T; amplitude = seeded(T, 1 / 10))
        st = open_state(prob, PML(; cells = 4); mode, solver = M4())
        ρbar0 = open_residuals(st, zero(T)).ρbar
        r = open_residuals(st, ρbar0)
        worst = r.free / (base_eps(T) * r.scale)
        worstB = layerB = 0.0
        n0 = alive_total(st)
        n = ncells(st.grid)
        free = (1:n[1], (4 + 2):(n[2] - 4 - 2))
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = open_residuals(st, ρbar0)
            worst = max(worst, r.free / (base_eps(T) * r.scale * sqrt(k + 1)))
            sB = m4_div_scale(st)
            if sB > 0
                worstB = max(worstB, Float64(m4_divergence(st, free)) / (base_eps(T) * sB * sqrt(k)))
                layerB = max(layerB, Float64(m4_divergence(st, (1:n[1], 1:4))) / (base_eps(T) * sB * sqrt(k)))
            end
        end
        left = n0 - alive_total(st)
        @info "T-1089" T mode worst worstB layerB left of = n0
        @test left > 50
        @test layerB > 1e4
        record!("T-1089 $(nameof(typeof(mode)))", T, worst)
        record!("T-1089 divB $(nameof(typeof(mode)))", T, worstB)
        @test worst <= C
        @test worstB <= CB
    end
    for T in TYPES, mode in (Reference(), Fast())
        case = load_case("landau")
        amplitude = T === D3 ? seeded(T, 0.01) : nothing
        prob = problem_from_case(case, T; ppc = 64, t_end = 2, amplitude)   # T-212's run, 214 steps
        st = EMState(prob, with_solver(numerics_from_case(case; mode), M4()))
        initialize!(st, prob)
        @test st.numerics.solver isa M4
        n = ncells(st.grid)
        p = params(prob)
        r = gauss_residual(st)
        worst = r.res / (base_eps(T) * n * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = gauss_residual(st)
            worst = max(worst, r.res / (base_eps(T) * n * r.scale * sqrt(k)))
        end
        record!("T-1089 1D $(nameof(typeof(mode)))", T, worst)
        @test worst <= emC("T-212")
    end
end

# --- T-1090 ---------------------------------------------------------------------------------

# The moving window with M4. T-472 and T-570: a vacuum pulse in the window is its static twin's
# before the window moves and, after, past the columns where the back edge's difference has reached
# (`back_comparison`, columns 25 on), to roundoff, and a PML on the back edge removes the back
# edge's reflection, to its measured fraction (C_back: the layer must be M4's, `back_state` stores
# the solver's stencil in it). The default window starts after the antenna has finished, so a run
# with the plane further forward (x = 20) and the window starting at t = 20, while the antenna
# still emits, checks the shifted antenna (`shift_antenna` keeps M4's stencil and ρ) against the
# twin, to the same roundoff. T-473: Gauss's law to roundoff across shifts with the laser driving the plasma,
# all three types, both modes, off the columns T-473 excludes.
@testset "T-1090 the moving window with M4: the vacuum pulse is its static twin's, the back-edge PML absorbs, Gauss's law across shifts [A2, A4, A7, A8, WP-079]" begin
    C = emC("T-472")
    bare = back_comparison(nothing; solver = M4())
    pml = back_comparison(PML(; cells = 8); solver = M4())
    @info "T-1090 back edge" bare = bare.near pml = pml.near before = pml.before far = pml.far bare_before = bare.before bare_far = bare.far
    @test pml.shifts > 64
    @test bare.near > 0.05
    @test pml.near < emC("T-570") * bare.near
    @test pml.near <= emC("T-1090", "C_back") * bare.near
    record!("T-1090 back edge", Float64, pml.near / bare.near)
    mid = back_comparison(nothing; solver = M4(), position = 20, start = 20)
    @info "T-1090 window moving while the antenna emits" before = mid.before far = mid.far shifts = mid.shifts
    for c in (bare, pml, mid)
        @test c.before <= C * eps(Float64)
        @test c.far <= C * eps(Float64)
    end
    record!("T-1090 twin", Float64, max(bare.far, pml.far, bare.before, pml.before) / eps(Float64))
    record!("T-1090 twin, antenna emitting", Float64, max(mid.far, mid.before) / eps(Float64))
    Cg = emC("T-473")
    for T in TYPES, mode in (Reference(), Fast())
        prob = window_lwfa(T)
        st = window_state(prob; mode, solver = M4())
        p = params(prob)
        ia = antenna_planes(st)[1].ia
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 3 == 0 || continue
            s_win = window_shifts(st, k)
            r = window_gauss(st, p, s_win)
            skip = (1, 2, ia - 1 - s_win, ia - s_win)
            cols = [i for i in axes(r.res, 1) if !(i in skip)]
            worst = max(worst, Float64(maximum(r.res[cols, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        @info "T-1090 Gauss" T mode worst shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-1090 $(nameof(typeof(mode)))", T, worst)
        @test worst <= Cg
    end
end

# --- T-1091 ---------------------------------------------------------------------------------

# Reference mode with M4 is bitwise across reruns and launch shapes: the laser–plasma run in the
# moving window with the back-edge PML (120 steps, past the window's start) and the open plasma
# with the PML, the 2D plasma with two passes of the current filter and the 1D Landau case with
# three (T-443's runs), per element type; and the two modes agree to roundoff in the fields (T-474's
# comparison, Float64).
function m4_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES
        prob = back_lwfa(T; nsteps = 120)
        st = back_state(prob; back = PML(; cells = 6, alpha_max = 0.2), workgroupsize, solver = M4(), be)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("window", T, state_digest_2d(st)))
        prob, _ = open_plasma_2d(T; amplitude = seeded(T, 1 / 10))
        st = open_state(prob, PML(; cells = 4, kappa_max = 2, alpha_max = 0.1); workgroupsize, solver = M4(), be)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("open", T, state_digest_2d(st)))
        prob = plasma_step_2d(T; nsteps = 20)
        st = EMState(prob, EMNumerics(; workgroupsize, filter = BinomialFilter(2), solver = M4()); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("filtered", T, state_digest_2d(st)))
        case = load_case("landau")
        case["numerics"]["filter_passes"] = 3
        prob = problem_from_case(case, T; ppc = 64, t_end = DETERMINISM_T_END)
        st = EMState(prob, with_solver(numerics_from_case(case; workgroupsize), M4()); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("1d filtered", T, state_digest(st)))
    end
    return out
end

function body_T1091(be)
    ref = m4_digests(; be)
    @test ref == m4_digests(; be)
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test m4_digests(; workgroupsize = wgs, be) == ref
    end
    prob = window_lwfa(Float64; nsteps = 120)
    a, b = window_state(prob; mode = Reference(), be, solver = M4()), window_state(prob; mode = Fast(), be, solver = M4())
    run!(a, params(prob), prob.nsteps)
    run!(b, params(prob), prob.nsteps)
    m = maximum(c -> maximum(abs, Array(getfield(a.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    d = maximum(c -> maximum(abs, Array(getfield(a.fields, c)) .- Array(getfield(b.fields, c))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    @info "T-1091 Reference against Fast" rel = d / m
    record!("T-1091", Float64, d / (m * eps(Float64)), be)
    @test d <= emC("T-474") * eps(Float64) * m
end

@testset "T-1091 the M4 step is bitwise across reruns and launch shapes in Reference mode, and the modes agree [A4, WP-079]" begin
    body_T1091(CPU())
end

# --- T-1092 ---------------------------------------------------------------------------------

# A3 through M4: the ω₀ intensity on the axis of a probe past the plane against the laser's focus,
# by ForwardDiff through the M4 run, agrees with the finite-difference floor (T-407's observable).
function m4_observable(x, mode, seed)
    T = typeof(x)
    prob = antenna_problem(T; length = (40, 30), waist = 3, duration = 3, position = 5, tend = 51, focus = x, chirp = 1 // 100)
    r = antenna_run(prob; mode, probes = (20,), solver = M4())
    A = r.A[1].Ey
    return abs2(A[length(A) ÷ 2])
end

@testset "T-1092 laser parameters enter through the M4 antenna: the gradient in focus agrees with the finite-difference floor [A3, A7, WP-079]" begin
    C = emC("T-407")
    hs = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    x0 = 12.0
    ad = ForwardDiff.derivative(x -> m4_observable(x, Reference(), 1), x0)
    s = fd_scan(m4_observable, x0, hs; seed = 1, ad)
    @info "T-1092" ad fd = s.fd floor = s.floor_rel rel = s.rel[s.istar]
    record!("T-1092", Float64, s.rel[s.istar] / s.floor_rel)
    @test abs(ad) > 0
    @test s.rel[s.istar] <= C * max(s.floor_rel, 1e-10)
end

# --- T-1093 ---------------------------------------------------------------------------------

# Input checks and the solver carried along: M4's half step refuses a grid of one guard, and the
# state's grid has the two M4 reads for either shape (the shapes' own guards already give two, so
# this checks the result, not the solver's term in it); an antenna plane needs one column more
# behind it with M4 (its B corrections reach ia − 2), and the window's check of the antennas counts
# that column: a plane at column 14 with a back-edge PML of 12 cells passes with Yee and is refused
# with M4; the window's shifted antenna and the layers rebuilt on other arrays (`on_arrays`, as a
# restart does) keep M4's stencil; the Courant limit is Yee's.
@testset "T-1093 input checks: guards, the antenna's columns and the Courant limit with M4 [A2, A10, WP-079]" begin
    grid1 = PeriodicGrid((8, 8), (0.5, 0.5), 1)
    f = em_fields(CPU(), Float64, grid1)
    @test_throws ArgumentError m4_b_half!(f, grid1, 0.1, faraday_stencil(M4(), grid1, 0.2))
    for shape in (Linear, Quadratic)
        prob = antenna_problem(Float64; length = (40, 20), tend = 1)
        @test guards(EMState(prob, EMNumerics(; shape, solver = M4())).grid) >= 2
    end
    grid = PeriodicGrid((40, 8), (0.5, 0.5), 3)
    dt = 0.3
    g = guards(grid)
    x(ia) = (ia - 1) * 0.5                       # the plane's position at the interior column ia
    @test AntennaPlane(x(g + 2), grid, dt, Yee()).ia == g + 2
    @test_throws ArgumentError AntennaPlane(x(g + 2), grid, dt, M4())
    @test AntennaPlane(x(g + 3), grid, dt, M4()).ia == g + 3
    @test courant_limit(M4(), grid) == courant_limit(Yee(), grid)
    wprob = back_lwfa(Float64; start = 60)
    @test back_state(wprob; back = PML(; cells = 12)) isa EMState
    @test_throws ArgumentError back_state(wprob; back = PML(; cells = 12), solver = M4())
    st = back_state(back_lwfa(Float64); back = PML(; cells = 6), solver = M4())
    a = Antenna(params(wprob).lasers[1], antenna_planes(st)[1])
    b = shift_antenna(a, 1)
    @test a.stencil isa M4Stencil
    @test b.stencil === a.stencil && b.ρ === a.ρ && b.ia == a.ia - 1
    ost = open_state(open_plasma_2d(Float64)[1], PML(; cells = 4); solver = M4())
    layers = [m for m in (st.modifiers..., ost.modifiers...) if m isa Union{PMLFields,BackPMLFields}]
    @test length(layers) == 2
    for m in layers
        @test m.stencil isa M4Stencil
        @test on_arrays(m, Tuple(values(m.ψ))).stencil === m.stencil
    end
    prob = antenna_problem(Float64; length = (40, 20), tend = 1, courant = 101 // 100)
    @test_throws ArgumentError EMState(prob, EMNumerics(; solver = M4()))
end
