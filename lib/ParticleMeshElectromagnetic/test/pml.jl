# WP-034: the convolutional PML of the open transverse axis, T-540 to T-548. The fields leave through
# the PML (`PML`, the outer rows of the box along y, where the y-differences are stretched and
# convolved), the particles that cross the box edge are retired as with WP-027's layer. The
# reflection runs are WP-027's vacuum slab packets with a PML, compared with the discrete theory of
# the same scheme (`pml_rt`, pml_cases.jl); the plasma runs are `open_plasma_2d` with a PML.

using ForwardDiff: ForwardDiff, Partials
using ParticleMeshElectromagnetic: PMLFields, pml_b!, pml_e!

"The vacuum slab run of `geo` with `pml` against `pml_rt`: (; run, pred, geo, k)."
function pml_slab(geo, pml, pol; T = Float64)
    run = slab_run(T, geo, pol; absorber = pml)
    return (; run, pred = pml_prediction(geo, pml, pol), geo, k = round(Int, geo.tmeas / geo.dt))
end

# A profile with every term active: κ > 1 graded at its own order and a frequency shift, Smilei's
# default grading (σ = 20u², κ = 1 + 79u⁴) with an α added.
mixed_pml(geo) = PML(; cells = geo.cells, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4, alpha_max = 0.1)

# --- T-540 ----------------------------------------------------------------------------------

# The reflection R and the transmission through both layers Tr of a slab packet against the PML's
# discrete theory, over the thickness (8, 16 cells) and the angle (0, 0.4, 0.8 rad), in both
# polarizations, at PIConGPU's profile; and one run each of Smilei's grading with α and of κ alone.
# The theory first: with a zero profile the slab passes the mode with the phase of its Yee k_y and
# reflects nothing, and at normal incidence TE and TM see the same |r| for every profile. The packet
# is 3 wavelengths wide up to 0.4 rad and 4 at 0.8 rad, so that its slowest modes have left the upper
# half of the interior by the measurement time (at 1 rad and 3 wavelengths they carry a few 10⁻⁹ of
# the energy there, which is a 10⁻³ error on R).
@testset "T-540 the PML's reflection and transmission against the scheme's discrete theory, over thickness, angle and profile [A2, A7, WP-034]" begin
    for θ in (0.0, 0.8), pol in (:TE, :TM)
        geo = slab_geometry(; theta = θ)
        zero_pml = PML(; cells = geo.cells, sigma_max = 0, kappa_max = 1, alpha_max = 0)
        for (ky, _) in slab_modes(geo)[1:7:end]
            r, t = pml_rt(geo, zero_pml, pol, ky)
            @test abs(r) < 1e-13
            @test abs(abs(t) - 1) < 1e-12
            @test abs(cis(ky * (2 * geo.cells + 6) * geo.dy) - t / abs(t)) < 1e-9
        end
    end
    for pml in (geo -> slab_pml(geo), mixed_pml, geo -> PML(; cells = geo.cells, order = 0, alpha_max = 0.3))
        geo = slab_geometry(; theta = 0.0, cells = 8)
        for (ky, _) in slab_modes(geo)[1:7:end]
            rTE, rTM = abs(first(pml_rt(geo, pml(geo), :TE, ky))), abs(first(pml_rt(geo, pml(geo), :TM, ky)))
            @test abs(rTE - rTM) <= 1e-10 * rTE
        end
    end
    C = emC("T-540")
    worst = 0.0
    Rs = Dict{Any,Float64}()
    cfgs = [(8, 0.0, 3, :picongpu), (16, 0.0, 3, :picongpu), (8, 0.4, 3, :picongpu), (16, 0.4, 3, :picongpu),
            (8, 0.8, 4, :picongpu), (16, 0.8, 4, :picongpu), (8, 0.4, 3, :mixed), (8, 0.4, 3, :kappa)]
    for (cells, θ, width, prof) in cfgs, pol in (:TE, :TM)
        geo = slab_geometry(; theta = θ, cells, width)
        pml = prof === :picongpu ? slab_pml(geo) : prof === :mixed ? mixed_pml(geo) : PML(; cells, kappa_max = 4)
        (; run, pred, k) = pml_slab(geo, pml, pol)
        eR = abs(run.R - pred.R) / slab_scale(Float64, k, pred.R)
        eT = abs(run.Tr - pred.Tr) / slab_scale(Float64, k, pred.Tr)
        @info "T-540" cells θ prof pol R = run.R R_theory = pred.R Tr = run.Tr Tr_theory = pred.Tr ratio_R = eR ratio_T = eT
        worst = max(worst, eR, eT)
        Rs[(cells, θ, prof, pol)] = run.R
        @test pred.R > 0
        @test eR <= C && eT <= C
    end
    record!("T-540", Float64, worst)
    for pol in (:TE, :TM), θ in (0.0, 0.4, 0.8)
        @test Rs[(16, θ, :picongpu, pol)] < Rs[(8, θ, :picongpu, pol)]
    end
end

# --- T-541 ----------------------------------------------------------------------------------

# Where the matched layer fails, at grazing incidence: at 1.2 rad and 16 cells WP-027's layer at its
# nominal attenuation reflects about 9 % (its discrete theory, T-450), the PML with PIConGPU's
# profile about 5·10⁻¹⁵. The run is against the PML's theory as in T-540, with a packet 12
# wavelengths wide (at 9 wavelengths the slowest modes of the incident packet, still on their way
# up, add 3·10⁻⁴ of that R, 400 times the roundoff scale; at 12 they are below it), and the
# PML has to beat the layer at the same thickness by more than two orders of magnitude at 0.8 and
# 1.2 rad, from the two theories.
@testset "T-541 at grazing incidence the PML matches its theory and reflects far less than the matched layer [A2, A7, WP-034]" begin
    C = emC("T-540")
    geo = slab_geometry(; theta = 1.2, cells = 16, width = 12)
    (; run, pred, k) = pml_slab(geo, slab_pml(geo), :TE)
    eR = abs(run.R - pred.R) / slab_scale(Float64, k, pred.R)
    eT = abs(run.Tr - pred.Tr) / slab_scale(Float64, k, pred.Tr)
    @info "T-541" R = run.R R_theory = pred.R Tr = run.Tr Tr_theory = pred.Tr ratio_R = eR ratio_T = eT
    record!("T-541", Float64, max(eR, eT))
    @test eR <= C && eT <= C
    for θ in (0.8, 1.2), pol in (:TE, :TM)
        g = slab_geometry(; theta = θ, cells = 16, strength = 2.0)
        Rl = slab_prediction(g, pol).R
        Rp = pml_prediction(g, slab_pml(g), pol).R
        @info "T-541 layer against PML" θ pol R_layer = Rl R_pml = Rp
        @test Rp < Rl / 100
    end
end

# --- T-542 ----------------------------------------------------------------------------------

# A long run stays bounded: random fields (every mode, the evanescent ones and the static ones
# included) in an open vacuum box of 16 × 40 cells with 8-cell PMLs, 8000 steps (about 40 crossing
# times of the box along y). The energy of E and B over the box stays within a few per cent of its
# start (it is not the PML's energy: the convolution fields hold some, and sampled every 20 steps
# Smilei's grading rises 1.4 % above the start early on; a static field reaching into the layer
# rose 8 % in a breaker's run), and what remains at
# the end is what cannot reach the layers (modes with k_y = 0 and the static fields), which
# fluctuates by a few per cent and does not grow: the mean over the last quarter of the run is not
# above that over the second quarter. A growing late-time mode would break both.
function pml_long_run(pml; nsteps = 8000, n = (16, 40), every = 100)
    L = (2 * π, 2 * π * n[2] / n[1])
    dt = 0.95 / sqrt(sum(d -> (n[d] / L[d])^2, 1:2))
    prob = EMProblem("long", L, n, dt, nsteps, (), 1, (1, 1); boundaries = (:periodic, :open))
    st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = pml))
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

@testset "T-542 a long run with the PML stays bounded, with and without frequency shift and stretching [A2, A7, WP-034]" begin
    for (name, pml) in (("PIConGPU", PML(; cells = 8)), ("α", PML(; cells = 8, alpha_max = 0.2)),
                        ("Smilei", PML(; cells = 8, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4)))
        w = pml_long_run(pml)
        q = length(w) ÷ 4
        late, mid = sum(w[(3q + 1):end]) / q, sum(w[(q + 1):2q]) / q
        @info "T-542" name max = maximum(w) last = w[end] mid late
        @test maximum(w) <= 1.05
        @test w[end] < 0.7
        @test late <= mid * (1 + 1e-2)
    end
end

# --- T-543 ----------------------------------------------------------------------------------

# Gauss's law with particles leaving through the PML: the correction changes Ex only on its layer
# Node() rows and Ey not at all, so D⁻·E = ρ − ρ̄₀ to roundoff on the node rows off the layer, and
# ∇·B on the cell centres whose Bx row is off it, as T-451 measures for the layer (`open_residuals`).
@testset "T-543 Gauss's law and ∇·B to roundoff off the PML with particles leaving, all three types, both modes [A2, A7, A8, WP-034]" begin
    C, CB = emC("T-451"), emC("T-451", "C_divB")
    for T in TYPES, mode in (Reference(), Fast())
        prob, _ = open_plasma_2d(T; amplitude = seeded(T, 1 / 10))
        st = open_state(prob, PML(; cells = 4); mode)
        ρbar0 = open_residuals(st, zero(T)).ρbar
        r = open_residuals(st, ρbar0)
        worst = r.free / (base_eps(T) * r.scale)
        worstB = worst_layer = 0.0
        n0 = alive_total(st)
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = open_residuals(st, ρbar0)
            worst = max(worst, r.free / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
            worst_layer = max(worst_layer, Float64(r.layer / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        left = n0 - alive_total(st)
        @info "T-543" T mode worst worstB worst_layer left of = n0
        @test left > 50
        @test worst_layer > 1e4
        record!("T-543 $(nameof(typeof(mode)))", T, worst)
        record!("T-543 divB $(nameof(typeof(mode)))", T, worstB)
        @test worst <= C
        @test worstB <= CB
    end
end

# --- T-544 ----------------------------------------------------------------------------------

# A zero profile adds exactly zero: a vacuum slab packet through the open box with a PML of
# σ = α = 0, κ = 1 is the periodic box's bit for bit. A real PML: its coefficients are the closed
# form of the profile at the layer rows (exactly b = 1, c = 0, q = 0 at the rows whose depth is
# zero), its convolution fields start at zero and are zero again at the start of a new run (a
# snapshot of the start, or `reset_convolution!`; no step zeroes them, WP-056), every guard
# is its interior image after a solve, and only the layer rows of Bx, Bz, Ex, Ez change against the
# plain solve.
@testset "T-544 a zero-profile PML is the periodic step bit for bit; coefficients, rows and guards [A2, A4, A8, WP-034]" begin
    for T in TYPES
        geo = slab_geometry(; theta = 0.0, cells = 8)
        prob, num = slab_problem(T, geo; absorber = PML(; cells = 8, sigma_max = 0, kappa_max = 1, alpha_max = 0))
        pp = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, (), 1, prob.kmode)
        so = EMState(prob, num)
        sp = EMState(pp, EMNumerics(; current = NoCurrent()))
        for s in (so, sp)
            initialize!(s, s === so ? prob : pp)
            slab_init!(s, geo, :TE)
        end
        run!(so, params(prob), prob.nsteps)
        run!(sp, params(pp), pp.nsteps)
        @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), keys(so.fields))
        # a real PML: guards refilled change no bit; only the layer rows differ from the plain solve
        sl = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = PML(; cells = 8, kappa_max = 3, alpha_max = 0.1)))
        initialize!(sl, prob)
        foreach(a -> (a .= T.(randn(Xoshiro(7), size(a))); fill_guards!(a, sl.grid)), values(sl.fields))
        plain = map(copy, sl.fields)
        solve_fields!(Yee(), sl.fields, sl.grid, prob.dt, sl.modifiers, nothing, 1)
        solve_fields!(Yee(), plain, sl.grid, prob.dt)
        refilled = map(a -> fill_guards!(copy(a), sl.grid), sl.fields)
        @test all(c -> bits(Array(getfield(sl.fields, c))) == bits(Array(getfield(refilled, c))), keys(sl.fields))
        g, n = guards(sl.grid), ncells(sl.grid)
        inner = (g + 1):(g + n[1])
        # the B after the second half step reads the corrected E, so off-layer B rows next to it
        # differ too; the E rows two rows into the interior and beyond have not seen the layer yet
        # at the end of one step: E's corrected rows are j₀ < 8 and j₀ ≥ n₂ − 8
        far = (g + 1 + 8 + 2):(g + n[2] - 8 - 2)
        for c in (:Ex, :Ey, :Ez)
            @test bits(Array(getfield(sl.fields, c))[inner, far]) == bits(Array(getfield(plain, c))[inner, far])
        end
        for c in (:Ex, :Ez, :Bx, :Bz)
            @test Array(getfield(sl.fields, c))[inner, (g + 1):(g + 8)] != Array(getfield(plain, c))[inner, (g + 1):(g + 8)]
        end
        # a state that ran before, set back to its start, takes the fresh state's first step bit for
        # bit. The convolution fields are part of what a run evolves (WP-056), so no step zeroes
        # them: the start is a snapshot, which holds them (zero there), or the initial fields
        # copied back by hand with reset_convolution!; the fields alone are not the start
        mk() = (s = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = PML(; cells = 8, kappa_max = 3, alpha_max = 0.1)));
                initialize!(s, prob); slab_init!(s, geo, :TM); s)
        same_fields(a, b) = all(c -> bits(Array(getfield(a.fields, c))) == bits(Array(getfield(b.fields, c))), keys(a.fields))
        fresh, used = mk(), mk()
        init = map(copy, used.fields)
        start = snapshot(used, 0)
        step!(fresh, params(prob), 1)
        run!(used, params(prob), 5)
        @test any(!iszero, Array(open_absorber(used).ψEx))
        foreach(copyto!, values(used.fields), values(init))
        step!(used, params(prob), 1)
        @test !same_fields(fresh, used)
        foreach(copyto!, values(used.fields), values(init))
        reset_convolution!(used)
        step!(used, params(prob), 1)
        @test same_fields(fresh, used)
        run!(used, params(prob), 4; first = 2)
        restore!(used, start)
        step!(used, params(prob), 1)
        @test same_fields(fresh, used)
    end
    # the coefficients: 4 cells on 16 rows with 3 guards; at the half row p the depth
    # d = 8 − min(p, 32 − p), r = d/8, and off the layer exactly (1, 0, 0)
    prob, _ = open_plasma_2d(Float64; layer = 4)
    pml = PML(; cells = 4, sigma_max = 1.5, order = 3, kappa_max = 2, kappa_order = 2, alpha_max = 0.2, alpha_order = 1)
    st = open_state(prob, pml)
    L = open_absorber(st)
    n2 = ncells(st.grid)[2]
    for (set, sh) in ((L.node, 0), (L.center, 1))
        τ = prob.dt
        b, c, q = map(Array, set)
        @test length(b) == 8
        for k in 1:8
            j0 = k <= 4 ? k - 1 : n2 - 8 + k - 1
            p = 2j0 + sh
            d = 8 - min(p, 2n2 - p)
            if d <= 0
                @test (b[k], c[k], q[k]) == (1.0, 0.0, 0.0)
            else
                r = d / 8
                σ, κ, α = 1.5 * r^3, 1 + r^2, 0.2 * (1 - r)
                be = exp(-(σ / κ + α) * τ)
                @test b[k] ≈ be rtol = 1e-14
                @test c[k] ≈ σ * (be - 1) / (κ * (σ + κ * α)) rtol = 1e-13
                @test q[k] ≈ 1 / κ - 1 rtol = 1e-14
            end
        end
    end
    @test all(iszero, Array(L.ψBx)) && all(iszero, Array(L.ψEx))
    @test size(L.ψBx) == (ncells(st.grid)[1], 8)
    # each correction on its own: the B correction changes Bx and Bz on their layer rows and
    # nothing else, advancing ψBx and ψBz only when asked (the second half step), and the E
    # correction changes Ex and Ez only, each advancing its ψ
    foreach(a -> (a .= randn(Xoshiro(11), size(a)); fill_guards!(a, st.grid)), values(st.fields))
    g, n = guards(st.grid), ncells(st.grid)
    inner, rows = (g + 1):(g + n[1]), vcat((g + 1):(g + 4), (g + n[2] - 3):(g + n[2]))
    for (fn, changed, ψs, adv) in ((pml_b!, (:Bx, :Bz), (:ψBx, :ψBz), false), (pml_b!, (:Bx, :Bz), (:ψBx, :ψBz), true),
                                   (pml_e!, (:Ex, :Ez), (:ψEx, :ψEz), true))
        Lc = PMLFields(pml, st.grid, prob.dt; backend = CPU())
        foreach(c -> (a = getfield(Lc, c); a .= randn(Xoshiro(13), size(a))), (:ψBx, :ψBz, :ψEx, :ψEz))
        ψ0 = map(c -> copy(getfield(Lc, c)), (ψBx = :ψBx, ψBz = :ψBz, ψEx = :ψEx, ψEz = :ψEz))
        f = map(copy, st.fields)
        fn === pml_b! ? fn(Lc, f, st.grid, prob.dt / 2, adv, nothing) : fn(Lc, f, st.grid, prob.dt, nothing)
        for c in keys(f)
            if c in changed
                @test all(j -> Array(getfield(f, c))[inner, j] != Array(getfield(st.fields, c))[inner, j], rows)
            else
                @test bits(Array(getfield(f, c))) == bits(Array(getfield(st.fields, c)))
            end
        end
        for c in keys(ψ0)
            @test (getfield(Lc, c) == ψ0[c]) == !(adv && c in ψs)
        end
    end
    # PIConGPU's optimum when sigma_max is not given
    @test PMLFields(PML(; cells = 4), st.grid, prob.dt; backend = CPU()).sigma_max ≈ 0.8 * 5 / cellsize(st.grid)[2]
end

# --- T-545 ----------------------------------------------------------------------------------

@testset "T-545 the PML step with particles is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-034]" begin
    ref = pml_step_digests()
    @test ref == pml_step_digests()
    for wgs in (32, 256)
        @test pml_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        lines = filter(startswith("EM2DP "), split(worker_output(nt), '\n'))
        got = Dict(String(f[3]) => parse(UInt64, f[4]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(ref)
        for (T, d) in ref
            @test get(got, type_label(T), nothing) == d
        end
    end
end

# --- T-546 ----------------------------------------------------------------------------------

# The element-type matrix on one reflection run with every profile term active (8 cells, θ = 0.4,
# R ≈ 10⁻⁴): each type within C·slab_scale of the discrete theory, and the dual's partials, seeded
# equal to the values in the first direction, stay the values bit for bit (the vacuum step with the
# PML is linear in the fields).
@testset "T-546 the PML's reflection in Float32, Float64 and Dual, and the dual's partials linear [A7, A8, WP-034]" begin
    C = emC("T-540")
    geo = slab_geometry(; theta = 0.4, cells = 8)
    pml = mixed_pml(geo)
    pred = pml_prediction(geo, pml, :TE)
    k = round(Int, geo.tmeas / geo.dt)
    for T in TYPES
        prob, num = slab_problem(T, geo; absorber = pml)
        st = EMState(prob, num)
        initialize!(st, prob)
        slab_init!(st, geo, :TE)
        if T === D3
            for c in keys(st.fields)
                a = getfield(st.fields, c)
                v = value.(a)
                a .= D3.(v, Partials.(tuple.(v, zero.(v), zero.(v))))
            end
        end
        rows = slab_rows(st, geo)
        W0 = slab_energy(st, rows.interior)
        run!(st, params(prob), prob.nsteps)
        R, Tr = slab_energy(st, rows.upper) / W0, slab_energy(st, rows.lower) / W0
        e = max(abs(R - pred.R) / slab_scale(T, k, pred.R), abs(Tr - pred.Tr) / slab_scale(T, k, pred.Tr))
        @info "T-546" T R R_theory = pred.R Tr Tr_theory = pred.Tr ratio = e
        record!("T-546", T, e)
        @test e <= C
        if T === D3
            for c in keys(st.fields)
                a = Array(getfield(st.fields, c))
                @test all(x -> partials(x)[1] == value(x) && iszero(partials(x)[2]) && iszero(partials(x)[3]), a)
            end
            ψ = Array(open_absorber(st).ψEz)
            @test all(x -> partials(x)[1] == value(x), ψ)
        end
    end
end

# --- T-547 ----------------------------------------------------------------------------------

@testset "T-547 input checks and the case file's PML [A1, A10, WP-034]" begin
    @test_throws ArgumentError PML(; cells = 0)
    @test_throws ArgumentError PML(; cells = 4, sigma_max = -1)
    @test_throws ArgumentError PML(; cells = 4, sigma_max = Inf)
    @test_throws ArgumentError PML(; cells = 4, kappa_max = 0.5)
    @test_throws ArgumentError PML(; cells = 4, alpha_max = -0.1)
    @test_throws ArgumentError PML(; cells = 4, order = -1)
    @test_throws ArgumentError PML(; cells = 4, kappa_order = -1)
    @test_throws ArgumentError PML(; cells = 4, alpha_order = -1)
    @test PML(; cells = 4).sigma_max === nothing && PML(; cells = 4, order = 3).kappa_order == 3
    prob, _ = open_plasma_2d(Float64)
    pp = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, prob.kmode)
    @test_throws ArgumentError EMState(pp, EMNumerics(; absorber = PML(; cells = 4)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = PML(; cells = 2)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = PML(; cells = 8)))
    st = open_state(prob, PML(; cells = 7))
    @test st isa EMState{2} && open_absorber(st) isa PMLFields
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = PML(; cells = 4), filter = BinomialFilter(1)))
    # the case file: [numerics.pml], exactly one absorber
    case = deepcopy(load_case("weibel-2d"))
    case["grid"]["boundaries"] = ["periodic", "open"]
    case["numerics"]["pml"] = Dict("cells" => 6)
    @test numerics_from_case(case).absorber == PML(; cells = 6)
    case["numerics"]["pml"] = Dict("cells" => 6, "order" => 2)       # κ's order follows σ's
    @test numerics_from_case(case).absorber == PML(; cells = 6, order = 2, kappa_order = 2, alpha_order = 1)
    case["numerics"]["pml"] = Dict("cells" => 6, "sigma_max" => 20, "order" => 2, "kappa_max" => 80, "kappa_order" => 4,
                                   "alpha_max" => 0.1, "alpha_order" => 2)
    @test numerics_from_case(case).absorber == PML(; cells = 6, sigma_max = 20, order = 2, kappa_max = 80, kappa_order = 4,
                                                   alpha_max = 0.1, alpha_order = 2)
    pc = problem_from_case(case, Float64; cells = (16, 24), dt = 0.05)
    @test EMState(pc, numerics_from_case(case)) isa EMState{2}
    case["numerics"]["absorbing_layer"] = Dict("cells" => 6, "strength" => 1.5)
    @test_throws ArgumentError numerics_from_case(case)        # both absorbers
    periodic = deepcopy(load_case("weibel-2d"))
    periodic["numerics"]["pml"] = Dict("cells" => 6)
    @test_throws ArgumentError numerics_from_case(periodic)    # a PML without an open boundary
end

# --- T-548 ----------------------------------------------------------------------------------

function step_bytes_pml(mode; ppc, cells)
    prob, _ = open_plasma_2d(Float64; ppc, cells, nsteps = 4)
    st = open_state(prob, PML(; cells = 4); mode)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-548 allocation per PML step does not grow with particle count or grid size [A7, WP-034]" begin
    for mode in (Reference(), Fast())
        small = step_bytes_pml(mode; ppc = (2, 2), cells = (16, 16))
        more_particles = step_bytes_pml(mode; ppc = (8, 8), cells = (16, 16))
        larger_grid = step_bytes_pml(mode; ppc = (2, 2), cells = (64, 48))
        @info "T-548 bytes per PML step" mode small more_particles larger_grid threads = Threads.nthreads()
        if Threads.nthreads() == 1 || mode isa Fast
            @test_alloc more_particles <= small
            @test_alloc larger_grid <= small
        end
    end
end

# --- T-549 ----------------------------------------------------------------------------------

# The PML on the open y axis of a moving window (WP-028): the window's x edges are applied after
# the PML's correction, and a shift moves the convolution fields one column back with E and B
# (front column zero). A zero profile is the window's step with a zero-strength matched layer bit
# for bit; and with a real PML and a warm plasma leaving through both y edges, Gauss's law holds
# to roundoff across the shifts on the nodes off the layer rows and off the back nodes 1 and 2
# (T-473's scale and constant).
@testset "T-549 the PML in the moving window: the convolution fields shift with the fields, Gauss's law off the layer across shifts [A2, A4, A8, WP-034]" begin
    warm(T) = window_plasma(T; cells = (24, 16), thermal = (T(1) / 10, T(1) / 10, T(1) / 10), nsteps = 60, open_y = true)
    for T in TYPES
        prob = warm(T)
        a = window_state(prob; absorber = PML(; cells = 4, sigma_max = 0, kappa_max = 1, alpha_max = 0))
        b = window_state(prob; absorber = AbsorbingLayer(; cells = 4, strength = 0))
        run!(a, params(prob), prob.nsteps)
        run!(b, params(prob), prob.nsteps)
        @test window_shifts(a, prob.nsteps) > 0
        @test all(c -> Array(getfield(a.fields, c)) == Array(getfield(b.fields, c)), keys(a.fields))
        @test particle_set(a.species[1]) == particle_set(b.species[1])
    end
    # the shift of the convolution fields
    st = window_state(warm(Float64); absorber = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1))
    L = open_absorber(st)
    foreach(c -> (getfield(L, c) .= randn(Xoshiro(17), size(getfield(L, c)))), (:ψBx, :ψBz, :ψEx, :ψEz))
    before = map(c -> copy(getfield(L, c)), (:ψBx, :ψBz, :ψEx, :ψEz))
    PME_shift = ParticleMeshElectromagnetic.shift_absorber!
    PME_shift(L, nothing)
    n1 = ncells(st.grid)[1]
    for (c, old) in zip((:ψBx, :ψBz, :ψEx, :ψEz), before)
        new = getfield(L, c)
        @test new[1:(n1 - 1), :] == old[2:n1, :]
        @test all(iszero, new[n1, :])
    end
    # Gauss's law across shifts with a real PML
    C = emC("T-473")
    for T in TYPES
        prob = warm(T)
        st = window_state(prob; absorber = PML(; cells = 4, kappa_max = 2, alpha_max = 0.2))
        n2 = ncells(st.grid)[2]
        off = 5:(n2 - 4)        # node rows j₀ = 4 … n₂ − 5, whose Ex is not corrected
        p = params(prob)
        n0 = sum(s -> count(==(ParticleMeshCore.ALIVE), Array(s.status)), st.species)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = window_gauss(st, p, window_shifts(st, k))
            worst = max(worst, Float64(maximum(r.res[3:end, off]) / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        @info "T-549" T worst shifts = window_shifts(st, prob.nsteps) alive = sum(s -> count(==(ParticleMeshCore.ALIVE), Array(s.status)), st.species) of = n0
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-549", T, worst)
        @test worst <= C
    end
end
