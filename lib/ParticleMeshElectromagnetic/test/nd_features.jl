# WP-093: what the 3D step lacked without a reason in the physics, T-1250 to T-1258 (T-1259 is
# their rerun on CUDA, in gpu.jl). The binomial current filter at N = 3 (T-1250 to T-1253, T-1258): the
# kernel and the step's use of it are one method over N, so the 2D theory of filter.jl carries
# over with one more axis, and in a moving window it keeps the window's edges (T-1252). The filter
# beside an open axis at N = 2 and 3 (T-1254). The sort from
# the order of the last sort at N = 3 (T-1255, T-1256). What a state of each N takes (T-1257).
# Reuses no other test file: the host references are filter_cases.jl's, the problems
# em3d_cases.jl's and boundaries_cases.jl's, the runs shared with the worker nd_features_cases.jl's.

using ParticleMeshElectromagnetic: check_filter_fits, state_view, current!, push_kernel!, retire_x!, rho_stagger, window_frame, zero_x_edges!,
                                   deposit_mode
using ParticleMeshCore: launcher

"Start nd_features_worker.jl at `nt` threads, its output going to a file; once per suite."
function start_features_worker(nt)
    get!(FEATURES_WORKERS, nt) do
        worker = joinpath(@__DIR__, "nd_features_worker.jl")
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(dirname(Base.active_project())) $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end
const FEATURES_WORKERS = Dict{Int,Tuple{Base.Process,String}}()

"The lines of the worker at `nt` threads that start with `prefix`, after it finished."
function features_worker_lines(nt, prefix = "ND3DF ")
    proc, out = start_features_worker(nt)
    wait(proc)
    success(proc) || error("nd_features_worker.jl with $nt threads failed: $proc")
    return filter(startswith(prefix), split(read(out, String), '\n'))
end

foreach(start_features_worker, (1, 4))     # the workers run beside this file's in-process runs

"A 3D grid with a different cell count and cell size on every axis, two of the counts even (a Nyquist mode) and one odd."
const FILTER_GRID_3D = PeriodicGrid((8, 6, 5), (1 // 4, 1 // 5, 1 // 3), 2)

# --- T-1250 ---------------------------------------------------------------------------------

# `binomial!` at N = 3, mode by mode as T-440: a random field (value and partials) whose guards
# start at junk and whose scratch starts at NaN, then the DFT of input and output: every
# coefficient is H_p = Π_d cos^{2p}(π m_d/n_d) times the input's, to the roundoff of the 3p axis
# passes. One pass over three axes is an odd number of axis passes, so the result comes back from
# the scratch (p = 1 and 3), two passes end in the field itself (p = 2). The output is also the
# host filter by `circshift` bit for bit (both add (l + r) + 2c). Exact identities: a constant
# stays the constant, the Nyquist mode of an even axis becomes +0.0, a field mirror-symmetric
# along an axis stays so, zero passes touch nothing.
function body_T1250(be)
    C = emC("T-1250")
    rng = Xoshiro(1250)
    for T in TYPES, p in (1, 2, 3)
        grid = grid_T(FILTER_GRID_3D, T)
        n = grid.n
        A = [filter_rand(rng, T) for _ in CartesianIndices(n)]
        a, s = guarded(grid, A, T(1000), T(NaN))
        a, s = todev(be, a), todev(be, s)
        binomial!(a, s, grid, p)
        out = Array(a)
        Bout = collect(interior_view(out, grid))
        worst = 0.0
        for (ci, co) in zip(component_arrays(A), component_arrays(Bout))
            Âi, Âo = dft(ci), dft(co)
            err = maximum(m -> abs(Âo[m] - transfer_closed(m, n, p) * Âi[m]), CartesianIndices(Âi))
            worst = max(worst, Float64(err / (base_eps(T) * maximum(abs, ci) * p * 3)))
        end
        @test same_bits(Bout, binomial_host(A, p))
        # the guards hold the periodic images on return
        ref = copy(out)
        fill_guards!(ref, grid)
        @test same_bits(ref, out)
        # the exported closed form is the same function of k = 2π m/L, a different m on every axis
        L = map(*, n, grid.dx)
        m = (1, 2, 3)
        k = map((md, Ld) -> 2π * md / Float64(value_of(Ld)), m, L)
        dx = map(d -> Float64(value_of(d)), grid.dx)
        @test binomial_transfer(k, dx, p) ≈ transfer_closed(CartesianIndex(m .+ 1), n, p) rtol = 64 * base_eps(T)   # k and Δ in T
        record!("T-1250", T, worst, be)
        @test worst <= C
    end
    for T in TYPES
        grid = grid_T(FILTER_GRID_3D, T)
        n = grid.n
        c = seeded(T, 3 / 7)
        a, s = guarded(grid, fill(c, n), zero(T), T(NaN))
        a, s = todev(be, a), todev(be, s)
        binomial!(a, s, grid, 3)
        @test all(x -> same_bits([x], [c]), collect(interior_view(Array(a), grid)))
        for d in 1:3
            mirror(I) = CartesianIndex(ntuple(k -> k == d ? mod(-(I[k] - 1), n[k]) + 1 : I[k], 3))
            B = [filter_rand(rng, T) for _ in CartesianIndices(n)]
            S = [B[I] + B[mirror(I)] for I in CartesianIndices(n)]
            a, s = guarded(grid, S, zero(T), T(NaN))
            a, s = todev(be, a), todev(be, s)
            binomial!(a, s, grid, 3)
            O = collect(interior_view(Array(a), grid))
            @test all(I -> same_bits([O[I]], [O[mirror(I)]]), CartesianIndices(n))
            @test !all(I -> same_bits([O[I]], [O[CartesianIndices(n)[1]]]), CartesianIndices(n))   # not trivially constant
            iseven(n[d]) || continue
            A = [isodd(I[d]) ? c : -c for I in CartesianIndices(n)]
            a, s = guarded(grid, A, zero(T), T(NaN))
            a, s = todev(be, a), todev(be, s)
            binomial!(a, s, grid, 1)
            @test all_plus_zero(collect(interior_view(Array(a), grid)))
        end
        A = [filter_rand(rng, T) for _ in CartesianIndices(n)]
        a, s = guarded(grid, A, T(7), T(NaN))
        a0, s0 = copy(a), copy(s)
        a, s = todev(be, a), todev(be, s)
        @test binomial!(a, s, grid, 0) === a
        @test same_bits(Array(a), a0) && same_bits(Array(s), s0)
    end
end

@testset "T-1250 the binomial filter at N = 3 transfers Π cos^(2p)(kΔ/2) mode by mode and is the host filter bit for bit, with its exact identities [A2, A7, A8, WP-093]" begin
    body_T1250(CPU())
end

# --- T-1251 ---------------------------------------------------------------------------------

# T-441 at N = 3. Each step changes D⁻·E by −Δt·D⁻·(FJ) = F(ρⁿ⁺¹ − ρⁿ): the filter is a periodic
# convolution, so it commutes with the backward differences, and the 3D `deposit_flux!` makes the
# continuity equation hold to roundoff (T-940). Hence Qⁿ = D⁻·Eⁿ − Fρⁿ is constant to T-950's
# roundoff, and the plain residual D⁻·Eⁿ − (ρⁿ − mean ρ) = (F − 1)(ρⁿ − ρ⁰) stands far above it,
# so the filter acts; its mean stays zero, since F keeps the mean. The plasma of T-950 with one
# and two passes, both modes, every element type.
function body_T1251(be)
    C = emC("T-1251")
    for T in TYPES, mode in (Reference(), Fast()), passes in (1, 2)
        prob = plasma_step_3d(T; amplitude = seeded(T, 1 / 10))
        st = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes)); backend = be)
        initialize!(st, prob)
        p = params(prob)
        n = st.grid.n
        gp = gauss_parts(st)
        Q0 = gp.divE .- binomial_host(gp.ρ, passes)
        ρ0 = gp.ρ
        worst = effect = meanr = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            gp = gauss_parts(st)
            Q = gp.divE .- binomial_host(gp.ρ, passes)
            scale = sum(n) * maximum(absmax, gp.ρ)
            worst = max(worst, maximum(absmax, Q .- Q0) / (base_eps(T) * scale * sqrt(k + 1)))
            ρ̄ = sum(gp.ρ) / prod(n)
            r = gp.divE .- (gp.ρ .- ρ̄)
            effect = max(effect, Float64(value_of(maximum(x -> abs(value_of(x)), r) / maximum(x -> abs(value_of(x)), gp.ρ .- ρ̄))))
            meanr = max(meanr, Float64(absmax(sum(r))) / (base_eps(T) * scale * prod(n) * sqrt(k + 1)))
        end
        @test maximum(absmax, gp.ρ .- ρ0) > 0             # the run has moved the charge
        label = "p=$passes $(nameof(typeof(mode)))"
        record!("T-1251 $label", T, worst, be)
        record!("T-1251 mean $label", T, meanr, be)
        d = get!(MEASURED, "T-1251 max|D⁻·E − (ρ − mean ρ)| / max|ρ − mean ρ|", Dict{String,Float64}())
        d["$label $(type_label(T))$(be isa CPU ? "" : " on GPU")"] = effect
        @test worst <= C
        @test meanr <= C
        @test effect > 1e4 * base_eps(T)
    end
end

@testset "T-1251 with p passes Gauss's law of the 3D step holds in its filtered form D⁻·E − Fρ = const to roundoff, and the plain residual is measured, both modes [A2, A7, A8, WP-093]" begin
    body_T1251(CPU())
end

# --- T-1252 ---------------------------------------------------------------------------------

# T-442 at N = 3, the wiring: one step from the same initialized state with no passes and with
# two. The push and the deposition are the same, so the filtered state's Jx, Jy, Jz are the host
# filter of the unfiltered ones (three Esirkepov components, after the fold), and since the first
# B half step is the same, E differs by −Δt times the filtered minus the unfiltered current. The
# guards of J hold their periodic images. In Fast mode the two states' atomics add in different
# orders, so they agree as the two modes do (T-951's scale and constant).
function body_T1252(be)
    C = emC("T-1252")
    for T in TYPES, mode in (Reference(), Fast())
        prob = plasma_step_3d(T; amplitude = seeded(T, 1 / 10))
        passes = 2
        s0 = EMState(prob, EMNumerics(; mode); backend = be)
        s1 = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes)); backend = be)
        initialize!(s0, prob)
        initialize!(s1, prob)
        step!(s0, params(prob), 1)
        step!(s1, params(prob), 1)
        grid = s0.grid
        int(a) = collect(interior_view(Array(a), grid))
        worstJ = worstE = 0.0
        for (j, e) in ((:Jx, :Ex), (:Jy, :Ey), (:Jz, :Ez))
            J0, J1 = int(getfield(s0.fields, j)), int(getfield(s1.fields, j))
            @test maximum(absmax, J0) > 0
            @test maximum(absmax, J1 .- J0) > 0                  # the filter changes every component
            worstJ = max(worstJ, maximum(absmax, J1 .- binomial_host(J0, passes)) / (base_eps(T) * maximum(absmax, J0) * passes * 3))
            E0, E1 = int(getfield(s0.fields, e)), int(getfield(s1.fields, e))
            scale = maximum(absmax, E0) + Float64(value_of(prob.dt)) * maximum(absmax, J0)
            worstE = max(worstE, maximum(absmax, (E1 .- E0) .+ prob.dt .* (J1 .- J0)) / (base_eps(T) * scale))
            full = Array(getfield(s1.fields, j))
            ref = copy(full)
            fill_guards!(ref, grid)
            @test same_bits(ref, full)
        end
        if mode isa Fast
            f = sqrt(prod(prob.species[1].ppc))
            worstJ, worstE = worstJ / f, worstE / f
        end
        CJ = mode isa Fast ? emC("T-951") : C
        record!("T-1252 J $(nameof(typeof(mode)))", T, worstJ, be)
        record!("T-1252 E $(nameof(typeof(mode)))", T, worstE, be)
        @test worstJ <= CJ
        @test worstE <= CJ
    end
end

# T-757's window part and T-478 at N = 3. The current of a 3D moving-window state with two filter
# passes against its pieces written out: the sort, the deposit, the x-guards of the deposited
# current dropped before the fold, and the filter with the window's edges (zero x-guards in every
# pass); a filter that wraps along x gives another current. The state is the one `current!` finds:
# fourteen steps in, then the push and the retirement at the x edges of the next step. And
# `binomial!` with the window's edges on a 3D field: a current in the back column stays out of the
# front columns, which the periodic filter reaches across the seam, and loses what leaves through
# the back edge, while away from the x edges it is the periodic filter, seams of y and z included.
function body_T1252_window(be)
    for T in TYPES
        tenth = one(T) / 10
        prob = window_plasma_3d(T; nsteps = 30, thermal = (tenth, tenth, tenth), drift = (2tenth, zero(T), -tenth))
        function mk()
            st = EMState(prob, EMNumerics(; filter = BinomialFilter(2), headroom = 1 / 2); backend = be)
            initialize!(st, prob)
            run!(st, params(prob), 14; observer = nothing)
            (; numerics, grid, dt) = st
            for s in st.species
                launcher(push_kernel!, get_backend(s), nothing)(s, st.fields, numerics.pusher, numerics.shape, Int32(guards(grid)), s.q / s.m * dt / 2,
                                                                map(h -> dt / h, grid.dx), Int32.(grid.n); ndrange = capacity(s))
            end
            retire_x!(window_frame(st), st)
            return st
        end
        function written_out!(st; edges = window_frame(st))
            (; numerics, grid, fields) = st
            for (s, ws) in zip(st.species, st.ws)
                sort_by_cell!(s, grid, ws)
            end
            foreach(J -> fill!(J, zero(T)), (fields.Jx, fields.Jy, fields.Jz))
            for (s, ws) in zip(st.species, st.ws)
                deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, rho_stagger(Val(3)), grid, ws; dt = st.dt)
            end
            for J in (fields.Jx, fields.Jy, fields.Jz)
                zero_x_edges!(window_frame(st), J)
                fold_guards!(J, grid)
            end
            filter_current!(numerics.filter, fields, grid, st.ρ; edges)
            return map(J -> bits(Array(J)), (fields.Jx, fields.Jy, fields.Jz))
        end
        a, b, c = mk(), mk(), mk()
        @test window_shifts(a, 14) > 4
        current!(Esirkepov(), a)
        @test all(J -> any(!iszero, Array(J)), (a.fields.Jx, a.fields.Jy, a.fields.Jz))
        ja = map(J -> bits(Array(J)), (a.fields.Jx, a.fields.Jy, a.fields.Jz))
        @test ja == written_out!(b)
        @test ja != written_out!(c; edges = nothing)          # a filter that wraps along x gives another current
    end
    st = EMState(window_plasma_3d(Float64), EMNumerics(); backend = be)
    grid = st.grid
    g, n = guards(grid), ncells(grid)
    inner = ntuple(d -> (g + 1):(g + n[d]), 3)
    filtered(h; kw...) = (a = copyto!(field(be, Float64, grid), h); binomial!(a, field(be, Float64, grid), grid, 2; kw...); Array(a))
    h = field(CPU(), Float64, grid)
    h[g + 1, g + 3, g + 2] = 1.0
    a, b = filtered(h; edges = st.aux.window), filtered(h)
    @test all(iszero, a[(g + n[1] - 2):(g + n[1]), :, :])
    @test maximum(abs, b[(g + n[1] - 1):(g + n[1]), :, :]) > 0
    @test sum(a[inner...]) < 1 == sum(b[inner...])            # what left through the back edge is gone; the periodic filter keeps the sum
    h = field(CPU(), Float64, grid)
    h[g + n[1] ÷ 2, g + 1, g + n[3]] = 1.0                    # away from the x edges, at the seams of y and z
    a, b = filtered(h; edges = st.aux.window), filtered(h)
    @test same_bits(a[inner...], b[inner...]) && sum(a[inner...]) == 1
    @test a[g + n[1] ÷ 2, g + n[2], g + 1] > 0                # across both seams
end

@testset "T-1252 the 3D step filters the folded Jx, Jy, Jz and the mesh solve takes the filtered current, both modes, and in a moving window the filter keeps the window's edges [A2, A7, A8, WP-093]" begin
    body_T1252(CPU())
    body_T1252_window(CPU())
end

# --- T-1253 ---------------------------------------------------------------------------------

# A4 with the filter and the sort from the last order at N = 3: the Reference runs of
# `features3d_digests` are bitwise across reruns, launch shapes and 1 and 4 threads, and the two
# deposition modes agree to roundoff with both switched on, measured as T-951.
function body_T1253_modes(be)
    C = emC("T-951")
    for T in (Float32, Float64)
        prob = plasma_step_3d(T)
        num(mode) = EMNumerics(; mode, filter = BinomialFilter(2), resort = true)
        sr, sf = EMState(prob, num(Reference()); backend = be), EMState(prob, num(Fast()); backend = be)
        initialize!(sr, prob)
        initialize!(sf, prob)
        worst = 0.0
        scale = sqrt(prod(prob.species[1].ppc))
        for k in 1:prob.nsteps
            step!(sr, params(prob), k)
            step!(sf, params(prob), k)
            for kind in ((:Ex, :Ey, :Ez, :Bx, :By, :Bz), (:Jx, :Jy, :Jz))
                A = map(c -> Array(getfield(sr.fields, c)), kind)
                F = map(c -> Array(getfield(sf.fields, c)), kind)
                m = maximum(a -> maximum(abs, a), A)
                worst = max(worst, maximum(ab -> maximum(abs, ab[1] .- ab[2]), zip(A, F)) / (eps(T) * m * k * scale))
            end
        end
        record!("T-1253 Reference against Fast", T, worst, be)
        @test worst <= C
    end
end

@testset "T-1253 with the filter and the sort from the last order the 3D Reference step is bitwise across reruns, launch shapes and 1 and 4 threads, and Fast agrees with it to roundoff [A4, A7, WP-093]" begin
    ref = features3d_digests()
    @test length(ref) == 2 * length(TYPES)
    @test allunique(last.(ref))
    @test features3d_digests() == ref
    for wgs in (32, 256)
        @test features3d_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        fields = split.(features_worker_lines(nt))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
    body_T1253_modes(CPU())
end

# --- T-1254 ---------------------------------------------------------------------------------

"The node rows between the two absorbers of `L` cells on an open axis of `n` cells, as interior indices (T-451's rule)."
free_node_rows(n, L) = (L + 2):(n - L)

# The filter beside an open axis. The grid under an open axis is periodic, and the deposit folds
# the current of a particle near one edge into the first rows of the other side: at most the
# guard width g of rows, which the absorber covers (cells ≥ g, T-451). Each filter pass carries
# that current one row further, so with cells ≥ g + p nothing of it reaches the rows between the
# layers (`check_filter_fits`), and there the filtered Gauss law of T-441 holds as on a periodic
# grid: a retired particle's charge, which leaves ρ while its field stays, lies within the
# shape's reach of the seam, and its filtered charge within p rows more, inside the absorbers.
# So Qⁿ = D⁻·Eⁿ − Fρⁿ (F the periodic filter of the step) keeps its value on the node rows off
# the absorbers to roundoff, with particles leaving, at the thinnest absorber the state takes
# (cells = g + p): N = 2 with two passes, N = 3 (open along y and z) with one, the layer and the
# PML, both modes. In the absorbers Q changes far above roundoff, which shows the check sees them,
# and the plain residual on the free rows shows the filter acting.
function body_T1254(be)
    C = emC("T-1254")
    for (N, absorbers, passes) in ((2, FILTER_OPEN2, 2), (3, OPEN3_ABSORBERS, 1)), T in TYPES, mode in (Reference(), Fast()),
        (label, ab) in absorbers

        prob = N == 2 ? open_plasma_2d(T; amplitude = seeded(T, 1 / 10))[1] : open_plasma_3d(T; amplitude = seeded(T, 1 / 10), nsteps = 20)
        st = EMState(prob, EMNumerics(; mode, absorber = ab, filter = BinomialFilter(passes)); backend = be)
        @test ab.cells == guards(st.grid) + passes
        initialize!(st, prob)
        n = st.grid.n
        free = CartesianIndices(ntuple(d -> prob.boundaries[d] === :open ? free_node_rows(n[d], ab.cells) : (1:n[d]), N))
        inside = [j in free for j in CartesianIndices(n)]
        gp = gauss_parts(st; sorted = true)
        Q0 = gp.divE .- binomial_host(gp.ρ, passes)
        r0 = gp.divE .- gp.ρ
        worst = worst_layer = effect = 0.0
        n0 = alive_total(st)
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            gp = gauss_parts(st; sorted = true)
            dQ = (gp.divE .- binomial_host(gp.ρ, passes)) .- Q0
            scale = sum(n) * maximum(absmax, gp.ρ)
            worst = max(worst, maximum(absmax, dQ[inside]) / (base_eps(T) * scale * sqrt(k + 1)))
            worst_layer = max(worst_layer, Float64(maximum(absmax, dQ[.!inside]) / (base_eps(T) * scale * sqrt(k + 1))))
            effect = max(effect, Float64(maximum(absmax, ((gp.divE .- gp.ρ) .- r0)[inside]) / maximum(absmax, gp.ρ)))
        end
        left = n0 - alive_total(st)
        @info "T-1254" N label T mode worst worst_layer effect left of = n0
        @test left > n0 ÷ 20                       # particles left through the open edges
        @test worst_layer > 1e4                    # the absorbers are not charge conserving
        @test effect > 1e4 * base_eps(T)           # the filter acts on the free rows
        record!("T-1254 $(N)D $label $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

"""
    edge_crossers(T, N; passes, mode, be) -> (state, problem)

A box open along its last axis (N = 2: 8 × 16 cells of 1/2; N = 3: 6 × 6 × 12 cells of 1 × 1 × 1/2)
behind an absorbing layer of guards + `passes` cells, with `passes` filter passes, initialized, in
which only the electrons of the last cell row of the open axis carry weight, placed at the offset
0.95 along it with u = 10 along it and 1 along the other two: with no field (a neutral, cold,
uniform plasma) one step moves them across the edge by more than half a cell, the farthest a
leaving particle's stencil reaches into the other side.
"""
function edge_crossers(::Type{T}, N; passes, mode, be) where {T}
    cells = N == 2 ? (8, 16) : (6, 6, 12)
    L = N == 2 ? (T(4), T(8)) : (T(6), T(6), T(6))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:N))) / 10
    m = ntuple(d -> d == 1 ? 1 : 0, N)
    ppc = ntuple(_ -> 2, N)
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), m, ppc, zero3(T), zero3(T), 1),
          SpeciesDescription{T}(:positive, one(T), one(T), one(T), zero(T), m, ppc, zero3(T), zero3(T), 2))
    prob = EMProblem("edge-crossers", L, cells, T(dt), 1, sp, 1, m; boundaries = ntuple(d -> d == N ? :open : :periodic, N))
    g = ParticleMeshElectromagnetic.em_guard_width(EMNumerics().shape, EMNumerics().solver)
    st = EMState(prob, EMNumerics(; mode, absorber = AbsorbingLayer(; cells = g + passes, strength = 1), filter = BinomialFilter(passes));
                 backend = be)
    initialize!(st, prob)
    e, positive = st.species
    w = Array(e.w)
    w[Array(e.cell[N]) .!= Int32(cells[N])] .= zero(T)
    copyto!(e.w, w)
    copyto!(positive.w, zero(Array(positive.w)))
    copyto!(e.offset[N], fill(T(19) / 20, capacity(e)))
    for c in 1:3
        copyto!(e.u[c], fill(c == N ? T(10) : one(T), capacity(e)))
    end
    return st, prob
end

# How far the current of a leaving particle reaches into the other side of an open axis, the
# number behind `check_filter_fits`: the grid under the open axis is periodic, so the last move of
# a particle that crosses the edge by more than half a cell is deposited on the node rows 1 to g
# of the other side (g = 3, the quadratic shape's guard width: the stencil about its new place),
# and p filter passes carry it to row g + p and no further. So for the two components at the
# nodes of the open axis row g + p holds current and the rows from g + p + 1 to the near side's
# hold exactly zero: with the thinnest absorber the state takes (g + p cells) all of it lies on
# rows inside the layer, as the deposit's does without a filter (cells ≥ g, T-451); an absorber of
# one cell less would hold its last row on the layer's inner edge (node row g + p, which the free
# rows of T-451 and of `body_T1254` leave out, so their Gauss law would not see it). The
# component along the open axis sits on its faces and ends one row earlier
# (face g + p − 1); behind it the Reference deposit leaves the roundoff sum of the weight changes
# on the stencil's last faces (T-840), at most to face g + p + 1, within the tolerance's `tail`
# times eps of the component, and exact zeros from there on. Every element type and mode, p = 0
# (the rule of T-451) to 2.
function body_T1254_reach(be)
    tail = emC("T-1254", "tail")
    for (N, ps) in ((2, (0, 1, 2)), (3, (0, 1))), passes in ps, T in TYPES, mode in (Reference(), Fast())
        st, prob = edge_crossers(T, N; passes, mode, be)
        step!(st, params(prob), 1)
        g, n = guards(st.grid), ncells(st.grid)[N]
        reach = g + passes
        J = map(c -> Array(getfield(st.fields, c)), (:Jx, :Jy, :Jz))
        rows(A, r) = selectdim(A, N, g .+ r)                       # interior rows r of the open axis
        @test alive_total(st) < sum(capacity, st.species)          # the particles of the last row left
        for c in 1:3
            if c == N                                              # along the open axis, on its faces
                top = Float64(maximum(absmax, J[c]))
                @test maximum(absmax, rows(J[c], reach - 1)) > 1e3 * base_eps(T) * top     # current, not a tail
                worst = Float64(maximum(absmax, rows(J[c], reach:(reach + 1)))) / (base_eps(T) * top)
                record!("T-1254 tail $(N)D", T, worst, be)
                @test worst <= tail
                @test all(iszero, rows(J[c], (reach + 2):(n - passes - 1)))
            else                                                   # at the nodes of the open axis
                @test any(!iszero, rows(J[c], reach))
                @test all(iszero, rows(J[c], (reach + 1):(n - passes - 1)))
                @test any(!iszero, rows(J[c], n - passes))         # the near side: the stencil about the old place, p rows on
            end
        end
    end
end

@testset "T-1254 beside an open axis the filtered Gauss law holds to roundoff off the absorbers with particles leaving, at N = 2 and 3, the layer and the PML, both modes; a leaving particle's current reaches guards + passes rows of the other side and no further [A2, A7, A8, WP-093]" begin
    body_T1254(CPU())
    body_T1254_reach(CPU())
end

# --- T-1255 ---------------------------------------------------------------------------------

"The hash `h` continued over the order of the workspaces `ws`: their permutation and segments."
order_hash(h, ws) = foldl((x, a) -> hash(Array(a), x), Iterators.flatten(map(w -> (w.perm, w.seg_start, w.seg_stop), ws)); init = h)

"""
    resort_digests_3d(T; resort, mode, nsteps, be) -> (digests, used, shifts, nborn, resorts)

`resort_window_3d`'s run (the window's refills behind the count after every shift, the dopant's
births behind the electrons' count, retirements at the back edge) with `EMNumerics(; resort)` and
one filter pass (the filter with the window and the dopant, which the 3D state refused before
WP-093): the digest of everything the state evolves after every step, and of the workspaces'
permutation and segments, as `resort_digests` at N = 2 (T-852).
"""
function resort_digests_3d(::Type{T}; resort, mode = Reference(), nsteps = 110, be = CPU()) where {T}
    prob = resort_window_3d(T; nsteps)
    st = EMState(prob, EMNumerics(; mode, resort, filter = BinomialFilter(1)); backend = be)
    initialize!(st, prob)
    p = params(prob)
    fr = st.aux.ionization
    out = UInt64[]
    nborn = 0
    for k in 1:nsteps
        step!(st, p, k)
        push!(out, order_hash(ion_digest_3d(st), (st.ws..., fr.electrons_ws)))
        nborn = max(nborn, born_3d(st).n)
    end
    used = all(w -> resort ? w.resort !== nothing && w.resort.ordered[] : w.resort === nothing, (st.ws..., fr.electrons_ws))
    resorts = resort ? map(w -> w.resort.resorts[], (st.ws..., fr.electrons_ws)) : nothing
    return out, used, window_shifts(st, nsteps), nborn, resorts
end

"The digest after every step of a periodic or open 3D plasma with `EMNumerics(; resort)`, its workspaces' order included, and the particles left."
function resort_box_digests_3d(prob, absorber; resort, mode = Reference(), be = CPU())
    st = EMState(prob, EMNumerics(; mode, resort, absorber); backend = be)
    initialize!(st, prob)
    p = params(prob)
    out = UInt64[]
    for k in 1:prob.nsteps
        step!(st, p, k)
        push!(out, order_hash(state_digest_3d(st), st.ws))
    end
    return out, alive_total(st), all(w -> resort ? w.resort.ordered[] && w.resort.resorts[] == prob.nsteps : w.resort === nothing, st.ws)
end

# T-852 at N = 3. The sort from the order of the last sort and the full sort give the same order,
# so the same bits of everything the step evolves, after every step: in the periodic plasma whose
# particles cross cells and the seams of every axis, in the same plasma open along y and z, where
# particles are retired at the open faces between the sorts, and in the window with a dopant and
# one filter pass, where each shift appends a column of fresh particles and the births append
# electrons. `runs` are the (element type, mode) pairs of the window's run.
function body_T1255(be; runs = ((Float32, Reference()), (Float64, Reference()), (D3, Reference()), (D3, Fast())))
    for T in TYPES
        prob = plasma_step_3d(T)
        inc, _, used = resort_box_digests_3d(prob, nothing; resort = true, be)
        full, _, _ = resort_box_digests_3d(prob, nothing; resort = false, be)
        @test used && inc == full && allunique(inc)
        op = open_plasma_3d(T; nsteps = 20)
        n0 = prod(op.cells) * sum(d -> prod(d.ppc), op.species)
        inc, left, used = resort_box_digests_3d(op, OPEN3_ABSORBERS[2][2]; resort = true, be)
        full, _, _ = resort_box_digests_3d(op, OPEN3_ABSORBERS[2][2]; resort = false, be)
        @test used && inc == full && left < n0 - n0 ÷ 20         # with particles retired at the open faces
    end
    for (T, mode) in runs
        inc, used_inc, shifts, nborn, resorts = resort_digests_3d(T; resort = true, mode, be)
        full, used_full, _, _, _ = resort_digests_3d(T; resort = false, mode, be)
        @info "T-1255" backend = backend_label(be) T mode shifts nborn resorts
        @test used_inc && used_full
        @test all(>=(110), resorts)              # every step's sort, and the shifts' where the step sorts the species
        @test shifts > 20 && nborn > 0           # the run refills and gives births
        @test inc == full
    end
end

@testset "T-1255 the 3D step sorting from the last order gives the bits of the full sort at every step, in the periodic box, with retirements at open faces, and with window refills and ionization births [A4, WP-093]" begin
    body_T1255(CPU())
end

# --- T-1256 ---------------------------------------------------------------------------------

"""
The status, the count and the bits of the alive slots of every species of a state with a dopant
(the plasma, the atoms and the born electrons), and the fields' bits: what a state initialized
again repeats, as T-855's record (its free slots keep what the run before left).
"""
function alive_record_3d(st)
    fr = st.aux.ionization
    rec(s) = (alive = Array(s.status) .== ParticleMeshCore.ALIVE;
              (Array(s.status), Array(s.count), map(a -> bits(Array(a)[alive]), (s.cell..., s.offset..., s.u..., s.w))...))
    return (map(rec, (st.species..., fr.atoms, fr.electrons)), map(a -> bits(Array(a)), values(st.fields)))
end

# T-854 and T-855 at N = 3: the order of the last sort belongs to the workspaces, not to the
# evolved state, so a restart into a state that ran on, into a new state, and a state over copies
# of saved arrays each sort anew and continue the uninterrupted run's bits, and a used state
# initialized again repeats the run (its alive particles and fields).
@testset "T-1256 a restart into a used 3D state, a used state initialized again, or a state over saved arrays, sorts anew and continues the run's bits [A4, WP-093]" begin
    nsteps, k0 = 110, 70
    prob = resort_window_3d(Float32; nsteps)
    p = params(prob)
    new_state() = (st = EMState(prob, EMNumerics(; mode = Reference(), resort = true); backend = CPU()); initialize!(st, prob); st)
    digest(st) = foldl((h, a) -> hash(bits(a), h), evolved(st); init = UInt64(0))
    ref = new_state()
    local snap
    nborn = 0
    for k in 1:nsteps
        step!(ref, p, k)
        k == k0 && (snap = snapshot(ref, k))
        k > k0 && (nborn = max(nborn, born_3d(ref).n))
    end
    @test window_shifts(ref, nsteps) - window_shifts(ref, k0) > 5 && nborn > 0
    want = digest(ref)
    want_alive = alive_record_3d(ref)
    for st in (ref, new_state())
        restore!(st, snap)
        foreach(k -> step!(st, p, k), (k0 + 1):nsteps)
        @test digest(st) == want
    end
    used = new_state()
    foreach(k -> step!(used, p, k), 1:nsteps)
    view = state_view(used, map(copy, snap.arrays))
    foreach(k -> step!(view, p, k), (k0 + 1):nsteps)
    @test digest(view) == want
    initialize!(used, prob)                                   # its workspaces hold the order of step 110's particles
    foreach(k -> step!(used, p, k), 1:nsteps)
    @test alive_record_3d(used) == want_alive
end

# --- T-1257 ---------------------------------------------------------------------------------

# What a state of each N takes. At N = 3: the filter and the sort from the last order. Beside an
# open axis at N = 2 and 3: the filter with an absorber of at least g + p cells, and no thinner
# one. At N = 1: a step without current (prescribed fields); its Ex then keeps the bits of the
# initial field, since no current changes it and the curl has no x component, while the particles
# move in it, and with the current on the same field changes; its kinetic energy density is the
# deposit of the particles where they are. One constructor serves N = 2 and 3,
# so a refusal names the state's N. What stays refused says why: a profile without a window (the
# static box is the window that never starts).
@testset "T-1257 what a state of each N takes: the filter and the resort at N = 3, the filter beside an open axis, the current-free step at N = 1 [A2, A10, WP-093]" begin
    T = Float64
    p3 = plasma_step_3d(T)
    st = EMState(p3, EMNumerics(; filter = BinomialFilter(1), resort = true))
    @test st isa EMState{3,T} && all(w -> w.resort !== nothing, st.ws)
    @test all(w -> w.resort === nothing, EMState(p3, EMNumerics()).ws)
    # the absorber of an open axis covers the rows the deposit and the filter reach across the edge
    for (prob, absorbers) in ((open_plasma_2d(T)[1], FILTER_OPEN2), (open_plasma_3d(T), OPEN3_ABSORBERS))
        N = ndims(prob)
        g = ParticleMeshElectromagnetic.em_guard_width(Quadratic)
        for (_, ab) in absorbers
            fits = ab.cells - g
            @test fits >= 1
            @test EMState(prob, EMNumerics(; absorber = ab, filter = BinomialFilter(fits))) isa EMState{N}
            @test_throws "needs at least $(g + fits + 1) cells" EMState(prob, EMNumerics(; absorber = ab, filter = BinomialFilter(fits + 1)))
            @test EMState(prob, EMNumerics(; absorber = ab)) isa EMState{N}              # no passes: any absorber the axis takes
        end
    end
    grid = PeriodicGrid((8, 12), (1 // 2, 1 // 2), 3)
    @test check_filter_fits(BinomialFilter(2), AbsorbingLayer(; cells = 5, strength = 1), grid) === nothing
    @test check_filter_fits(BinomialFilter(0), AbsorbingLayer(; cells = 3, strength = 1), grid) === nothing
    @test_throws ArgumentError check_filter_fits(BinomialFilter(3), AbsorbingLayer(; cells = 5, strength = 1), grid)
    @test_throws ArgumentError check_filter_fits(BinomialFilter(3), PML(; cells = 5), grid)
    # a refusal names the state's N
    @test_throws "a 2D state takes current" EMState(plasma_step_2d(T), EMNumerics(; current = nothing))
    @test_throws "a 3D state takes current" EMState(p3, EMNumerics(; current = nothing))
    @test_throws "axis 2 of the problem is open" EMState(open_plasma_2d(T)[1], EMNumerics())
    @test_throws "axes 2 and 3 of the problem are open" EMState(open_plasma_3d(T), EMNumerics())
    # N = 1 without current: Ex keeps its bits, the particles move in it; with the current it changes
    p1 = beams_1d(T)
    @test_throws ArgumentError EMState(p1, EMNumerics(; current = nothing))
    runs = map((NoCurrent(), Esirkepov())) do current
        s1 = EMState(p1, EMNumerics(; current))
        initialize!(s1, p1)
        E0, u0 = copy(s1.fields.Ex), copy(s1.species[1].u[1])
        run!(s1, params(p1), 10)
        (; s1, E0, u0)
    end
    free, driven = runs
    @test maximum(abs, free.E0) > 0
    @test same_bits(free.s1.fields.Ex, free.E0) && !same_bits(driven.s1.fields.Ex, driven.E0)
    @test all(c -> all_plus_zero(getfield(free.s1.fields, c)), (:Jx, :Jy, :Jz, :Ey, :Ez, :By, :Bz))
    @test maximum(abs, sort(Array(free.s1.species[1].u[1])) .- sort(Array(free.u0))) > 1e-4   # as sets: the step's sort permutes the slots
    @test same_bits(free.E0, driven.E0) && same_bits(free.u0, driven.u0)                 # the same start
    # the kinetic energy at N = 1 is a deposition from the segments of the step's sort, and a step
    # without current sorts all the same: K as the step leaves it has the bits of a sort and a new
    # deposit (`refresh_caches!`, the identity permutation), the two modes agree on it to the
    # roundoff of their sums (the momenta are the same bits: the pushes read a field nothing
    # changes); the particles have changed cells, so the order of `initialize!` is stale
    kin = map((Reference(), Fast())) do mode
        s1 = EMState(p1, EMNumerics(; current = NoCurrent(), mode))
        initialize!(s1, p1)
        cells() = map(s -> copy(Array(s.cell[1])), s1.species)
        folded() = (a = copy(s1.K); fold_guards!(a, s1.grid); collect(interior_view(Array(a), s1.grid)))
        start = cells()
        run!(s1, params(p1), 30)
        ParticleMeshElectromagnetic.kinetic!(s1)
        K, order = folded(), cells()
        ParticleMeshElectromagnetic.refresh_caches!(s1)
        (; K, again = folded(), same_order = order == cells(), moved = order != start)
    end
    @test same_bits(kin[1].K, kin[1].again) && kin[1].same_order && kin[1].moved
    worstK = maximum(abs, kin[1].K .- kin[2].K) / (eps(T) * maximum(abs, kin[1].K))
    record!("T-1257 K", T, worstK)
    @test worstK <= emC("T-1257")
    @test sum(kin[1].K) ≈ sum(kin[2].K) rtol = 64 * eps(T)
    # a profile without a window stays refused, and the message says how to get the static box
    wp = window_plasma_3d(T)
    @test_throws "MovingWindow(; start = Inf)" EMProblem("p", wp.length, wp.cells, wp.dt, 2, wp.species, 1, wp.kmode; profile = wp.profile)
    static = EMProblem("p", wp.length, wp.cells, wp.dt, 2, wp.species, 1, wp.kmode; boundaries = wp.boundaries,
                       window = MovingWindow(; start = T(Inf)), profile = wp.profile)
    ss = EMState(static, EMNumerics())
    initialize!(ss, static)
    run!(ss, params(static), 2)
    @test window_shifts(ss, 2) == 0
end

# --- T-1258 ---------------------------------------------------------------------------------

"Bytes allocated by the third step of the 3D plasma with `passes` filter passes and the sort from the last order, and the step's sorts."
function step_bytes_features_3d(mode; ppc, cells, passes = 2)
    prob = plasma_step_3d(Float64; ppc, cells, nsteps = 4)
    st = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes), resort = true))
    initialize!(st, prob)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    bytes = @allocated step!(st, p, 3)
    return bytes, count(s -> ParticleMeshElectromagnetic.sorts(st, s), st.species)
end

@testset "T-1258 allocation per 3D step with the filter and the sort from the last order does not grow with particle count, grid size or passes [A7, WP-093]" begin
    for mode in (Reference(), Fast())
        small, sorts = step_bytes_features_3d(mode; ppc = (1, 1, 2), cells = (8, 7, 6))
        more_particles, _ = step_bytes_features_3d(mode; ppc = (3, 3, 2), cells = (8, 7, 6))
        larger_grid, _ = step_bytes_features_3d(mode; ppc = (1, 1, 2), cells = (16, 14, 12))
        more_passes, _ = step_bytes_features_3d(mode; ppc = (1, 1, 2), cells = (8, 7, 6), passes = 5)
        @info "T-1258 bytes per 3D step with the filter and the resort" mode small more_particles larger_grid more_passes sorts threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-956); CI runs one thread
        # each sort may add Base's radix count buffer, bounded whatever the size (`SORT_COUNTS_BYTES`, as T-956)
        bound = small + sorts * SORT_COUNTS_BYTES
        if Threads.nthreads() == 1 || mode isa Fast
            @test_alloc more_particles <= bound
            @test_alloc larger_grid <= bound
            @test_alloc more_passes <= bound       # the passes reuse the scratch and launch with fixed arguments
        end
    end
end
