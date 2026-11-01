# The engine contracts of WP-011: T-171 to T-177.

"The output lines of `threads_worker.jl` run with `nt` threads (cached: one subprocess per thread count)."
const WORKER_OUTPUT = Dict{Int,Vector{String}}()
function worker_lines(nt)
    get!(WORKER_OUTPUT, nt) do
        worker = joinpath(@__DIR__, "threads_worker.jl")
        project = dirname(Base.active_project())
        out = read(`$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$project $worker`, String)
        split(out, '\n'; keepempty = false)
    end
end
worker_digests(nt) = [chopprefix(l, "DIGEST ") for l in worker_lines(nt) if startswith(l, "DIGEST ")]
"The digest lines of one kind (`scan`, `sort`, `sortmatch`, `deposit`)."
of_kind(lines, kind) = filter(l -> startswith(l, kind * "|"), lines)
"Lines grouped by everything except the launch shape (the fourth field)."
function by_case(lines)
    groups = Dict{String,Vector{String}}()
    for l in lines
        f = split(l, '|')
        push!(get!(groups, join(f[[1, 2, 3]], '|'), String[]), f[5])
    end
    return groups
end

const INPROCESS_DIGESTS = thread_digests()

@testset "T-171 the engine holds no physics [A9, WP-011]" begin
    @test !isdefined(ParticleMeshCore, :Electrostatic)
    for name in names(ParticleMeshCore)
        s = lowercase(string(name))
        @test !any(w -> occursin(w, s), ("gauss", "poisson", "electrostatic"))
    end
end

@testset "T-172 cumsum_periodic! identities [A7, A8, WP-011]" begin
    C = tolC("T-172")
    tile = ParticleMeshCore.SCAN_TILE
    @test SCAN_GRIDS[1] <= tile && SCAN_GRIDS[2] > 2tile          # one tile; three tiles
    for T in TYPES, n in SCAN_GRIDS
        ε = base_eps(T)
        out, a, grid = scan_output(T, n)
        g, dx = guards(grid), cellsize(grid)
        ai, oi = a[(g + 1):(g + n)], out[(g + 1):(g + n)]
        amax = maximum(absmax, ai)
        ā = sum(ai) / n
        mean_err = absmax(sum(oi) / n) / (ε * n * amax * dx)
        diff_err = maximum(absdiff((oi[j] - oi[mod1(j - 1, n)]) / dx, ai[j] - ā) for j in 1:n) / (ε * n * amax)
        record!("T-172", T, max(mean_err, diff_err))
        @test mean_err <= C
        @test diff_err <= C
        T === D3 && @test any(x -> !iszero(partials(x)), oi)               # the partials are exercised
        @test same_bits(out[1:g], out[(n + 1):(n + g)])                    # guards filled
        @test same_bits(out[(n + g + 1):end], out[(g + 1):(2g)])
        @test same_bits(oi, serial_tiled_cumsum(ai, dx, tile))             # the documented order
        @test same_bits(out, scan_output(T, n)[1])                         # rerun
    end
    # Shapes are checked on the host before any launch.
    let T = Float64, bad = (PeriodicGrid(0, 1.0, 2), PeriodicGrid(4, 1.0, 0))
        for grid in bad
            v = field(backend, T, PeriodicGrid(4, 1.0, 2))
            @test_throws ArgumentError cumsum_periodic!(similar(v), v, grid)
        end
    end
    # Thread counts: the subprocess digests at 1 and 4 threads equal the in-process ones.
    for nt in (1, 4)
        @test of_kind(worker_digests(nt), "scan") == of_kind(INPROCESS_DIGESTS, "scan")
    end
end

@testset "T-173 extras follow the sort [A4, WP-011]" begin
    for T in TYPES
        s, ws, grid = sorted_extras(T)
        @test keys(s.extras) == (:old_slot, :old_offset)
        @test eltype(s.extras.old_slot) === Int32 && eltype(s.extras.old_offset) === T
        @test length(s.extras.old_slot) == capacity(s) == length(s.extras.old_offset)
        @test alive_count(s) == 500 - length(DEAD_SLOTS)
        @test extras_match(s, ws)
        # The two extras reuse the core permutation buffers: no extra scratch.
        bare = test_species(T, grid, 500; seed = 13)
        @test ParticleMeshCore.scratch_bytes(ws) == ParticleMeshCore.scratch_bytes(Workspace(bare, grid))
    end
    # An extra whose element type no core array has gets a buffer of its own, and follows too.
    grid = PeriodicGrid(5, 1.0f0, 2)
    s = test_species(Float32, grid, 40; seed = 3, extras = (tag = UInt16, wide = Float64))
    for i in 1:capacity(s)
        s.extras.tag[i] = UInt16(i)
        s.extras.wide[i] = Float64(s.offset[1][i])
    end
    ws = Workspace(s, grid)
    sort_by_cell!(s, grid, ws)
    @test all(i -> s.extras.tag[i] == ws.perm[i], 1:capacity(s))
    @test all(i -> s.extras.wide[i] === Float64(s.offset[1][i]), 1:capacity(s))
    @test ParticleMeshCore.scratch_bytes(ws) ==
          ParticleMeshCore.scratch_bytes(Workspace(test_species(Float32, grid, 40), grid)) + capacity(s) * (2 + 8)
    for nt in (1, 4)
        d = worker_digests(nt)
        @test of_kind(d, "sort") == of_kind(INPROCESS_DIGESTS, "sort")
        @test all(l -> endswith(l, "|true"), of_kind(d, "sortmatch"))
    end
end

@testset "T-174 deposit! quantity [A4, A7, A8, WP-011]" begin
    C = tolC("T-174")
    Cnode = tolC("T-135")
    # The default quantity is the spike's q·w deposition, bitwise: Reference (and Fast for Dual,
    # which falls back to it) in process; Fast with atomics on one thread, in the subprocess.
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast())
        effective_mode(mode, T) isa Reference || continue
        @test same_bits(deposit_case(T, mode, shape, stagger), deposit_case(T, mode, shape, stagger; spike = true))
    end
    fast = filter(startswith("FASTSPIKE|"), worker_lines(1))
    @test length(fast) == 2 * length(SHAPES) * length(STAGGERS)
    @test all(endswith("|true"), fast)
    # q·w·u_y: the deposited total is Σ q·w·u_y.
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast()), np in (100, 2000)
        ε = base_eps(T)
        grid = PeriodicGrid(10, one(basetype(T)), guard_width(shape))
        s = quantity_species(T, grid, np; seed = 7)
        ws = Workspace(s, grid)
        effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
        ρ = fold_guards!(deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws; quantity = ChargeUy()), grid)
        expect = sum(ChargeUy()(s, i) for i in 1:np)
        scale = np * maximum(i -> absmax(ChargeUy()(s, i)), 1:np)
        r = record!("T-174", T, absdiff(sum(ρ), expect) / (ε * scale))
        @test r <= C
    end
    # Reference and Fast agree per node within T-135's bound, C·eps(T)·Σ|contributions|.
    for T in (Float32, Float64), shape in SHAPES, stagger in STAGGERS
        grid = PeriodicGrid(16, one(T), guard_width(shape))
        dep(mode, quantity) = begin
            s = quantity_species(T, grid, 16 * 400; seed = 11)
            ws = Workspace(s, grid)
            mode isa Reference && sort_by_cell!(s, grid, ws)
            deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws; quantity)
        end
        ρF, ρR, ρabs = dep(Fast(), ChargeUy()), dep(Reference(), ChargeUy()), dep(Reference(), AbsChargeUy())
        worst = maximum(abs(ρF[a] - ρR[a]) / (eps(T) * ρabs[a]) for a in eachindex(ρR) if ρabs[a] != 0)
        record!("T-174 nodes", T, worst)
        @test worst <= Cnode
    end
end

@testset "T-175 deposit! with a quantity is the adjoint of gather! [A3, A7, A8, WP-011]" begin
    # Σᵢ quantity(s, i)·gather(fill f)ᵢ = ⟨fold(deposit(quantity)), f⟩ over the interior.
    C = tolC("T-175")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in (Reference(), Fast())
        ε = base_eps(T)
        grid = PeriodicGrid(12, basetype(T)(1 // 2), guard_width(shape))
        s = quantity_species(T, grid, 300; seed = 5)
        f = test_field(T, grid, j -> seeded(T, sin(0.7j) + 0.3))
        out = similar(s.w)
        gather!(out, f, s, shape, stagger, grid)
        lhs = sum(ChargeUy()(s, i) * out[i] for i in 1:300)
        ws = Workspace(s, grid)
        effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
        ρ = fold_guards!(deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws; quantity = ChargeUy()), grid)
        rhs = sum(ρ[guards(grid) + j] * f[guards(grid) + j] for j in 1:ncells(grid))
        scale = sum(absmax(ChargeUy()(s, i)) for i in 1:300) * maximum(absmax, f)
        r = record!("T-175", T, absdiff(lhs, rhs) / (ε * scale))
        @test r <= C
    end
    # Dual weights: the partials of the deposited q·w·u_y are the depositions of q·∂w·u_y,
    # computed independently in Float64 at the same positions.
    Cp = tolC("T-175", "C_partials")
    for shape in SHAPES, stagger in STAGGERS
        grid = PeriodicGrid(12, 0.5f0, guard_width(shape))
        sD = quantity_species(D3, grid, 300; seed = 5)
        for i in 1:300
            w = value(sD.w[i])
            sD.w[i] = mkdual(w, cos(0.3i), w^2, 0)
            sD.offset[1][i] = mkdual(value(sD.offset[1][i]), 0, 0, 0)
            sD.u[2][i] = mkdual(value(sD.u[2][i]), 0, 0, 0)
        end
        ρD = deposited_with(Reference(), sD, shape, stagger, grid, ChargeUy())
        g64 = PeriodicGrid(12, 0.5, guard_width(shape))
        for dir in 1:3
            s64 = Species(backend, Float64, capacity(sD); q = -1, m = 1)
            for i in 1:300
                s64.cell[1][i] = sD.cell[1][i]
                s64.offset[1][i] = value(sD.offset[1][i])
                s64.w[i] = partials(sD.w[i])[dir]
                s64.u[2][i] = value(sD.u[2][i])
                s64.status[i] = ParticleMeshCore.ALIVE
            end
            fill!(s64.count, Int32(300))
            exact = deposited_with(Reference(), s64, shape, stagger, g64, ChargeUy())
            ∂ = [partials(x)[dir] for x in ρD]
            if dir == 3
                @test all(iszero, ∂)
            else
                r = record!("T-175 partials", D3, maximum(abs.(∂ .- exact)) / (eps(Float32) * maximum(abs, exact)))
                @test r <= Cp
            end
        end
    end
end

@testset "T-176 fd_scan reports the A7 floor and runs in Float64 only [A7, WP-011]" begin
    hs = 10.0 .^ (-1:-0.5:-9)
    seen = Set{Any}()
    floors = Dict{String,Float64}()
    # Two analytic functions with exact derivatives, and the first with a deterministic
    # perturbation of known size η that raises the cancellation error to about η/h.
    η = 1e-10
    bump(x) = η * (isodd(reinterpret(UInt64, x) >> 7) ? 1 : -1)
    cases = (("sin", x -> sin(x), cos, 1.0), ("exp/3", x -> exp(x) / 3, x -> exp(x) / 3, 0.7),
             ("sin + η·noise", x -> sin(x) + bump(x), cos, 1.0))
    for (label, f, df, x0) in cases
        logged(x, mode, seed) = (push!(seen, (typeof(mode), seed)); f(x))
        ad = df(x0)
        r = fd_scan(logged, x0, hs; seed = 42, ad)
        @test r.fd == [(f(x0 + h) - f(x0 - h)) / 2h for h in hs]
        spread = [maximum(r.fd[(k - 1):(k + 1)]) - minimum(r.fd[(k - 1):(k + 1)]) for k in 2:(length(hs) - 1)]
        @test r.spread == spread
        @test r.istar == argmin(spread) + 1
        @test r.floor_rel == spread[r.istar - 1] / abs(ad)
        # The exact derivative within C times the floor. For a smooth function the floor at its
        # minimum is about the error itself, so the measured ratio sits near 1 (through the module's step, T-219: 0.176).
        @test r.rel[r.istar] <= tolC("T-176") * r.floor_rel
        MEASURED["T-176 rel/floor"] = merge(get(MEASURED, "T-176 rel/floor", Dict{String,Float64}()), Dict(label => r.rel[r.istar] / r.floor_rel))
        floors[label] = r.floor_rel
    end
    @test seen == Set([(Reference, 42)])                  # Reference mode, one seed, every evaluation
    @test floors["sin"] < 1e-8 && floors["exp/3"] < 1e-8  # Float64 roundoff and truncation balance far below
    @test floors["sin + η·noise"] > 10 * floors["sin"]    # the floor follows the injected rounding
    @info "T-176 floors" floors
    gsin(x, mode, seed) = sin(x)
    @test_throws ArgumentError fd_scan(gsin, 1.0f0, hs; seed = 1, ad = cos(1.0))
    @test_throws ArgumentError fd_scan((x, m, s) -> Float32(sin(x)), 1.0, hs; seed = 1, ad = cos(1.0))
    @test_throws ArgumentError fd_scan(gsin, 1.0, Float32.(hs); seed = 1, ad = cos(1.0))
    @test_throws ArgumentError fd_scan(gsin, 1.0, hs[1:4]; seed = 1, ad = cos(1.0))
end

@testset "T-177 the launch keyword [A4, WP-011]" begin
    # Every engine launch accepts `workgroupsize`.
    for T in TYPES, wg in WG_SIZES
        grid = PeriodicGrid(12, basetype(T)(1 // 2), 2)
        s = Species(backend, T, 12 * 8; q = -1, m = 1)
        load_quiet!(s, grid, (x, p) -> p, one(T), 8; seed = 1, stream = 1, workgroupsize = wg)
        @test alive_count(s) == 96
        ws = Workspace(s, grid)
        sort_by_cell!(s, grid, ws; workgroupsize = wg)
        a = test_field(T, grid, j -> seeded(T, cos(j)))
        @test fill_guards!(copy(a), grid; workgroupsize = wg) == a
        @test fold_guards!(copy(a), grid; workgroupsize = wg) isa typeof(a)
        out = similar(s.w)
        @test gather!(out, a, s, Quadratic, Center(), grid; workgroupsize = wg) === out
        for mode in (Reference(), Fast())
            @test deposit!(mode, field(backend, T, grid), s, Quadratic, Center(), grid, ws; workgroupsize = wg) isa typeof(a)
        end
        @test cumsum_periodic!(field(backend, T, grid), a, grid; workgroupsize = wg) isa typeof(a)
    end
    # The scan is one serial work item (it ignores `workgroupsize`), so only the guard fill's bits
    # depend on the launch shape; the digests below cover both together.
    # Reference mode: the same bits for the default and two explicit sizes, per element type...
    for (case, digests) in by_case(filter(l -> !startswith(l, "sortmatch"), INPROCESS_DIGESTS))
        @test length(digests) == length(WG_SIZES)
        @test allequal(digests)
    end
    # ...and at 1 and 4 threads.
    for nt in (1, 4)
        @test worker_lines(nt)[1] == "THREADS $nt"
        @test worker_digests(nt) == INPROCESS_DIGESTS
    end
end
