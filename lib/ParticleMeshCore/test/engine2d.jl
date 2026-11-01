# WP-018, the engine at N = 2 on the CPU backend: T-260 to T-269.

@testset "T-260 the port keeps N = 1 bit for bit [A0, A4, WP-018]" begin
    # The N = 1 engine digests recorded before the port (n1_digests.jl), in process and in the
    # subprocesses at 1 and 4 threads. Atomic Fast() bits are an A4 claim on one thread only.
    nonatomic(lines) = filter(l -> !startswith(l, "depositfast|"), lines)
    inprocess = n1_lines()
    @test length(N1_RECORDED) == 81
    @test nonatomic(inprocess) == nonatomic(N1_RECORDED)
    Threads.nthreads() == 1 && @test inprocess == N1_RECORDED
    n1_worker(nt) = [chopprefix(l, "N1 ") for l in worker_lines(nt) if startswith(l, "N1 ")]
    @test n1_worker(1) == N1_RECORDED
    @test nonatomic(n1_worker(4)) == nonatomic(N1_RECORDED)
    # T-136's subprocess digests (one Float32 step, the whole T-141 run) at 1 and 4 threads;
    # `Base.hash` is fixed within one Julia version only, as in T-170.
    results = T136_RESULTS[]
    if results === nothing
        @info "T-260: T-136 did not run first; its recorded digests are not rechecked here"
    else
        for r in results
            @test r.one == T136_RECORDED.one skip = VERSION >= v"1.14-"
            @test r.full == T136_RECORDED.full skip = VERSION >= v"1.14-"
        end
    end
end

# The 2D tests run on the non-square grids of engine2d_cases.jl: 16 × 12 cells of 1/4 × 3/8,
# 2 × 3 particles per cell when loading, a few hundred hand-placed particles otherwise.

const INPROCESS_2D = engine2d_digests()
e2d_worker(nt) = [chopprefix(l, "E2D ") for l in worker_lines(nt) if startswith(l, "E2D ")]

@testset "T-261 the 2D grid, linearization and advance [A7, A8, WP-018]" begin
    for T in TYPES
        grid = grid2(T, 2)
        @test grid isa PeriodicGrid{2,basetype(T)}
        @test ncells(grid) == (16, 12) && cellsize(grid) == dx2(T) && guards(grid) == 2
        a = field(backend, T, grid)
        @test size(a) == (20, 16) && eltype(a) === T && all(iszero, a)
        @test interior(grid) == (3:18, 3:14)
        v = interior_view(a, grid)
        @test size(v) == (16, 12) && parentindices(v) == (3:18, 3:14)
    end
    # the per-axis check n_d ≥ 2g, before any allocation
    @test_throws ArgumentError PeriodicGrid((16, 3), (1.0, 1.0), 2)
    @test_throws ArgumentError PeriodicGrid((3, 12), (1.0, 1.0), 2)
    @test PeriodicGrid((4, 4), (1.0, 1.0), 2) isa PeriodicGrid{2,Float64}
    # column-major linear cell index (axis 1 fastest) and its inverse
    n = Int32.(N2)
    Ls = Int32[]
    for c2 in Int32(1):n[2], c1 in Int32(1):n[1]
        L = ParticleMeshCore.linear_cell((c1, c2), n)
        push!(Ls, L)
        @test L == c1 + (c2 - 1) * n[1]
        @test ParticleMeshCore.cartesian_cell(L, n) === (c1, c2)
    end
    @test Ls == 1:prod(n)
    # Free streaming at dyadic displacements is exact per axis, through each seam and diagonally
    # through the corner; offsets stay in [0, 1).
    for T in TYPES
        starts = (((16, 12), (15 // 16, 15 // 16)), ((1, 1), (1 // 16, 0 // 1)), ((16, 5), (1 // 2, 3 // 8)), ((7, 12), (0 // 1, 13 // 16)))
        disps = ((1 // 8, 1 // 8), (-3 // 16, -1 // 4), (1 // 4, 0 // 1), (0 // 1, 5 // 16))
        steps = 300
        for ((c0, δ0), Δr) in zip(starts, disps)
            c = Int32.(c0)
            δ = (seeded(T, δ0[1]), seeded(T, δ0[2]))
            Δ = (seeded(T, Δr[1]), seeded(T, Δr[2]))
            inside = true
            for _ in 1:steps
                c, δ = advance(c, δ, Δ, n)
                inside &= all(x -> 0 <= value_of(x) < 1, δ)
            end
            @test inside
            for d in 1:2
                exact = (c0[d] - 1) + δ0[d] + steps * Δr[d]
                exact -= n[d] * floor(exact / n[d])
                @test (c[d] - 1) + value_of(δ[d]) == exact
            end
        end
        # a tiny negative step from offset 0: `δ + Δ - floor(δ + Δ)` rounds up to one, so the offset
        # becomes 0 in the same cell (contract 3), never 1; one cell back through the seam is exact
        c, δ = advance((Int32(1), Int32(1)), (zero(T), zero(T)), (-T(eps(basetype(T))) / 64, zero(T)), n)
        @test c == (Int32(1), Int32(1)) && value_of(δ[1]) == 0 && value_of(δ[2]) == 0
        c, δ = advance((Int32(1), Int32(1)), (zero(T), zero(T)), (-one(T) / 16, -one(T) / 16), n)
        @test c == n && value_of(δ[1]) == 15 // 16 && value_of(δ[2]) == 15 // 16
    end
    # A Dual offset carries the partials of the continuous trajectory; cells stay integers.
    c, δ = (Int32(16), Int32(12)), (mkdual(15 / 16, 1, 0, 0), mkdual(1 / 2, 0, 1, 0))
    Δ = (mkdual(1 / 8, 0, 0, 1), mkdual(-1 / 4, 0, 0, 1 / 2))
    for _ in 1:100
        c, δ = advance(c, δ, Δ, n)
    end
    @test c isa NTuple{2,Int32}
    @test Tuple(partials(δ[1])) == (1.0f0, 0.0f0, 100.0f0)
    @test Tuple(partials(δ[2])) == (0.0f0, 1.0f0, 50.0f0)
end

@testset "T-262 2D shape weights [A2, A7, A8, WP-018]" begin
    C = tolC("T-262")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS
        ε = base_eps(T)
        worst = 0.0
        offsets = (0.0, 0.25, 0.5, 0.7, Float64(prevfloat(one(basetype(T)))), (k / 13 for k in 0:12)...)
        for a in offsets, b in offsets
            δ = (seeded(T, a), seeded(T, b))
            f, W = ParticleMeshCore.tensor_weights(shape, st, δ)
            worst = max(worst, absdiff(sum(W), one(T)) / ε)
            (f1, w1), (f2, w2) = shape_weights(shape, st[1], δ[1]), shape_weights(shape, st[2], δ[2])
            P1 = order(shape) + 1
            @test f === (f1, f2)
            @test same_bits(collect(W), [w1[k1] * w2[k2] for k2 in 1:P1 for k1 in 1:P1])   # the outer product, bitwise
            @test shape_weights(shape, st, δ) === ((f1, w1), (f2, w2))
        end
        record!("T-262", T, worst)
        @test worst <= C
        @test ParticleMeshCore.stencil_reach(shape, st) ==
              (ParticleMeshCore.stencil_reach(shape, st[1]), ParticleMeshCore.stencil_reach(shape, st[2]))
    end
    for T in (Float32, D3), shape in SHAPES, st in STAGGER_PAIRS
        δ = (seeded(T, 0.3), seeded(T, 0.8))
        f, W = @inferred ParticleMeshCore.tensor_weights(shape, st, δ)
        @test eltype(W) === T && f isa NTuple{2,Int32}
        r = @inferred shape_weights(shape, st, δ)
        @test eltype(r[1][2]) === T && eltype(r[2][2]) === T
    end
end

@testset "T-263 2D fill and fold [A7, A8, WP-018]" begin
    C = tolC("T-263")
    for T in TYPES, g in (1, 2), n in ((4, 9), (9, 5))
        ε = base_eps(T)
        grid = PeriodicGrid(n, dx2(T), g)
        x, y = field(backend, T, grid), field(backend, T, grid)
        for i in eachindex(x)
            x[i] = seeded(T, cos(1.3i))
            y[i] = seeded(T, sin(2.1i) + 0.1)
        end
        fx = fill_guards!(copy(x), grid)
        # every entry, guards and corners included, is a bitwise copy of its periodic interior image
        image = [x[mod1(a1 - g, n[1]) + g, mod1(a2 - g, n[2]) + g] for a1 in axes(x, 1), a2 in axes(x, 2)]
        @test same_bits(fx, image)
        fy = fold_guards!(copy(y), grid)
        r = record!("T-263", T, absdiff(sum(fx .* y), sum(x .* fy)) / (ε * sum(absmax.(x)) * maximum(absmax, y)))
        @test r <= C
        inner = falses(size(y))
        inner[(g + 1):(g + n[1]), (g + 1):(g + n[2])] .= true
        @test all(iszero, fy[.!inner])
    end
end

@testset "T-264 2D gather [A7, A8, WP-018]" begin
    # The whole array, guards included, holds a + b·x + c·y + d·x·y at the node positions, so every
    # particle is away from the seam for the polynomial (as T-131 does in 1D).
    C = tolC("T-264")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS
        ε = base_eps(T)
        B = basetype(T)
        grid = grid2(T, guard_width(shape))
        s = test_species_2d(T, grid, 300; seed = 3)
        g = guards(grid)
        h = cellsize(grid)
        L = ncells(grid) .* h
        sv = (ParticleMeshCore.stagger_offset(st[1], B), ParticleMeshCore.stagger_offset(st[2], B))
        xnode(idx, d) = (idx - g - 1 + sv[d]) * h[d]
        worst = 0.0
        for (a, b, c, d) in ((B(3 // 2), zero(B), zero(B), zero(B)), (B(3 // 2), B(-5 // 4), B(7 // 8), B(1 // 2)))
            f = field(backend, T, grid)
            for i2 in axes(f, 2), i1 in axes(f, 1)
                x, y = xnode(i1, 1), xnode(i2, 2)
                f[i1, i2] = a + b * x + c * y + d * x * y
            end
            out = similar(s.w)
            gather!(out, f, s, shape, st, grid)
            scale = abs(a) + abs(b) * L[1] + abs(c) * L[2] + abs(d) * L[1] * L[2]
            for i in 1:300
                x = (s.cell[1][i] - 1 + s.offset[1][i]) * h[1]
                y = (s.cell[2][i] - 1 + s.offset[2][i]) * h[2]
                worst = max(worst, absdiff(out[i], a + b * x + c * y + d * x * y) / (ε * scale))
            end
        end
        record!("T-264", T, worst)
        @test worst <= C
    end
    for T in (Float32, D3), shape in SHAPES, st in STAGGER_PAIRS
        grid = grid2(T, guard_width(shape))
        f = test_field_2d(T, grid, (a, b) -> seeded(T, a + b / 3))
        v = @inferred ParticleMeshCore.gather_at(f, (Int32(3), Int32(4)), (seeded(T, 0.3), seeded(T, 0.6)), shape, st, Int32(guards(grid)))
        @test v isa T
    end
end

@testset "T-265 2D deposition [A3, A7, A8, WP-018]" begin
    C, Cadj, Cp, Cnode = tolC("T-265"), tolC("T-265", "C_adjoint"), tolC("T-265", "C_partials"), tolC("T-265", "C_nodes")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in (Reference(), Fast())
        ε = base_eps(T)
        grid = grid2(T, guard_width(shape))
        for (np, quantity) in ((200, ChargeWeight()), (2000, ChargeWeight()), (2000, ChargeUy2()))
            s = moved_species_2d(T, grid, np)
            ρ = fold_guards!(deposited_2d(mode, s, shape, st, grid; quantity), grid)
            q = [quantity(s, i) for i in 1:np]
            r = record!("T-265", T, absdiff(sum(ρ), sum(q)) / (ε * np * maximum(absmax, q)))
            @test r <= C
        end
        # adjoint of gather with the same weighting, fill and fold included
        s = moved_species_2d(T, grid, 400)
        f = test_field_2d(T, grid, (a, b) -> seeded(T, sin(0.7a) + cos(0.4b) / 2 + 0.3))
        out = similar(s.w)
        gather!(out, f, s, shape, st, grid)
        lhs = sum(ChargeUy2()(s, i) * out[i] for i in 1:400)
        ρ = interior_view(fold_guards!(deposited_2d(mode, s, shape, st, grid; quantity = ChargeUy2()), grid), grid)
        rhs = sum(ρ .* interior_view(f, grid))
        scale = sum(absmax(ChargeUy2()(s, i)) for i in 1:400) * maximum(absmax, f)
        r = record!("T-265 adjoint", T, absdiff(lhs, rhs) / (ε * scale))
        @test r <= Cadj
    end
    # Dual weights: exact partials, the primal values bitwise equal to the Float32 run.
    for shape in SHAPES, st in STAGGER_PAIRS
        grid = grid2(D3, guard_width(shape))
        sD = moved_species_2d(D3, grid, 400)
        for i in 1:400
            w = value(sD.w[i])
            sD.w[i] = mkdual(w, cos(0.3i), w^2, 0)
            for d in 1:2
                sD.offset[d][i] = mkdual(value(sD.offset[d][i]), 0, 0, 0)
            end
        end
        ρD = deposited_2d(Reference(), sD, shape, st, grid)
        g32 = grid2(Float32, guard_width(shape))
        weighted(wf) = begin
            s32 = Species(backend, Float32, capacity(sD); q = -1, m = 1, N = 2)
            for i in 1:400
                for d in 1:2
                    s32.cell[d][i] = sD.cell[d][i]
                    s32.offset[d][i] = value(sD.offset[d][i])
                end
                s32.w[i] = wf(sD.w[i])
                s32.status[i] = ParticleMeshCore.ALIVE
            end
            fill!(s32.count, Int32(400))
            deposited_2d(Reference(), s32, shape, st, g32)
        end
        @test same_bits(value.(ρD), weighted(value))
        for dir in 1:3
            ∂ = [partials(x)[dir] for x in ρD]
            if dir == 3
                @test all(iszero, ∂)
            else
                g64 = PeriodicGrid(N2, Float64.(dx2(Float32)), guard_width(shape))
                s64 = Species(backend, Float64, capacity(sD); q = -1, m = 1, N = 2)
                for i in 1:400
                    for d in 1:2
                        s64.cell[d][i] = sD.cell[d][i]
                        s64.offset[d][i] = value(sD.offset[d][i])
                    end
                    s64.w[i] = partials(sD.w[i])[dir]
                    s64.status[i] = ParticleMeshCore.ALIVE
                end
                fill!(s64.count, Int32(400))
                exact = deposited_2d(Reference(), s64, shape, st, g64)
                r = record!("T-265 partials", D3, maximum(abs.(∂ .- exact)) / (eps(Float32) * maximum(abs, exact)))
                @test r <= Cp
            end
        end
    end
    # Reference and Fast agree per node within C·eps(T)·Σ|contributions|.
    for T in (Float32, Float64), shape in SHAPES, st in STAGGER_PAIRS
        grid = grid2(T, guard_width(shape))
        dep(mode, quantity) = deposited_2d(mode, moved_species_2d(T, grid, 192 * 40; seed = 11), shape, st, grid; quantity)
        ρF, ρR, ρabs = dep(Fast(), ChargeUy2()), dep(Reference(), ChargeUy2()), dep(Reference(), AbsChargeUy2())
        worst = maximum(abs(ρF[a] - ρR[a]) / (eps(T) * ρabs[a]) for a in eachindex(ρR) if ρabs[a] != 0)
        record!("T-265 nodes", T, worst)
        @test worst <= Cnode
    end
end

@testset "T-266 2D sort and Reference determinism [A4, WP-018]" begin
    for T in TYPES
        s, ws, grid = sorted_2d(T)
        n = Int32.(ncells(grid))
        alive = 300 - length(DEAD2)
        @test alive_count(s) == alive
        @test all(==(ParticleMeshCore.ALIVE), s.status[1:alive]) && all(==(ParticleMeshCore.FREE), s.status[(alive + 1):end])
        Lc(i) = ParticleMeshCore.linear_cell((s.cell[1][i], s.cell[2][i]), n)
        @test issorted([(Lc(i), ws.perm[i]) for i in 1:alive])          # linear cell, then the old slot
        @test issorted(ws.perm[(alive + 1):end])                        # free slots last, by slot
        @test allunique(ws.key)
        @test sum(max.(0, ws.seg_stop .- ws.seg_start .+ 1)) == alive
        @test length(ws.seg_start) == prod(n)
        @test all(L -> all(i -> Lc(i) == L, ws.seg_start[L]:ws.seg_stop[L]), 1:prod(n))
        empty = [L for L in 1:prod(n) if !any(i -> Lc(i) == L, 1:alive)]
        @test !isempty(empty) && all(L -> ws.seg_start[L] == 1 && ws.seg_stop[L] == 0, empty)
        # two extras of different element types follow the permutation
        @test all(i -> s.extras.old_slot[i] == ws.perm[i], 1:capacity(s))
        @test same_bits(s.extras.old_offset, s.offset[1])
        @test all(i -> s.extras.old_slot[i] > 300 || Int(value_of(s.u[3][i])) == s.extras.old_slot[i], 1:capacity(s))
        # sorting a sorted species is the identity
        before = map(copy, species_arrays(s))
        sort_by_cell!(s, grid, ws)
        @test all(same_bits.(before, species_arrays(s)))
        @test ws.perm == 1:capacity(s)
    end
    # Reference results: one digest per case for reruns and the three launch shapes, in process and at 1 and 4 threads.
    @test engine2d_digests() == INPROCESS_2D
    groups = Dict{String,Vector{String}}()
    for l in INPROCESS_2D
        f = split(l, '|')
        push!(get!(groups, join(f[1:3], '|'), String[]), f[5])
    end
    @test all(v -> length(v) == length(WG_SIZES) && allequal(v), values(groups))
    for nt in (1, 4)
        @test e2d_worker(nt) == INPROCESS_2D
    end
end

@testset "T-267 2D loading [A3, A4, A8, WP-018]" begin
    C = tolC("T-267")
    for T in TYPES
        grid = grid2(T, 2)
        n, h = ncells(grid), cellsize(grid)
        K = prod(PPC2)
        p = oblique_params(T, grid)
        s = load_2d(T, grid, PPC2; seed = 4, stream = 2)
        np = prod(n) * K
        @test alive_count(s) == np && count(==(ParticleMeshCore.ALIVE), s.status) == np
        key = (UInt32(4), UInt32(2))
        vdrift, vth = (T(1 // 10), zero(T), T(-1 // 10)), (T(1 // 20), T(1 // 30), T(1 // 40))
        pos_ok, ctr_ok, w_ok = true, true, true
        wmax = zero(basetype(T))
        for L in 1:prod(n), k in 1:K
            i = (L - 1) * K + k
            c = ParticleMeshCore.cartesian_cell(L, n)
            kk = (mod1(k, PPC2[1]), (k - 1) ÷ PPC2[1] + 1)
            δ = ntuple(d -> (T(kk[d]) - one(T) / 2) / T(PPC2[d]), 2)
            pos_ok &= (s.cell[1][i], s.cell[2][i]) == Int32.(c) && components(s.offset[1][i]) == components(δ[1]) &&
                      components(s.offset[2][i]) == components(δ[2])
            # the counter (cell₁, cell₂, 0, k << 16) through the known-answer philox4x32
            r = philox4x32((UInt32(c[1]), UInt32(c[2]), UInt32(0), UInt32(k) << 16), key)
            z1, z2 = box_muller(T, r[1], r[2])
            z3, _ = box_muller(T, r[3], r[4])
            u = (vdrift[1] + vth[1] * z1, vdrift[2] + vth[2] * z2, vdrift[3] + vth[3] * z3)
            # within 2 ulp of |vdrift| + |vth·z| (the kernel may contract to FMAs; u can cancel to near zero)
            ctr_ok &= all(d -> absdiff(s.u[d][i], u[d]) <= 2eps(basetype(T)) * (absmax(vdrift[d]) + absmax(vth[d] * (z1, z2, z3)[d])), 1:3)
            x = ((T(c[1] - 1) + δ[1]) * h[1], (T(c[2] - 1) + δ[2]) * h[2])
            wh = ObliqueDensity()(x, p) / T(K)
            w_ok &= absdiff(s.w[i], wh) <= 2eps(basetype(T)) * absmax(wh)
            wmax = max(wmax, absmax(wh))
        end
        @test pos_ok
        @test ctr_ok
        @test w_ok
        # Σw against the lattice sum: the oblique mode sums to zero over the product lattice, so Σw = n₀·cells.
        r = record!("T-267", T, absdiff(sum(s.w[1:np]), p[1] * prod(n)) / (base_eps(T) * np * wmax))
        @test r <= C
        # same seed: the same bits; another stream or seed: other draws, the same positions and weights
        a, b, c2, d = load_2d(T, grid, PPC2), load_2d(T, grid, PPC2), load_2d(T, grid, PPC2; stream = 3), load_2d(T, grid, PPC2; seed = 5)
        @test all(same_bits.(species_arrays(a), species_arrays(b)))
        @test a.u[1] != c2.u[1] && a.u[1] != d.u[1] && same_bits(a.w, c2.w) && same_bits(a.offset[2], d.offset[2])
    end
    # Positions are bitwise independent of p: Dual parameters appear in the weights only.
    grid = grid2(D3, 2)
    pD = (mkdual(1, 1, 0, 0), mkdual(1 / 10, 0, 1, 0), oblique_params(D3, grid)[3:4]...)
    sD = load_2d(D3, grid, PPC2; p = pD)
    sF = load_2d(Float32, grid2(Float32, 2), PPC2)
    np = prod(N2) * prod(PPC2)
    @test all(d -> all(i -> iszero(partials(sD.offset[d][i])) && value(sD.offset[d][i]) === sF.offset[d][i], 1:np), 1:2)
    @test all(d -> sD.cell[d] == sF.cell[d], 1:2)
    @test any(i -> !iszero(partials(sD.w[i])), 1:np)
    # The N = 1 counter is unchanged: (cell, 0, 0, k << 16).
    g1 = PeriodicGrid(16, 0.25, 2)
    s1 = load_quiet!(Species(backend, Float64, 16 * 3; q = -1, m = 1), g1, (x, p) -> p, 1.0, 3; vth = (1.0, 1.0, 1.0), seed = 4, stream = 2)
    ok1 = true
    for c in 1:16, k in 1:3
        r = philox4x32((UInt32(c), UInt32(0), UInt32(0), UInt32(k) << 16), (UInt32(4), UInt32(2)))
        z1, z2 = box_muller(Float64, r[1], r[2])
        ok1 &= abs(s1.u[1][(c - 1) * 3 + k] - z1) <= 2eps() * abs(z1) && abs(s1.u[2][(c - 1) * 3 + k] - z2) <= 2eps() * abs(z2)
    end
    @test ok1
    # the in-cell index must fit the counter layout
    @test_throws ArgumentError load_quiet!(Species(backend, Float64, 2^17; q = -1, m = 1, N = 2), PeriodicGrid((4, 4), (1.0, 1.0), 2),
                                           ObliqueDensity(), (1.0, 0.0, 0.0, 0.0), (256, 256); seed = 1, stream = 1)
    # the 1- and 4-thread loads are T-266's digests (`load2d`)
end

@testset "T-268 allocation does not grow in 2D [A7, A9, WP-018]" begin
    # Bytes per call after warm-up at two particle counts and two grid sizes, with and without
    # extras, measured as T-144 does: the CPU backend allocates a constant amount per launch;
    # what must not happen is growth with the problem size. With more than one thread the sort
    # allocates in AcceleratedKernels' threaded sample sort (#82), so that check is broken there.
    for T in (Float32, Float64), extras in (NamedTuple(), (old_cell = Int32, old_offset = T))
        bytes = Dict{Tuple{String,Tuple{Int,Int},Tuple{Int,Int}},Int}()
        for (n, ppc) in (((16, 12), (2, 3)), ((16, 12), (4, 6)), ((32, 24), (2, 3)))
            grid = grid2(T, 2; n)
            s = load_2d(T, grid, ppc; extras)
            ws = Workspace(s, grid)
            ρ = field(backend, T, grid)
            out = similar(s.w)
            p = oblique_params(T, grid)
            ops = (sort = () -> sort_by_cell!(s, grid, ws),
                   reference = () -> deposit!(Reference(), ρ, s, Quadratic, (Center(), Center()), grid, ws),
                   fast = () -> deposit!(Fast(), ρ, s, Quadratic, (Center(), Center()), grid, ws),
                   gather = () -> gather!(out, ρ, s, Quadratic, (Center(), Center()), grid),
                   fill = () -> fill_guards!(ρ, grid), fold = () -> fold_guards!(ρ, grid),
                   load = () -> load_quiet!(s, grid, ObliqueDensity(), p, ppc; seed = 1, stream = 1))
            for (name, op) in pairs(ops)
                op(); op()
                bytes[(string(name), n, ppc)] = @allocated op()
            end
        end
        @info "T-268 bytes per call" T extras = keys(extras) bytes nthreads = Threads.nthreads()
        for name in ("sort", "reference", "fast", "gather", "fill", "fold", "load")
            broken = name == "sort" && Threads.nthreads() > 1
            small = bytes[(name, (16, 12), (2, 3))]
            if VERSION >= v"1.14-"
                @test_skip bytes[(name, (16, 12), (4, 6))] <= small
            else
                @test bytes[(name, (16, 12), (4, 6))] <= small broken = broken
            end
            @test bytes[(name, (32, 24), (2, 3))] <= small broken = broken
        end
    end
end

@testset "T-269 2D reduces to 1D [A7, WP-018]" begin
    C, Cg = tolC("T-269"), tolC("T-269", "C_gather")
    for T in TYPES, shape in SHAPES, st1 in STAGGERS
        ε = base_eps(T)
        st = (st1, Center())
        grid = grid2(T, guard_width(shape))
        n, g = ncells(grid), guards(grid)
        p = (seeded(T, 1), T(1 // 10), T(2π / (n[1] * cellsize(grid)[1])))
        s2 = load_quiet!(Species(backend, T, prod(n) * prod(PPC2); q = -1, m = 1, N = 2), grid, XDensity(), p, PPC2; seed = 1, stream = 1)
        np = capacity(s2)
        g1 = PeriodicGrid(n[1], cellsize(grid)[1], g)
        s1 = Species(backend, T, np; q = -1, m = 1)
        copyto!(s1.cell[1], s2.cell[1]); copyto!(s1.offset[1], s2.offset[1]); copyto!(s1.w, s2.w); copyto!(s1.status, s2.status)
        fill!(s1.count, Int32(np))
        # gather of a y-constant field (before the deposits: the Reference deposits sort s1 and s2 by different keys)
        f1 = test_field(T, g1, j -> seeded(T, sin(0.7j) + 0.3))
        f2 = field(backend, T, grid)
        for a2 in axes(f2, 2)
            f2[:, a2] .= f1
        end
        o1, o2 = similar(s1.w), similar(s2.w)
        gather!(o1, f1, s1, shape, st1, g1)
        gather!(o2, f2, s2, shape, st, grid)
        rg = record!("T-269 gather", T, maximum(absdiff.(o1, o2)) / (ε * maximum(absmax, f1)))
        @test rg <= Cg
        ρ2 = interior_view(fold_guards!(deposited_2d(Reference(), s2, shape, st, grid), grid), grid)
        ρ1 = interior_view(fold_guards!(deposited(Reference(), s1, shape, st1, g1), g1), g1)
        ysum = vec(sum(ρ2; dims = 2))
        scale = n[2] * maximum(absmax, ρ1)
        r = record!("T-269", T, maximum(absdiff.(ysum, ρ1)) / (ε * scale))
        @test r <= C
    end
end
