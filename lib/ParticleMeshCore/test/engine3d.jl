# WP-042, the engine at N = 3 outside the flux deposition, on the CPU backend: T-870 to T-875
# (T-876 is their rerun on CUDA, in gpu.jl). The grid, guard fill and fold, the quiet load, the
# sort with the scalar deposition and the gather, and the periodic solve with its gradient.
#
# The 3D tests run on the grids of engine3d_cases.jl: 8 × 6 × 5 cells of 1/4 × 3/8 × 1/2, with
# 2 × 1 × 3 particles per cell when loading and a few hundred hand-placed particles otherwise.

foreach(start_worker, WORKER_THREADS)   # the thread-count workers (workers.jl) run beside this file's in-process runs

const INPROCESS_3D = engine3d_digests()
e3d_worker(nt) = [chopprefix(l, "E3D ") for l in worker_lines(nt) if startswith(l, "E3D ")]

@testset "T-870 the 3D grid, linearization and advance [A7, A8, WP-042]" begin
    for T in TYPES
        grid = grid3(T, 2)
        @test grid isa PeriodicGrid{3,basetype(T)}
        @test ncells(grid) == (8, 6, 5) && cellsize(grid) == dx3(T) && guards(grid) == 2
        a = field(backend, T, grid)
        @test size(a) == (12, 10, 9) && eltype(a) === T && all(iszero, a)
        @test interior(grid) == (3:10, 3:8, 3:7)
        v = interior_view(a, grid)
        @test size(v) == (8, 6, 5) && parentindices(v) == (3:10, 3:8, 3:7)
    end
    # the per-axis check n_d ≥ 2g, before any allocation
    for n in ((3, 6, 5), (8, 3, 5), (8, 6, 3))
        @test_throws ArgumentError PeriodicGrid(n, (1.0, 1.0, 1.0), 2)
    end
    @test PeriodicGrid((4, 4, 4), (1.0, 1.0, 1.0), 2) isa PeriodicGrid{3,Float64}
    # column-major linear cell index (axis 1 fastest) and its inverse
    n = Int32.(N3)
    Ls = Int32[]
    for c3 in Int32(1):n[3], c2 in Int32(1):n[2], c1 in Int32(1):n[1]
        L = ParticleMeshCore.linear_cell((c1, c2, c3), n)
        push!(Ls, L)
        @test L == c1 + (c2 - 1) * n[1] + (c3 - 1) * n[1] * n[2]
        @test ParticleMeshCore.cartesian_cell(L, n) === (c1, c2, c3)
    end
    @test Ls == 1:prod(n)
    # Free streaming at dyadic displacements is exact per axis, through each seam and diagonally
    # through the corner; offsets stay in [0, 1).
    for T in TYPES
        starts = (((8, 6, 5), (15 // 16, 15 // 16, 15 // 16)), ((1, 1, 1), (1 // 16, 0 // 1, 1 // 2)), ((8, 3, 1), (1 // 2, 3 // 8, 0 // 1)))
        disps = ((1 // 8, 1 // 8, 1 // 8), (-3 // 16, -1 // 4, -1 // 16), (1 // 4, 0 // 1, -5 // 16))
        steps = 300
        for ((c0, δ0), Δr) in zip(starts, disps)
            c = Int32.(c0)
            δ = map(x -> seeded(T, x), δ0)
            Δ = map(x -> seeded(T, x), Δr)
            inside = true
            for _ in 1:steps
                c, δ = advance(c, δ, Δ, n)
                inside &= all(x -> 0 <= value_of(x) < 1, δ)
            end
            @test inside
            for d in 1:3
                exact = (c0[d] - 1) + δ0[d] + steps * Δr[d]
                exact -= n[d] * floor(exact / n[d])
                @test (c[d] - 1) + value_of(δ[d]) == exact
            end
        end
        # one cell back through the corner is exact
        c, δ = advance((Int32(1), Int32(1), Int32(1)), (zero(T), zero(T), zero(T)), ntuple(_ -> -one(T) / 16, 3), n)
        @test c == n && all(x -> value_of(x) == 15 // 16, δ)
    end
end

"The periodic image of every entry of `x`, guards, edges and corners included: the interior entry it aliases."
periodic_image(x, grid::PeriodicGrid{N}) where {N} =
    [x[CartesianIndex(ntuple(d -> mod1(I[d] - guards(grid), grid.n[d]) + guards(grid), Val(N)))] for I in CartesianIndices(x)]

function body_T871(be)
    C = tolC("T-871")
    for T in TYPES, g in (1, 2, 3), n in ((6, 7, 9), (9, 6, 7))
        ε = base_eps(T)
        grid = PeriodicGrid(n, dx3(T), g)
        x, y = indexed_field(T, grid, i -> cos(1.3i)), indexed_field(T, grid, i -> sin(2.1i) + 0.1)
        fx = on(be, fill_guards!, copy(x), grid)
        # every entry, guards, edges and corners included, is a bitwise copy of its periodic interior image
        @test same_bits(fx, periodic_image(x, grid))
        fy = on(be, fold_guards!, copy(y), grid)
        r = record!("T-871", T, absdiff(sum(fx .* y), sum(x .* fy)) / (ε * sum(absmax.(x)) * maximum(absmax, y)), be)
        @test r <= C
        inner = falses(size(y))
        inner[interior(grid)...] .= true
        @test all(iszero, fy[.!inner])
        # A field of small integers folds exactly, in any order: every interior entry is the sum
        # of the entries that alias it (itself, and up to seven guard entries at a corner).
        z = field(backend, T, grid)
        expect = zeros(Int, n)
        for I in CartesianIndices(z)
            v = mod(7 * LinearIndices(z)[I] + 3, 11) - 5
            z[I] = T(v)
            expect[CartesianIndex(ntuple(d -> mod1(I[d] - g, n[d]), 3))] += v
        end
        fz = on(be, fold_guards!, z, grid)
        @test value_of.(interior_view(fz, grid)) == expect
        # fill after fold: the guards are again the images of the folded interior
        @test same_bits(on(be, fill_guards!, copy(fz), grid), periodic_image(fz, grid))
    end
    # g = 0: nothing to fill or fold
    for T in TYPES
        grid = PeriodicGrid((3, 2, 4), dx3(T), 0)
        x = indexed_field(T, grid, i -> cos(1.3i))
        @test same_bits(on(be, fill_guards!, copy(x), grid), x) && same_bits(on(be, fold_guards!, copy(x), grid), x)
    end
end

@testset "T-871 3D fill and fold [A7, A8, WP-042]" begin
    body_T871(CPU())
    # The one kernel of every N ≥ 2 at N = 2, against the image and the exact fold (T-263 has its adjointness).
    for T in TYPES, g in (1, 2), n in ((4, 9), (9, 5))
        grid = PeriodicGrid(n, dx2(T), g)
        x = indexed_field(T, grid, i -> cos(1.3i))
        @test same_bits(fill_guards!(copy(x), grid), periodic_image(x, grid))
        z = field(backend, T, grid)
        expect = zeros(Int, n)
        for I in CartesianIndices(z)
            v = mod(7 * LinearIndices(z)[I] + 3, 11) - 5
            z[I] = T(v)
            expect[CartesianIndex(mod1(I[1] - g, n[1]), mod1(I[2] - g, n[2]))] += v
        end
        @test value_of.(interior_view(fold_guards!(z, grid), grid)) == expect
    end
end

"The in-cell lattice indices `(k₁, k₂, k₃)` of the in-cell index `k`, axis 1 fastest."
incell_3d(k, ppc) = Tuple(CartesianIndices(ppc)[k])

"T-872's load checks on `be`."
function body_T872(be)
    C, Cw = tolC("T-872"), tolC("T-872", "C_weights")
    for T in TYPES, at in (:particle, :cell_centre)
        B = basetype(T)
        grid = grid3(T, 2)
        n, h = ncells(grid), cellsize(grid)
        K = prod(PPC3)
        np = prod(n) * K
        p = oblique_params_3d(T, grid)
        first = 5
        s = load_3d(T, grid, PPC3; seed = 4, stream = 2, first, density_at = at, be)
        @test alive_count(s) == first - 1 + np
        @test all(==(ParticleMeshCore.FREE), s.status[1:(first - 1)]) && all(==(ParticleMeshCore.ALIVE), s.status[first:(first - 1 + np)]) &&
              all(==(ParticleMeshCore.FREE), s.status[(first + np):end])
        @test all(iszero, s.w[1:(first - 1)]) && all(iszero, s.w[(first + np):end])        # the other slots are not written
        key = (UInt32(4), UInt32(2))
        vdrift, vth = T.(LOAD_VDRIFT_3D), T.(LOAD_VTH_3D)
        pos_ok, u_ok, w_ok = true, true, true
        wmax = zero(B)
        for L in 1:prod(n), k in 1:K
            i = first - 1 + (L - 1) * K + k
            c = Tuple(CartesianIndices(n)[L])                                  # axis 1 fastest
            kk = incell_3d(k, PPC3)
            δ = ntuple(d -> (T(kk[d]) - one(T) / 2) / T(PPC3[d]), 3)
            pos_ok &= all(d -> s.cell[d][i] == c[d] && components(s.offset[d][i]) == components(δ[d]), 1:3)
            # the counter (cell₁, cell₂, cell₃, k << 16) through the known-answer philox4x32
            r = philox4x32((UInt32(c[1]), UInt32(c[2]), UInt32(c[3]), UInt32(k) << 16), key)
            z = (box_muller(T, r[1], r[2])..., box_muller(T, r[3], r[4])[1])
            u = ntuple(d -> vdrift[d] + vth[d] * z[d], 3)
            # within 2 ulp of |vdrift| + |vth·z| (the kernel may contract to FMAs; u can cancel to near zero)
            u_ok &= all(d -> absdiff(s.u[d][i], u[d]) <= 2eps(B) * (absmax(vdrift[d]) + absmax(vth[d] * z[d])), 1:3)
            # the weight against the density in Float64, with the parameters as `T` holds them, at the
            # particle (or at its cell's centre)
            x = ntuple(d -> (c[d] - 1 + (at === :particle ? Float64(value_of(δ[d])) : 0.5)) * Float64(h[d]), 3)
            ρ = 1 + Float64(value_of(p[2])) * cos(sum(d -> Float64(value_of(p[2 + d])) * x[d], 1:3))
            wref = Float64.(components(p[1])) .* (ρ / K)
            wr = maximum(abs.(Float64.(components(s.w[i])) .- wref)) / (eps(B) * maximum(abs, wref))
            record!("T-872 weights", T, wr, be)
            w_ok &= wr <= Cw
            wmax = max(wmax, absmax(s.w[i]))
        end
        @test pos_ok
        @test u_ok
        @test w_ok
        # Σw against the lattice sum: the oblique mode sums to zero over the product lattice (and
        # over the cell centres), so Σw = n₀·cells.
        r = record!("T-872", T, absdiff(sum(s.w[first:(first - 1 + np)]), p[1] * prod(n)) / (base_eps(T) * np * wmax), be)
        @test r <= C
        # same seed: the same bits; another stream or seed: other draws, the same positions and weights
        a, b = load_3d(T, grid, PPC3; density_at = at, be), load_3d(T, grid, PPC3; density_at = at, be)
        c2, d = load_3d(T, grid, PPC3; stream = 3, density_at = at, be), load_3d(T, grid, PPC3; seed = 5, density_at = at, be)
        @test all(same_bits.(species_arrays(a), species_arrays(b)))
        @test a.u[1] != c2.u[1] && a.u[1] != d.u[1] && same_bits(a.w, c2.w) && same_bits(a.offset[3], d.offset[3])
    end
end

@testset "T-872 3D quiet load [A3, A4, A8, WP-042]" begin
    body_T872(CPU())
    key = (UInt32(4), UInt32(2))
    for T in TYPES
        B = basetype(T)
        grid = grid3(T, 2)
        n = ncells(grid)
        K = prod(PPC3)
        np = prod(n) * K
        s = load_3d(T, grid, PPC3)
        # the loaded slots are in sorted order: the sort is the identity
        ws = Workspace(s, grid)
        before = map(copy, species_arrays(s))
        sort_by_cell!(s, grid, ws)
        @test ws.perm == 1:capacity(s) && all(same_bits.(before, species_arrays(s)))
        # the distribution form: GaussianVelocity is the keyword form bit for bit
        gv = load_3d(T, grid, PPC3; dist = GaussianVelocity(T.(LOAD_VDRIFT_3D), T.(LOAD_VTH_3D)))
        @test all(same_bits.(species_arrays(gv), before))
        # and with the density at either point, written from slot 3: the same slots and the same count
        for at in (:particle, :cell_centre)
            kw = load_3d(T, grid, PPC3; density_at = at, first = 3)
            gv = load_3d(T, grid, PPC3; density_at = at, first = 3, dist = GaussianVelocity(T.(LOAD_VDRIFT_3D), T.(LOAD_VTH_3D)))
            @test all(same_bits.(species_arrays(gv), species_arrays(kw)))
            @test Array(gv.count) == Array(kw.count) == [Int32(2 + prod(ncells(grid)) * prod(PPC3))]
            at === :cell_centre && @test !same_bits(gv.w, load_3d(T, grid, PPC3; first = 3).w)   # the point matters
        end
        # the other distributions: the same lattice and weights, the draws of the counter (c₁, c₂, c₃, k << 16)
        for dist in (GaussianMomentum(T.(LOAD_VDRIFT_3D), T.(LOAD_VTH_3D)), MaxwellJuttner(B(1 // 20), B.((1 // 10, 0, -1 // 5))))
            exhausted = zeros(Int32, 1)
            m = load_quiet!(Species(backend, T, np; q = -1, m = 1, N = 3), grid, ObliqueDensity3(), oblique_params_3d(T, grid), PPC3, dist;
                            seed = 4, stream = 2, exhausted)
            @test exhausted[1] == 0 && alive_count(m) == np
            @test all(d -> same_bits(m.cell[d], s.cell[d][1:np]) && same_bits(m.offset[d], s.offset[d][1:np]), 1:3) && same_bits(m.w, s.w[1:np])
            ok = true
            for L in 1:prod(n), k in 1:K
                c = Tuple(CartesianIndices(n)[L])
                u, accepted = ParticleMeshCore.draw(dist, T, (UInt32(c[1]), UInt32(c[2]), UInt32(c[3]), UInt32(k) << 16), key)
                i = (L - 1) * K + k
                # the kernel's arithmetic may differ from the host's in the last bits; another counter gives another draw altogether
                ok &= accepted && all(d -> absdiff(m.u[d][i], u[d]) <= 64eps(B), 1:3)
            end
            @test ok
        end
        # A cell's particles do not depend on the grid that loads them (A4): the cells of the
        # 8 × 6 × 5 grid inside a 9 × 7 × 6 grid of the same cell sizes have the same draws.
        big = PeriodicGrid((9, 7, 6), dx3(T), 2)
        sb = load_3d(T, big, PPC3; p = oblique_params_3d(T, grid))
        same = true
        for L in 1:prod(n), k in 1:K
            c = Tuple(CartesianIndices(n)[L])
            i, j = (L - 1) * K + k, (LinearIndices((9, 7, 6))[c...] - 1) * K + k
            same &= all(d -> components(s.u[d][i]) == components(sb.u[d][j]) && components(s.offset[d][i]) == components(sb.offset[d][j]), 1:3) &&
                    components(s.w[i]) == components(sb.w[j])
        end
        @test same
        # every axis' cell is in the counter: two cells that differ along one axis only draw differently
        cellslots(c) = (LinearIndices(n)[c...] - 1) * K .+ (1:K)
        for (ca, cb) in (((2, 3, 4), (3, 3, 4)), ((2, 3, 4), (2, 4, 4)), ((2, 3, 4), (2, 3, 5)))
            @test value_of.(s.u[1][cellslots(ca)]) != value_of.(s.u[1][cellslots(cb)])
        end
    end
    # Dual parameters enter through the weights only (A3): positions and draws carry no partials,
    # and the partials of a weight are those of the density.
    grid = grid3(D3, 2)
    s = load_3d(D3, grid, PPC3)
    live = 1:(prod(N3) * prod(PPC3))
    @test all(d -> all(x -> iszero(partials(x)), s.offset[d][live]) && all(x -> iszero(partials(x)), s.u[d][live]), 1:3)
    @test all(x -> all(abs.(Tuple(partials(x)) .- value(x) .* (1.0f0, 2.0f0, 3.0f0) ./ 8) .<= 4 * eps(value(x))), s.w[live])   # w ∝ p[1] = 1 + (1, 2, 3)/8·ε
    # input checks
    cap = Species(backend, Float64, 10; q = -1, m = 1, N = 3)
    g64 = grid3(Float64, 2)
    p64 = oblique_params_3d(Float64, g64)
    @test_throws ArgumentError load_quiet!(cap, g64, ObliqueDensity3(), p64, PPC3; seed = 1, stream = 1)                       # capacity
    full = Species(backend, Float64, prod(N3) * 6; q = -1, m = 1, N = 3)
    @test_throws ArgumentError load_quiet!(full, g64, ObliqueDensity3(), p64, PPC3; seed = 1, stream = 1, first = 2)           # one slot short
    @test_throws ArgumentError load_quiet!(full, g64, ObliqueDensity3(), p64, (2, 0, 3); seed = 1, stream = 1)
    @test_throws ArgumentError load_quiet!(full, g64, ObliqueDensity3(), p64, (64, 32, 32); seed = 1, stream = 1)             # prod = 2^16
    @test_throws ArgumentError load_quiet!(full, g64, ObliqueDensity3(), p64, PPC3; seed = 1, stream = 1, density_at = :node)
    @test_throws ArgumentError load_quiet!(full, g64, ObliqueDensity3(), p64, PPC3, GaussianMomentum((0.0, 0.0, 0.0), (0.1, 0.1, 0.1));
                                           seed = 1, stream = 1, first = 2)
end

"q·w·u₂, a module-supplied quantity."
struct ChargeU2 end
@inline (::ChargeU2)(s, i) = s.q * s.w[i] * s.u[2][i]
struct AbsChargeU2 end
@inline (::AbsChargeU2)(s, i) = abs(s.q * s.w[i] * s.u[2][i])

function body_T873(be)
    C, Cg, Cadj, Cnode = tolC("T-873"), tolC("T-873", "C_gather"), tolC("T-873", "C_adjoint"), tolC("T-873", "C_nodes")
    # the sort: by column-major linear cell, then slot; free slots last; extras follow; idempotent
    for T in TYPES
        s, ws, grid = sorted_3d(T; be)
        n = Int32.(ncells(grid))
        alive = 300 - length(DEAD3)
        @test alive_count(s) == alive
        @test all(==(ParticleMeshCore.ALIVE), s.status[1:alive]) && all(==(ParticleMeshCore.FREE), s.status[(alive + 1):end])
        Lc(i) = ParticleMeshCore.linear_cell((s.cell[1][i], s.cell[2][i], s.cell[3][i]), n)
        @test issorted([(Lc(i), ws.perm[i]) for i in 1:alive])
        @test issorted(ws.perm[(alive + 1):end]) && allunique(ws.key)
        @test sum(max.(0, ws.seg_stop .- ws.seg_start .+ 1)) == alive && length(ws.seg_start) == prod(n)
        @test all(L -> all(i -> Lc(i) == L, ws.seg_start[L]:ws.seg_stop[L]), 1:prod(n))
        @test all(i -> s.extras.old_slot[i] == ws.perm[i], 1:capacity(s)) && same_bits(s.extras.old_offset, s.offset[3])
        before = map(copy, species_arrays(s))
        on(be, sort_by_cell!, s, grid, ws)
        @test all(same_bits.(before, species_arrays(s))) && ws.perm == 1:capacity(s)
    end
    for T in TYPES, shape in SHAPES, st in STAGGER_TRIPLES
        ε = base_eps(T)
        B = basetype(T)
        grid = grid3(T, guard_width(shape))
        g, h = guards(grid), cellsize(grid)
        Lbox = ncells(grid) .* h
        # The gather of a + b·x + c·y + d·z + e·x·y·z, held on the whole array (guards included) at
        # the node positions of the stagger, so no particle sees the seam of the polynomial.
        s = test_species_3d(T, grid, 300; seed = 3)
        sv = map(x -> ParticleMeshCore.stagger_offset(x, B), st)
        a, b, c, d, e = B(3 // 2), B(-5 // 4), B(7 // 8), B(1 // 2), B(-3 // 8)
        poly(x) = a + b * x[1] + c * x[2] + d * x[3] + e * x[1] * x[2] * x[3]
        f = field(backend, T, grid)
        for I in CartesianIndices(f)
            f[I] = poly(ntuple(k -> (I[k] - g - 1 + sv[k]) * h[k], 3))
        end
        out = similar(s.w)
        on(be, gather!, out, f, s, shape, st, grid)
        scale = abs(a) + abs(b) * Lbox[1] + abs(c) * Lbox[2] + abs(d) * Lbox[3] + abs(e) * prod(Lbox)
        worst = maximum(absdiff(out[i], poly(ntuple(k -> (s.cell[k][i] - 1 + s.offset[k][i]) * h[k], 3))) / (ε * scale) for i in 1:300)
        record!("T-873 gather", T, worst, be)
        @test worst <= Cg
        for mode in (Reference(), Fast())
            # the deposited total is Σ quantity
            for (np, quantity) in ((200, ChargeWeight()), (2000, ChargeU2()))
                m = moved_species_3d(T, grid, np)
                ρ = on(be, fold_guards!, deposited_3d(mode, m, shape, st, grid; quantity, be), grid)
                q = [quantity(m, i) for i in 1:np]
                r = record!("T-873", T, absdiff(sum(ρ), sum(q)) / (ε * np * maximum(absmax, q)), be)
                @test r <= C
            end
            # the deposition is the adjoint of the gather with the same weighting, fill and fold included
            m = moved_species_3d(T, grid, 400)
            fld = test_field_3d(T, grid, (j1, j2, j3) -> seeded(T, sin(0.7j1) + cos(0.4j2) / 2 + sin(0.9j3) / 3 + 0.3))
            out = similar(m.w)
            on(be, gather!, out, fld, m, shape, st, grid)
            lhs = sum(ChargeU2()(m, i) * out[i] for i in 1:400)
            ρ = interior_view(on(be, fold_guards!, deposited_3d(mode, m, shape, st, grid; quantity = ChargeU2(), be), grid), grid)
            rhs = sum(ρ .* interior_view(fld, grid))
            r = record!("T-873 adjoint", T, absdiff(lhs, rhs) / (ε * sum(absmax(ChargeU2()(m, i)) for i in 1:400) * maximum(absmax, fld)), be)
            @test r <= Cadj
        end
    end
    # Reference against Fast per node, within C·eps(T)·Σ|contributions| (20 particles per cell)
    for T in (Float32, Float64), shape in SHAPES, st in DIGEST_TRIPLES
        grid = grid3(T, guard_width(shape))
        dep(mode, quantity) = deposited_3d(mode, moved_species_3d(T, grid, prod(N3) * 20; seed = 11), shape, st, grid; quantity, be)
        ρF, ρR, ρabs = dep(Fast(), ChargeU2()), dep(Reference(), ChargeU2()), dep(Reference(), AbsChargeU2())
        @test count(!iszero, ρabs) > prod(N3)
        worst = maximum(abs(ρF[i] - ρR[i]) / (eps(T) * ρabs[i]) for i in eachindex(ρR) if ρabs[i] != 0)
        record!("T-873 nodes", T, worst, be)
        @test worst <= Cnode
        @test all(i -> ρabs[i] != 0 || (ρF[i] == 0 && ρR[i] == 0), eachindex(ρR))
    end
end

@testset "T-873 3D sort, gather and deposition of a quantity [A3, A4, A7, A8, WP-042]" begin
    body_T873(CPU())
    # One particle deposits the tensor product of its per-axis weights on its stencil and nothing
    # elsewhere, at the nodes the weights name (axis 1 fastest), in both modes.
    for T in TYPES, shape in SHAPES, st in STAGGER_TRIPLES, mode in (Reference(), Fast())
        grid = grid3(T, guard_width(shape))
        g = guards(grid)
        s = Species(backend, T, 3; q = -2, m = 1, N = 3)
        c, δ = (Int32(8), Int32(1), Int32(3)), (seeded(T, 0.3), seeded(T, 0.8), seeded(T, 0.55))
        for d in 1:3
            s.cell[d][2] = c[d]
            s.offset[d][2] = δ[d]
        end
        s.w[2] = seeded(T, 1.5)
        s.status[2] = ParticleMeshCore.ALIVE
        ρ = deposited_3d(mode, s, shape, st, grid)
        expect = field(backend, T, grid)
        fw = shape_weights(shape, st, δ)
        P1 = order(shape) + 1
        for k3 in 1:P1, k2 in 1:P1, k1 in 1:P1
            k = (k1, k2, k3)
            expect[ntuple(d -> c[d] + fw[d][1] + g + k[d] - 1, 3)...] = T(-2) * seeded(T, 1.5) * (fw[1][2][k1] * fw[2][2][k2] * fw[3][2][k3])
        end
        @test all(i -> absdiff(ρ[i], expect[i]) <= 4base_eps(T) * absmax(expect[i]), eachindex(ρ))
    end
end

"`2 sin(π m/n)/Δ` in BigFloat (the sign of `m` kept), one axis of the Yee-consistent symbol."
symbol_axis_big(m, n, Δ) = 2 * sinpi(BigFloat(m // n)) / BigFloat(Δ)

"T-874's grids: three sizes, non-cubic with three cell sizes, so that the ratio's growth is visible."
const T874_GRIDS = ((8, 6, 5), (16, 12, 10), (32, 24, 20))

@testset "T-874 the 3D periodic solve: div grad φ = s - mean(s) at every node, and an oblique mode against the closed form [A7, A8, WP-042]" begin
    C, Cm, Cmode = tolC("T-874"), tolC("T-874", "C_mean"), tolC("T-874", "C_mode")
    for T in TYPES, n in T874_GRIDS
        ε = base_eps(T)
        grid = grid3(T, 2; n)
        s = solve_input(T, grid)
        φ, G = solved(s, grid)
        φmax = maximum(absmax, interior_of(φ, grid))
        smax = maximum(absmax, interior_of(s, grid))
        nsum = sum(grid.n)
        r = maximum(absmax, divergence_residual(G, s, grid)) / (ε * nsum * (smax + laplacian_norm(grid) * φmax))
        record!("T-874", T, r)
        @test r <= C
        rm = absmax(sum(interior_of(φ, grid)) / prod(grid.n)) / (ε * nsum * φmax)
        record!("T-874 mean", T, rm)
        @test rm <= Cm
        # outputs have their guards filled; the NaN guards of s were not read
        @test guards_are_images(φ, grid) && all(Gd -> guards_are_images(Gd, grid), G)
        @test all(x -> all(isfinite, components(x)), φ) && all(Gd -> all(x -> all(isfinite, components(x)), Gd), G)
        T === D3 && @test any(x -> !iszero(partials(x)), φ)
    end
    # s = A·cos θ_j, θ_j = 2π Σ_d m_d(j_d - 1)/n_d, on 8 × 6 × 5 cells of 1/4 × 3/8 × 1/2. Closed form
    # of the discrete problem: φ = -s/K² with K² = Σ_d (2 sin(π m_d/n_d)/Δ_d)², and
    # G_d[j] = 2 sin(π m_d/n_d)/(Δ_d K²)·A·sin(θ_j + π m_d/n_d). References in BigFloat.
    for T in TYPES, m in ((1, 2, 1), (3, -2, 2), (0, 0, 1), (0, 1, 0), (1, 0, 0))
        ε = base_eps(T)
        grid = grid3(T, 2)
        n, h, g = grid.n, grid.dx, guards(grid)
        A = seeded(T, 1)
        cA = Float64.(components(A))
        K2 = sum(d -> symbol_axis_big(m[d], n[d], h[d])^2, 1:3)
        θ(j) = 2 * sum(d -> m[d] * (j[d] - 1) // n[d], 1:3)                    # θ/π, exact
        s = field(backend, T, grid)
        for j in CartesianIndices(n)
            s[j + CartesianIndex(g, g, g)] = A * basetype(T)(cospi(BigFloat(θ(j))))
        end
        φ, G = solved(s, grid)
        refs = ([Float64(-cospi(BigFloat(θ(j))) / K2) for j in CartesianIndices(n)],
                ntuple(d -> [Float64(symbol_axis_big(m[d], n[d], h[d]) / K2 * sinpi(BigFloat(θ(j) + m[d] // n[d]))) for j in CartesianIndices(n)], 3)...)
        gmax = maximum(ref -> maximum(abs, ref), refs[2:4])
        for (out, ref, scale) in zip((φ, G...), refs, (maximum(abs, refs[1]), gmax, gmax, gmax))
            oi = interior_of(out, grid)
            err = maximum(maximum(abs.(Float64.(components(oi[j])) .- cA .* ref[j])) for j in eachindex(ref))
            rr = err / (ε * sum(n) * maximum(cA) * scale)
            record!("T-874 mode", T, rr)
            @test rr <= Cmode
        end
    end
    # A reused workspace and an in-place solve (φ === s) give the same bits as a fresh one; s is not written.
    for T in TYPES
        grid = grid3(T, 2)
        s = solve_input(T, grid)
        φ, G = solved(s, grid)
        ws = SpectralWorkspace(backend, T, grid)
        @test length(ws.re) == 2 && size(ws.re[1]) == grid.n
        solved(solve_input(T, grid3(T, 2)), grid; ws)
        φ2, G2 = solved(s, grid; ws)
        @test same_bits(φ, φ2) && all(same_bits.(G, G2))
        s3 = copy(s)
        inverse_laplacian_periodic!(s3, s3, grid, ws)
        @test same_bits(s3, φ) && same_bits(s, solve_input(T, grid))
        # input checks: another grid's workspace, a field of another size, a gradient into φ
        @test_throws ArgumentError inverse_laplacian_periodic!(φ, s, grid, SpectralWorkspace(backend, T, grid3(T, 2; n = (8, 6, 6))))
        @test_throws ArgumentError inverse_laplacian_periodic!(field(backend, T, grid3(T, 1)), s, grid, ws)
        @test_throws ArgumentError gradient_periodic!((φ, G[2], G[3]), φ, grid)
    end
    @test_throws ArgumentError SpectralWorkspace(backend, Float64, PeriodicGrid((4, 4, 4, 4), (1.0, 1.0, 1.0, 1.0), 1))
end

@testset "T-875 the 3D engine in Reference mode is bitwise across reruns, thread counts and launch shapes [A4, WP-042]" begin
    @test engine3d_digests() == INPROCESS_3D                             # rerun in process
    groups = by_case(INPROCESS_3D)
    @test length(groups) == length(TYPES) * (1 + 2 * length(SHAPES) * length(DIGEST_TRIPLES) + 2 + 2 + length(SOLVE_GRIDS_3D))
    @test all(v -> length(v) == length(WG_SIZES) && allequal(v), values(groups))   # every launch shape
    for nt in (1, 4)
        @test e3d_worker(nt) == INPROCESS_3D                             # 1 and 4 threads
    end
    # and the bits recorded on a commit (recorded_3d.jl), so a change to them fails here
    backend isa CPU && @test recorded_2d_lines(INPROCESS_3D) == E3D_RECORDED
end
