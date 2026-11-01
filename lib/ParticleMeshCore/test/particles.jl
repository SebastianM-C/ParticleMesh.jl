# Position advance and loading: T-138 and T-139.

"A species of `np` uncharged particles with offsets `k/16`, velocity `v` or `-v`, and an id in `u[2]`."
function streaming_species(::Type{T}, n, np, v) where {T}
    s = Species(backend, T, np + 5; q = 0, m = 1)
    for i in 1:np
        s.cell[1][i] = Int32(mod1(3i, n))
        s.offset[1][i] = seeded(T, mod(i, 16) / 16)
        s.u[1][i] = isodd(i) ? v : -v
        s.u[2][i] = T(i)
        s.status[i] = ParticleMeshCore.ALIVE
    end
    fill!(s.count, Int32(np))
    return s
end

position(s, i) = (s.cell[1][i] - 1) + s.offset[1][i]

@testset "T-138 free streaming at zero field and periodic wrap [A2, A7, A8, WP-010]" begin
    # Uncharged particles: every step sorts by cell (as a Reference step does) and advances each
    # particle by v·dt/dx with dt = dx = 1, so x(N) = x₀ + N·v exactly when the displacement is
    # dyadic.
    C = tolC("T-138")
    n, np, N = 8, 40, 400
    for T in TYPES
        grid = PeriodicGrid(n, one(basetype(T)), 2)
        for (label, v) in (("dyadic", seeded(T, 1 // 8)), ("general", seeded(T, 0.3)))
            s = streaming_species(T, n, np, v)
            x0 = Dict(Int(value_of(s.u[2][i])) => position(s, i) for i in 1:np)
            ws = Workspace(s, grid)
            for _ in 1:N
                sort_by_cell!(s, grid, ws)
                for i in 1:capacity(s)
                    s.status[i] == ParticleMeshCore.ALIVE || continue
                    s.cell[1][i], s.offset[1][i] = advance(s.cell[1][i], s.offset[1][i], s.u[1][i], Int32(n))
                end
            end
            @test alive_count(s) == np
            @test count(==(ParticleMeshCore.ALIVE), s.status) == np
            worst = 0.0
            for i in 1:np
                id = Int(value_of(s.u[2][i]))
                Δ = isodd(id) ? v : -v
                if label == "dyadic"
                    exact = x0[id] + N * Δ
                    exact -= n * floor(value_of(exact) / n)
                    @test components(position(s, i)) == components(exact)
                else
                    e = Float64(value_of(x0[id])) + N * Float64(value_of(Δ))
                    d = Float64(value_of(position(s, i))) - mod(e, n)
                    d -= n * round(d / n)
                    worst = max(worst, abs(d) / (base_eps(T) * N))
                end
            end
            label == "general" && (record!("T-138", T, worst); @test worst <= C)
        end
    end
end

@testset "T-139 loading: same seed bitwise, Philox known answers [A4, A8, WP-010]" begin
    # Known-answer vectors of Philox4x32-10 from the Random123 distribution, file
    # `tests/kat_vectors`, github.com/DEShawResearch/random123 at commit
    # 9545ff6413f258be2f04c1d319d99aaef7521150 (lines `philox4x32 10 ...`: counter, key, expected).
    kat = (
        ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000),
         (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
        ((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff), (0xffffffff, 0xffffffff),
         (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
        ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
         (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
    )
    for (ctr, key, expected) in kat
        @test philox4x32(ctr, key) == expected
    end
    # Same seed and stream: bitwise identical loads; another stream or seed: different draws.
    for T in TYPES
        grid = PeriodicGrid(16, basetype(T)(1 // 8), 2)
        p = (seeded(T, 1), seeded(T, 1 // 20), T(2π / length(grid)))
        vth = (T(1 // 20), T(1 // 20), T(1 // 20))
        load(seed, stream) = load_quiet!(Species(backend, T, 16 * 32; q = -1, m = 1), grid, CosineDensity(), p, 32;
                                         vth, seed, stream)
        a, b, c, d = load(4, 1), load(4, 1), load(4, 2), load(5, 1)
        for f in (s -> s.cell[1], s -> s.offset[1], s -> s.u[1], s -> s.u[2], s -> s.u[3], s -> s.w, s -> s.status)
            @test isequal(f(a), f(b))
        end
        @test a.u[1] != c.u[1] && a.u[1] != d.u[1]
        @test a.w == c.w                                   # weights do not depend on the draws
        @test alive_count(a) == 16 * 32
        # Weights are n(x_p)/ppc at the lattice positions: bitwise for Float32 and Float64; for
        # Dual{Float32} the kernel's value part may differ from a host evaluation in the last bit
        # (the Dual `cos` is inlined into the kernel, where Base's `muladd`s may be contracted to
        # FMAs; suspected, not verified at the LLVM level), so within 2 ulp there.
        hostw(i) = CosineDensity()((T(a.cell[1][i] - 1) + a.offset[1][i]) * cellsize(grid), p) / T(32)
        if T === D3
            @test all(i -> absdiff(a.w[i], hostw(i)) <= 2eps(Float32) * absmax(a.w[i]), 1:(16 * 32))
        else
            @test all(i -> a.w[i] == hostw(i), 1:(16 * 32))
        end
    end
    # The Gaussian draws have the right first two moments (a loose sanity check, not a tolerance).
    grid = PeriodicGrid(64, 1.0, 2)
    s = load_quiet!(Species(backend, Float64, 64 * 256; q = -1, m = 1), grid, CosineDensity(), (1.0, 0.0, 1.0), 256;
                    vth = (1.0, 1.0, 1.0), seed = 1, stream = 1)
    for d in 1:3
        m1 = sum(s.u[d]) / length(s.u[d])
        m2 = sum(abs2, s.u[d]) / length(s.u[d])
        @test abs(m1) < 5 / sqrt(length(s.u[d])) && abs(m2 - 1) < 5 * sqrt(2 / length(s.u[d]))
    end
end
