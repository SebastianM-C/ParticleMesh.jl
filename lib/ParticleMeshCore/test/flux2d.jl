# WP-020, the charge-conserving flux deposition at N = 2: T-300 to T-309 (computations in
# flux2d_cases.jl).

const INPROCESS_FLUX2D = flux2d_digests()
f2d_worker(nt) = [chopprefix(l, "F2D ") for l in worker_lines(nt) if startswith(l, "F2D ")]

@testset "T-300 discrete continuity at N = 2 [A2, A7, WP-020]" begin
    C = tolC("T-300")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
        c = folded_2d(continuity_case_2d(T, mode, shape, st))
        r = record!("T-300", T, continuity_residual_2d(c))
        @test r <= C
    end
    # The displacements cross faces on both axes, both seams and the corner.
    grid = grid2(Float64, 3)
    s = flux_species_2d(Float64, grid, 400)
    alive = [i for i in 1:400 if s.status[i] == ParticleMeshCore.ALIVE]
    crossed(d) = count(i -> s.cell[d][i] != (d == 1 ? s.extras.old_cell_1 : s.extras.old_cell_2)[i], alive)
    seam(d) = count(i -> abs(s.cell[d][i] - (d == 1 ? s.extras.old_cell_1 : s.extras.old_cell_2)[i]) > 1, alive)
    @test crossed(1) > 100 && crossed(2) > 100 && seam(1) > 5 && seam(2) > 5
    @test (s.cell[1][1], s.cell[2][1]) == (1, 1) && (s.extras.old_cell_1[1], s.extras.old_cell_2[1]) == (16, 12)
    @test (s.cell[1][2], s.cell[2][2]) == (16, 12) && (s.extras.old_cell_1[2], s.extras.old_cell_2[2]) == (1, 1)
    # The per-axis guard width: one more than guard_width except for order 2 at Center().
    @test flux_guard_width(Quadratic, (Node(), Center())) == (3, 2)
    @test flux_guard_width(Linear, (Center(), Node())) == (2, 2)
    @test flux_guard_width(Quadratic, (Node(), Node())) == (flux_guard_width(Quadratic, Node()), flux_guard_width(Quadratic, Node()))
    # Argument errors, before any launch: too few guards, missing extras, a quantity_y, a
    # Reference workspace without flux scratch (or with too little), fewer than 3 cells.
    st = (Node(), Node())
    s = flux_species_2d(Float64, grid2(Float64, 2), 50)
    J = fields3(Float64, grid2(Float64, 2))
    @test_throws ArgumentError deposit_flux!(Fast(), J..., s, Quadratic, st, grid2(Float64, 2), Workspace(s, grid2(Float64, 2));
                                             dt = 0.25, quantity_z = QZ2)
    g3 = grid2(Float64, 3)
    s = flux_species_2d(Float64, g3, 50)
    J = fields3(Float64, g3)
    bare = test_species_2d(Float64, g3, 50)
    @test_throws ArgumentError deposit_flux!(Fast(), J..., bare, Quadratic, st, g3, Workspace(bare, g3); dt = 0.25, quantity_z = QZ2)
    @test_throws ArgumentError deposit_flux!(Fast(), J..., s, Quadratic, st, g3, Workspace(s, g3); dt = 0.25, quantity_y = QZ2,
                                             quantity_z = QZ2)
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, g3, Workspace(s, g3); dt = 0.25, quantity_z = QZ2)
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, g3, Workspace(s, g3; flux = (Linear, st));
                                             dt = 0.25, quantity_z = QZ2)
    @test deposit_flux!(Fast(), J..., s, Quadratic, st, g3, Workspace(s, g3); dt = 0.25, quantity_z = QZ2) isa Tuple
    small = PeriodicGrid((16, 2), (0.25, 0.375), 1)
    s = flux_species_2d(Float64, small, 20)
    J = fields3(Float64, small)
    @test_throws ArgumentError deposit_flux!(Fast(), J..., s, Linear, st, small, Workspace(s, small); dt = 0.25, quantity_z = QZ2)
end

@testset "T-301 the out-of-plane weights; the 1D transverse average [A2, WP-020]" begin
    C = tolC("T-301")
    # A particle moving only in z deposits Jz = quantity × its tensor-product weights, exactly,
    # and no Jx, Jy; one particle per run, so both modes add to zero.
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES, δ in ((0.3, 0.8), (0.0, 0.55), (0.97, 0.5))
        grid = grid2(T, flux_g2(shape, st))
        s = Species(backend, T, 2; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
        s.cell[1][1], s.cell[2][1] = Int32(5), Int32(7)
        s.offset[1][1], s.offset[2][1] = seeded(T, δ[1]), seeded(T, δ[2])
        s.w[1], s.u[2][1], s.status[1] = seeded(T, 0.8), seeded(T, -0.6), ParticleMeshCore.ALIVE
        fill!(s.count, Int32(1))
        store_old_2d!(s)
        Jx, Jy, Jz, _ = flux2d(mode, s, shape, st, grid)
        (f1, f2), W = tensor_weights(shape, st, (s.offset[1][1], s.offset[2][1]))
        expect = zero(Jz)
        P1 = order(shape) + 1
        for k in eachindex(W)
            expect[5 + f1 + mod1(k, P1) - 1 + guards(grid), 7 + f2 + (k - 1) ÷ P1 + guards(grid)] = QZ2(s, 1) * W[k]
        end
        @test all(a -> components(Jz[a]) == components(expect[a]), eachindex(Jz))
        @test all(iszero, Jx) && all(iszero, Jy)
    end
    # Moving in x, y and z: Σ Jz = q·w·u (the quantity), to roundoff, for every particle of a moved species.
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS
        c = folded_2d(continuity_case_2d(T, Reference(), shape, st))
        alive = [i for i in 1:capacity(c.s) if c.s.status[i] == ParticleMeshCore.ALIVE]
        expect = sum(QZ2(c.s, i) for i in alive)
        r = record!("T-301", T, absdiff(sum(c.Jz), expect) / (base_eps(T) * length(alive) * maximum(i -> absmax(QZ2(c.s, i)), alive)))
        @test r <= C
    end
    # One particle crossing a face on each axis, per node against Esirkepov's weights evaluated by
    # hand (PIConGPU's form of W_z, with the ⅓ ΔSx·ΔSy term), Float64, in both modes.
    Ch = tolC("T-301", "C_hand")
    for shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
        grid = grid2(Float64, flux_g2(shape, st))
        g = guards(grid)
        s = Species(backend, Float64, 1; q = -1, m = 1, N = 2, extras = flux_extras(Float64, 2))
        c0, δ0, Δ = (6, 5), (0.7, 0.2), (0.55, -0.45)
        s.extras.old_cell_1[1], s.extras.old_cell_2[1] = Int32.(c0)
        s.extras.old_offset_1[1], s.extras.old_offset_2[1] = δ0
        c, δ = advance(Int32.(c0), δ0, Δ, Int32.(ncells(grid)))
        @test c == (7, 4)
        s.cell[1][1], s.cell[2][1] = c
        s.offset[1][1], s.offset[2][1] = δ
        s.w[1], s.u[2][1], s.status[1] = 0.8, -0.6, ParticleMeshCore.ALIVE
        fill!(s.count, Int32(1))
        Jx, Jy, Jz, _ = flux2d(mode, s, shape, st, grid; dt = 0.25)
        sx = cellsize(grid) ./ 0.25
        nodes = (1:12, 0:9)
        X0, X1 = c0 .+ δ0, c .+ δ
        hx, hy, hz = hand_flux_2d(shape, st, X0, X1, s.q * s.w[1], QZ2(s, 1), sx, nodes)
        worst = 0.0
        for j2 in nodes[2], j1 in nodes[1]
            a = (j1 + g, j2 + g)
            for (J, H, sc) in ((Jx, hx, abs(s.q * s.w[1] * sx[1])), (Jy, hy, abs(s.q * s.w[1] * sx[2])), (Jz, hz, abs(QZ2(s, 1))))
                worst = max(worst, abs(J[a...] - H[(j1, j2)]) / (eps() * sc))
            end
        end
        @test sum(Jz) ≈ QZ2(s, 1)
        @test any(k -> abs(hz[k] - 0.5 * (hand_weight(shape, st[1], X0[1], k[1]) * hand_weight(shape, st[2], X0[2], k[2]) +
                                          hand_weight(shape, st[1], X1[1], k[1]) * hand_weight(shape, st[2], X1[2], k[2])) * QZ2(s, 1)) >
                         1e-3 * abs(QZ2(s, 1)), keys(hz))           # W_z is not the average of the old and new products
        record!("T-301 hand", Float64, worst)
        @test worst <= Ch
    end
    # N = 1: a particle crossing a face deposits Jy and Jz equal to quantity·(W_old + W_new)/2 on
    # every node of its flux stencil, against the B-splines evaluated by hand.
    C1 = tolC("T-301", "C_1d")
    for shape in SHAPES, stagger in STAGGERS, mode in MODES, (c0, δ0, Δ) in ((4, 0.8, 0.45), (5, 0.1, -0.35), (4, 0.45, 0.6))
        grid = PeriodicGrid(10, 0.5, flux_guard_width(shape, stagger))
        g = guards(grid)
        s = Species(backend, Float64, 1; q = -1, m = 1, extras = old_extras(Float64))
        s.extras.old_cell[1], s.extras.old_offset[1] = Int32(c0), δ0
        c, δ = ParticleMeshCore.advance(Int32(c0), δ0, Δ, Int32(10))
        s.cell[1][1], s.offset[1][1] = c, δ
        s.w[1], s.u[2][1], s.u[3][1], s.status[1] = 0.7, 0.4, -0.9, ParticleMeshCore.ALIVE
        fill!(s.count, Int32(1))
        ws = Workspace(s, grid)
        sort_by_cell!(s, grid, ws)
        Jx, Jy, Jz = field(backend, Float64, grid), field(backend, Float64, grid), field(backend, Float64, grid)
        deposit_flux!(mode, Jx, Jy, Jz, s, shape, stagger, grid, ws; dt = 0.25, quantity_y = ChargeUy(), quantity_z = ChargeUz())
        worst, touched = 0.0, 0
        for j in -1:12
            W = (hand_weight(shape, stagger, c0 + δ0, j) + hand_weight(shape, stagger, c + δ, j)) / 2
            for (J, q) in ((Jy, ChargeUy()(s, 1)), (Jz, ChargeUz()(s, 1)))
                worst = max(worst, abs(J[j + g] - q * W) / (eps() * abs(q)))
            end
            touched += W > 0
        end
        @test c != c0 && touched >= order(shape) + 1
        record!("T-301 1D average", Float64, worst)
        @test worst <= C1
    end
end

@testset "T-302 Reference and Fast agree per node at N = 2 [A4, WP-020]" begin
    C = tolC("T-302")
    for T in (Float32, Float64), shape in SHAPES, st in STAGGER_PAIRS
        cR = continuity_case_2d(T, Reference(), shape, st; np = 192 * 20, seed = 11)
        cF = continuity_case_2d(T, Fast(), shape, st; np = 192 * 20, seed = 11)
        A = flux_abs_2d(cR.s, shape, st, cR.grid, cR.dt)
        worst = 0.0
        for (d, (JR, JF)) in enumerate(((cR.Jx, cF.Jx), (cR.Jy, cF.Jy), (cR.Jz, cF.Jz))), a in eachindex(JR)
            A[d][a] == 0 && (@test JR[a] == 0 && JF[a] == 0; continue)
            worst = max(worst, abs(JR[a] - JF[a]) / (eps(T) * A[d][a]))
        end
        record!("T-302", T, worst)
        @test worst <= C
    end
end

@testset "T-303 the 2D Reference flux is bitwise across reruns, threads and launch shapes [A4, WP-020]" begin
    groups = Dict{String,Vector{String}}()
    for l in INPROCESS_FLUX2D
        f = split(l, '|')
        push!(get!(groups, join(f[1:3], '|'), String[]), f[5])
    end
    @test length(groups) == length(TYPES) * length(SHAPES) * length(FLUX2D_DIGEST_PAIRS)
    @test all(v -> length(v) == length(WG_SIZES) && allequal(v), values(groups))
    @test flux2d_digests() == INPROCESS_FLUX2D
    for nt in (1, 4)
        @test f2d_worker(nt) == INPROCESS_FLUX2D
    end
end

@testset "T-304 the 2D flux reduces to 1D [A7, WP-020]" begin
    # A load constant along y moving along x only: the y-sum of Jx (and of Jz) is the 1D flux of
    # the same particles, and Jy is exactly zero.
    C = tolC("T-304")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
        ε = base_eps(T)
        grid = grid2(T, flux_g2(shape, st))
        n, g = ncells(grid), guards(grid)
        p = (seeded(T, 1), T(1 // 10), T(2π / (n[1] * cellsize(grid)[1])))
        s2 = load_quiet!(Species(backend, T, prod(n) * prod(PPC2); q = -1, m = 1, N = 2, extras = flux_extras(T, 2)), grid, XDensity(), p,
                         PPC2; seed = 1, stream = 1)
        np = capacity(s2)
        store_old_2d!(s2)
        nn = Int32.(n)
        for i in 1:np
            s2.u[2][i] = seeded(T, cos(1.3i))                  # the Jz quantity q·w·u₂ (the load has no thermal spread)
            c, δ = advance((s2.cell[1][i], s2.cell[2][i]), (s2.offset[1][i], s2.offset[2][i]), (seeded(T, 0.85 * sin(0.37i)), zero(T)), nn)
            s2.cell[1][i], s2.cell[2][i] = c
            s2.offset[1][i], s2.offset[2][i] = δ
        end
        g1 = PeriodicGrid(n[1], cellsize(grid)[1], g)
        s1 = Species(backend, T, np; q = -1, m = 1, extras = old_extras(T))
        copyto!(s1.cell[1], s2.cell[1]); copyto!(s1.offset[1], s2.offset[1]); copyto!(s1.w, s2.w); copyto!(s1.status, s2.status)
        copyto!(s1.extras.old_cell, s2.extras.old_cell_1); copyto!(s1.extras.old_offset, s2.extras.old_offset_1)
        copyto!(s1.u[2], s2.u[2]); copyto!(s1.u[3], s2.u[2])
        fill!(s1.count, Int32(np))
        Jx2, Jy2, Jz2, _ = flux2d(mode, s2, shape, st, grid)
        foreach(a -> fold_guards!(a, grid), (Jx2, Jy2, Jz2))
        ws1 = Workspace(s1, g1)
        effective_mode(mode, T) isa Reference && sort_by_cell!(s1, g1, ws1)
        Jx1, Jy1, Jz1 = field(backend, T, g1), field(backend, T, g1), field(backend, T, g1)
        deposit_flux!(mode, Jx1, Jy1, Jz1, s1, shape, st[1], g1, ws1; dt = T(1 // 4), quantity_y = ChargeUy(), quantity_z = ChargeUz())
        foreach(a -> fold_guards!(a, g1), (Jx1, Jz1))
        @test all(iszero, Jy2)
        for (J2, J1) in ((Jx2, Jx1), (Jz2, Jz1))
            ysum = vec(sum(interior_view(J2, grid); dims = 2))
            ref = interior_view(J1, g1)
            r = record!("T-304", T, maximum(absdiff.(ysum, ref)) / (ε * n[2] * maximum(absmax, ref)))
            @test r <= C
        end
    end
end

@testset "T-305 Dual weights: exact partials, the primal the Float32 run [A8, WP-020]" begin
    C = tolC("T-305")
    for shape in SHAPES, st in STAGGER_PAIRS
        grid = grid2(Float32, flux_g2(shape, st))
        s32 = flux_species_2d(Float32, grid, 400; seed = 4)
        makeD(wf, ::Type{E}) where {E} = begin
            sD = Species(backend, E, capacity(s32); q = -1, m = 1, N = 2, extras = flux_extras(E, 2))
            for i in 1:capacity(s32)
                for d in 1:2
                    sD.cell[d][i] = s32.cell[d][i]
                    sD.offset[d][i] = E(s32.offset[d][i])
                end
                sD.extras.old_cell_1[i], sD.extras.old_cell_2[i] = s32.extras.old_cell_1[i], s32.extras.old_cell_2[i]
                sD.extras.old_offset_1[i], sD.extras.old_offset_2[i] = E(s32.extras.old_offset_1[i]), E(s32.extras.old_offset_2[i])
                sD.u[2][i] = E(s32.u[2][i])
                sD.w[i] = wf(i)
                sD.status[i] = s32.status[i]
            end
            fill!(sD.count, Int32(400))
            sD
        end
        wdual(i) = mkdual(s32.w[i], cos(0.3i), s32.w[i]^2, 0)
        JD = flux2d(Reference(), makeD(wdual, D3), shape, st, grid)[1:3]
        J32 = flux2d(Reference(), makeD(i -> s32.w[i], Float32), shape, st, grid)[1:3]
        primal_bitwise = all(same_bits(value.(a), b) for (a, b) in zip(JD, J32))
        MEASURED["T-305 primal bitwise"] = merge(get(MEASURED, "T-305 primal bitwise", Dict{String,Float64}()),
                                                 Dict("$(order(shape)),$(stagger_label(st))" => Float64(primal_bitwise)))
        @test primal_bitwise
        g64 = PeriodicGrid(N2, Float64.(dx2(Float32)), flux_g2(shape, st))
        for dir in 1:3
            J64 = flux2d(Reference(), makeD(i -> Float64(partials(wdual(i))[dir]), Float64), shape, st, g64; dt = 0.25)[1:3]
            for (JDd, J64d) in zip(JD, J64)
                ∂ = [partials(x)[dir] for x in JDd]
                if dir == 3
                    @test all(iszero, ∂)
                else
                    r = record!("T-305", D3, maximum(abs.(∂ .- J64d)) / (eps(Float32) * maximum(abs, J64d)))
                    @test r <= C
                end
            end
        end
    end
end

@testset "T-306 allocation of the 2D flux does not grow with particle count or grid size [A7, A9, WP-020]" begin
    for T in (Float32, Float64), mode in MODES, st in ((Node(), Node()), (Center(), Center()))
        bytes = Dict{Tuple{Tuple{Int,Int},Int},Int}()
        for (n, np) in (((16, 12), 500), ((16, 12), 8000), ((32, 24), 500))
            grid = grid2(T, flux_g2(Quadratic, st); n)
            s = flux_species_2d(T, grid, np; dead = ())
            ws = Workspace(s, grid; flux = (Quadratic, st))
            sort_by_cell!(s, grid, ws)
            J = fields3(T, grid)
            run() = deposit_flux!(mode, J..., s, Quadratic, st, grid, ws; dt = T(1 // 4), quantity_z = QZ2)
            run(); run()
            bytes[(n, np)] = @allocated run()
        end
        @info "T-306 bytes per 2D deposit_flux!" T mode st bytes
        @test bytes[((16, 12), 8000)] <= bytes[((16, 12), 500)]
        @test bytes[((32, 24), 500)] <= bytes[((16, 12), 500)]
    end
end

@testset "T-307 the 2D flux of order 2 is C¹ through a crossing on each axis and the corner [A2, A7, WP-020]" begin
    # A smooth functional of Jx, Jy, Jz against one displacement along axis 1, axis 2 and the
    # diagonal, ForwardDiff against fd_scan (Float64, Reference, one seed), end points on both
    # sides of a face (or the corner) as close as 1e-7. Order 2 is asserted; order 1 at its
    # natural pair is recorded only (C⁰ at a node).
    hs = 10.0 .^ (-1:-0.5:-9)
    Δ0 = 0.3
    cases = vcat([(Quadratic, st) for st in STAGGER_PAIRS], [(Linear, (Node(), Node()))])
    for (shape, st) in cases, dir in ((1, 0), (0, 1), (1, 1))
        f(x, mode, seed) = flux_functional_2d(x, mode, seed; shape, st, dir)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 7), Δ0)
        r = fd_scan(f, Δ0, hs; seed = 7, ad)
        label = "order $(order(shape)), $(stagger_label(st)), dir $(dir)"
        MEASURED["T-307 rel/floor"] = merge(get(MEASURED, "T-307 rel/floor", Dict{String,Float64}()), Dict(label => r.rel[r.istar] / r.floor_rel))
        MEASURED["T-307 floor_rel"] = merge(get(MEASURED, "T-307 floor_rel", Dict{String,Float64}()), Dict(label => r.floor_rel))
        @test isfinite(ad) && ad != 0
        if shape === Quadratic
            @test r.rel[r.istar] <= tolC("T-307") * r.floor_rel
            @test r.floor_rel <= tolC("T-307", "floor_max")
        end
    end
end

@testset "T-308 a particle at rest deposits exactly zero Jx and Jy at N = 2 [A7, WP-020]" begin
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
        grid = grid2(T, flux_g2(shape, st))
        s = flux_species_2d(T, grid, 400; seed = 2, maxstep = 0)
        Jx, Jy, Jz, _ = flux2d(mode, s, shape, st, grid)
        @test all(iszero, Jx) && all(iszero, Jy)
        @test any(!iszero, Jz)
    end
end

@testset "T-309 the sums of the 2D flux [A7, WP-020]" begin
    # Σ Jx = Σ q·w·Δ₁·Δx₁/Δt and Σ Jy = Σ q·w·Δ₂·Δx₂/Δt, Δ_d the displacement in cells (the first
    # moment of the shape per axis; densities per node, as deposit!), and Σ Jz = Σ quantity_z.
    C = tolC("T-309")
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
        ε = base_eps(T)
        c = folded_2d(continuity_case_2d(T, mode, shape, st))
        (; s, grid, dt) = c
        n = ncells(grid)
        alive = [i for i in 1:capacity(s) if s.status[i] == ParticleMeshCore.ALIVE]
        np = length(alive)
        old = ((s.extras.old_cell_1, s.extras.old_offset_1), (s.extras.old_cell_2, s.extras.old_offset_2))
        disp(i, d) = begin
            e = mod(old[d][1][i] - s.cell[d][i] + 1, n[d]) - 1
            s.offset[d][i] - (e + old[d][2][i])
        end
        for (d, J) in ((1, c.Jx), (2, c.Jy))
            sx = cellsize(grid)[d] / dt
            expect = sum(s.q * s.w[i] * disp(i, d) * sx for i in alive)
            scale = np * maximum(i -> absmax(s.q * s.w[i] * sx), alive)
            @test record!("T-309", T, absdiff(sum(J), expect) / (ε * scale)) <= C
        end
        expect = sum(QZ2(s, i) for i in alive)
        @test record!("T-309", T, absdiff(sum(c.Jz), expect) / (ε * np * maximum(i -> absmax(QZ2(s, i)), alive))) <= C
    end
end
