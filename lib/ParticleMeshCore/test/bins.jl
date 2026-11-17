# WP-041, the particles binned by the first node of their stencil (`NodeBins`): T-860 to T-864.

"The bins of a shape and stagger (one per axis): the first node of the stencil."
node_bins(shape, st::ParticleMeshCore.Stagger) = NodeBins(shape, st)
node_bins(shape, st::Tuple) = NodeBins(shape, st)

# Per node of the folded fields, |bins - cells| against the folded per-node scale `A` (zero only where
# both are zero): the bins' frames put a seam's contributions on the other side's guards, so only
# the folded fields compare.
function bins_vs_cells(b, c, A, T)
    worst = 0.0
    for (d, k) in enumerate((:Jx, :Jy, :Jz))
        Jb, Jc = getfield(b, k), getfield(c, k)
        worst = max(worst, maximum(a -> A[d][a] == 0 ? (Jb[a] == Jc[a] == 0 ? 0.0 : Inf) :
                                         absdiff(Jb[a], Jc[a]) / (base_eps(T) * A[d][a]), eachindex(Jb)))
    end
    return max(worst, maximum(absdiff.(b.ρnew, c.ρnew)) / (base_eps(T) * maximum(absmax, c.ρnew)))
end

@testset "T-860 the Reference deposits by nearest-node bins conserve charge and agree with the cells' to roundoff, N = 1 and 2, every type, order and stagger [A2, A4, WP-041]" begin
    C, Cc = tolC("T-860"), tolC("T-860", "C_cells")
    for T in TYPES, shape in SHAPES, st in (Node(), Center())
        c = continuity_case(T, Reference(), shape, st)
        A = map(a -> (fold_guards!(a, c.grid); a), flux_abs(c, shape, st))
        b = folded(continuity_case(T, Reference(), shape, st; bins = node_bins(shape, st)))
        @test record!("T-860 1D", T, continuity_residual(b)) <= C
        @test record!("T-860 1D cells", T, bins_vs_cells(b, folded(c), A, T)) <= Cc
    end
    for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS
        c = continuity_case_2d(T, Reference(), shape, st)
        A = map(a -> (fold_guards!(a, c.grid); a), flux_abs_2d(c.s, shape, st, c.grid, c.dt))
        b = folded_2d(continuity_case_2d(T, Reference(), shape, st; bins = node_bins(shape, st)))
        @test record!("T-860", T, continuity_residual_2d(b)) <= C
        @test record!("T-860 cells", T, bins_vs_cells(b, folded_2d(c), A, T)) <= Cc
    end
end

@testset "T-861 a nearest-node bin holds the particles whose stencil starts at its first node, wraps at the seam, and shrinks the flux patch [A4, WP-041]" begin
    n = Int32.((16, 12))
    for T in (Float32, Float64), shape in SHAPES, st in (Node(), Center())
        b = NodeBins(shape, (st, st))
        lo, hi = ParticleMeshCore.deposit_reach(b, shape, st, 1)
        @test hi - lo == ParticleMeshCore.order(shape)                     # P + 1 nodes per bin
        δs = T[0, nextfloat(T(0)), T(1 // 4), prevfloat(T(1 // 2)), T(1 // 2), T(3 // 4), prevfloat(T(1))]
        moving = (shape, st) in ((Quadratic, Node()), (Linear, Center()))  # the pairs whose stencil moves within a cell
        for δ in δs
            h = ParticleMeshCore.bin_shift(b, 1, δ)
            f, _ = shape_weights(shape, st, δ)
            @test f - h == lo                                               # the stencil starts at the bin's first node
            @test h in (moving ? (0, 1) : (0,))
        end
        moving && @test ParticleMeshCore.bin_shift(b, 1, T(1 // 4)) == 0 && ParticleMeshCore.bin_shift(b, 1, T(3 // 4)) == 1
        # the bin of the last cell's particle past its stencil's midpoint is the first cell's
        s = Species(backend, T, 2; q = -1, m = 1, N = 2)
        s.cell[1] .= Int32[16, 3]; s.cell[2] .= Int32[12, 5]
        s.offset[1] .= T[3 // 4, 1 // 4]; s.offset[2] .= T[3 // 4, 1 // 4]
        @test ParticleMeshCore.particle_bin(b, s, 1, n) == (moving ? (Int32(1), Int32(1)) : (Int32(16), Int32(12)))
        @test ParticleMeshCore.particle_bin(b, s, 2, n) == (Int32(3), Int32(5))
        @test ParticleMeshCore.particle_bin(CellBins(), s, 1, n) == (Int32(16), Int32(12))
    end
    # the flux patch: 3·5·5 per bin at order 2 on nodes where a cell's is 3·6·6, 3·4·4 at order 1 at
    # centres where a cell's is 3·5·5; the other two pairs keep theirs
    PV = ParticleMeshCore.flux_patch_values
    @test PV(Quadratic, (Node(), Node()), NodeBins(Quadratic, (Node(), Node()))) == 75 && PV(Quadratic, (Node(), Node())) == 108
    @test PV(Linear, (Center(), Center()), NodeBins(Linear, (Center(), Center()))) == 48 && PV(Linear, (Center(), Center())) == 75
    @test PV(Quadratic, (Center(), Center()), NodeBins(Quadratic, (Center(), Center()))) == PV(Quadratic, (Center(), Center())) == 75
    @test PV(Linear, (Node(), Node()), NodeBins(Linear, (Node(), Node()))) == PV(Linear, (Node(), Node())) == 48
    # a workspace takes bins of its own number of axes only
    grid = grid2(Float64, 3)
    s = flux_species_2d(Float64, grid, 40)
    @test_throws ArgumentError Workspace(s, grid; bins = NodeBins(Quadratic, Node()))
    @test size(Workspace(s, grid; flux = (Quadratic, (Node(), Node())), bins = NodeBins(Quadratic, (Node(), Node()))).flux_patch, 1) == 75
end

@testset "T-862 the sort and the sort from the last order by nearest-node bins give the same bits, N = 1, 2, 3, every particle lies in its bin's segment, and junk in free slots changes nothing [A4, WP-041]" begin
    for T in TYPES, n in ((7,), (16, 12), (3, 4, 3)), (shape, st) in ((Quadratic, Node()), (Linear, Center()))
        same = resort_run(T, n; bins = NodeBins(shape, ntuple(_ -> st, length(n))))
        @test length(same) == 7 && all(same)
    end
    grid = PeriodicGrid((16, 12), (1 / 2, 1 / 2), 1)
    s = resort_species(Float64, (16, 12), 2304, 1536)
    b = NodeBins(Quadratic, (Node(), Node()))
    ws = Workspace(s, grid; bins = b, append = 40)
    sort_by_cell!(s, grid, ws)
    resort_move!(s, (16, 12), ResortLCG(5))
    resort_append!(s, (16, 12), 40, ResortLCG(6); tag0 = 5000)
    resort_by_cell!(s, grid, ws)
    n = Int32.((16, 12))
    bin(i) = linear_cell(ParticleMeshCore.particle_bin(b, s, i, n), n)
    cell(i) = linear_cell((s.cell[1][i], s.cell[2][i]), n)
    alive = Int(s.count[1])
    @test all(i -> ws.seg_start[bin(i)] <= i <= ws.seg_stop[bin(i)], 1:alive)
    @test count(i -> bin(i) != cell(i), 1:alive) > alive ÷ 4                 # the bins are not the cells
    # a free slot's offset is not state: NaN and Inf there leave the sort's bits alone (the bin is read
    # for alive slots only), at N = 1 (its own key kernel) and N = 2
    for n in ((16,), (16, 12))
        N = length(n)
        g = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 1)
        bn = NodeBins(Quadratic, ntuple(_ -> Node(), N))
        x = resort_species(Float64, n, 12 * prod(n), 8 * prod(n))
        resort_move!(x, n, ResortLCG(7))
        y = deepcopy(x)
        free = findall(!=(ParticleMeshCore.ALIVE), y.status)
        @test length(free) > 4 * prod(n)
        for d in 1:N
            y.offset[d][free] .= ifelse.(isodd.(free), NaN, Inf)
        end
        wx, wy = Workspace(x, g; bins = bn), Workspace(y, g; bins = bn)
        sort_by_cell!(x, g, wx)
        sort_by_cell!(y, g, wy)
        m = Int(x.count[1])
        @test y.count[1] == m && wx.perm[1:m] == wy.perm[1:m] && wx.seg_start == wy.seg_start && wx.seg_stop == wy.seg_stop
        @test all(d -> same_bits(x.offset[d][1:m], y.offset[d][1:m]) && x.cell[d][1:m] == y.cell[d][1:m], 1:N)
    end
end

@testset "T-863 the Reference deposits by nearest-node bins are bitwise across reruns and launch shapes [A4, WP-041]" begin
    for T in TYPES, shape in SHAPES, st in FLUX2D_DIGEST_PAIRS
        b = node_bins(shape, st)
        runs = map((WG_SIZES..., nothing)) do wgs
            c = continuity_case_2d(T, Reference(), shape, st; bins = b, workgroupsize = wgs)
            (c.ρnew, c.Jx, c.Jy, c.Jz)
        end
        @test all(r -> all(same_bits.(r, runs[1])), runs)
    end
    for T in TYPES, shape in SHAPES, st in (Node(), Center())
        runs = map(wgs -> continuity_case(T, Reference(), shape, st; bins = node_bins(shape, st), workgroupsize = wgs), (WG_SIZES..., nothing))
        @test all(r -> all(same_bits.((r.ρnew, r.Jx, r.Jy, r.Jz), (runs[1].ρnew, runs[1].Jx, runs[1].Jy, runs[1].Jz))), runs)
    end
end

"The largest of ρ, Jx, Jy, Jz by `NodeBins(bshape, bst)` against by cells at N = 1, folded, in eps·max|field|."
function bins_vs_cells_1d(bshape, bst, shape, st; g = 3)
    grid = PeriodicGrid(10, 1 / 2, g)
    s = moved_species(Float64, grid, 400; seed = 3, dead = (40, 41, 300))
    out = map((CellBins(), NodeBins(bshape, bst))) do b
        x = deepcopy(s)
        ws = Workspace(x, grid; bins = b)
        sort_by_cell!(x, grid, ws)
        ρ = deposit!(Reference(), field(backend, Float64, grid), x, shape, st, grid, ws)
        J = (field(backend, Float64, grid), field(backend, Float64, grid), field(backend, Float64, grid))
        deposit_flux!(Reference(), J..., x, shape, st, grid, ws; dt = 0.25, quantity_y = ChargeUy(), quantity_z = ChargeUz())
        foreach(a -> fold_guards!(a, grid), (ρ, J...))
        (ρ, J...)
    end
    return maximum(k -> maximum(abs.(out[1][k] .- out[2][k])) / (eps(Float64) * maximum(abs, out[1][k])), 1:4)
end

@testset "T-864 bins of another stagger than the deposit's still conserve charge at N = 2, agree with the cells' at N = 1, and need one more guard cell, which is checked at N = 1 and 2 [A2, WP-041]" begin
    # bins of order 2 on nodes under a flux at centres: a bin holds the particles of two cells
    b = NodeBins(Quadratic, (Node(), Node()))
    st = (Center(), Center())
    grid = grid2(Float64, flux_g2(Quadratic, st))                        # 2 guards: the cells' need
    s = flux_species_2d(Float64, grid, 400)
    ws = Workspace(s, grid; flux = (Quadratic, st), bins = b)
    sort_by_cell!(s, grid, ws)
    J = fields3(Float64, grid)
    @test_throws "guard cells" deposit_flux!(Reference(), J..., s, Quadratic, st, grid, ws; dt = 0.25, quantity_z = QZ2)
    grid3 = grid2(Float64, 3)
    s = flux_species_2d(Float64, grid3, 400)
    ws = Workspace(s, grid3; flux = (Quadratic, st), bins = b)
    sort_by_cell!(s, grid3, ws)
    ρold = deposited_2d(Reference(), at_old_positions_2d(s), Quadratic, st, grid3)
    ρnew = deposit!(Reference(), field(backend, Float64, grid3), s, Quadratic, st, grid3, ws)
    Jx, Jy, Jz = fields3(Float64, grid3)
    deposit_flux!(Reference(), Jx, Jy, Jz, s, Quadratic, st, grid3, ws; dt = 0.25, quantity_z = QZ2)
    c = folded_2d((; ρold, ρnew, Jx, Jy, Jz, grid = grid3, s, ws, dt = 0.25))
    @test record!("T-864", Float64, continuity_residual_2d(c)) <= tolC("T-860")
    # the plain deposit refuses too few guard cells as well, at N = 2 and N = 1
    g2 = grid2(Float64, 1)
    s2 = flux_species_2d(Float64, g2, 400)
    w2 = Workspace(s2, g2; bins = b)
    sort_by_cell!(s2, g2, w2)
    @test_throws "guard cells" deposit!(Reference(), field(backend, Float64, g2), s2, Quadratic, st, g2, w2)
    g1 = PeriodicGrid(10, 1 / 2, 1)
    s1 = moved_species(Float64, g1, 100; seed = 3)
    w1 = Workspace(s1, g1; bins = NodeBins(Quadratic, Node()))
    sort_by_cell!(s1, g1, w1)
    @test_throws "guard cells" deposit!(Reference(), field(backend, Float64, g1), s1, Quadratic, Center(), g1, w1)
    # at N = 1, bins of another shape or stagger than the deposit's: ρ and the flux by bins against
    # by cells, folded, in eps·max|field| (a looser scale than T-860's per node), and too few guards refused
    for (bshape, bst, shape, dst) in ((Quadratic, Node(), Quadratic, Center()), (Linear, Center(), Linear, Node()),
                                      (Quadratic, Center(), Quadratic, Node()))
        @test bins_vs_cells_1d(bshape, bst, shape, dst) <= tolC("T-860")
    end
    @test_throws "guard cells" bins_vs_cells_1d(Quadratic, Node(), Quadratic, Center(); g = 2)
end
