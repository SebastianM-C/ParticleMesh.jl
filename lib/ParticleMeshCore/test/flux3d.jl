# WP-042, the charge-conserving flux deposition at N = 3: T-940 to T-949 (computations in
# flux3d_cases.jl; T-946, the GPU rerun, in gpu.jl).

foreach(start_worker, WORKER_THREADS)   # the thread-count workers (workers.jl) run beside this file's in-process runs

const INPROCESS_FLUX3D = flux3d_digests()
f3d_worker(nt) = [chopprefix(l, "F3D ") for l in worker_lines(nt) if startswith(l, "F3D ")]

"""
The bins of the 3D flux tests for `shape` and `st`: the cells, and for order 2 on ρ's nodes and
on (Node, Node, Center) also `NodeBins` of that shape and stagger (the second has another reach
on axis 3 than on the others).
"""
flux3d_bins(shape, st) = shape === Quadratic && st in (NODES3, (Node(), Node(), Center())) ? (CellBins(), NodeBins(Quadratic, st)) : (CellBins(),)

function body_T940(be)
    C = tolC("T-940")
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, mode in FLUX3D_MODES, bins in flux3d_bins(shape, st)
        c = folded_3d(continuity_case_3d(T, mode, shape, st; be, bins); be)
        @test record!("T-940", T, continuity_residual_3d(c), be) <= C
    end
    # bins of another stagger than the flux's (T-864 at N = 3): node bins of ρ's nodes under a flux
    # at the cell centres, with the one more guard cell they need
    CCC = (Center(), Center(), Center())
    for T in TYPES, mode in FLUX3D_MODES
        c = folded_3d(continuity_case_3d(T, mode, Quadratic, CCC; be, bins = NodeBins(Quadratic, NODES3), g = 3); be)
        @test record!("T-940", T, continuity_residual_3d(c), be) <= C
    end
end

@testset "T-940 discrete continuity at N = 3 [A2, A7, WP-042]" begin
    body_T940(CPU())
    # The displacements cross faces on every axis, every seam, and particle 8 the corner.
    grid = flux_grid3(Float64, Quadratic, NODES3)
    s = flux_species_3d(Float64, grid, 400)
    alive = [i for i in 1:400 if s.status[i] == ParticleMeshCore.ALIVE]
    crossed(d) = count(i -> s.cell[d][i] != old_position_3d(s, i)[1][d], alive)
    seam(d) = count(i -> abs(s.cell[d][i] - old_position_3d(s, i)[1][d]) > 1, alive)
    @test all(d -> crossed(d) > 100 && seam(d) > 5, 1:3)
    @test ntuple(d -> s.cell[d][8], 3) == (1, 1, 1) && old_position_3d(s, 8)[1] == N3F
    @test ntuple(d -> s.cell[d][1], 3) == N3F && old_position_3d(s, 1)[1] == (1, 1, 1)
    @test flux_guard_width(Quadratic, (Node(), Center(), Node())) == (3, 2, 3)
    @test flux_extras(Float32, 3) == (old_cell_1 = Int32, old_cell_2 = Int32, old_cell_3 = Int32, old_offset_1 = Float32,
                                      old_offset_2 = Float32, old_offset_3 = Float32)
    @test_throws ArgumentError flux_extras(Float32, 4)
    # Argument errors, before any launch: too few guards, missing extras, a quantity, a Reference
    # workspace without flux scratch, for another shape or stagger or with a pair of staggers, a
    # bad `planes`. (Fewer than 3 cells cannot be reached: the flux guards are at least 2 and a grid
    # has at least twice its guards per axis.)
    st = NODES3
    J = fields3(Float64, grid)
    w(s, g = grid; kw...) = Workspace(s, g; kw...)
    g2 = grid3(Float64, 2; n = N3F)
    s2 = flux_species_3d(Float64, g2, 50)
    @test_throws ArgumentError deposit_flux!(Fast(), fields3(Float64, g2)..., s2, Quadratic, st, g2, w(s2, g2); dt = 0.25)
    s = flux_species_3d(Float64, grid, 50)
    bare = test_species_3d(Float64, grid, 50)
    @test_throws ArgumentError deposit_flux!(Fast(), J..., bare, Quadratic, st, grid, w(bare); dt = 0.25)
    for k in keys(flux_extras(Float64, 3))           # any one extra missing
        part = test_species_3d(Float64, grid, 50; extras = Base.structdiff(flux_extras(Float64, 3), NamedTuple{(k,)}))
        @test_throws r"reads the old position" deposit_flux!(Fast(), J..., part, Quadratic, st, grid, w(part); dt = 0.25)
    end
    @test_throws ArgumentError deposit_flux!(Fast(), J..., s, Quadratic, st, grid, w(s); dt = 0.25, quantity_z = ChargeWeight())
    @test_throws ArgumentError deposit_flux!(Fast(), J..., s, Quadratic, st, grid, w(s); dt = 0.25, quantity_y = ChargeWeight())
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, grid, w(s); dt = 0.25)
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, grid, w(s; flux = (Linear, st)); dt = 0.25)
    # another stagger triple, with another patch size (150 values per cell against 180) and with
    # the same (150 for both (Node, Node, Center) and (Center, Node, Node))
    NNC, CNN = (Node(), Node(), Center()), (Center(), Node(), Node())
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, grid, w(s; flux = (Quadratic, NNC)); dt = 0.25)
    @test ParticleMeshCore.flux_patch_values(Quadratic, NNC) == ParticleMeshCore.flux_patch_values(Quadratic, CNN) == 150
    @test_throws r"workspace was built with" deposit_flux!(Reference(), J..., s, Quadratic, NNC, grid, w(s; flux = (Quadratic, CNN)); dt = 0.25)
    # node bins of ρ's nodes under a flux at the cell centres need one more guard than the flux's 2
    CCC = (Center(), Center(), Center())
    gC = flux_grid3(Float64, Quadratic, CCC)
    sC = flux_species_3d(Float64, gC, 50)
    @test guards(gC) == 2
    @test_throws r"with these bins needs 3 guard cells" deposit_flux!(Reference(), fields3(Float64, gC)..., sC, Quadratic, CCC, gC,
                                                                      w(sC, gC; flux = (Quadratic, CCC), bins = NodeBins(Quadratic, NODES3)); dt = 0.25)
    @test_throws ArgumentError w(s; flux = (Quadratic, (Node(), Node())))
    @test_throws ArgumentError deposit_flux!(Reference(), J..., s, Quadratic, st, grid, w(s; flux = (Quadratic, st)); dt = 0.25, planes = 0)
    @test deposit_flux!(Fast(), J..., s, Quadratic, st, grid, w(s); dt = 0.25) isa Tuple
    # The patch sizes: one component at a time, (M₁ - 1)·M₂·M₃ values per bin and cyclically.
    @test ParticleMeshCore.flux_patch_values(Quadratic, NODES3) == 5 * 6 * 6
    @test ParticleMeshCore.flux_patch_values(Quadratic, NODES3, NodeBins(Quadratic, NODES3)) == 4 * 5 * 5
    @test ParticleMeshCore.flux_patch_values(Linear, NODES3) == 3 * 4 * 4
    @test ParticleMeshCore.flux_patch_values(Quadratic, (Center(), Node(), Node())) == 5 * 5 * 6   # widths (5, 6, 6): Jx 4·6·6, Jy and Jz 5·5·6
    @test size(w(s; flux = (Quadratic, st)).flux_patch) == ParticleMeshCore.flux_patch_size(ParticleMeshCore.flux_patch_layout(backend), 180,
                                                                                                 prod(ncells(grid)))
end

function body_T941(be)
    C = tolC("T-941")
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, mode in FLUX3D_MODES
        ε = base_eps(T)
        c = folded_3d(continuity_case_3d(T, mode, shape, st; be); be)
        (; s, grid, dt) = c
        n = ncells(grid)
        alive = [i for i in 1:capacity(s) if s.status[i] == ParticleMeshCore.ALIVE]
        disp(i, d) = begin
            oc, oδ = old_position_3d(s, i)
            e = mod(oc[d] - s.cell[d][i] + 1, n[d]) - 1
            s.offset[d][i] - (e + oδ[d])
        end
        for (d, J) in enumerate((c.Jx, c.Jy, c.Jz))
            sx = cellsize(grid)[d] / dt
            expect = sum(s.q * s.w[i] * disp(i, d) * sx for i in alive)
            scale = length(alive) * maximum(i -> absmax(s.q * s.w[i] * sx), alive)
            @test record!("T-941", T, absdiff(sum(J), expect) / (ε * scale), be) <= C
        end
    end
    # A particle at rest deposits exactly zero, in every component.
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, mode in FLUX3D_MODES
        grid = flux_grid3(T, shape, st)
        s = flux_species_3d(T, grid, 400; seed = 2, maxstep = 0)
        J = flux3d(mode, s, shape, st, grid; be)[1:3]
        @test all(a -> all(iszero, a), J)
    end
end

@testset "T-941 the sums of the 3D flux; a particle at rest deposits exactly zero [A7, WP-042]" begin
    # Σ J_d = Σ q·w·Δ_d·Δx_d/Δt, Δ_d the displacement in cells (the first moment of the shape per axis).
    body_T941(CPU())
end

@testset "T-942 the 3D flux reduces to 2D without a move along z, and to 1D with a move along z only [A7, WP-042]" begin
    C = tolC("T-942")
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, mode in FLUX3D_MODES
        ε = base_eps(T)
        grid = flux_grid3(T, shape, st)
        n, g = ncells(grid), guards(grid)
        # No move along axis 3: Jz is exactly zero, and the axis-3 sums of Jx and Jy are the 2D flux
        # of the same particles on the first two axes.
        s3 = flux_species_3d(T, grid, 400; axes = (true, true, false))
        J3 = flux3d(mode, s3, shape, st, grid)[1:3]
        @test all(iszero, J3[3])
        foreach(a -> fold_guards!(a, grid), J3)
        g2 = PeriodicGrid(n[1:2], cellsize(grid)[1:2], g)
        s2 = Species(backend, T, capacity(s3); q = s3.q, m = 1, N = 2, extras = flux_extras(T, 2))
        for d in 1:2
            copyto!(s2.cell[d], s3.cell[d]); copyto!(s2.offset[d], s3.offset[d])
        end
        copyto!(s2.extras.old_cell_1, s3.extras.old_cell_1); copyto!(s2.extras.old_cell_2, s3.extras.old_cell_2)
        copyto!(s2.extras.old_offset_1, s3.extras.old_offset_1); copyto!(s2.extras.old_offset_2, s3.extras.old_offset_2)
        copyto!(s2.w, s3.w); copyto!(s2.status, s3.status); copyto!(s2.count, s3.count)
        J2 = flux2d(mode, s2, shape, st[1:2], g2)[1:2]
        foreach(a -> fold_guards!(a, g2), J2)
        for d in 1:2
            zsum = dropdims(sum(interior_view(J3[d], grid); dims = 3); dims = 3)
            ref = interior_view(J2[d], g2)
            @test record!("T-942", T, maximum(absdiff.(zsum, ref)) / (ε * n[3] * maximum(absmax, ref))) <= C
        end
        # A move along axis 3 only: Jx and Jy are exactly zero, and the sums of Jz over the first two
        # axes are the 1D flux of the same particles along axis 3.
        sz = flux_species_3d(T, grid, 400; axes = (false, false, true))
        Jz3 = flux3d(mode, sz, shape, st, grid)[1:3]
        @test all(iszero, Jz3[1]) && all(iszero, Jz3[2])
        fold_guards!(Jz3[3], grid)
        g1 = PeriodicGrid(n[3], cellsize(grid)[3], g)
        s1 = Species(backend, T, capacity(sz); q = sz.q, m = 1, extras = old_extras(T))
        copyto!(s1.cell[1], sz.cell[3]); copyto!(s1.offset[1], sz.offset[3])
        copyto!(s1.extras.old_cell, sz.extras.old_cell_3); copyto!(s1.extras.old_offset, sz.extras.old_offset_3)
        copyto!(s1.w, sz.w); copyto!(s1.status, sz.status); copyto!(s1.count, sz.count)
        ws1 = Workspace(s1, g1)
        effective_mode(mode, T) isa Reference && sort_by_cell!(s1, g1, ws1)
        J1 = (field(backend, T, g1), field(backend, T, g1), field(backend, T, g1))
        deposit_flux!(mode, J1..., s1, shape, st[3], g1, ws1; dt = T(1 // 4), quantity_y = ChargeWeight(), quantity_z = ChargeWeight())
        fold_guards!(J1[1], g1)
        xysum = vec(sum(interior_view(Jz3[3], grid); dims = (1, 2)))
        ref = interior_view(J1[1], g1)
        @test record!("T-942", T, maximum(absdiff.(xysum, ref)) / (ε * n[1] * n[2] * maximum(absmax, ref))) <= C
    end
end

@testset "T-943 the 3D flux of one particle against Esirkepov's weights by hand [A2, WP-042]" begin
    # One particle crossing a face on each axis (or none, or all by a hair, or landing on a face),
    # per node against Esirkepov's 3D weights evaluated by hand, Float64, in both modes and both
    # bins: the moves keep the stencil inside `nodes`.
    C = tolC("T-943")
    moves = (((5, 4, 3), (0.7, 0.2, 0.5), (0.55, -0.45, 0.3)), ((4, 3, 3), (0.1, 0.9, 0.95), (-0.8, 0.35, 0.2)),
             ((5, 4, 2), (0.999, 0.001, 0.5), (0.002, -0.003, 0.0)), ((4, 4, 3), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)),
             ((3, 4, 3), (0.3, 0.6, 0.2), (0.2, 0.1, -0.25)))
    for shape in SHAPES, st in FLUX3D_TRIPLES, mode in FLUX3D_MODES, bins in flux3d_bins(shape, st), (c0, δ0, Δ) in moves
        grid = flux_grid3(Float64, shape, st)
        g = guards(grid)
        s = Species(backend, Float64, 1; q = -1, m = 1, N = 3, extras = flux_extras(Float64, 3))
        s.extras.old_cell_1[1], s.extras.old_cell_2[1], s.extras.old_cell_3[1] = Int32.(c0)
        s.extras.old_offset_1[1], s.extras.old_offset_2[1], s.extras.old_offset_3[1] = δ0
        c, δ = advance(Int32.(c0), δ0, Δ, Int32.(ncells(grid)))
        @test all(c .+ δ .≈ c0 .+ δ0 .+ Δ)
        for d in 1:3
            s.cell[d][1], s.offset[d][1] = c[d], δ[d]
        end
        s.w[1], s.status[1] = 0.8, ParticleMeshCore.ALIVE
        fill!(s.count, Int32(1))
        J = flux3d(mode, s, shape, st, grid; dt = 0.25, bins)[1:3]
        sx = cellsize(grid) ./ 0.25
        nodes = (0:9, 0:8, 0:7)
        H = hand_flux_3d(shape, st, c0 .+ δ0, c .+ δ, s.q * s.w[1], sx, nodes)
        worst = 0.0
        for j3 in nodes[3], j2 in nodes[2], j1 in nodes[1], d in 1:3
            worst = max(worst, abs(J[d][j1 + g, j2 + g, j3 + g] - H[d][(j1, j2, j3)]) / (eps() * abs(s.q * s.w[1] * sx[d])))
        end
        @test record!("T-943", Float64, worst) <= C
    end
end

function body_T944(be)
    C = tolC("T-944")
    for T in (Float32, Float64), shape in SHAPES, st in FLUX3D_TRIPLES, bins in flux3d_bins(shape, st)
        cR = continuity_case_3d(T, Reference(), shape, st; np = prod(N3F) * 8, seed = 11, bins, be)
        cF = continuity_case_3d(T, Fast(), shape, st; np = prod(N3F) * 8, seed = 11, bins, be)
        A = flux_abs_3d(cR.s, shape, st, cR.grid, cR.dt, bins)
        # Both add into the fields they are given: into a pattern, Reference() gives the pattern
        # plus its current into zeros bit for bit (its sum is added once per node); Fast() differs
        # from the pattern plus its current into zeros by at most the roundoff of its k atomic adds
        # on the node, in either run, each within eps/2 of |pattern| + A there (`flux_hits_3d`).
        # Unfolded, where Fast()'s nodes are the particles' own (those of `A`).
        J0 = pattern_fields_3d(T, cR.grid, map(J -> maximum(abs, J), (cR.Jx, cR.Jy, cR.Jz)))
        pR = continuity_case_3d(T, Reference(), shape, st; np = prod(N3F) * 8, seed = 11, bins, be, J0)
        pF = continuity_case_3d(T, Fast(), shape, st; np = prod(N3F) * 8, seed = 11, bins, be, J0)
        @test all(same_bits(p, J0[d] .+ J) for (d, (p, J)) in enumerate(((pR.Jx, cR.Jx), (pR.Jy, cR.Jy), (pR.Jz, cR.Jz))))
        k = flux_hits_3d(cF.s, shape, st, cF.grid, cF.dt)
        added = 0.0
        for (d, (p, J)) in enumerate(((pF.Jx, cF.Jx), (pF.Jy, cF.Jy), (pF.Jz, cF.Jz))), a in eachindex(p)
            r = abs(Float64(p[a]) - Float64(J0[d][a]) - Float64(J[a]))
            added = max(added, r / (eps(T) * (k[d][a] + 1) * (abs(J0[d][a]) + A[d][a])))
        end
        @test record!("T-944 added", T, added, be) <= tolC("T-944", "C_added")
        # compared folded: a node bin wraps periodically (a particle near an axis' last node is in
        # its first bin), so Reference() writes such a particle's values into the other periodic
        # image of the guards than Fast(), and the two agree once the guards are folded
        foreach(a -> fold_guards!(a, cR.grid), (cR.Jx, cR.Jy, cR.Jz, cF.Jx, cF.Jy, cF.Jz, A...))
        worst, zeros_ok = 0.0, true
        for (d, (JR, JF)) in enumerate(((cR.Jx, cF.Jx), (cR.Jy, cF.Jy), (cR.Jz, cF.Jz))), a in eachindex(JR)
            A[d][a] == 0 && (zeros_ok &= JR[a] == 0 && JF[a] == 0; continue)
            worst = max(worst, abs(JR[a] - JF[a]) / (eps(T) * A[d][a]))
        end
        @test zeros_ok
        @test record!("T-944", T, worst, be) <= C
    end
end

@testset "T-944 Reference and Fast agree per node at N = 3 and add into the fields they are given; a particle's values are the same bits in its own frame and its bin's [A4, WP-042]" begin
    body_T944(CPU())
    # Fast()'s particle frame against the bin frame of Reference(): on the nodes both hold the same
    # bits; the bin frame's other nodes hold zeros, and along the component's axis past the
    # particle's frame the roundoff prefix sum of all weight changes times the transverse factor.
    Ct = tolC("T-944", "C_tail")
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, bins in flux3d_bins(shape, st)
        P = order(shape)
        grid = flux_grid3(T, shape, st)
        s = flux_species_3d(T, grid, 400)
        n = Int32.(ncells(grid))
        sx = map(h -> h / T(basetype(T)(1 // 4)), cellsize(grid))
        shared, zeros_ok, frames_ok, tails, worst = true, true, true, 0, 0.0
        for i in 1:capacity(s)
            s.status[i] == ParticleMeshCore.ALIVE || continue
            VB = particle_values_3d(s, i, shape, st, n, sx, bin_frame(bins))
            VP = particle_values_3d(s, i, shape, st, n, sx, ParticleMeshCore.ParticleFrame())
            for C in 1:3
                lo, D, v = VB[C]
                lp, Dp, vp = VP[C]
                frames_ok &= Dp == ntuple(d -> P + 2 - (d == C), 3)
                frames_ok &= all(d -> lo[d] <= lp[d] && lp[d] + Dp[d] <= lo[d] + D[d] + (d == C), 1:3)
                scale = absmax(s.q * s.w[i] * sx[C])
                for (k, K) in enumerate(CartesianIndices(D))
                    m = ntuple(d -> lo[d] + K[d] - lp[d], 3)              # the node in the particle's frame, from 1
                    if all(d -> 1 <= m[d] <= Dp[d], 1:3)
                        shared &= same_bits([v[k]], [vp[LinearIndices(Dp)[m...]]])
                    elseif m[C] > Dp[C] && all(d -> d == C || 1 <= m[d] <= Dp[d], 1:3)
                        tails += 1
                        worst = max(worst, absmax(v[k]) / (base_eps(T) * scale))
                    else
                        zeros_ok &= iszero(v[k])
                    end
                end
            end
        end
        @test frames_ok
        @test shared
        @test zeros_ok
        @test tails > 0
        @test record!("T-944 tail", T, worst) <= Ct
    end
end

@testset "T-945 the 3D Reference flux is bitwise across reruns, threads, launch shapes and planes per launch, sums in the stated order, and equals recorded bits [A4, WP-042]" begin
    groups = by_case(INPROCESS_FLUX3D)
    @test length(groups) == length(TYPES) * (length(SHAPES) * length(FLUX3D_DIGEST_TRIPLES) + 1)
    @test all(v -> length(v) == length(WG_SIZES) && allequal(v), values(groups))
    @test flux3d_digests(; wgs = (nothing,)) == filter(l -> split(l, '|')[4] == "nothing", INPROCESS_FLUX3D)   # rerun in process
    for nt in (1, 4)
        @test f3d_worker(nt) == INPROCESS_FLUX3D
    end
    # The planes of the frame's third axis per launch of the patch kernel: one, two, all, and the
    # default (two of five for the Dual by NodeBins, all for the floats); the patch and the current
    # are the same bits.
    for T in TYPES, (shape, st, bins) in ((Quadratic, NODES3, NodeBins(Quadratic, NODES3)), (Quadratic, (Center(), Node(), Center()), CellBins()),
                                         (Linear, NODES3, CellBins()))
        runs = [continuity_case_3d(T, Reference(), shape, st; bins, planes) for planes in (nothing, 1, 2, 6)]
        @test all(r -> same_bits(r.Jx, runs[1].Jx) && same_bits(r.Jy, runs[1].Jy) && same_bits(r.Jz, runs[1].Jz), runs)
        @test all(r -> same_bits(r.ws.flux_patch, runs[1].ws.flux_patch), runs)
    end
    # The order of the sums, which the recorded bits pin only on one Julia version and the CPU:
    # per bin its particles' values in slot order, per node the bins in increasing index.
    for T in TYPES, shape in SHAPES, st in FLUX3D_TRIPLES, bins in flux3d_bins(shape, st)
        c = continuity_case_3d(T, Reference(), shape, st; bins)
        @test same_bits.(reference_by_hand_3d(c, shape, st, bins), (c.Jx, c.Jy, c.Jz)) == (true, true, true)
    end
    M = ParticleMeshCore.flux_widths(Quadratic, NODES3, NodeBins(Quadratic, NODES3))
    @test M == (5, 5, 5)
    @test ParticleMeshCore.flux_planes(Float32, 20, 5, nothing) == 5 && ParticleMeshCore.flux_planes(Float64, 20, 5, nothing) == 5
    @test ParticleMeshCore.flux_planes(D3, 20, 5, nothing) == 2 && ParticleMeshCore.flux_planes(D3, 25, 4, nothing) == 2
    # the bits recorded on a commit (recorded_3d.jl)
    backend isa CPU && @test_recorded recorded_2d_lines(INPROCESS_FLUX3D) == F3D_RECORDED
end

@testset "T-947 the 3D flux of order 2 is C¹ through a crossing on an axis, an edge and the corner [A2, A7, WP-042]" begin
    # A smooth functional of Jx, Jy, Jz against one displacement along axis 1, axis 3, the
    # diagonal of axes 1 and 2 and the diagonal of the box, ForwardDiff against fd_scan (Float64,
    # Reference, one seed), end points on both sides of a face (or an edge, or the corner) as close
    # as 1e-7. Order 2 is asserted; order 1 at its natural triple is recorded only (C⁰ at a node).
    hs = 10.0 .^ (-1:-0.5:-9)
    Δ0 = 0.3
    cases = ((Quadratic, NODES3), (Quadratic, (Center(), Node(), Center())), (Linear, NODES3))
    for (shape, st) in cases, dir in ((1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1))
        f(x, mode, seed) = flux_functional_3d(x, mode, seed; shape, st, dir)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 7), Δ0)
        r = fd_scan(f, Δ0, hs; seed = 7, ad)
        label = "order $(order(shape)), $(stagger_label(st)), dir $(dir)"
        MEASURED["T-947 rel/floor"] = merge(get(MEASURED, "T-947 rel/floor", Dict{String,Float64}()), Dict(label => r.rel[r.istar] / r.floor_rel))
        MEASURED["T-947 floor_rel"] = merge(get(MEASURED, "T-947 floor_rel", Dict{String,Float64}()), Dict(label => r.floor_rel))
        @test isfinite(ad) && ad != 0
        if shape === Quadratic
            @test r.rel[r.istar] <= tolC("T-947") * r.floor_rel
            @test r.floor_rel <= tolC("T-947", "floor_max")
        end
    end
end

@testset "T-948 Dual weights at N = 3: exact partials, the primal the Float32 run; Fast() needs no prior sort [A8, WP-042]" begin
    C = tolC("T-948")
    for shape in SHAPES, st in FLUX3D_TRIPLES, bins in flux3d_bins(shape, st)
        grid = flux_grid3(Float32, shape, st)
        s32 = flux_species_3d(Float32, grid, 400; seed = 4)
        makeD(wf, ::Type{E}) where {E} = begin
            sD = Species(backend, E, capacity(s32); q = -1, m = 1, N = 3, extras = flux_extras(E, 3))
            for i in 1:capacity(s32)
                for d in 1:3
                    sD.cell[d][i] = s32.cell[d][i]
                    sD.offset[d][i] = E(s32.offset[d][i])
                end
                oc, oδ = old_position_3d(s32, i)
                sD.extras.old_cell_1[i], sD.extras.old_cell_2[i], sD.extras.old_cell_3[i] = oc
                sD.extras.old_offset_1[i], sD.extras.old_offset_2[i], sD.extras.old_offset_3[i] = E.(oδ)
                sD.w[i] = wf(i)
                sD.status[i] = s32.status[i]
            end
            fill!(sD.count, s32.count[1])
            sD
        end
        wdual(i) = mkdual(s32.w[i], cos(0.3i), s32.w[i]^2, 0)
        JD = flux3d(Reference(), makeD(wdual, D3), shape, st, grid; bins)[1:3]
        J32 = flux3d(Reference(), makeD(i -> s32.w[i], Float32), shape, st, grid; bins)[1:3]
        primal_bitwise = all(same_bits(value.(a), b) for (a, b) in zip(JD, J32))
        MEASURED["T-948 primal bitwise"] = merge(get(MEASURED, "T-948 primal bitwise", Dict{String,Float64}()),
                                                 Dict("$(order(shape)),$(stagger_label(st)),$(nameof(typeof(bins)))" => Float64(primal_bitwise)))
        @test primal_bitwise
        g64 = PeriodicGrid(N3F, Float64.(dx3(Float32)), flux_g3(shape, st))
        for dir in 1:3
            J64 = flux3d(Reference(), makeD(i -> Float64(partials(wdual(i))[dir]), Float64), shape, st, g64; dt = 0.25, bins)[1:3]
            for (JDd, J64d) in zip(JD, J64)
                ∂ = [partials(x)[dir] for x in JDd]
                if dir == 3
                    @test all(iszero, ∂)
                else
                    @test record!("T-948", D3, maximum(abs.(∂ .- J64d)) / (eps(Float32) * maximum(abs, J64d))) <= C
                end
            end
        end
    end
    # Fast() on a Dual species falls back to Reference() and sorts first (T-345 at N = 3): never
    # sorted, it gives the sorted Reference() run bit for bit
    for bins in (CellBins(), NodeBins(Quadratic, NODES3))
        grid = flux_grid3(D3, Quadratic, NODES3)
        s = flux_species_3d(D3, grid, 400; seed = 5)
        expect = flux3d(Reference(), flux_species_3d(D3, grid, 400; seed = 5), Quadratic, NODES3, grid; bins)[1:3]
        J = fields3(D3, grid)
        deposit_flux!(Fast(), J..., s, Quadratic, NODES3, grid, Workspace(s, grid; flux = (Quadratic, NODES3), bins); dt = D3(Float32(1 // 4)))
        @test same_bits.(J, expect) == (true, true, true)
    end
end

# The larger grid has more than 1023 cells: an `Int` boxed by a call that does not infer allocates
# from 1024 on (Julia caches the smaller ones), so a size passed through such a call shows here.
@testset "T-949 allocation of the 3D flux does not grow with particle count or grid size [A7, A9, WP-042]" begin
    for T in (Float32, Float64), mode in FLUX3D_MODES, bins in (CellBins(), NodeBins(Quadratic, NODES3))
        bytes = Dict{Tuple{NTuple{3,Int},Int},Int}()
        for (n, np) in ((N3F, 500), (N3F, 4000), ((16, 14, 12), 500))
            grid = flux_grid3(T, Quadratic, NODES3; n)
            s = flux_species_3d(T, grid, np; dead = ())
            ws = Workspace(s, grid; flux = (Quadratic, NODES3), bins)
            sort_by_cell!(s, grid, ws)
            J = fields3(T, grid)
            run() = deposit_flux!(mode, J..., s, Quadratic, NODES3, grid, ws; dt = T(1 // 4))
            run(); run()
            bytes[(n, np)] = @allocated run()
        end
        @info "T-949 bytes per 3D deposit_flux!" T mode bins bytes
        @test_alloc bytes[(N3F, 4000)] <= bytes[(N3F, 500)]
        @test_alloc bytes[((16, 14, 12), 500)] <= bytes[(N3F, 500)]
    end
end
