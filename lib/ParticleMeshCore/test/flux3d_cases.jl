# The computations of the 3D flux tests (WP-042: T-940 to T-948), shared by flux3d.jl, gpu.jl and
# threads_worker.jl, which repeats the Reference-mode digests with 1 and 4 threads. Needs setup.jl,
# helpers.jl, digest.jl, engine_cases.jl, engine2d_cases.jl, flux2d_cases.jl and engine3d_cases.jl.
#
# The grids have the cell sizes of the 3D engine tests (1/4, 3/8 and 1/2 per axis) and 8 × 7 × 6
# cells, the smallest box of distinct sizes per axis that holds the flux guards of order 2 on nodes
# (3 per side), so an exchange of two axes fails a test.

const N3F = (8, 7, 6)
"The guard width of a 3D flux test grid: the largest per-axis `flux_guard_width`."
flux_g3(shape, st) = maximum(flux_guard_width(shape, st))
"The 3D flux test grid of `T` for `shape` and `st`."
flux_grid3(::Type{T}, shape, st; n = N3F) where {T} = grid3(T, flux_g3(shape, st); n)

const FLUX3D_MODES = (Reference(), Fast())
"""
The stagger triples of the 3D flux tests: ρ's nodes (the module's), the cell centres, and two
mixed triples. (Center, Node, Center) tells axis 2 from the others and (Node, Node, Center) axis 3,
so a stagger read from another axis fails a test, which no palindrome triple shows.
"""
const FLUX3D_TRIPLES = ((Node(), Node(), Node()), (Center(), Center(), Center()), (Center(), Node(), Center()),
                        (Node(), Node(), Center()))
const NODES3 = (Node(), Node(), Node())

"Store the current position of every slot as the old one (the extras of `flux_extras(T, 3)`)."
function store_old_3d!(s)
    copyto!(s.extras.old_cell_1, s.cell[1]); copyto!(s.extras.old_cell_2, s.cell[2]); copyto!(s.extras.old_cell_3, s.cell[3])
    copyto!(s.extras.old_offset_1, s.offset[1]); copyto!(s.extras.old_offset_2, s.offset[2]); copyto!(s.extras.old_offset_3, s.offset[3])
    return s
end

"The old cell and offset of slot `i` per axis."
old_position_3d(s, i) = ((s.extras.old_cell_1[i], s.extras.old_cell_2[i], s.extras.old_cell_3[i]),
                         (s.extras.old_offset_1[i], s.extras.old_offset_2[i], s.extras.old_offset_3[i]))

"""
    flux_species_3d(T, grid, np; seed, maxstep = 9/10, dead = DEAD3, axes = (true, true, true))

`test_species_3d` with the old-position extras of `flux_extras(T, 3)`: the old position is stored,
then every alive particle moves through the engine's per-axis `advance` by a displacement in
`(-maxstep, maxstep)` cells on each axis that `axes` enables (a `Dual` displacement for `Dual`
types), so particles cross every seam; particle 8 starts in the last corner cell at offsets
15/16 and moves by 1/8 on every axis, through the corner into cell (1, 1, 1), and particle 1
makes the opposite move. `maxstep = 0` leaves every particle at rest. `dead` slots are freed.
"""
function flux_species_3d(::Type{T}, grid, np; seed = 3, maxstep = 9 / 10, dead = DEAD3, axes = (true, true, true)) where {T}
    s = test_species_3d(T, grid, np; seed, extras = flux_extras(T, 3))
    n = Int32.(ncells(grid))
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    for d in 1:3
        s.offset[d][8] = seeded(T, 15 / 16)
        s.offset[d][1] = seeded(T, 1 / 16)
    end
    foreach(i -> i <= capacity(s) && (s.status[i] = ParticleMeshCore.FREE), dead)
    store_old_3d!(s)
    maxstep == 0 && return s
    for i in 1:np
        Δ = i == 8 ? (1 / 8, 1 / 8, 1 / 8) : i == 1 ? (-1 / 8, -1 / 8, -1 / 8) : ntuple(_ -> maxstep * (2nextu() - 1), 3)
        Δ = ntuple(d -> axes[d] ? seeded(T, Δ[d]) : zero(T), 3)
        c, δ = advance(ntuple(d -> s.cell[d][i], 3), ntuple(d -> s.offset[d][i], 3), Δ, n)
        for d in 1:3
            s.cell[d][i] = c[d]
            s.offset[d][i] = δ[d]
        end
    end
    return s
end

"A copy of the 3D species `s` without extras whose positions are the old ones (for ρ_old)."
function at_old_positions_3d(s)
    o = Species(backend, eltype(s), capacity(s); q = s.q, m = s.m, N = 3)
    copyto!(o.cell[1], s.extras.old_cell_1); copyto!(o.cell[2], s.extras.old_cell_2); copyto!(o.cell[3], s.extras.old_cell_3)
    copyto!(o.offset[1], s.extras.old_offset_1); copyto!(o.offset[2], s.extras.old_offset_2); copyto!(o.offset[3], s.extras.old_offset_3)
    foreach(copyto!, o.u, s.u)
    copyto!(o.w, s.w); copyto!(o.status, s.status); copyto!(o.count, s.count)
    return o
end

"""
    flux3d(mode, s, shape, st, grid; dt, workgroupsize, be, bins, planes, flux_layout) -> (Jx, Jy, Jz, ws)

The 3D flux of `s` in `mode` on `be` into fresh fields, unfolded, with a flux workspace for
(`shape`, `st`) by `bins`, sorting first when the mode needs it.
"""
function flux3d(mode, s, shape, st, grid; dt = basetype(eltype(s))(1 // 4), workgroupsize = nothing, be = CPU(), bins = CellBins(),
                planes = nothing, flux_layout = ParticleMeshCore.flux_patch_layout(get_backend(s)))
    T = eltype(s)
    ws = Workspace(s, grid; flux = (shape, st), flux_layout, bins)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    Jx, Jy, Jz = fields3(T, grid)
    on(be, deposit_flux!, mode, Jx, Jy, Jz, s, shape, st, grid, ws; dt = T(dt), planes, workgroupsize)
    return Jx, Jy, Jz, ws
end

"""
    continuity_case_3d(T, mode, shape, st; np, seed, workgroupsize, be, dt, bins, planes, g, J0)

One step of `flux_species_3d` on the 3D flux test grid (with `g` guard cells if given): ρ at the
old positions, then, after a sort, ρ at the new positions in `mode` and the flux, added into
fields of zeros or into copies of the fields `J0`. Unfolded host fields.
"""
function continuity_case_3d(::Type{T}, mode, shape, st; np = 400, seed = 3, workgroupsize = nothing, be = CPU(),
                            dt = basetype(T)(1 // 4), bins = CellBins(), planes = nothing, g = nothing, J0 = nothing) where {T}
    grid = g === nothing ? flux_grid3(T, shape, st) : grid3(T, g; n = N3F)
    s = flux_species_3d(T, grid, np; seed)
    ρold = deposited_3d(Reference(), at_old_positions_3d(s), shape, st, grid; be)
    ws = Workspace(s, grid; flux = (shape, st), bins)
    on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    ρnew = on(be, deposit!, mode, field(backend, T, grid), s, shape, st, grid, ws)
    Jx, Jy, Jz = J0 === nothing ? fields3(T, grid) : map(copy, J0)
    on(be, deposit_flux!, mode, Jx, Jy, Jz, s, shape, st, grid, ws; dt = T(dt), planes, workgroupsize)
    return (; ρold, ρnew, Jx, Jy, Jz, grid, s, ws, dt = T(dt))
end

"Fields of `T` on `grid` holding a fixed pattern of values up to `scale[c]` in component `c`, guards included."
pattern_fields_3d(::Type{T}, grid, scale) where {T} =
    ntuple(c -> T[T(scale[c] * sin(0.7 * k + 1.3 * c)) for k in reshape(1:prod(ncells(grid) .+ 2guards(grid)), ncells(grid) .+ 2guards(grid))], 3)

"""
    reference_by_hand_3d(c, shape, st, bins) -> (Jx, Jy, Jz)

The Reference() flux of the case `c` (a `continuity_case_3d` into fields of zeros) summed on the
host in the order contract 13 states, from the engine's own per-particle values
(`flux_values_3d`): per bin, its particles' values in slot order from zero; per node, the bins'
sums in increasing column-major linear bin index from zero. The engine's bits on every backend.
"""
function reference_by_hand_3d(c, shape, st, bins)
    (; s, grid, ws, dt) = c
    T = eltype(s)
    n, g = Int32.(ncells(grid)), guards(grid)
    nb = ParticleMeshCore.nbins(bins, n)
    sx = map(h -> h / dt, cellsize(grid))
    frame = bin_frame(bins)
    seg_start, seg_stop = Array(ws.seg_start), Array(ws.seg_stop)
    J = fields3(T, grid)
    bins_ci = CartesianIndices(Int.(nb))
    for C in 1:3
        vC = Val(C)
        for L in 1:prod(nb)
            seg_stop[L] >= seg_start[L] || continue
            b = Tuple(bins_ci[L])
            patch, lo, D = nothing, nothing, nothing
            for i in seg_start[L]:seg_stop[L]
                l, parts = ParticleMeshCore.flux_parts_3d(vC, s, i, shape, st, n, sx, frame)
                a, _ = ParticleMeshCore.transverse_axes(vC)
                M = ntuple(d -> length(d == C ? parts[1] : d == a ? parts[2] : parts[3]), 3)
                D = ParticleMeshCore.flux_dims_3d(vC, M)
                v = ParticleMeshCore.flux_values_3d(vC, parts, Val(0), Val(prod(D)))
                patch = patch === nothing ? map(x -> zero(T) + x, v) : map(+, patch, v)
                lo = Int.(l)
            end
            for (k, K) in enumerate(CartesianIndices(D))
                J[C][ntuple(d -> b[d] + lo[d] + K[d] - 1 + g, 3)...] += patch[k]
            end
        end
    end
    return J
end

"Fold the guards of every field of a 3D case (in place, on `be`) and return it."
function folded_3d(c; be = CPU())
    foreach(a -> on(be, fold_guards!, a, c.grid), (c.ρold, c.ρnew, c.Jx, c.Jy, c.Jz))
    return c
end

"""
    continuity_residual_3d(c) -> ratio

max over interior nodes of |(ρ_new - ρ_old)/Δt + Σ_d (J_d[j] - J_d[j - e_d])/Δx_d| (periodically)
over `eps·max|ρ|/Δt`, on a folded case: T-940's discrete continuity in units of its bound.
"""
function continuity_residual_3d(c)
    (; grid, dt) = c
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    J = (c.Jx, c.Jy, c.Jz)
    ρmax = max(maximum(absmax, c.ρold), maximum(absmax, c.ρnew))
    worst = 0.0
    for j3 in 1:n[3], j2 in 1:n[2], j1 in 1:n[1]
        a = (g + j1, g + j2, g + j3)
        r = (c.ρnew[a...] - c.ρold[a...]) / dt
        for d in 1:3
            b = ntuple(e -> e == d ? g + mod1(a[e] - g - 1, n[e]) : a[e], 3)
            r += (J[d][a...] - J[d][b...]) / h[d]
        end
        worst = max(worst, absmax(r))
    end
    return worst / (base_eps(eltype(c.Jx)) * ρmax / value_of(dt))
end

"""
    particle_values_3d(s, i, shape, st, n, sx, frame) -> ((lo, D, v), (lo, D, v), (lo, D, v))

Per component, the engine's values of particle `i` on its nodes in `frame`: the first node
relative to the particle's cell, the node counts per axis and the values, axis 1 fastest. The
engine gives a bin frame's first node relative to the bin, the cell shifted by `bin_shift`.
"""
function particle_values_3d(s, i, shape, st, n, sx, frame)
    h = frame_shift(frame, s, i)
    return ntuple(3) do C
        lo, parts = ParticleMeshCore.flux_parts_3d(Val(C), s, i, shape, st, n, sx, frame)
        a, _ = ParticleMeshCore.transverse_axes(Val(C))
        M = ntuple(d -> length(d == C ? parts[1] : d == a ? parts[2] : parts[3]), 3)
        D = ParticleMeshCore.flux_dims_3d(Val(C), M)
        (Int.(lo) .+ h, D, ParticleMeshCore.flux_values_3d(Val(C), parts, Val(0), Val(prod(D))))
    end
end

"The shift of particle `i`'s bin from its cell per axis, in which `frame` places its first node."
frame_shift(::Any, s, i) = (0, 0, 0)
frame_shift(fr::ParticleMeshCore.BinFrame, s, i) = ntuple(d -> Int(ParticleMeshCore.bin_shift(fr.bins, d, s.offset[d][i])), 3)

"The frame of `bins` for the engine's 3D patch (`CellFrame()` for cells)."
bin_frame(bins) = bins isa CellBins ? ParticleMeshCore.CellFrame() : ParticleMeshCore.BinFrame(bins)

"""
    flux_abs_3d(s, shape, st, grid, dt, bins) -> (Ax, Ay, Az)

Per node and component, the scale of T-944's per-node bound, on the host from the engine's
per-particle values: the sum over the particles whose bin frame holds the node of the particle's
|value| there where the node is in the particle's own frame too, and of its largest |value| where
it is not (the bin frame's roundoff tail along the component's axis, which Fast() leaves out).
"""
function flux_abs_3d(s, shape, st, grid, dt, bins)
    T = eltype(s)
    sz = ncells(grid) .+ 2guards(grid)
    A = ntuple(_ -> zeros(basetype(T), sz), 3)
    n, g = Int32.(ncells(grid)), guards(grid)
    sx = cellsize(grid) ./ dt
    for i in 1:capacity(s)
        s.status[i] == ParticleMeshCore.ALIVE || continue
        VB = particle_values_3d(s, i, shape, st, n, sx, bin_frame(bins))
        VP = particle_values_3d(s, i, shape, st, n, sx, ParticleMeshCore.ParticleFrame())
        c = ntuple(d -> s.cell[d][i], 3)
        for C in 1:3
            lo, D, v = VB[C]
            lp, Dp, _ = VP[C]
            m = maximum(absmax, v)
            for (k, K) in enumerate(CartesianIndices(D))
                r = ntuple(d -> lo[d] + K[d] - 1 - lp[d], 3)              # the node in the particle's frame
                shared = all(d -> 0 <= r[d] < Dp[d], 1:3)
                A[C][ntuple(d -> c[d] + lo[d] + K[d] - 1 + g, 3)...] += shared ? absmax(v[k]) : m
            end
        end
    end
    return A
end

"""
    flux_hits_3d(s, shape, st, grid, dt) -> (kx, ky, kz)

Per node and component, how many particles' own frames (`ParticleFrame()`, Fast()'s atomics)
hold the node, unfolded.
"""
function flux_hits_3d(s, shape, st, grid, dt)
    k = ntuple(_ -> zeros(Int, ncells(grid) .+ 2guards(grid)), 3)
    n, g = Int32.(ncells(grid)), guards(grid)
    sx = cellsize(grid) ./ dt
    for i in 1:capacity(s)
        s.status[i] == ParticleMeshCore.ALIVE || continue
        c = ntuple(d -> s.cell[d][i], 3)
        for (C, (lo, D, _)) in enumerate(particle_values_3d(s, i, shape, st, n, sx, ParticleMeshCore.ParticleFrame())), K in CartesianIndices(D)
            k[C][ntuple(d -> c[d] + lo[d] + K[d] - 1 + g, 3)...] += 1
        end
    end
    return k
end

"The stagger triples of the 3D flux digests: ρ's nodes and the mixed triple."
const FLUX3D_DIGEST_TRIPLES = ((Node(), Node(), Node()), (Center(), Node(), Center()))

"""
    flux3d_digests(be = CPU(); wgs = WG_SIZES) -> Vector{String}

`"flux3d|type|order,stagger,bins|workgroupsize|digest"` of Jx, Jy and Jz of the Reference continuity
case, for every element type, both orders, the triples of `FLUX3D_DIGEST_TRIPLES` by cell, and order
2 on nodes by `NodeBins` of that shape and stagger, at the launch shapes `wgs`, on the backend `be`.
"""
function flux3d_digests(be = CPU(); wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs
        L = type_label(T)
        for shape in SHAPES, st in FLUX3D_DIGEST_TRIPLES
            c = continuity_case_3d(T, Reference(), shape, st; workgroupsize = wg, be)
            push!(lines, "flux3d|$L|$(order(shape)),$(stagger_label(st)),cells|$wg|$(fnv_digest(c.Jx, c.Jy, c.Jz))")
        end
        c = continuity_case_3d(T, Reference(), Quadratic, NODES3; workgroupsize = wg, be, bins = NodeBins(Quadratic, NODES3))
        push!(lines, "flux3d|$L|2,$(stagger_label(NODES3)),nodes|$wg|$(fnv_digest(c.Jx, c.Jy, c.Jz))")
    end
    return lines
end

# Hand evaluation (T-943): Esirkepov's 3D weights from the B-splines written out on the host
# (`bspline`, `hand_weight` of flux2d_cases.jl), independent of the engine's `shape_weights`.

"""
    hand_flux_3d(shape, st, X0, X1, qw, sx, nodes) -> (Jx, Jy, Jz)

Esirkepov's 3D decomposition evaluated by hand for one particle moving from `X0` to `X1` (cell
units, no seam): per node of `nodes`, with S0, S1 the old and new weights and ΔS = S1 - S0 per
axis,

    W_x = ΔSx·(S0y·S0z + (ΔSy·S0z + S0y·ΔSz)/2 + ΔSy·ΔSz/3),   Jx[j] = -qw·sx₁ Σ_{i ≤ j₁} W_x(i, j₂, j₃),

and cyclically (W_y = ΔSy·(S0x·S0z + ...), W_z = ΔSz·(S0x·S0y + ...)).
"""
function hand_flux_3d(shape, st, X0, X1, qw, sx, nodes)
    S0(d, j) = hand_weight(shape, st[d], X0[d], j)
    D(d, j) = hand_weight(shape, st[d], X1[d], j) - S0(d, j)
    W(c, a, b, jc, ja, jb) = D(c, jc) * (S0(a, ja) * S0(b, jb) + (D(a, ja) * S0(b, jb) + S0(a, ja) * D(b, jb)) / 2 + D(a, ja) * D(b, jb) / 3)
    lo = map(r -> first(r) - 3, nodes)
    J = ntuple(_ -> Dict{NTuple{3,Int},Float64}(), 3)
    for j3 in nodes[3], j2 in nodes[2], j1 in nodes[1]
        j = (j1, j2, j3)
        J[1][j] = -qw * sx[1] * sum(W(1, 2, 3, i, j2, j3) for i in lo[1]:j1)
        J[2][j] = -qw * sx[2] * sum(W(2, 1, 3, i, j1, j3) for i in lo[2]:j2)
        J[3][j] = -qw * sx[3] * sum(W(3, 1, 2, i, j1, j2) for i in lo[3]:j3)
    end
    return J
end

"""
    flux_functional_3d(Δ, mode, seed; shape, st, dir)

T-947's smooth functional of the 3D flux, Σ φ·Jx + Σ ψ·Jy + Σ χ·Jz with fixed smooth weights of
the array index, as a function of one displacement `Δ` shared by every particle, along `dir`. The
particles start in cell (5, 4, 3) at offsets `0.7 ± ε` (ε = 10^-1, ..., 10^-7) along the moving
axes, rotated among the axes of a diagonal move, and 0.4 along a fixed one, so with Δ = 0.3 their
end points lie on both sides of a face (or of an edge and the corner), as close as 10^-7; the
weights come from `seed` (common random numbers).
"""
function flux_functional_3d(Δ, mode, seed; shape, st, dir)
    T = typeof(Δ)
    grid = PeriodicGrid(N3F, (1 / 4, 3 / 8, 1 / 2), flux_g3(shape, st))
    εs = [10.0^(-k) for k in 1:7]
    offs = vcat(0.7 .- εs, 0.7 .+ εs)
    K = length(offs)
    starts = [ntuple(d -> dir[d] == 1 ? offs[mod1(k + 5(d - 1), K)] : 0.4, 3) for k in 1:K]
    np = length(starts)
    s = Species(backend, T, np + 2; q = -1, m = 1, N = 3, extras = flux_extras(T, 3))
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    n = Int32.(ncells(grid))
    for (i, δ0) in enumerate(starts)
        c0 = (Int32(5), Int32(4), Int32(3))
        s.extras.old_cell_1[i], s.extras.old_cell_2[i], s.extras.old_cell_3[i] = c0
        s.extras.old_offset_1[i], s.extras.old_offset_2[i], s.extras.old_offset_3[i] = T.(δ0)
        s.w[i] = T(1 / 2 + nextu())
        s.status[i] = ParticleMeshCore.ALIVE
        c, δ = advance(c0, T.(δ0), ntuple(d -> dir[d] * Δ, 3), n)
        for d in 1:3
            s.cell[d][i] = c[d]
            s.offset[d][i] = δ[d]
        end
    end
    fill!(s.count, Int32(np))
    Jx, Jy, Jz, _ = flux3d(mode, s, shape, st, grid; dt = 1 / 4)
    foreach(a -> fold_guards!(a, grid), (Jx, Jy, Jz))
    return sum((sin(0.9a[1] + 0.4a[2] - 0.2a[3]) + 1 / 3) * Jx[a] + (cos(0.7a[1] - 0.3a[2] + 0.5a[3]) + 1 / 5) * Jy[a] +
               (sin(0.5a[1]) * cos(0.6a[2] + 0.3a[3]) + 1 / 7) * Jz[a] for a in CartesianIndices(Jx))
end
