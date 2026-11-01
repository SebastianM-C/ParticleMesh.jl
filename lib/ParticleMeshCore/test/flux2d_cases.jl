# The computations of the 2D flux tests (WP-020: T-300 to T-309), shared by flux2d.jl and by
# threads_worker.jl, which repeats the Reference-mode digests with 1 and 4 threads. Needs
# setup.jl, helpers.jl, digest.jl, engine_cases.jl, flux_cases.jl and engine2d_cases.jl.
#
# The grids are those of the 2D engine tests (16 × 12 cells of 1/4 × 3/8, engine2d_cases.jl), so a
# transposition of the axes fails a test; the guards are the larger of the two per-axis flux
# guard widths.

"The guard width of a 2D flux test grid: the larger per-axis `flux_guard_width`."
flux_g2(shape, st) = maximum(flux_guard_width(shape, st))

"The module-supplied quantity of Jz in the 2D tests: q·w·u₂ (`ChargeUy2`; 2D species keep the id in u₃)."
const QZ2 = ChargeUy2()

"Store the current position of every slot as the old one (the 2D extras of `flux_extras`)."
function store_old_2d!(s)
    copyto!(s.extras.old_cell_1, s.cell[1]); copyto!(s.extras.old_cell_2, s.cell[2])
    copyto!(s.extras.old_offset_1, s.offset[1]); copyto!(s.extras.old_offset_2, s.offset[2])
    return s
end

"""
    flux_species_2d(T, grid, np; seed, maxstep = 9/10, dead = DEAD2, axes = (true, true))

`test_species_2d` with the old-position extras of `flux_extras(T, 2)`: the old position is
stored, then every alive particle moves through the engine's per-axis `advance` by a
displacement in `(-maxstep, maxstep)` cells on each axis that `axes` enables (a `Dual`
displacement for `Dual` types), so particles cross both seams; particle 1 starts in the last
corner cell at (15/16, 15/16) and moves by (1/8, 1/8) diagonally through the corner, particle 2
makes the opposite move. `maxstep = 0` leaves every particle at rest.
"""
function flux_species_2d(::Type{T}, grid, np; seed = 3, maxstep = 9 / 10, dead = DEAD2, axes = (true, true)) where {T}
    s = test_species_2d(T, grid, np; seed, extras = flux_extras(T, 2))
    n = Int32.(ncells(grid))
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    s.offset[1][1], s.offset[2][1] = seeded(T, 15 / 16), seeded(T, 15 / 16)
    s.offset[1][2], s.offset[2][2] = seeded(T, 1 / 16), seeded(T, 1 / 16)
    foreach(i -> s.status[i] = ParticleMeshCore.FREE, dead)
    store_old_2d!(s)
    maxstep == 0 && return s
    for i in 1:np
        Δ = i == 1 ? (1 / 8, 1 / 8) : i == 2 ? (-1 / 8, -1 / 8) : (maxstep * (2nextu() - 1), maxstep * (2nextu() - 1))
        Δ = ntuple(d -> axes[d] ? seeded(T, Δ[d]) : zero(T), 2)
        c, δ = advance((s.cell[1][i], s.cell[2][i]), (s.offset[1][i], s.offset[2][i]), Δ, n)
        s.cell[1][i], s.cell[2][i] = c
        s.offset[1][i], s.offset[2][i] = δ
    end
    return s
end

"A copy of the 2D species `s` without extras whose positions are the old ones (for ρ_old)."
function at_old_positions_2d(s)
    o = Species(backend, eltype(s), capacity(s); q = s.q, m = s.m, N = 2)
    copyto!(o.cell[1], s.extras.old_cell_1); copyto!(o.cell[2], s.extras.old_cell_2)
    copyto!(o.offset[1], s.extras.old_offset_1); copyto!(o.offset[2], s.extras.old_offset_2)
    foreach(copyto!, o.u, s.u)
    copyto!(o.w, s.w); copyto!(o.status, s.status); copyto!(o.count, s.count)
    return o
end

"Three fresh fields on `grid`."
fields3(::Type{T}, grid) where {T} = (field(backend, T, grid), field(backend, T, grid), field(backend, T, grid))

"""
    flux2d(mode, s, shape, st, grid; dt, workgroupsize, be) -> (Jx, Jy, Jz, ws)

The 2D flux of `s` in `mode` on `be` into fresh fields, unfolded, with a flux workspace for
(`shape`, `st`), sorting first when the mode needs it.
"""
function flux2d(mode, s, shape, st, grid; dt = basetype(eltype(s))(1 // 4), workgroupsize = nothing, be = CPU(), quantity_z = QZ2)
    T = eltype(s)
    ws = Workspace(s, grid; flux = (shape, st))
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    Jx, Jy, Jz = fields3(T, grid)
    on(be, deposit_flux!, mode, Jx, Jy, Jz, s, shape, st, grid, ws; dt = T(dt), quantity_z, workgroupsize)
    return Jx, Jy, Jz, ws
end

"""
    continuity_case_2d(T, mode, shape, st; np, seed, workgroupsize, be)

One step of `flux_species_2d` on the 16 × 12 test grid: ρ at the old positions, then, after a
sort, ρ at the new positions in `mode` and the flux. Unfolded host fields.
"""
function continuity_case_2d(::Type{T}, mode, shape, st; np = 400, seed = 3, workgroupsize = nothing, be = CPU(),
                            dt = basetype(T)(1 // 4)) where {T}
    grid = grid2(T, flux_g2(shape, st))
    s = flux_species_2d(T, grid, np; seed)
    ρold = deposited_2d(Reference(), at_old_positions_2d(s), shape, st, grid; be)
    ws = Workspace(s, grid; flux = (shape, st))
    on(be, sort_by_cell!, s, grid, ws; workgroupsize)
    ρnew = on(be, deposit!, mode, field(backend, T, grid), s, shape, st, grid, ws)
    Jx, Jy, Jz = fields3(T, grid)
    on(be, deposit_flux!, mode, Jx, Jy, Jz, s, shape, st, grid, ws; dt = T(dt), quantity_z = QZ2, workgroupsize)
    return (; ρold, ρnew, Jx, Jy, Jz, grid, s, ws, dt = T(dt))
end

"Fold the guards of every field of a 2D case (in place, on `be`) and return it."
function folded_2d(c; be = CPU())
    foreach(a -> on(be, fold_guards!, a, c.grid), (c.ρold, c.ρnew, c.Jx, c.Jy, c.Jz))
    return c
end

"""
    continuity_residual_2d(c) -> ratio

max over interior nodes of |(ρ_new - ρ_old)/Δt + (Jx[j] - Jx[j - e₁])/Δx₁ + (Jy[j] - Jy[j - e₂])/Δx₂|
(periodically) over `eps·max|ρ|/Δt`, on a folded case: T-300's discrete continuity in units of
its bound. max|ρ| is the count of particles a node sees times their max|q·w|, to within the
shape's weights.
"""
function continuity_residual_2d(c)
    (; grid, dt) = c
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    ρmax = max(maximum(absmax, c.ρold), maximum(absmax, c.ρnew))
    worst = 0.0
    for j2 in 1:n[2], j1 in 1:n[1]
        a1, a2 = g + j1, g + j2
        b1, b2 = g + mod1(j1 - 1, n[1]), g + mod1(j2 - 1, n[2])
        r = (c.ρnew[a1, a2] - c.ρold[a1, a2]) / dt + (c.Jx[a1, a2] - c.Jx[b1, a2]) / h[1] + (c.Jy[a1, a2] - c.Jy[a1, b2]) / h[2]
        worst = max(worst, absmax(r))
    end
    return worst / (base_eps(eltype(c.Jx)) * ρmax / value_of(dt))
end

"""
    flux_abs_2d(s, shape, st, grid, dt) -> (Ax, Ay, Az)

Per node, Σ|contribution| of the 2D flux of `s`, on the host from the engine's per-particle
stencil (`ParticleMeshCore.flux_stencil`): the scale of T-302's per-node bound.
"""
function flux_abs_2d(s, shape, st, grid, dt)
    T = eltype(s)
    sz = ncells(grid) .+ 2guards(grid)
    A = ntuple(_ -> zeros(basetype(T), sz), 3)
    n, g = Int32.(ncells(grid)), guards(grid)
    sx = cellsize(grid) ./ dt
    for i in 1:capacity(s)
        s.status[i] == ParticleMeshCore.ALIVE || continue
        lo, M, J = ParticleMeshCore.flux_stencil(s, i, shape, st, n, sx, QZ2)
        for d in 1:3, k in eachindex(J[d])
            k1, k2 = mod1(k, M[1]), (k - 1) ÷ M[1] + 1
            A[d][s.cell[1][i] + lo[1] + k1 - 1 + g, s.cell[2][i] + lo[2] + k2 - 1 + g] += absmax(J[d][k])
        end
    end
    return A
end

"The stagger pairs of the 2D flux digests: ρ at (Node, Node), the module's, and (Center, Center)."
const FLUX2D_DIGEST_PAIRS = ((Node(), Node()), (Center(), Center()))

"""
    flux2d_digests(be = CPU(); wgs = WG_SIZES) -> Vector{String}

`"flux2d|type|order,stagger|workgroupsize|digest"` of Jx, Jy and Jz of the Reference continuity
case, for every element type, both orders, the stagger pairs of `FLUX2D_DIGEST_PAIRS` and the
launch shapes `wgs`, on the backend `be`. Every stagger pair is covered in process by T-300 to
T-302; this set is kept small because each subprocess compiles it anew.
"""
function flux2d_digests(be = CPU(); wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs, shape in SHAPES, st in FLUX2D_DIGEST_PAIRS
        c = continuity_case_2d(T, Reference(), shape, st; workgroupsize = wg, be)
        push!(lines, "flux2d|$(type_label(T))|$(order(shape)),$(stagger_label(st))|$wg|$(fnv_digest(c.Jx, c.Jy, c.Jz))")
    end
    return lines
end

# Hand evaluation (T-301): the B-splines of order 1 and 2 written out on the host, independent of
# the engine's `shape_weights`, at the node positions of a stagger in cell units.

"The B-spline of order `P` at distance `t` (in cells) from a node."
bspline(::Shape{1}, t) = max(1 - abs(t), 0.0)
function bspline(::Shape{2}, t)
    a = abs(t)
    return a < 1 / 2 ? 3 / 4 - a^2 : a < 3 / 2 ? (3 / 2 - a)^2 / 2 : 0.0
end
"Node `j`'s position in cell units for a stagger: `Node()` at `j`, `Center()` at `j + 1/2` (cell `c` covers `[c, c + 1)`)."
node_pos(::Node, j) = Float64(j)
node_pos(::Center, j) = j + 1 / 2
"The weight of a particle at cell-unit position `X` on node `j`."
hand_weight(shape, st, X, j) = bspline(shape, X - node_pos(st, j))

"""
    hand_flux_2d(shape, st, X0, X1, qw, qz, sx, nodes) -> (Jx, Jy, Jz)

Esirkepov's 2D decomposition evaluated by hand for one particle moving from `X0` to `X1` (cell
units, no seam): per node `(j1, j2)` of `nodes`, with S0, S1 the old and new weights and
ΔS = S1 - S0 per axis, in PIConGPU's form,

    Jx[j1, j2] = -qw·sx₁ Σ_{i ≤ j1} ΔSx(i)·(S0y(j2) + ΔSy(j2)/2)
    Jy[j1, j2] = -qw·sx₂ Σ_{j ≤ j2} ΔSy(j)·(S0x(j1) + ΔSx(j1)/2)
    Jz[j1, j2] = qz·(S0x·S0y + (ΔSx·S0y + S0x·ΔSy)/2 + ΔSx·ΔSy/3).
"""
function hand_flux_2d(shape, st, X0, X1, qw, qz, sx, nodes)
    S0(d, j) = hand_weight(shape, st[d], X0[d], j)
    S1(d, j) = hand_weight(shape, st[d], X1[d], j)
    D(d, j) = S1(d, j) - S0(d, j)
    lo1, lo2 = first(nodes[1]) - 3, first(nodes[2]) - 3
    Jx = Dict{Tuple{Int,Int},Float64}(); Jy = similar(Jx); Jz = similar(Jx)
    for j2 in nodes[2], j1 in nodes[1]
        Jx[(j1, j2)] = -qw * sx[1] * sum(D(1, i) * (S0(2, j2) + D(2, j2) / 2) for i in lo1:j1)
        Jy[(j1, j2)] = -qw * sx[2] * sum(D(2, j) * (S0(1, j1) + D(1, j1) / 2) for j in lo2:j2)
        Jz[(j1, j2)] = qz * (S0(1, j1) * S0(2, j2) + (D(1, j1) * S0(2, j2) + S0(1, j1) * D(2, j2)) / 2 + D(1, j1) * D(2, j2) / 3)
    end
    return Jx, Jy, Jz
end

"""
    flux_functional_2d(Δ, mode, seed; shape, st, dir)

T-307's smooth functional of the 2D flux, Σ φ·Jx + Σ ψ·Jy + Σ χ·Jz with fixed smooth weights
φ, ψ, χ of the array index, as a function of one displacement `Δ` shared by every particle,
along `dir` ((1, 0), (0, 1) or the diagonal (1, 1)). The particles start in cell (6, 5) at
offsets `0.7 ± ε` (ε = 10^-1, ..., 10^-7) along each moving axis and 0.4 along a fixed one, so
with Δ = 0.3 their end points lie on both sides of a face (or of the corner), as close as 10^-7;
the weights come from `seed` (common random numbers).
"""
function flux_functional_2d(Δ, mode, seed; shape, st, dir)
    T = typeof(Δ)
    grid = PeriodicGrid((12, 10), (1 / 4, 3 / 8), flux_g2(shape, st))
    εs = [10.0^(-k) for k in 1:7]
    offs = vcat(0.7 .- εs, 0.7 .+ εs)
    starts = [(dir[1] == 1 ? a : 0.4, dir[2] == 1 ? (dir[1] == 1 ? b : a) : 0.4) for a in offs for b in (dir == (1, 1) ? offs : (0.0,))]
    np = length(starts)
    s = Species(backend, T, np + 2; q = -1, m = 1, N = 2, extras = flux_extras(T, 2))
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    n = Int32.(ncells(grid))
    for (i, δ0) in enumerate(starts)
        c0 = (Int32(6), Int32(5))
        s.extras.old_cell_1[i], s.extras.old_cell_2[i] = c0
        s.extras.old_offset_1[i], s.extras.old_offset_2[i] = T(δ0[1]), T(δ0[2])
        s.w[i] = T(1 / 2 + nextu())
        s.u[2][i] = T(2nextu() - 1)
        s.status[i] = ParticleMeshCore.ALIVE
        c, δ = advance(c0, (T(δ0[1]), T(δ0[2])), (dir[1] * Δ, dir[2] * Δ), n)
        s.cell[1][i], s.cell[2][i] = c
        s.offset[1][i], s.offset[2][i] = δ
    end
    fill!(s.count, Int32(np))
    Jx, Jy, Jz, _ = flux2d(mode, s, shape, st, grid; dt = 1 / 4)
    foreach(a -> fold_guards!(a, grid), (Jx, Jy, Jz))
    I = CartesianIndices(Jx)
    return sum((sin(0.9a[1] + 0.4a[2]) + 1 / 3) * Jx[a] + (cos(0.7a[1] - 0.3a[2]) + 1 / 5) * Jy[a] +
               (sin(0.5a[1]) * cos(0.6a[2]) + 1 / 7) * Jz[a] for a in I)
end
