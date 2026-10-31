# The computations of the flux tests (T-110 to T-117), shared by flux.jl and by
# threads_worker.jl, which repeats the Reference-mode ones with 1 and 4 threads. Needs setup.jl,
# helpers.jl, digest.jl and engine_cases.jl.

"The extras a module declares for `deposit_flux!`: the position before the push."
old_extras(::Type{T}) where {T} = (old_cell = Int32, old_offset = T)

"q·w·u_z, the second transverse quantity of the flux tests (`ChargeUy` is in engine_cases.jl)."
struct ChargeUz end
@inline (::ChargeUz)(s, i) = s.q * s.w[i] * s.u[3][i]

"Store the current position of every slot as the old one, as a module does before its push."
function store_old!(s)
    copyto!(s.extras.old_cell, s.cell[1])
    copyto!(s.extras.old_offset, s.offset[1])
    return s
end

"""
    moved_species(T, grid, np; seed, maxstep = 9/10, dead = ())

`test_species` with the old-position extras and `u_y`, `u_z` in [-1, 1): the old position is
stored, then every alive particle moves by a displacement in `(-maxstep, maxstep)` cells (a
`Dual` displacement for `Dual` types) through the engine's `advance`, so many particles cross a
cell face and those in the first and last cell cross the periodic seam; `maxstep = 0` leaves
every particle at rest. `dead` slots are freed before the move.
"""
function moved_species(::Type{T}, grid, np; seed, maxstep = 9 / 10, dead = ()) where {T}
    s = test_species(T, grid, np; seed, extras = old_extras(T))
    n = Int32(ncells(grid))
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    for i in 1:np
        s.u[2][i] = seeded(T, cos(1.7i + seed))
        s.u[3][i] = seeded(T, sin(0.9i + seed))
    end
    for i in dead
        s.status[i] = ParticleMeshCore.FREE
    end
    store_old!(s)
    maxstep == 0 && return s                    # at rest: no displacement, not even in the partials
    for i in 1:np
        Δ = seeded(T, maxstep * (2nextu() - 1))
        c, δ = ParticleMeshCore.advance(s.cell[1][i], s.offset[1][i], Δ, n)
        s.cell[1][i] = c
        s.offset[1][i] = δ
    end
    return s
end

"A copy of `s` without extras whose positions are the old ones (for depositing ρ_old)."
function at_old_positions(s)
    T = eltype(s)
    o = Species(backend, T, capacity(s); q = s.q, m = s.m)
    copyto!(o.cell[1], s.extras.old_cell)
    copyto!(o.offset[1], s.extras.old_offset)
    foreach(copyto!, o.u, s.u)
    copyto!(o.w, s.w)
    copyto!(o.status, s.status)
    copyto!(o.count, s.count)
    return o
end

"ρ of `s` in `mode` on `grid`, sorting a private copy's workspace first when the mode needs it; guards not folded."
function charge(mode, s, shape, stagger, grid)
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
    return deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws)
end

"""
    continuity_case(T, mode, shape, stagger; n, np, seed, dt, sort_between, workgroupsize)

One step of `moved_species` on `n` cells: ρ at the old positions, then, after a sort when the
mode needs it or `sort_between` asks for it, ρ at the new positions and the flux. Returns the
unfolded fields and the grid.
"""
function continuity_case(::Type{T}, mode, shape, stagger; n = 10, np = 400, seed = 3, dt = basetype(T)(1 // 4),
                         sort_between = true, workgroupsize = nothing, dead = (40, 41, 300)) where {T}
    grid = PeriodicGrid(n, basetype(T)(1 // 2), flux_guard_width(shape, stagger))
    s = moved_species(T, grid, np; seed, dead)
    ρold = charge(Reference(), at_old_positions(s), shape, stagger, grid)
    ws = Workspace(s, grid)
    (sort_between || effective_mode(mode, T) isa Reference) && sort_by_cell!(s, grid, ws; workgroupsize)
    ρnew = deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws)
    Jx, Jy, Jz = field(backend, T, grid), field(backend, T, grid), field(backend, T, grid)
    deposit_flux!(mode, Jx, Jy, Jz, s, shape, stagger, grid, ws; dt = T(dt), quantity_y = ChargeUy(), quantity_z = ChargeUz(),
                  workgroupsize)
    return (; ρold, ρnew, Jx, Jy, Jz, grid, s, ws, dt = T(dt))
end

"Fold the guards of every field of a continuity case (in place) and return it."
function folded(c)
    foreach(a -> fold_guards!(a, c.grid), (c.ρold, c.ρnew, c.Jx, c.Jy, c.Jz))
    return c
end

"""
    continuity_residual(c) -> ratio

max over interior nodes of |(ρ_new - ρ_old)/Δt + (Jx[j] - Jx[j-1])/Δx| / (eps·max|ρ|/Δt), on a
folded case: the discrete continuity of T-110 in units of its bound.
"""
function continuity_residual(c)
    (; grid, dt) = c
    g, n = guards(grid), ncells(grid)
    div = staggered_divergence(c.Jx, grid)
    ρmax = max(maximum(absmax, c.ρold), maximum(absmax, c.ρnew))
    ε = base_eps(eltype(c.Jx))
    return maximum(absdiff((c.ρnew[g + j] - c.ρold[g + j]) / dt, -div[j]) for j in 1:n) / (ε * ρmax / value_of(dt))
end

"""
    flux_abs(c, shape, stagger) -> (Ax, Ay, Az)

Per face and node, Σ|contribution| of the flux of `c.s`, computed on the host from the engine's
per-particle stencil (`ParticleMeshCore.flux_stencil`): the scale of T-113's per-face bound.
"""
function flux_abs(c, shape, stagger)
    (; grid, s, dt) = c
    T = eltype(s)
    A = ntuple(_ -> zeros(basetype(T), length(c.Jx)), 3)
    n, g = Int32(ncells(grid)), guards(grid)
    for i in 1:capacity(s)
        s.status[i] == ParticleMeshCore.ALIVE || continue
        lo, jx, jy, jz = ParticleMeshCore.flux_stencil(s, i, shape, stagger, n, cellsize(grid) / dt, ChargeUy(), ChargeUz())
        for (d, j) in enumerate((jx, jy, jz)), k in eachindex(j)
            A[d][s.cell[1][i] + lo + k - 1 + g] += absmax(j[k])
        end
    end
    return A
end

"""
    flux_digests() -> Vector{String}

`"flux|type|order,stagger|workgroupsize|digest"` of Jx, Jy and Jz of the Reference continuity
case, for every element type, shape, stagger and launch shape of T-113.
"""
function flux_digests()
    lines = String[]
    for T in TYPES, wg in WG_SIZES, shape in SHAPES, stagger in STAGGERS
        c = continuity_case(T, Reference(), shape, stagger; workgroupsize = wg)
        push!(lines, "flux|$(type_label(T))|$(order(shape)),$(nameof(typeof(stagger)))|$wg|$(fnv_digest(c.Jx, c.Jy, c.Jz))")
    end
    return lines
end

"""
    flux_functional(Δ, mode, seed; shape, stagger)

T-116's smooth functional of the flux, Σ_a φ_a·Jx_a with φ_a = sin(0.9a) + 1/3, as a function of
one displacement `Δ` shared by every particle (Float64 or a `Dual` of it). The particles start
in cell 3 at offsets `0.7 ± ε` for ε = 10^-1, ..., 10^-7, so with Δ = 0.3 their end points lie
on both sides of the face between cells 3 and 4, as close as 10^-7; the weights come from
`seed` (common random numbers across evaluations).
"""
function flux_functional(Δ, mode, seed; shape, stagger)
    T = typeof(Δ)
    grid = PeriodicGrid(8, 1 / 2, flux_guard_width(shape, stagger))
    εs = [10.0^(-k) for k in 1:7]
    offsets = vcat(0.7 .- εs, 0.7 .+ εs)
    np = length(offsets)
    s = Species(backend, T, np + 2; q = -1, m = 1, extras = old_extras(T))
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    n = Int32(ncells(grid))
    for (i, δ0) in enumerate(offsets)
        s.extras.old_cell[i] = Int32(3)
        s.extras.old_offset[i] = T(δ0)
        s.w[i] = T(1 / 2 + nextu())
        s.status[i] = ParticleMeshCore.ALIVE
        c, δ = ParticleMeshCore.advance(Int32(3), T(δ0), Δ, n)
        s.cell[1][i] = c
        s.offset[1][i] = δ
    end
    fill!(s.count, Int32(np))
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
    Jx, Jy, Jz = field(backend, T, grid), field(backend, T, grid), field(backend, T, grid)
    deposit_flux!(mode, Jx, Jy, Jz, s, shape, stagger, grid, ws; dt = T(1 / 4), quantity_y = ChargeUy(), quantity_z = ChargeUz())
    fold_guards!(Jx, grid)
    return sum((sin(0.9a) + 1 / 3) * Jx[a] for a in eachindex(Jx))
end
