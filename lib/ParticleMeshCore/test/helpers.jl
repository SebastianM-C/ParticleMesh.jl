# Test scaffolding: species with hand-placed particles on the CPU backend (plain `Array`s, so
# host writes are direct; no kernel code is involved in setting up a test).

"""
    test_species(T, grid, np; seed, q = -1)

`np` alive particles spread over every cell (including the first and last, so the stencils
cross the periodic seam), with offsets from a fixed LCG sequence that includes 0 and values
just below 1, positive weights, and velocities in `[-1, 1)`.
"""
function test_species(::Type{T}, grid::PeriodicGrid{1}, np::Integer; seed::Integer = 1, q = -1) where {T}
    n = ncells(grid)
    s = Species(backend, T, np + 3; q, m = 1)          # three free slots at the end
    state = UInt64(seed)
    nextu() = (state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; Float64(state >> 11) / 2.0^53)
    for i in 1:np
        s.cell[1][i] = Int32(mod1(i, n))
        δ = i == 1 ? 0.0 : i == 2 ? Float64(prevfloat(one(basetype(T)))) : nextu()
        s.offset[1][i] = seeded(T, δ)
        s.w[i] = seeded(T, 0.5 + nextu())
        s.u[1][i] = seeded(T, 2nextu() - 1)
        s.u[2][i] = T(i)                                # particle id
        s.status[i] = ParticleMeshCore.ALIVE
    end
    fill!(s.count, Int32(np))
    return s
end

"A field on `grid` with interior values from `f(j)` and filled guards."
function test_field(::Type{T}, grid::PeriodicGrid{1}, f) where {T}
    a = field(backend, T, grid)
    for j in 1:ncells(grid)
        a[guards(grid) + j] = f(j)
    end
    return fill_guards!(a, grid)
end

"Deposit `q·w` of `s` into a fresh field in `mode` (sorting first when the mode needs it); guards not folded."
function deposited(mode, s, shape, stagger, grid)
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
    ρ = field(backend, T, grid)
    deposit!(mode, ρ, s, shape, stagger, grid, ws)
    return ρ
end

const SHAPES = (Linear, Quadratic)
const STAGGERS = (Node(), Center())
