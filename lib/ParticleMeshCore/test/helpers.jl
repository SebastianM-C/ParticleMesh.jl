# Test scaffolding: species with hand-placed particles on the CPU backend (plain `Array`s, so
# host writes are direct; no kernel code is involved in setting up a test).

"""
    test_species(T, grid, np; seed, q = -1, extras = NamedTuple())

`np` alive particles spread over every cell (including the first and last, so the stencils
cross the periodic seam), with offsets from a fixed LCG sequence that includes 0 and values
just below 1, positive weights, and velocities in `[-1, 1)`.
"""
function test_species(::Type{T}, grid::PeriodicGrid{1}, np::Integer; seed::Integer = 1, q = -1,
                      extras = NamedTuple()) where {T}
    n = ncells(grid)
    s = Species(backend, T, np + 3; q, m = 1, extras)  # three free slots at the end
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

"""
Deposit `q·w` of `s` into a fresh field in `mode` (sorting first when the mode needs it), on the
backend `be` through `on` (gpu_support.jl); guards not folded. The species and the field stay host
arrays.
"""
function deposited(mode, s, shape, stagger, grid; be = CPU())
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws)
    ρ = field(backend, T, grid)
    on(be, deposit!, mode, ρ, s, shape, stagger, grid, ws)
    return ρ
end

const SHAPES = (Linear, Quadratic)
const STAGGERS = (Node(), Center())

"""
    fd_scan(f, x0, hs; seed, ad) -> (; fd, spread, istar, floor_rel, rel)

The finite-difference floor of A7 (`docs/goals/A7-acceptance.md`), as every gradient check of
the package measures it. A7's operational definition: scan the step over at least five points;
the floor is the spread (maximum minus minimum) of three neighbouring difference quotients,
minimised over the scan, relative to the AD value; the AD derivative must agree within a stated
multiple of it (the multiple measured and proposed in `tolerances.toml`). No formula in `eps`
and the step stands in for it: the particle dynamics amplify rounding, so the floor is measured.

A7's three conditions, which this function enforces where it can:

1. Float64, even where production is Float32: it throws an `ArgumentError` unless `x0`, every
   step and every value `f` returns are `Float64`.
2. The deterministic Reference mode on both sides: every evaluation is `f(x, Reference(), seed)`.
3. Common random numbers: every evaluation gets the same `seed`.

`f(x, mode, seed)` must build its whole state from `x` (so that a `Dual` `x` gives the AD
value). The caller's obligation, which `fd_scan` cannot check: `ad` is computed with the same
`f`, `Reference()` and `seed`, for example `ForwardDiff.derivative(x -> f(x, Reference(), seed), x0)`.

Returns the central difference quotients `fd[k] = (f(x0 + h) - f(x0 - h)) / 2h` for
`h = hs[k]`; `spread[k - 1]` for the three quotients `fd[k-1:k+1]`, `k = 2:length(hs)-1`; `istar`,
the `k` of the smallest spread; `floor_rel = spread[istar - 1] / |ad|`; and the relative
differences `rel[k] = |fd[k] - ad| / |ad|`. A check asserts `rel[istar] <= C·floor_rel`.
"""
function fd_scan(f, x0, hs; seed, ad)
    x0 isa Float64 || throw(ArgumentError("fd_scan runs in Float64 (A7); x0 is a $(typeof(x0))"))
    eltype(hs) === Float64 || throw(ArgumentError("fd_scan runs in Float64 (A7); the steps are $(eltype(hs))"))
    length(hs) >= 5 || throw(ArgumentError("A7 asks for at least five steps; got $(length(hs))"))
    function evaluate(x)
        y = f(x, Reference(), seed)
        y isa Float64 || throw(ArgumentError("fd_scan runs in Float64 (A7); f returned a $(typeof(y))"))
        return y
    end
    fd = [(evaluate(x0 + h) - evaluate(x0 - h)) / 2h for h in hs]
    spread = [maximum(fd[(k - 1):(k + 1)]) - minimum(fd[(k - 1):(k + 1)]) for k in 2:(length(hs) - 1)]
    istar = argmin(spread) + 1
    floor_rel = spread[istar - 1] / abs(ad)
    rel = abs.(fd .- ad) ./ abs(ad)
    return (; fd, spread, istar, floor_rel, rel)
end

"Deposit `quantity` of `s` into a fresh field in `mode` (sorting first when the mode needs it), on `be`; guards not folded."
function deposited_with(mode, s, shape, stagger, grid, quantity; be = CPU())
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws)
    return on(be, deposit!, mode, field(backend, T, grid), s, shape, stagger, grid, ws; quantity)
end

"""
    staggered_divergence(Jx, grid) -> Vector

`(Jx[j] - Jx[j-1]) / dx` for every interior node `j`, periodically (`Jx[0] ≡ Jx[n]`), on a
folded flux whose entry `j` is the face half a cell to the right of ρ's node `j`: the discrete
divergence that matches `deposit_flux!`'s staggering. Test support only; the engine has no
divergence operator.
"""
function staggered_divergence(Jx, grid::PeriodicGrid{1})
    n, g, dx = ncells(grid), guards(grid), cellsize(grid)
    return [(Jx[g + j] - Jx[g + mod1(j - 1, n)]) / dx for j in 1:n]
end
