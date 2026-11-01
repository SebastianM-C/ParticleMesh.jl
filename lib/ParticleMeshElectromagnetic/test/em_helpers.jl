# Test support of the module's step: case runs, the Gauss residual and bitwise digests. The state
# lives on the backend it was built for (`be`, CPU by default; CUDA in gpu.jl, WP-013); the host-side
# analysis reads copies (`Array`, `collect`), which on the CPU are plain copies of the same values.

"""
    case_state(name, T; mode = Reference(), workgroupsize = nothing, be = CPU(), kw...) -> (state, params, problem, case)

The state of a case file on the backend `be`, initialized: `problem_from_case(case, T; kw...)`
with the case's numerics in `mode`.
"""
function case_state(name, ::Type{T}; mode = Reference(), workgroupsize = nothing, be = CPU(), kw...) where {T}
    case = load_case(name)
    prob = problem_from_case(case, T; kw...)
    st = EMState(prob, numerics_from_case(case; mode, workgroupsize); backend = be)
    initialize!(st, prob)
    return st, params(prob), prob, case
end

"""
    gauss_residual(state) -> (; res, scale)

The discrete Gauss law of the layout: `res = max_j |(Ex[j] − Ex[j−1])/Δx − (ρ[j] − mean ρ)|`
with ρ deposited now in the state's mode (the species are sorted by their current cells after
a step, as `deposit!`'s Reference mode needs), and `scale = max|ρ|`.
"""
function gauss_residual(st::EMState)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, RHO_STAGGER, grid, ws)
    end
    fold_guards!(ρd, grid)
    ρ, Ex = Array(ρd), Array(st.fields.Ex)
    g, n, dx = guards(grid), ncells(grid), cellsize(grid)
    ρ̄ = sum(interior_view(ρ, grid)) / n
    res = maximum(j -> absmax((Ex[g + j] - Ex[g + j - 1]) / dx - (ρ[g + j] - ρ̄)), 1:n)
    return (; res, scale = maximum(absmax, interior_view(ρ, grid)))
end

bits(a::AbstractArray) = reinterpret(UInt8, vec(collect(a)))

"""
    state_digest(state) -> UInt64

A hash of the bits of every particle array, every field and every history of the state (the
bits, not the values: `-0.0` and the partials of duals count).
"""
function state_digest(st::EMState)
    h = UInt64(0)
    for s in st.species
        for a in (s.cell[1], s.offset[1], s.u..., s.w, s.status, s.count, values(s.extras)...)
            h = hash(bits(a), h)
        end
    end
    for a in values(st.fields)
        h = hash(bits(a), h)
    end
    for a in values(st.hist)
        h = hash(bits(a), h)
    end
    return h
end

"Times of the history entries: entry k at t = (k − 1)Δt."
history_times(prob) = (0:(prob.nsteps - 1)) .* Float64(value_of(prob.dt))

"True when every entry of `a` is `+0.0` (of its element type) bit for bit."
all_plus_zero(a) = all(x -> bits([x]) == bits([zero(x)]), Array(a))
