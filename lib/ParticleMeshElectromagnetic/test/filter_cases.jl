# WP-029: the runs with the binomial current filter shared by filter.jl and the subprocess worker
# (determinism_worker.jl, T-443 at 1 and 4 threads), and the host-side reference filter.

"""
    binomial_host(a, passes) -> Array

The reference of the binomial filter, on the host, independent of the module's kernels and
guards: `passes` times, along each axis in turn, `(a[j − 1] + a[j + 1] + 2a[j])/4` with periodic
neighbours by `circshift`, on an interior array (no guards).
"""
function binomial_host(a::AbstractArray{T,N}, passes) where {T,N}
    b = copy(a)
    for _ in 1:passes, d in 1:N
        b = (circshift(b, ntuple(k -> k == d ? 1 : 0, N)) .+ circshift(b, ntuple(k -> k == d ? -1 : 0, N)) .+ 2 .* b) ./ 4
    end
    return b
end

"The landau.toml case at test size (64 particles per cell, one time unit) with the case's `filter_passes` set to `passes`."
function landau_filtered(T, passes; mode = Reference(), workgroupsize = nothing, be = CPU())
    case = load_case("landau")
    case["numerics"]["filter_passes"] = passes
    prob = problem_from_case(case, T; ppc = 64, t_end = DETERMINISM_T_END)
    st = EMState(prob, numerics_from_case(case; mode, workgroupsize); backend = be)
    initialize!(st, prob)
    return st, prob
end

"""
    filter_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(label, T, digest)}

T-443: Reference runs with the filter, per element type: the 1D Landau case with three passes
(an odd count, so the result returns from the scratch) and the 2D plasma of
[`plasma_step_2d`](@ref) with two passes, 20 steps; one digest each.
"""
function filter_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for T in TYPES
        st, prob = landau_filtered(T, 3; workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        push!(out, ("1d", T, state_digest(st)))
        p2 = plasma_step_2d(T; nsteps = 20)
        s2 = EMState(p2, EMNumerics(; workgroupsize, filter = BinomialFilter(2)); backend = be)
        initialize!(s2, p2)
        run!(s2, params(p2), p2.nsteps)
        push!(out, ("2d", T, state_digest_2d(s2)))
    end
    return out
end
