# WP-029: the runs with the binomial current filter shared by filter.jl and the subprocess worker
# (determinism_worker.jl, T-443 at 1 and 4 threads), and the host-side references of filter.jl and
# nd_features.jl (WP-093): the filter by `circshift`, the DFT and the two sides of Gauss's law.

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

"A random value of `T` in [−1, 1) (each partial of a dual drawn too)."
filter_rand(rng, ::Type{T}) where {T<:AbstractFloat} = T(2rand(rng) - 1)
filter_rand(rng, ::Type{Dual{Nothing,V,N}}) where {V,N} = Dual{Nothing}(V(2rand(rng) - 1), ntuple(_ -> V(2rand(rng) - 1), N)...)

"The value and each partial of an array of `T`, as BigFloat arrays (exact conversions)."
component_arrays(A::AbstractArray{<:AbstractFloat}) = [BigFloat.(A)]
component_arrays(A::AbstractArray{<:Dual}) = [BigFloat.(map(x -> components(x)[c], A)) for c in 1:length(components(first(A)))]

"""
The normalized DFT of a real array in N dimensions, Â[m] = Σ_j A[j]·exp(−2πi m·j/n)/Π n, one axis
after the other, in BigFloat, so that its own rounding stays far below eps(Float64) at any test
size (in Float64 it reached 1.4 of T-440's scale at n = 33, measured by the WP-029 physics breaker).
"""
function dft(A::AbstractArray{BigFloat,N}) where {N}
    out = Complex{BigFloat}.(A)
    for d in 1:N
        n = size(A, d)
        W = [cispi(-2 * big((k - 1) * (j - 1)) / n) for k in 1:n, j in 1:n]
        out = mapslices(v -> W * v, out; dims = d)
    end
    return out ./ length(A)
end

"The closed form H_p of the DFT index `m` (1-based) on `n` cells: Π_d cos^{2p}(π(m_d − 1)/n_d)."
transfer_closed(m::CartesianIndex, n, p) = prod(d -> cospi(big(m[d] - 1) / n[d])^(2p), 1:length(n))

"A field of `grid` with the interior `A` and every guard set to `junk`, and a scratch field filled with `junk2`."
function guarded(grid, A::AbstractArray{T}, junk, junk2) where {T}
    a = field(CPU(), T, grid)
    fill!(a, junk)
    interior_view(a, grid) .= A
    s = field(CPU(), T, grid)
    fill!(s, junk2)
    return a, s
end

"The grid `g` with its cell sizes in the element type `T`."
grid_T(g::PeriodicGrid{N}, ::Type{T}) where {N,T} = PeriodicGrid(g.n, map(d -> T(d), g.dx), guards(g))

"""
    gauss_parts(state; sorted = false) -> (; divE, ρ)

The backward-difference divergence of E over the resolved axes and the charge deposited now in
the state's mode at the primal nodes, on the host, as interior arrays, at N = 1, 2 and 3. The
species are sorted by their current cells after every step; `sorted = true` sorts them first in
Reference mode, as a state with an open axis needs (its retired particles stay in their segments
until the next sort).
"""
function gauss_parts(st; sorted = false)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    N = ndims(st)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        sorted && numerics.mode isa Reference && sort_by_cell!(s, grid, ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, N == 1 ? RHO_STAGGER : ntuple(_ -> RHO_STAGGER, N), grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), grid.n, grid.dx
    ρ = collect(interior_view(Array(ρd), grid))
    E = map(c -> Array(getfield(st.fields, c)), (:Ex, :Ey, :Ez)[1:N])
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), N)), N)
    divE = [sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:N) for j in CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), N))]
    return (; divE, ρ)
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
