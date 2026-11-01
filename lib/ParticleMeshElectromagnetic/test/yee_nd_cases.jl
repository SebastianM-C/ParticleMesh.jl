# Shared by runtests.jl and the T-327 worker: the grids, the random fields and the bitwise digests
# of the WP-022 solver tests (yee_nd.jl), at N = 1 and N = 2.

const E_KEYS = (:Ex, :Ey, :Ez)
const B_KEYS = (:Bx, :By, :Bz)
const J_KEYS = (:Jx, :Jy, :Jz)
const EB_KEYS = (E_KEYS..., B_KEYS...)

"""
    nd_grid(Val(N), B; scale = 1, g = 2) -> PeriodicGrid{N,B}

The grids of the WP-022 tests, cell sizes in `B`: at N = 1 `48·scale` cells on 2π; at N = 2
`(24, 16)·scale` cells on (2π, 3π/2), so Δx = π/12 and Δy = 3π/32 differ (a non-square grid).
"""
nd_grid(::Val{1}, ::Type{B}; scale = 1, g = 2) where {B} = PeriodicGrid(48scale, 2 * B(π) / (48scale), g)
nd_grid(::Val{2}, ::Type{B}; scale = 1, g = 2) where {B} =
    PeriodicGrid((24scale, 16scale), (2 * B(π) / (24scale), 3 * B(π) / 2 / (16scale)), g)

"A random value of `T` in [−1/2, 1/2), with independent random partials for duals."
rand_T(rng, ::Type{T}) where {T<:AbstractFloat} = T(rand(rng) - 1 / 2)
rand_T(rng, ::Type{Dual{Nothing,V,P}}) where {V,P} = Dual{Nothing}(V(rand(rng) - 1 / 2), ntuple(_ -> V(rand(rng) - 1 / 2), P)...)

"""
    random_fields(T, grid; seed, current = true) -> fields

The fields of `field_layout` on `grid`, every E and B component random on the interior (and J
when `current`), guards filled: a generic state, no mode in particular.
"""
function random_fields(::Type{T}, grid::PeriodicGrid; seed, current = true) where {T}
    rng = Xoshiro(seed)
    f = em_fields(CPU(), T, grid)
    keys = current ? (EB_KEYS..., J_KEYS...) : EB_KEYS
    for k in keys
        a = getfield(f, k)
        v = interior_view(a, grid)
        for I in eachindex(v)
            v[I] = rand_T(rng, T)
        end
        fill_guards!(a, grid)
    end
    return f
end

"A hash of the bits of the nine field arrays (guards included)."
function fields_digest(f)
    h = UInt64(0)
    for a in values(f)
        h = hash(bits(a), h)
    end
    return h
end

"""
    yee_nd_digests(; workgroupsize = nothing) -> [(N, T, digest)]

T-327: 50 steps of `solve_fields!` at 0.95 of the Courant limit from random fields with a random
current, at N = 1 and N = 2 for every element type, one digest per run.
"""
function yee_nd_digests(; workgroupsize = nothing)
    out = Tuple{Int,DataType,UInt64}[]
    for N in (1, 2), T in TYPES
        grid = nd_grid(Val(N), basetype(T))
        f = random_fields(T, grid; seed = 3270 + N)
        dt = basetype(T)(95 // 100) * courant_limit(Yee(), grid)
        for _ in 1:50
            solve_fields!(Yee(), f, grid, dt; workgroupsize)
        end
        push!(out, (N, T, fields_digest(f)))
    end
    return out
end
