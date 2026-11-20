# WP-069: gather! and deposit! as a transposed pair, on random data and for every bins choice
# (T-980). T-132, T-265 and T-873 hold the pair with the cells as bins on smooth fields; this file
# adds the nearest-node bins of WP-041 (of the deposit's own shape and stagger, and of another),
# random fields and random signed quantities with independent partials, and dead slots, at N = 1,
# 2 and 3 in one body. The body is a function of the backend (gpu_support.jl).

"""
    unit_stream(T, seed) -> () -> value of T

Uniform values in [−1, 1) from the suites' fixed LCG, as values of `T`; for duals the value and
every partial are independent draws.
"""
function unit_stream(::Type{T}, seed) where {T}
    state = Ref(UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d)
    u() = (state[] = state[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; 2 * (Float64(state[] >> 11) / 2.0^53) - 1)
    return () -> unit_value(T, u)
end
unit_value(::Type{T}, u) where {T<:AbstractFloat} = T(u())
unit_value(::Type{Dual{Nothing,V,P}}, u) where {V,P} = Dual{Nothing}(V(u()), ntuple(_ -> V(u()), P)...)

"A field of `T` on `grid` with random interior values from `next` and filled guards."
function random_field(::Type{T}, grid, next) where {T}
    a = field(backend, T, grid)
    v = interior_view(a, grid)
    for I in eachindex(v)
        v[I] = next()
    end
    return fill_guards!(a, grid)
end

transpose_grid(::Val{1}, ::Type{T}, g) where {T} = PeriodicGrid(12, basetype(T)(1 // 2), g)
transpose_grid(::Val{2}, ::Type{T}, g) where {T} = grid2(T, g)
transpose_grid(::Val{3}, ::Type{T}, g) where {T} = grid3(T, g; n = (8, 7, 6))   # room for three guard cells
# moved once, so that particles sit on both sides of every seam; two slots freed
transpose_species(::Val{1}, ::Type{T}, grid) where {T} = moved_species(T, grid, 300; seed = 9, dead = (4, 77))
transpose_species(::Val{2}, ::Type{T}, grid) where {T} = moved_species_2d(T, grid, 400; seed = 9, dead = (5, 77))
transpose_species(::Val{3}, ::Type{T}, grid) where {T} = moved_species_3d(T, grid, 400; seed = 9, dead = (5, 77))
transpose_staggers(::Val{1}) = STAGGERS
transpose_staggers(::Val{2}) = STAGGER_PAIRS
transpose_staggers(::Val{3}) = STAGGER_TRIPLES
"The same stagger on every one of N axes, bare at N = 1 as the engine's 1D calls take it."
all_axes(::Val{1}, st) = st
all_axes(::Val{N}, st) where {N} = ntuple(_ -> st, Val(N))
"`st` with the stagger of axis 1 exchanged (Node ↔ Center): a wrong partner for the control."
flip_first(st::ParticleMeshCore.Stagger) = st isa Node ? Center() : Node()
flip_first(st::Tuple) = (flip_first(st[1]), Base.tail(st)...)

"""
    transpose_partners(Val(N), shape, st) -> ((mode, bins, extra guards), ...)

The deposits T-980 holds against the gather: `Fast()` (atomics; for duals the sorted fallback),
`Reference()` by cells, by `NodeBins` of the deposit's own shape and stagger, and by `NodeBins`
of another (order 2 on nodes, or order 1 at centres where the deposit is that), which needs one
guard cell more (`check_deposit_guards`).
"""
function transpose_partners(vN::Val, shape, st)
    other = shape === Quadratic && st == all_axes(vN, Node()) ? NodeBins(Linear, all_axes(vN, Center())) :
            NodeBins(Quadratic, all_axes(vN, Node()))
    return ((Fast(), CellBins(), 0), (Reference(), CellBins(), 0), (Reference(), NodeBins(shape, st), 0), (Reference(), other, 1))
end

"""
    transposed_pair(be, T, vN, shape, st, mode, bins, extra; dep_st = st, seed)

`(lhs, rhs, scale)` of one dot-product test: `lhs = Σᵢ yᵢ · gather(fill x)ᵢ` over the alive
particles and `rhs = ⟨fold(deposit(y)), x⟩` over the interior, for a random field `x` and random
signed per-particle values `yᵢ = q·wᵢ`; `scale = Σᵢ|yᵢ|·max|x|`. The deposit runs at `dep_st`
(the control passes a wrong stagger).
"""
function transposed_pair(be, ::Type{T}, vN::Val, shape, st, mode, bins, extra; dep_st = st, seed) where {T}
    next = unit_stream(T, seed)
    grid = transpose_grid(vN, T, guard_width(shape) + extra)
    s = transpose_species(vN, T, grid)
    alive = [i for i in 1:capacity(s) if s.status[i] == ParticleMeshCore.ALIVE]
    for i in alive
        s.w[i] = next()                       # signed: the probe y is any vector on the particles
    end
    x = random_field(T, grid, next)
    out = similar(s.w)
    on(be, gather!, out, x, s, shape, st, grid)
    lhs = sum(s.q * s.w[i] * out[i] for i in alive)
    scale = sum(absmax(s.q * s.w[i]) for i in alive) * maximum(absmax, interior_view(x, grid))
    ws = Workspace(s, grid; bins)
    effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws)   # permutes the slots; lhs is already summed
    ρ = on(be, fold_guards!, on(be, deposit!, mode, field(backend, T, grid), s, shape, dep_st, grid, ws), grid)
    rhs = sum(interior_view(ρ, grid) .* interior_view(x, grid))
    return lhs, rhs, scale
end

function body_T980(be)
    C = tolC("T-980")
    for N in (1, 2, 3), T in TYPES, shape in SHAPES, st in transpose_staggers(Val(N))
        ε = base_eps(T)
        for (k, (mode, bins, extra)) in enumerate(transpose_partners(Val(N), shape, st))
            lhs, rhs, scale = transposed_pair(be, T, Val(N), shape, st, mode, bins, extra; seed = 980 + 10N + k)
            r = record!("T-980", T, absdiff(lhs, rhs) / (ε * scale), be)
            @test r <= C
        end
        # control: the deposit at another stagger is not the gather's transpose
        lhs, rhs, scale = transposed_pair(be, T, Val(N), shape, st, Reference(), CellBins(), 0; dep_st = flip_first(st), seed = 980)
        @test absdiff(lhs, rhs) > 100 * C * ε * scale
    end
end

@testset "T-980 deposit! is the transpose of gather! on random data for every bins choice, N = 1, 2, 3 [A2, A7, A8, WP-069]" begin
    body_T980(CPU())
end
