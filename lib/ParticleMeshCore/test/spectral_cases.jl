# The computations of the WP-021 tests (T-310 to T-316: the periodic spectral inverse of the
# discrete Laplacian and the forward-difference gradient), shared by spectral.jl and by
# threads_worker.jl, which repeats the digests with 1 and 4 threads. Needs setup.jl, digest.jl,
# engine_cases.jl (`WG_SIZES`, `varied`) and engine2d_cases.jl (`grid2`).

"The interior source of T-310 and T-312 at node `j`: a smooth part, a constant (a nonzero mean) and a high-frequency part."
solve_value(j1, j2) = sin(0.37j1) + cos(1.1j2) / 5 + 3 / 10 + sin(0.9j1 * j2) / 2
solve_value(j) = solve_value(j, j % 5 + 1)

"""
    solve_input(T, grid) -> s

A field on `grid` holding `solve_value` on the interior (for duals with partials that vary per
node) and NaN in every guard, which the solve must not read.
"""
function solve_input(::Type{T}, grid::PeriodicGrid{2}) where {T}
    s = fill!(field(backend, T, grid), T(NaN))
    g = guards(grid)
    for j2 in 1:ncells(grid)[2], j1 in 1:ncells(grid)[1]
        s[g + j1, g + j2] = varied(T, solve_value(j1, j2), j1 + 7j2)
    end
    return s
end
function solve_input(::Type{T}, grid::PeriodicGrid{1}) where {T}
    s = fill!(field(backend, T, grid), T(NaN))
    for j in 1:ncells(grid)
        s[guards(grid) + j] = varied(T, solve_value(j), j)
    end
    return s
end

"`(φ, G)` of `s` on `grid`: `inverse_laplacian_periodic!` into a fresh field, then `gradient_periodic!`."
function solved(s, grid::PeriodicGrid{N}; workgroupsize = nothing,
                ws = SpectralWorkspace(backend, eltype(s), grid; workgroupsize)) where {N}
    T = eltype(s)
    φ = inverse_laplacian_periodic!(field(backend, T, grid), s, grid, ws; workgroupsize)
    G = gradient_periodic!(ntuple(_ -> field(backend, T, grid), Val(N)), φ, grid; workgroupsize)
    return φ, G
end

"The interior of `a` as a plain array (`Array` of the interior view)."
interior_of(a, grid) = collect(interior_view(a, grid))

"""
    divergence_residual(G, s, grid) -> Array

`Σ_d (G[d][j] - G[d][j - e_d])/Δ_d - (s[j] - mean(s))` at every interior node, periodically: the
backward-difference divergence that matches the forward-difference gradient. Test support only.
"""
function divergence_residual(G, s, grid::PeriodicGrid{N}) where {N}
    n, h = grid.n, grid.dx
    si = interior_of(s, grid)
    s̄ = sum(si) / prod(n)
    Gi = map(Gd -> interior_of(Gd, grid), G)
    r = similar(si)
    for j in CartesianIndices(si)
        acc = -(si[j] - s̄)
        for d in 1:N
            jm = CartesianIndex(ntuple(e -> e == d ? mod1(j[e] - 1, n[e]) : j[e], Val(N)))
            acc += (Gi[d][j] - Gi[d][jm]) / h[d]
        end
        r[j] = acc
    end
    return r
end

"‖L‖ = Σ_d 4/Δ_d², the largest magnitude of the discrete Laplacian's symbol on `grid`."
laplacian_norm(grid) = sum(4 ./ Float64.(grid.dx) .^ 2)

"Every guard of `a` is a bitwise copy of its periodic interior image."
function guards_are_images(a, grid::PeriodicGrid{N}) where {N}
    n, g = grid.n, guards(grid)
    image = [a[CartesianIndex(ntuple(d -> mod1(I[d] - g, n[d]) + g, Val(N)))] for I in CartesianIndices(a)]
    return same_bits(a, image)
end

"Grids of the T-312 digests: even and odd sizes per axis (the odd ones have no Nyquist mode)."
const SOLVE_GRIDS_1D = (64, 37)
const SOLVE_GRIDS_2D = ((16, 12), (13, 7))

"""
    spectral_digests(; wgs = WG_SIZES) -> Vector{String}

`"psolve|type|case|workgroupsize|digest"` of φ and every `G[d]` (guards included) for the T-312
grids, N = 1 and N = 2, every type of the matrix and the launch shapes `wgs`.
"""
function spectral_digests(; wgs = WG_SIZES)
    lines = String[]
    for T in TYPES, wg in wgs
        L = type_label(T)
        for n in SOLVE_GRIDS_1D
            grid = PeriodicGrid(n, basetype(T)(3 // 8), 2)
            φ, G = solved(solve_input(T, grid), grid; workgroupsize = wg)
            push!(lines, "psolve|$L|n=$n|$wg|$(fnv_digest(φ, G...))")
        end
        for n in SOLVE_GRIDS_2D
            grid = grid2(T, 2; n)
            φ, G = solved(solve_input(T, grid), grid; workgroupsize = wg)
            push!(lines, "psolve|$L|n=$(n[1])x$(n[2])|$wg|$(fnv_digest(φ, G...))")
        end
    end
    return lines
end
