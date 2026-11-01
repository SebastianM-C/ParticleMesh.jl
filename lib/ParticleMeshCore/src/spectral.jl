# The periodic spectral inverse of the discrete Laplacian and the forward-difference gradient:
# physics-free mesh operators. A module names what they solve.
#
# Every kernel below runs on two-dimensional arrays; an N = 1 field enters as an n × 1 matrix
# (`as_lines`) with no guards on the second axis, so N = 1 and N = 2 share one set of kernels.

"""
    SpectralWorkspace(backend, T, grid; workgroupsize = nothing)

The scratch of [`inverse_laplacian_periodic!`](@ref) on `grid`, allocated once on `backend`
with element type `T` (that of the fields it will solve): per axis `d` the twiddle tables
`cos(2πm/n_d)` and `sin(2πm/n_d)` for `m = 0:n_d-1`, and `N` pairs of real and imaginary
spectrum arrays of `prod(ncells(grid))` entries each (for `N = 1` stored as `n × 1`).

The tables are written by a kernel from `sincospi(T(2m′)/T(n_d))` with `m′ = min(m, n_d - m)`,
the sine negated for `m > n_d/2`, so that the integer reduction of the angle is exact and the
tables are exactly conjugate-symmetric. They depend on `ncells(grid)` only; the solve reads the
cell sizes from the grid it is given. `workgroupsize` selects the launch shape of the table
kernel only (contract 12).
"""
struct SpectralWorkspace{N,A<:AbstractMatrix,V<:AbstractVector}
    n::NTuple{N,Int}
    cos::NTuple{N,V}
    sin::NTuple{N,V}
    re::NTuple{N,A}
    im::NTuple{N,A}
end

"`(n₁, n₂)` of the line layout: an `N = 1` grid is `n × 1`."
@inline line_shape(n::NTuple{1,Int}) = (n[1], 1)
@inline line_shape(n::NTuple{2,Int}) = n
"An `N = 1` field as an `(n + 2g) × 1` matrix sharing its memory; an `N = 2` field as itself."
@inline as_lines(a::AbstractVector) = reshape(a, length(a), 1)
@inline as_lines(a::AbstractMatrix) = a
"The guard offset of axis 2 in the line layout: none for `N = 1`."
@inline line_guard2(grid::PeriodicGrid{1}) = 0
@inline line_guard2(grid::PeriodicGrid{2}) = guards(grid)

@kernel function twiddle_kernel!(c, s, n::Int)
    i = @index(Global, Linear)
    T = eltype(c)
    m = i - 1
    r = min(m, n - m)
    @inbounds begin
        sn, cs = sincospi(T(2r) / T(n))
        c[i] = cs
        s[i] = ifelse(m > n - m, -sn, sn)
    end
end

function SpectralWorkspace(backend, ::Type{T}, grid::PeriodicGrid{N}; workgroupsize = nothing) where {T,N}
    N <= 2 || throw(ArgumentError("SpectralWorkspace: N = 1 and N = 2 are implemented, not N = $N"))
    n = grid.n
    tables = ntuple(Val(N)) do d
        c = KernelAbstractions.allocate(backend, T, n[d])
        s = KernelAbstractions.allocate(backend, T, n[d])
        launcher(twiddle_kernel!, backend, workgroupsize)(c, s, n[d]; ndrange = n[d])
        (c, s)
    end
    shape = line_shape(n)
    re = ntuple(_ -> KernelAbstractions.allocate(backend, T, shape...), Val(N))
    im = ntuple(_ -> KernelAbstractions.allocate(backend, T, shape...), Val(N))
    return SpectralWorkspace{N,eltype(re),eltype(first.(tables))}(n, first.(tables), last.(tables), re, im)
end

# One work item per output mode k of axis 1 and line j of axis 2: the DFT of the real interior
# line, e^{-2πi k j₁/n₁}, summed serially over j₁ = 0:n₁-1 in increasing order. The twiddle index
# m = k·j₁ mod n₁ advances by k per term with one conditional subtraction, exact in integers.
@kernel function dft_real_axis1_kernel!(re, im, @Const(s), @Const(c), @Const(sn), n1::Int, g1::Int, g2::Int)
    k, j = @index(Global, NTuple)
    T = eltype(re)
    step = k - 1
    m = 0
    ar, ai = zero(T), zero(T)
    @inbounds begin
        for j1 in 1:n1
            x = s[g1 + j1, g2 + j]
            ar += x * c[m + 1]
            ai -= x * sn[m + 1]
            m += step
            m = ifelse(m >= n1, m - n1, m)
        end
        re[k, j] = ar
        im[k, j] = ai
    end
end

# One work item per entry k₁ of axis 1 and output mode k of axis 2: the complex DFT along axis 2
# with e^{σ·2πi k j₂/n₂} (σ = -1 forward, +1 inverse), summed serially over j₂ = 0:n₂-1 in
# increasing order.
@kernel function dft_axis2_kernel!(ore, oim, @Const(ire), @Const(iim), @Const(c), @Const(sn), n2::Int, σ)
    i, k = @index(Global, NTuple)
    T = eltype(ore)
    step = k - 1
    m = 0
    ar, ai = zero(T), zero(T)
    @inbounds begin
        for j2 in 1:n2
            x, y = ire[i, j2], iim[i, j2]
            cc, ss = c[m + 1], σ * sn[m + 1]
            ar += x * cc - y * ss
            ai += y * cc + x * ss
            m += step
            m = ifelse(m >= n2, m - n2, m)
        end
        ore[i, k] = ar
        oim[i, k] = ai
    end
end

# The spectrum divided by minus the Yee-consistent symbol K² = Σ_d (2 sin(π k_d/n_d)/Δ_d)², with
# the sine's argument reduced exactly to min(k_d, n_d - k_d)/n_d; the k = 0 mode set to zero.
@kernel function inverse_symbol_kernel!(re, im, n1::Int, n2::Int, h1, h2)
    k1, k2 = @index(Global, NTuple)
    T = eltype(re)
    r1 = min(k1 - 1, n1 - k1 + 1)
    r2 = min(k2 - 1, n2 - k2 + 1)
    a = T(2) * sinpi(T(r1) / T(n1)) / h1
    b = T(2) * sinpi(T(r2) / T(n2)) / h2
    K2 = a * a + b * b
    origin = k1 == 1 && k2 == 1
    @inbounds begin
        re[k1, k2] = ifelse(origin, zero(T), -re[k1, k2] / ifelse(origin, one(T), K2))
        im[k1, k2] = ifelse(origin, zero(T), -im[k1, k2] / ifelse(origin, one(T), K2))
    end
end

# One work item per output node j₁ of axis 1 and line j of axis 2: the real part of the inverse
# DFT along axis 1, e^{+2πi k j₁/n₁}, summed serially over k = 0:n₁-1 in increasing order, then
# divided by n₁n₂ and written into the interior of φ.
@kernel function idft_real_axis1_kernel!(φ, @Const(re), @Const(im), @Const(c), @Const(sn), n1::Int, g1::Int, g2::Int, nrm)
    j1, j = @index(Global, NTuple)
    T = eltype(φ)
    step = j1 - 1
    m = 0
    acc = zero(T)
    @inbounds begin
        for k in 1:n1
            acc += re[k, j] * c[m + 1] - im[k, j] * sn[m + 1]
            m += step
            m = ifelse(m >= n1, m - n1, m)
        end
        φ[g1 + j1, g2 + j] = acc / nrm
    end
end

function check_spectral(φ, s, grid::PeriodicGrid{N}, ws::SpectralWorkspace{N}) where {N}
    guards(grid) >= 1 || throw(ArgumentError("inverse_laplacian_periodic! needs at least 1 guard cell, the grid has $(guards(grid))"))
    ws.n == grid.n || throw(ArgumentError("the SpectralWorkspace is for $(ws.n) cells, the grid has $(grid.n)"))
    full = grid.n .+ 2guards(grid)
    size(φ) == full || throw(ArgumentError("φ has size $(size(φ)), the grid's fields have $full"))
    size(s) == full || throw(ArgumentError("s has size $(size(s)), the grid's fields have $full"))
    T = eltype(φ)
    eltype(s) === T && eltype(first(ws.re)) === T ||
        throw(ArgumentError("φ, s and the SpectralWorkspace must share one element type; got $T, $(eltype(s)), $(eltype(first(ws.re)))"))
    return nothing
end

"""
    inverse_laplacian_periodic!(φ, s, grid, ws::SpectralWorkspace; workgroupsize = nothing)

The zero-mean periodic solution `φ` of `L φ = s - mean(s)` on the interior of `grid`, written
into the field `φ` with its guards filled; `s` is a field (with guards, which are not read) and
may be `φ` itself. `L` is the discrete Laplacian `Σ_d D⁻_d D⁺_d` built from the forward difference
`D⁺_d φ[j] = (φ[j + e_d] - φ[j])/Δ_d` and the backward difference `D⁻_d G[j] = (G[j] - G[j - e_d])/Δ_d`,
periodically. With [`gradient_periodic!`](@ref), which returns `G_d = D⁺_d φ`, this gives, up to
roundoff, `Σ_d D⁻_d G_d = s - mean(s)` at every interior node and `Σ_j φ[j] = 0` (T-310, T-311).
The sign convention is the engine's: `φ` solves `+L φ = s - mean(s)` and `G = +∇φ`; a module
that wants a field `E` with `div E = s` takes `E = G` (with `s = ρ`, the electric field in the
convention `div E = ρ`), and the engine does not name the equation.

Method: the DFT of `s` per axis (axis 1 from the real interior, then axis 2 for `N = 2`),
division of every mode `k ≠ 0` by `-K²` with the Yee-consistent symbol
`K² = Σ_d (2 sin(π k_d/n_d)/Δ_d)²`, the `k = 0` mode set to zero, and the inverse DFTs in the
reverse axis order, the real part divided by `prod(n)`. The twiddles come from the tables of
`ws` and the symbol from `sinpi(min(k_d, n_d - k_d)/n_d)`, both with the angle reduced exactly in
integers.

Association order, part of the contract: every transform runs one work item per output entry
(an output mode or node of the transformed axis and one index of each other axis), which sums
the `n_d` terms of its line serially in increasing order of the summed index, starting from zero;
so the bits are fixed by the operator, not by the launch configuration, the thread count or the
backend, and backends differ only by their own arithmetic. Each transform is O(n_d) work per
entry, O(prod(n)·Σ_d n_d) per solve: meant for the field at t = 0, at test sizes
(`reports/wp-021-periodic-solve-2d.md`).

`N = 1` and `N = 2`; `ws` must be `SpectralWorkspace(backend, T, grid)` for the element type of
`φ` and `s` and the same `ncells(grid)`, and the grid needs at least one guard cell (checked).
Allocation per call does not grow with the grid size (T-315). `workgroupsize` selects the launch
shape only (contract 12).
"""
function inverse_laplacian_periodic!(φ::AbstractArray{T,N}, s::AbstractArray{T,N}, grid::PeriodicGrid{N}, ws::SpectralWorkspace{N};
                                     workgroupsize = nothing) where {T,N}
    check_spectral(φ, s, grid, ws)
    backend = get_backend(φ)
    n1, n2 = line_shape(grid.n)
    g1, g2 = guards(grid), line_guard2(grid)
    h1 = T(grid.dx[1])
    h2 = N == 1 ? one(T) : T(grid.dx[N])
    φl, sl = as_lines(φ), as_lines(s)
    re1, im1 = ws.re[1], ws.im[1]
    launcher(dft_real_axis1_kernel!, backend, workgroupsize)(re1, im1, sl, ws.cos[1], ws.sin[1], n1, g1, g2; ndrange = (n1, n2))
    if N == 2
        re2, im2 = ws.re[N], ws.im[N]
        axis2! = launcher(dft_axis2_kernel!, backend, workgroupsize)
        axis2!(re2, im2, re1, im1, ws.cos[N], ws.sin[N], n2, -one(T); ndrange = (n1, n2))
        launcher(inverse_symbol_kernel!, backend, workgroupsize)(re2, im2, n1, n2, h1, h2; ndrange = (n1, n2))
        axis2!(re1, im1, re2, im2, ws.cos[N], ws.sin[N], n2, one(T); ndrange = (n1, n2))
    else
        launcher(inverse_symbol_kernel!, backend, workgroupsize)(re1, im1, n1, n2, h1, h2; ndrange = (n1, n2))
    end
    launcher(idft_real_axis1_kernel!, backend, workgroupsize)(φl, re1, im1, ws.cos[1], ws.sin[1], n1, g1, g2, T(n1 * n2);
                                                             ndrange = (n1, n2))
    fill_guards!(φ, grid; workgroupsize)
    return φ
end

# One work item per interior node: the periodic forward difference along the axis (d1, d2),
# (φ[j + e_d] - φ[j])/h, stored at j; the neighbour of the last node is the first (no guard read).
@kernel function forward_difference_kernel!(G, @Const(φ), n1::Int, n2::Int, g1::Int, g2::Int, d1::Int, d2::Int, h)
    i, j = @index(Global, NTuple)
    ip = ifelse(i + d1 > n1, 1, i + d1)
    jp = ifelse(j + d2 > n2, 1, j + d2)
    @inbounds G[g1 + i, g2 + j] = (φ[g1 + ip, g2 + jp] - φ[g1 + i, g2 + j]) / h
end

"""
    gradient_periodic!(G::NTuple{N}, φ, grid; workgroupsize = nothing)

The periodic forward-difference gradient of the field `φ`: `G[d][j] = (φ[j + e_d] - φ[j])/Δ_d` at
every interior index `j`, written into the fields `G[d]` with their guards filled (the guards of
`φ` are not read: the neighbour of the last interior node is the first). `G[d][j]` is the face
half a cell to the right of node `j` along axis `d`, stored at index `j`: with `φ` on ρ's nodes
it is `Center()` on axis `d` and `Node()` on the others, which for `N = 2` is the Yee layout of
`Ex` at (Center, Node) and `Ey` at (Node, Center) (the module's paragraph of `docs/plan/architecture-draft.md` §0.7),
and for `N = 1` the layout of `Ex` and of [`cumsum_periodic!`](@ref)'s output. Its
backward-difference divergence `Σ_d (G[d][j] - G[d][j - e_d])/Δ_d` is the discrete Laplacian of
[`inverse_laplacian_periodic!`](@ref), so after that solve it equals `s - mean(s)` to roundoff
(T-310); at `N = 1`, `G[1]` agrees with `cumsum_periodic!(out, s, grid)` to roundoff (T-313).

One work item per interior node and axis, each writing its own entry, so the bits do not depend
on the launch shape. No `G[d]` may be `φ` (checked); `N = 1` and `N = 2`, at least one guard cell.
"""
function gradient_periodic!(G::NTuple{N,AbstractArray{T,N}}, φ::AbstractArray{T,N}, grid::PeriodicGrid{N};
                            workgroupsize = nothing) where {T,N}
    guards(grid) >= 1 || throw(ArgumentError("gradient_periodic! needs at least 1 guard cell, the grid has $(guards(grid))"))
    full = grid.n .+ 2guards(grid)
    size(φ) == full || throw(ArgumentError("φ has size $(size(φ)), the grid's fields have $full"))
    all(Gd -> size(Gd) == full, G) || throw(ArgumentError("every G[d] must have the grid's field size $full"))
    any(Gd -> Gd === φ, G) && throw(ArgumentError("gradient_periodic! cannot write the gradient into φ"))
    backend = get_backend(φ)
    n1, n2 = line_shape(grid.n)
    g1, g2 = guards(grid), line_guard2(grid)
    φl = as_lines(φ)
    kernel! = launcher(forward_difference_kernel!, backend, workgroupsize)
    for d in 1:N
        kernel!(as_lines(G[d]), φl, n1, n2, g1, g2, Int(d == 1), Int(d == 2), T(grid.dx[d]); ndrange = (n1, n2))
        fill_guards!(G[d], grid; workgroupsize)
    end
    return G
end
