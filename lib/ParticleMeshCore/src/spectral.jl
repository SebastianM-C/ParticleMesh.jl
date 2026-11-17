# The periodic spectral inverse of the discrete Laplacian and the forward-difference gradient:
# physics-free mesh operators. A module names what they solve.
#
# Every kernel below is written once for N axes: a work item is a Cartesian index of the
# spectrum (`prod(n)` entries, no guards) and sums one line along the transformed axis.

"""
    SpectralWorkspace(backend, T, grid; workgroupsize = nothing)

The scratch of [`inverse_laplacian_periodic!`](@ref) on `grid`, allocated once on `backend`
with element type `T` (that of the fields it will solve): per axis `d` the twiddle tables
`cos(2πm/n_d)` and `sin(2πm/n_d)` for `m = 0:n_d-1`, and `min(N, 2)` pairs of real and imaginary
spectrum arrays of size `ncells(grid)` (the transforms of the axes after the first go back and
forth between two pairs).

The tables are written by a kernel from `sincospi(T(2m′)/T(n_d))` with `m′ = min(m, n_d - m)`,
the sine negated for `m > n_d/2`, so that the integer reduction of the angle is exact and the
tables are exactly conjugate-symmetric. They depend on `ncells(grid)` only; the solve reads the
cell sizes from the grid it is given. `workgroupsize` selects the launch shape of the table
kernel only (contract 12).
"""
struct SpectralWorkspace{N,A<:AbstractArray,V<:AbstractVector,M}
    n::NTuple{N,Int}
    cos::NTuple{N,V}
    sin::NTuple{N,V}
    re::NTuple{M,A}
    im::NTuple{M,A}
end

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
    N <= 3 || throw(ArgumentError("SpectralWorkspace: N = 1, 2 and 3 are implemented, not N = $N"))
    n = grid.n
    tables = ntuple(Val(N)) do d
        c = KernelAbstractions.allocate(backend, T, n[d])
        s = KernelAbstractions.allocate(backend, T, n[d])
        launcher(twiddle_kernel!, backend, workgroupsize)(c, s, n[d]; ndrange = n[d])
        (c, s)
    end
    pairs = Val(min(N, 2))
    re = ntuple(_ -> KernelAbstractions.allocate(backend, T, n...), pairs)
    im = ntuple(_ -> KernelAbstractions.allocate(backend, T, n...), pairs)
    return SpectralWorkspace{N,eltype(re),eltype(first.(tables)),min(N, 2)}(n, first.(tables), last.(tables), re, im)
end

# The index of the entry `j` of axis `d` on the line of the work item `I`, shifted by `g` guards
# on every axis (g = 0 for the spectrum arrays).
@inline line_index(I::CartesianIndex{N}, ::Val{d}, j::Int, g::Int) where {N,d} =
    CartesianIndex(ntuple(e -> (e == d ? j : I[e]) + g, Val(N)))

# One work item per output mode k of axis 1 and index of the other axes: the DFT of the real
# interior line, e^{-2πi k j₁/n₁}, summed serially over j₁ = 0:n₁-1 in increasing order. The
# twiddle index m = k·j₁ mod n₁ advances by k per term with one conditional subtraction, exact in
# integers.
@kernel function dft_real_axis1_kernel!(re, im, @Const(s), @Const(c), @Const(sn), n1::Int, g::Int)
    I = @index(Global, Cartesian)
    T = eltype(re)
    step = I[1] - 1
    m = 0
    ar, ai = zero(T), zero(T)
    @inbounds begin
        for j1 in 1:n1
            x = s[line_index(I, Val(1), j1, g)]
            ar += x * c[m + 1]
            ai -= x * sn[m + 1]
            m += step
            m = ifelse(m >= n1, m - n1, m)
        end
        re[I] = ar
        im[I] = ai
    end
end

# One work item per output mode k of axis d and index of the other axes: the complex DFT along
# axis d with e^{σ·2πi k j/n_d} (σ = -1 forward, +1 inverse), summed serially over j = 0:n_d-1 in
# increasing order.
@kernel function dft_axis_kernel!(ore, oim, @Const(ire), @Const(iim), @Const(c), @Const(sn), nd::Int, σ, vd::Val{d}) where {d}
    I = @index(Global, Cartesian)
    T = eltype(ore)
    step = I[d] - 1
    m = 0
    ar, ai = zero(T), zero(T)
    @inbounds begin
        for j in 1:nd
            J = line_index(I, vd, j, 0)
            x, y = ire[J], iim[J]
            cc, ss = c[m + 1], σ * sn[m + 1]
            ar += x * cc - y * ss
            ai += y * cc + x * ss
            m += step
            m = ifelse(m >= nd, m - nd, m)
        end
        ore[I] = ar
        oim[I] = ai
    end
end

# One axis of the symbol, (2 sin(π k/n)/Δ)², with the sine's argument reduced exactly to
# min(k, n - k)/n.
@inline function symbol_axis(::Type{T}, k::Int, n::Int, h) where {T}
    a = T(2) * sinpi(T(min(k - 1, n - k + 1)) / T(n)) / h
    return a * a
end

# The entries of a tuple added from the first to the last.
@inline sum_in_order(t::Tuple{Any}) = t[1]
@inline sum_in_order(t::Tuple) = sum_in_order(Base.front(t)) + t[end]

# The spectrum divided by minus the Yee-consistent symbol K² = Σ_d (2 sin(π k_d/n_d)/Δ_d)², the
# axes added in increasing order; the k = 0 mode set to zero.
@kernel function inverse_symbol_kernel!(re, im, n::NTuple{N,Int}, h::NTuple{N,Any}) where {N}
    I = @index(Global, Cartesian)
    T = eltype(re)
    K2 = sum_in_order(ntuple(d -> symbol_axis(T, I[d], n[d], h[d]), Val(N)))
    origin = all(ntuple(d -> I[d] == 1, Val(N)))
    @inbounds begin
        re[I] = ifelse(origin, zero(T), -re[I] / ifelse(origin, one(T), K2))
        im[I] = ifelse(origin, zero(T), -im[I] / ifelse(origin, one(T), K2))
    end
end

# One work item per output node j₁ of axis 1 and index of the other axes: the real part of the
# inverse DFT along axis 1, e^{+2πi k j₁/n₁}, summed serially over k = 0:n₁-1 in increasing order,
# then divided by prod(n) and written into the interior of φ.
@kernel function idft_real_axis1_kernel!(φ, @Const(re), @Const(im), @Const(c), @Const(sn), n1::Int, g::Int, nrm)
    I = @index(Global, Cartesian)
    T = eltype(φ)
    step = I[1] - 1
    m = 0
    acc = zero(T)
    @inbounds begin
        for k in 1:n1
            K = line_index(I, Val(1), k, 0)
            acc += re[K] * c[m + 1] - im[K] * sn[m + 1]
            m += step
            m = ifelse(m >= n1, m - n1, m)
        end
        φ[line_index(I, Val(1), I[1], g)] = acc / nrm
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

Method: the DFT of `s` per axis (axis 1 from the real interior, then the other axes in increasing order),
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

`N = 1`, `2` and `3`; `ws` must be `SpectralWorkspace(backend, T, grid)` for the element type of
`φ` and `s` and the same `ncells(grid)`, and the grid needs at least one guard cell (checked).
Allocation per call does not grow with the grid size (T-315). `workgroupsize` selects the launch
shape only (contract 12).
"""
# The transform along each axis of `vals` in turn (static axes, so every launch is inferred): the
# spectrum moves from the pair `cur` to the pair `other`, which then swap. Returns `(cur, other)`.
dft_axes!(axis!, ws, n, σ, cur::Int, other::Int, ::Tuple{}) = (cur, other)
function dft_axes!(axis!, ws, n, σ, cur::Int, other::Int, vals::Tuple{Val{d},Vararg}) where {d}
    axis!(ws.re[other], ws.im[other], ws.re[cur], ws.im[cur], ws.cos[d], ws.sin[d], n[d], σ, first(vals); ndrange = n)
    return dft_axes!(axis!, ws, n, σ, other, cur, Base.tail(vals))
end

function inverse_laplacian_periodic!(φ::AbstractArray{T,N}, s::AbstractArray{T,N}, grid::PeriodicGrid{N}, ws::SpectralWorkspace{N};
                                     workgroupsize = nothing) where {T,N}
    check_spectral(φ, s, grid, ws)
    backend = get_backend(φ)
    n, g = grid.n, guards(grid)
    h = map(T, grid.dx)
    # The spectrum goes back and forth between two pairs of arrays: `cur` holds it, `other` is free.
    cur, other = 1, min(N, 2)
    launcher(dft_real_axis1_kernel!, backend, workgroupsize)(ws.re[cur], ws.im[cur], s, ws.cos[1], ws.sin[1], n[1], g; ndrange = n)
    axis! = launcher(dft_axis_kernel!, backend, workgroupsize)
    cur, other = dft_axes!(axis!, ws, n, -one(T), cur, other, Base.tail(axis_vals(Val(N))))
    launcher(inverse_symbol_kernel!, backend, workgroupsize)(ws.re[cur], ws.im[cur], n, h; ndrange = n)
    cur, other = dft_axes!(axis!, ws, n, one(T), cur, other, reverse(Base.tail(axis_vals(Val(N)))))
    launcher(idft_real_axis1_kernel!, backend, workgroupsize)(φ, ws.re[cur], ws.im[cur], ws.cos[1], ws.sin[1], n[1], g, T(prod(n));
                                                             ndrange = n)
    fill_guards!(φ, grid; workgroupsize)
    return φ
end

# One work item per interior node: the periodic forward difference along axis d,
# (φ[j + e_d] - φ[j])/h, stored at j; the neighbour of the last node is the first (no guard read).
@kernel function forward_difference_kernel!(G, @Const(φ), nd::Int, g::Int, vd::Val{d}, h) where {d}
    I = @index(Global, Cartesian)
    jp = ifelse(I[d] + 1 > nd, 1, I[d] + 1)
    J = line_index(I, vd, I[d], g)
    @inbounds G[J] = (φ[line_index(I, vd, jp, g)] - φ[J]) / h
end

"""
    gradient_periodic!(G::NTuple{N}, φ, grid; workgroupsize = nothing)

The periodic forward-difference gradient of the field `φ`: `G[d][j] = (φ[j + e_d] - φ[j])/Δ_d` at
every interior index `j`, written into the fields `G[d]` with their guards filled (the guards of
`φ` are not read: the neighbour of the last interior node is the first). `G[d][j]` is the face
half a cell to the right of node `j` along axis `d`, stored at index `j`: with `φ` on ρ's nodes
it is `Center()` on axis `d` and `Node()` on the others, which for `N ≥ 2` is the Yee layout of
the electric field (at `N = 2`, `Ex` at (Center, Node) and `Ey` at (Node, Center)) (the module's paragraph of `docs/plan/architecture-draft.md` §0.7),
and for `N = 1` the layout of `Ex` and of [`cumsum_periodic!`](@ref)'s output. Its
backward-difference divergence `Σ_d (G[d][j] - G[d][j - e_d])/Δ_d` is the discrete Laplacian of
[`inverse_laplacian_periodic!`](@ref), so after that solve it equals `s - mean(s)` to roundoff
(T-310); at `N = 1`, `G[1]` agrees with `cumsum_periodic!(out, s, grid)` to roundoff (T-313).

One work item per interior node and axis, each writing its own entry, so the bits do not depend
on the launch shape. No `G[d]` may be `φ` (checked); `N = 1`, `2` and `3`, at least one guard cell.
"""
function gradient_periodic!(G::NTuple{N,AbstractArray{T,N}}, φ::AbstractArray{T,N}, grid::PeriodicGrid{N};
                            workgroupsize = nothing) where {T,N}
    guards(grid) >= 1 || throw(ArgumentError("gradient_periodic! needs at least 1 guard cell, the grid has $(guards(grid))"))
    full = grid.n .+ 2guards(grid)
    size(φ) == full || throw(ArgumentError("φ has size $(size(φ)), the grid's fields have $full"))
    all(Gd -> size(Gd) == full, G) || throw(ArgumentError("every G[d] must have the grid's field size $full"))
    any(Gd -> Gd === φ, G) && throw(ArgumentError("gradient_periodic! cannot write the gradient into φ"))
    backend = get_backend(φ)
    kernel! = launcher(forward_difference_kernel!, backend, workgroupsize)
    foreach_axis(axis_vals(Val(N))) do vd
        d = axis_of(vd)
        kernel!(G[d], φ, grid.n[d], guards(grid), vd, T(grid.dx[d]); ndrange = grid.n)
        fill_guards!(G[d], grid; workgroupsize)
    end
    return G
end
