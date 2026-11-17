# Periodic guard cells: `fill_guards!` copies the interior into the guards, `fold_guards!` adds
# the guards back into the interior cells they alias and zeroes them. They are adjoints:
# ⟨fill x, y⟩ = ⟨x, fold y⟩ over the full arrays (test T-133). Requires n ≥ 2g.

@kernel function fill_guards_kernel!(a, n, g)
    k = @index(Global, Linear)          # 1:2g
    @inbounds if k <= g
        a[k] = a[k + n]                 # left guard k ← interior k - g + n
    else
        j = k - g                       # right guard n + g + j ← interior j
        a[n + g + j] = a[g + j]
    end
end

@kernel function fold_guards_kernel!(a, n, g)
    k = @index(Global, Linear)          # 1:2g, each interior target receives from one guard
    @inbounds if k <= g
        a[k + n] += a[k]
        a[k] = zero(eltype(a))
    else
        j = k - g
        a[g + j] += a[n + g + j]
        a[n + g + j] = zero(eltype(a))
    end
end

"""
    fill_guards!(a, grid; workgroupsize = nothing)

Copy the periodic images of the interior into the guard cells of `a`. For N ≥ 2 axis by axis,
the edges and corners included.
"""
function fill_guards!(a::AbstractVector, grid::PeriodicGrid{1}; workgroupsize = nothing)
    n, g = ncells(grid), guards(grid)
    launcher(fill_guards_kernel!, get_backend(a), workgroupsize)(a, n, g; ndrange = 2g)
    return a
end

"""
    fold_guards!(a, grid; workgroupsize = nothing)

Add the guard cells of `a` into the interior cells they alias periodically, then zero them.
For N ≥ 2 axis by axis in the reverse order of `fill_guards!`, so the two stay adjoint (T-263,
T-871).
"""
function fold_guards!(a::AbstractVector, grid::PeriodicGrid{1}; workgroupsize = nothing)
    n, g = ncells(grid), guards(grid)
    launcher(fold_guards_kernel!, get_backend(a), workgroupsize)(a, n, g; ndrange = 2g)
    return a
end

# N ≥ 2: axis by axis, one kernel for every axis and every N. The pass of axis d runs over the
# 2g guard layers of that axis, over the whole extent (guards included) of the axes before d and
# over the interior of the axes after d (`guard_pass`). `fill_guards!` takes the axes in
# increasing order, so the pass of axis d also copies the guards the earlier passes filled, and
# the edges and corners receive the images of the interior's; at N = 2 that is axis 1 on the
# interior rows of axis 2, then axis 2 on every column. `fold_guards!` is its transpose: the same
# passes in decreasing order of the axis (a corner moves into the guards of the axes before, and
# from there into the interior). Each work item writes its own entries, so the result does not
# depend on the launch shape.

# The launch range of the pass of axis d and the offset that turns a work item's index into an
# array index (zero along d and before it, g after it).
@inline guard_pass(n::NTuple{N,Int}, g::Int, d::Int) where {N} =
    (ntuple(e -> e == d ? 2g : e < d ? n[e] + 2g : n[e], Val(N)), ntuple(e -> e > d ? g : 0, Val(N)))

# The array index of a work item with its index along axis d replaced by `i`.
@inline along(I::CartesianIndex{N}, off::NTuple{N,Int}, ::Val{d}, i::Int) where {N,d} =
    CartesianIndex(ntuple(e -> e == d ? i : I[e] + off[e], Val(N)))

@kernel function fill_guards_axis_kernel!(a, vd::Val{d}, nd::Int, g::Int, off) where {d}
    I = @index(Global, Cartesian)
    k = I[d]
    @inbounds if k <= g
        a[along(I, off, vd, k)] = a[along(I, off, vd, k + nd)]
    else
        m = k - g
        a[along(I, off, vd, nd + g + m)] = a[along(I, off, vd, g + m)]
    end
end

@kernel function fold_guards_axis_kernel!(a, vd::Val{d}, nd::Int, g::Int, off) where {d}
    I = @index(Global, Cartesian)
    k = I[d]
    @inbounds if k <= g
        a[along(I, off, vd, k + nd)] += a[along(I, off, vd, k)]
        a[along(I, off, vd, k)] = zero(eltype(a))
    else
        m = k - g
        a[along(I, off, vd, g + m)] += a[along(I, off, vd, nd + g + m)]
        a[along(I, off, vd, nd + g + m)] = zero(eltype(a))
    end
end

"The axes 1…N as a tuple of `Val`s: a loop over them (`foreach_axis`) has a static axis in every iteration."
@inline axis_vals(::Val{N}) where {N} = ntuple(Val, Val(N))
@inline axis_of(::Val{d}) where {d} = d
@inline foreach_axis(f::F, ::Tuple{}) where {F} = nothing
@inline foreach_axis(f::F, vals::Tuple) where {F} = (f(first(vals)); foreach_axis(f, Base.tail(vals)))

function guard_pass!(kernel, a, grid::PeriodicGrid{N}, vd::Val{d}, workgroupsize) where {N,d}
    n, g = ncells(grid), guards(grid)
    range, off = guard_pass(n, g, d)
    launcher(kernel, get_backend(a), workgroupsize)(a, vd, n[d], g, off; ndrange = range)
    return a
end

function fill_guards!(a::AbstractArray{<:Any,N}, grid::PeriodicGrid{N}; workgroupsize = nothing) where {N}
    guards(grid) == 0 && return a
    foreach_axis(vd -> guard_pass!(fill_guards_axis_kernel!, a, grid, vd, workgroupsize), axis_vals(Val(N)))
    return a
end

function fold_guards!(a::AbstractArray{<:Any,N}, grid::PeriodicGrid{N}; workgroupsize = nothing) where {N}
    guards(grid) == 0 && return a
    foreach_axis(vd -> guard_pass!(fold_guards_axis_kernel!, a, grid, vd, workgroupsize), reverse(axis_vals(Val(N))))
    return a
end
