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

Copy the periodic images of the interior into the guard cells of `a`. For N = 2 axis by axis,
the corners included.
"""
function fill_guards!(a::AbstractVector, grid::PeriodicGrid{1}; workgroupsize = nothing)
    n, g = ncells(grid), guards(grid)
    launcher(fill_guards_kernel!, get_backend(a), workgroupsize)(a, n, g; ndrange = 2g)
    return a
end

"""
    fold_guards!(a, grid; workgroupsize = nothing)

Add the guard cells of `a` into the interior cells they alias periodically, then zero them.
For N = 2 axis by axis in the reverse order of `fill_guards!`, so the two stay adjoint (T-263).
"""
function fold_guards!(a::AbstractVector, grid::PeriodicGrid{1}; workgroupsize = nothing)
    n, g = ncells(grid), guards(grid)
    launcher(fold_guards_kernel!, get_backend(a), workgroupsize)(a, n, g; ndrange = 2g)
    return a
end

# N = 2: axis by axis. `fill_guards!` fills axis 1 on the interior rows of axis 2, then axis 2 on
# every column, axis-1 guards included, so the corners receive the images of the interior
# corners. `fold_guards!` is its transpose: axis 2 on every column first (the corners move into
# the axis-1 guards of interior rows), then axis 1 on the interior rows. Each work item writes
# its own entries, so the result does not depend on the launch shape.

@kernel function fill_guards_axis1_kernel!(a, n1, g)    # ndrange (2g, n2)
    k, j = @index(Global, NTuple)
    r = j + g
    @inbounds if k <= g
        a[k, r] = a[k + n1, r]
    else
        m = k - g
        a[n1 + g + m, r] = a[g + m, r]
    end
end

@kernel function fill_guards_axis2_kernel!(a, n2, g)    # ndrange (n1 + 2g, 2g)
    i, k = @index(Global, NTuple)
    @inbounds if k <= g
        a[i, k] = a[i, k + n2]
    else
        m = k - g
        a[i, n2 + g + m] = a[i, g + m]
    end
end

@kernel function fold_guards_axis1_kernel!(a, n1, g)    # ndrange (2g, n2)
    k, j = @index(Global, NTuple)
    r = j + g
    @inbounds if k <= g
        a[k + n1, r] += a[k, r]
        a[k, r] = zero(eltype(a))
    else
        m = k - g
        a[g + m, r] += a[n1 + g + m, r]
        a[n1 + g + m, r] = zero(eltype(a))
    end
end

@kernel function fold_guards_axis2_kernel!(a, n2, g)    # ndrange (n1 + 2g, 2g)
    i, k = @index(Global, NTuple)
    @inbounds if k <= g
        a[i, k + n2] += a[i, k]
        a[i, k] = zero(eltype(a))
    else
        m = k - g
        a[i, g + m] += a[i, n2 + g + m]
        a[i, n2 + g + m] = zero(eltype(a))
    end
end

function fill_guards!(a::AbstractMatrix, grid::PeriodicGrid{2}; workgroupsize = nothing)
    (n1, n2), g = ncells(grid), guards(grid)
    g == 0 && return a
    backend = get_backend(a)
    launcher(fill_guards_axis1_kernel!, backend, workgroupsize)(a, n1, g; ndrange = (2g, n2))
    launcher(fill_guards_axis2_kernel!, backend, workgroupsize)(a, n2, g; ndrange = (n1 + 2g, 2g))
    return a
end

function fold_guards!(a::AbstractMatrix, grid::PeriodicGrid{2}; workgroupsize = nothing)
    (n1, n2), g = ncells(grid), guards(grid)
    g == 0 && return a
    backend = get_backend(a)
    launcher(fold_guards_axis2_kernel!, backend, workgroupsize)(a, n2, g; ndrange = (n1 + 2g, 2g))
    launcher(fold_guards_axis1_kernel!, backend, workgroupsize)(a, n1, g; ndrange = (2g, n2))
    return a
end
