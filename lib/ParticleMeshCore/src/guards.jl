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
    fill_guards!(a, grid)

Copy the periodic images of the interior into the guard cells of `a`.
"""
function fill_guards!(a::AbstractVector, grid::PeriodicGrid{1})
    n, g = ncells(grid), guards(grid)
    fill_guards_kernel!(get_backend(a))(a, n, g; ndrange = 2g)
    return a
end

"""
    fold_guards!(a, grid)

Add the guard cells of `a` into the interior cells they alias periodically, then zero them.
"""
function fold_guards!(a::AbstractVector, grid::PeriodicGrid{1})
    n, g = ncells(grid), guards(grid)
    fold_guards_kernel!(get_backend(a))(a, n, g; ndrange = 2g)
    return a
end
