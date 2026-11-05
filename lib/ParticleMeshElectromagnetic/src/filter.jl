# The binomial current filter (A2): `passes` applications of the 3-point stencil (1, 2, 1)/4 along
# every resolved axis to Jx, Jy and Jz, after the deposited current is folded and before the mesh
# solve. On a periodic grid one pass along axis d multiplies the Fourier mode of wavenumber k_d by
#     h(k_d) = (1 + cos(k_dΔ_d))/2 = cos²(k_dΔ_d/2),
# on every stagger (a shift of the sample positions by half a cell does not change a convolution),
# so p passes over both axes transfer Π_d cos^{2p}(k_dΔ_d/2) (T-440): the mean (k = 0) is kept,
# the Nyquist mode of each axis is removed, and every mode keeps its phase (the stencil is
# symmetric). The filter is a convolution, so it commutes with the backward differences of the
# continuity equation: with the current filtered, D⁻·Eⁿ⁺¹ − D⁻·Eⁿ = F(ρⁿ⁺¹ − ρⁿ) instead of
# ρⁿ⁺¹ − ρⁿ, and Gauss's law becomes D⁻·Eⁿ − Fρⁿ = D⁻·E⁰ − Fρ⁰ for all n, to roundoff (T-441).

"""
    BinomialFilter(passes = 0)

The current filter of an [`EMNumerics`](@ref) (A2): after the deposition, `passes` applications
of the binomial stencil `(J[j − 1] + 2J[j] + J[j + 1])/4` along every resolved axis to each of
Jx, Jy and Jz, on their own staggers. One pass along axis d multiplies the mode of wavenumber
`k_d` by `cos²(k_dΔ_d/2)`, so the filter transfers `Π_d cos^{2·passes}(k_dΔ_d/2)`: the mean is
kept, each axis' Nyquist mode removed. The charge is not filtered, so with `passes > 0` Gauss's
law holds in its filtered form, `D⁻·Eⁿ − F ρⁿ` constant in time (F the filter), and
`D⁻·Eⁿ − (ρⁿ − mean ρ) = (F − 1)(ρⁿ − ρ⁰)` (T-441). With `passes = 0` (the default) the step does
no filter work at all and keeps its bits. `passes` is a host integer of the configuration, not a
parameter (§0.4); a negative count is an `ArgumentError`.
"""
struct BinomialFilter
    passes::Int
    function BinomialFilter(passes::Integer = 0)
        passes >= 0 || throw(ArgumentError("filter passes must be ≥ 0, got $passes"))
        return new(Int(passes))
    end
end

# One pass of (1, 2, 1)/4 along the axis of the unit step `e`, at the interior node I0 + g (guards
# of `src` filled). (l + r) + 2c is symmetric in l and r, so the pass keeps a mirror symmetry of
# the data bit for bit, maps a constant to itself and an alternating sequence to +0.0 exactly.
@kernel function binomial_pass_kernel!(dst, @Const(src), e, g::Int)
    I0 = @index(Global, Cartesian)
    I = I0 + CartesianIndex(ntuple(_ -> g, Val(length(I0))))
    @inbounds dst[I] = (src[I - e] + src[I + e] + 2 * src[I]) / 4
end

"""
    binomial!(a, scratch, grid::PeriodicGrid{N}, passes; workgroupsize = nothing) -> a

`passes` binomial passes along every axis of `grid` (axis 1 to N in each pass) on the field `a`,
in place, through the field `scratch` of the same size, whose contents it overwrites and never
reads before writing. Each pass fills the guards of its source (periodic images), writes every
interior node once from its two neighbours along the axis, and alternates between `a` and
`scratch`; on return the interior of `a` holds the filtered field and its guards the periodic
images. Deterministic: every output is written by one work item from fixed inputs, so the bits do
not depend on the launch shape. `passes = 0` returns `a` untouched. With `edges` a moving
window's [`WindowFrame`](@ref) the x-guards are set to zero after every guard fill (the window's
outside is field-free), so the filter does not wrap along axis 1; `nothing` (the default) changes
nothing.
"""
function binomial!(a, scratch, grid::PeriodicGrid{N}, passes::Integer; workgroupsize = nothing, edges = nothing) where {N}
    passes == 0 && return a
    size(scratch) == size(a) || throw(DimensionMismatch("scratch of size $(size(scratch)) for a field of size $(size(a))"))
    src, dst = a, scratch
    launch = launcher(binomial_pass_kernel!, get_backend(a), workgroupsize)
    for _ in 1:passes, d in 1:N
        fill_guards!(src, grid; workgroupsize)
        zero_x_edges!(edges, src; workgroupsize)
        launch(dst, src, CartesianIndex(ntuple(k -> k == d ? 1 : 0, Val(N))), guards(grid); ndrange = grid.n)
        src, dst = dst, src
    end
    src === a || copyto!(a, src)            # an odd number of axis passes ends in the scratch (N = 1)
    fill_guards!(a, grid; workgroupsize)
    zero_x_edges!(edges, a; workgroupsize)
    return a
end

"""
    filter_current!(filter, fields, grid, scratch; workgroupsize = nothing) -> fields

The step's current filter: [`binomial!`](@ref) with `filter.passes` on Jx, Jy and Jz of `fields`
(folded, guards zero), through `scratch`. With zero passes nothing is touched, so the step keeps
its bits; otherwise the guards of the three components hold their periodic images on return.
"""
function filter_current!(f::BinomialFilter, fields::NamedTuple, grid::PeriodicGrid, scratch; workgroupsize = nothing, edges = nothing)
    f.passes == 0 && return fields
    binomial!(fields.Jx, scratch, grid, f.passes; workgroupsize, edges)
    binomial!(fields.Jy, scratch, grid, f.passes; workgroupsize, edges)
    binomial!(fields.Jz, scratch, grid, f.passes; workgroupsize, edges)
    return fields
end

"""
    binomial_transfer(k, dx, passes) -> H

The closed form of [`binomial!`](@ref)'s transfer function for the wave vector `k` (a number at
N = 1, a tuple at N = 2) on cells `dx`: `Π_d cos(k_dΔ_d/2)^(2·passes)`.
"""
binomial_transfer(k::Number, dx::Number, passes::Integer) = cos(k * dx / 2)^(2 * passes)
binomial_transfer(k::Tuple, dx::Tuple, passes::Integer) = prod(map((kd, d) -> binomial_transfer(kd, d, passes), k, dx))
