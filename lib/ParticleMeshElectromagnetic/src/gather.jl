# The shape-compensated gather (WP-067). A particle of shape order p reads a field mode of
# wavenumber k, averaged over its offset in the cell, at sinc(kΔx/2)^(p+1) of its amplitude (the
# Fourier transform of the B-spline): for the laser, k = ω₀/c = 1, at Δx = λ₀/16 that is 1.9 % less
# field with the quadratic shape, and an ionization-injected charge that rises with a high power
# of the field feels it many times over. The compensation sharpens a copy of E and B along axis 1
# (the lasers' axis) with the symmetric three-point stencil
#     E'_i = E_i + β(2E_i − E_{i−1} − E_{i+1}),   transfer  1 + 4β sin²(kΔx/2),
# with β chosen so that the product of the two transfers is one at the wavenumber k of
# [`CompensatedGather`](@ref):
#     β = (sinc(kΔx/2)^−(p+1) − 1) / (4 sin²(kΔx/2)),   small-cell limit (p + 1)/24.
# Only the particles read the copy (the push, the ionization rates and the births' half-step
# kick); the mesh solve, the deposition and every diagnostic keep the grid's own fields, so the
# current and with it Gauss's law are the scheme's as before. The stencil is symmetric, so no
# mode's phase changes; its price is a gain of 1 + 4β at the grid's shortest wave (kΔx = π), 1.5
# for the quadratic shape at small Δx, on the grid noise the particles feel.

"""
    CompensatedGather(; k = 1)

The `gather` of an [`EMNumerics`](@ref) (WP-067): the particles read a copy of E and B sharpened
along axis 1, `E + β(2E_i − E_{i−1} − E_{i+1})`, whose transfer `1 + 4β sin²(k_xΔx/2)` undoes the
shape's mean transfer `sinc(k_xΔx/2)^(p+1)` exactly at the wavenumber `k` (ω₀/c; the laser's is 1,
the default) and to within O((kΔx)⁴) below it ([`sharpening_beta`](@ref), [`gather_transfer`](@ref)).
That is the mean over the particle's offset in the cell: at a given offset the amplitude read
still ripples about it, and more so for the linear shape (at k and 8 cells per wavelength, 0.973
to 1.053 compensated against 0.924 to 1 without; for the quadratic shape 0.998 to 1.0015).
The grid's own dispersion of the laser is not compensated, only the shape's averaging.
The push, the ionization rates and the births' kick read the copy; the mesh solve, the current
and the diagnostics read the grid's fields. `k` is a host number of the configuration; it must be
positive and resolved (`kΔx < π`, which the state checks). With `gather = nothing` (the default)
the particles read the grid's fields and the step keeps its bits.
"""
struct CompensatedGather
    k::Float64
    function CompensatedGather(; k::Real = 1)
        isfinite(k) && k > 0 || throw(ArgumentError("CompensatedGather needs a finite wavenumber k > 0, got $k"))
        return new(Float64(k))
    end
end

# The Galerkin gather (WP-070). The charge-conserving current of a particle of shape order p moving
# along axis d has weights one order lower along d, at the faces half a cell from ρ's nodes: the
# derivative of the order-p B-spline is the difference of two order-(p − 1) splines half a cell
# apart. A gather with those weights for E_d (order p − 1 along d, at its `Center()` stagger there,
# order p along the other axes) is the transpose of the current deposition for a vanishing move,
# so the work the field does on the particles is the work Σ J·E booked on the grid: the energy of
# fields and particles is kept to the time integrator's error. B does no work; its components are
# read one order lower along their two `Center()` axes too, which makes the gathered B follow
# Faraday's law with the gathered E: the discrete curl of the grid's E, read with B's weights, is
# the curl of E read with E's weights (B. B. Godfrey and J.-L. Vay, J. Comput. Phys. 248, 33
# (2013), arXiv:1211.0232, abstract read; the rule is WarpX's `interpolation.galerkin_scheme`,
# its default on staggered grids with Esirkepov deposition, parameter documentation read). The
# rule is one line: along every axis on which a component sits at `Center()`, one order lower.

"""
    GalerkinGather(; compensate = nothing)

The `gather` of an [`EMNumerics`](@ref) that reads every field component one shape order lower
along each axis on which the component sits at `Center()` in [`field_layout`](@ref) (WP-070):
with the quadratic shape, Ex with the linear shape along x and the quadratic along y and z, Bz
with the linear shape along x and y, and so on; with the linear shape, the lower order is the
nearest grid point (`Shape{0}`). E's weights are then those of the charge-conserving current for
a vanishing move, so the gather is the current deposition's transpose and the total energy of
fields and particles is kept to the time integrator's error; B's weights make the gathered B
follow Faraday's law with the gathered E. Momentum is not kept (a lone particle pushes itself).
The mesh solve, the current and the diagnostics are unchanged.

`compensate` is `nothing` or a [`CompensatedGather`](@ref): the particles then read the sharpened
copy of E and B, each component's β for the order it is read with along axis 1
([`sharpening_beta`](@ref) of the lower order where the component sits at `Center()` along x),
so that the mean transfer at the compensation's wavenumber is one for every component.
"""
struct GalerkinGather{C<:Union{Nothing,CompensatedGather}}
    compensate::C
end
function GalerkinGather(; compensate = nothing)
    compensate === nothing || compensate isa CompensatedGather ||
        throw(ArgumentError("GalerkinGather(; compensate = $compensate): nothing or a CompensatedGather(...)"))
    return GalerkinGather(compensate)
end

"""
    GalerkinRead{P}()

What the particles' gather kernels receive as their shape with a [`GalerkinGather`](@ref) of the
shape order `P`: [`component_shape`](@ref) turns it into one shape per axis for each component's
stagger. Without it they receive the numerics' `Shape` itself, and every component is read with it.
"""
struct GalerkinRead{P} end

"""
    read_shape(gather, shape)

The shape argument of the particles' gathers (the push, the initial kick, the ionization): the
`shape` itself without a Galerkin gather, [`GalerkinRead`](@ref) of its order with one.
"""
read_shape(::Union{Nothing,CompensatedGather}, shape::Shape) = shape
read_shape(::GalerkinGather, ::Shape{P}) where {P} = GalerkinRead{P}()
read_shape(numerics) = read_shape(numerics.gather, numerics.shape)

"""
    component_shape(read, stagger)

The shape a component on `stagger` is read with: `read` itself when it is a `Shape`; for a
[`GalerkinRead`](@ref)`{P}`, `Shape{P−1}()` on `Center()` and `Shape{P}()` on `Node()`, per axis
for a tuple of staggers.
"""
@inline component_shape(s::Shape, stagger) = s
@inline component_shape(::GalerkinRead{P}, ::Center) where {P} = Shape{P - 1}()
@inline component_shape(::GalerkinRead{P}, ::Node) where {P} = Shape{P}()
@inline component_shape(r::GalerkinRead, st::Tuple) = map(s -> component_shape(r, s), st)

# The shape a component is read with along axis 1, where the compensation sharpens.
axis1_shape(read, st::Stagger) = component_shape(read, st)
axis1_shape(read, st::Tuple) = component_shape(read, st[1])

"""
    sharpening_beta(gather::CompensatedGather, shape, dx) -> β

The weight β of the sharpening stencil for `shape` (order p) and cell size `dx` along axis 1, in
the type of `dx`: `(1/sinc(x)^(p+1) − 1)/(4 sin²x)` with `x = k·dx/2`, so that
`sinc(x)^(p+1)·(1 + 4β sin²x) = 1`. An `ArgumentError` unless `0 < x < π/2` (the wavenumber resolved
by the grid).
"""
function sharpening_beta(c::CompensatedGather, shape::Shape, dx)
    x = convert(typeof(dx), c.k) * dx / 2
    0 < x < π / 2 || throw(ArgumentError("CompensatedGather(k = $(c.k)) is not resolved by Δx = $dx: k·Δx = $(2x) must be < π"))
    # 1/sinc^n − 1 = (1 − sⁿ)/sⁿ = d·Σ_{j<n} sʲ/sⁿ with s = sinc x = 1 − d: no difference of two
    # numbers near one, so β keeps the precision of its type down to the finest cells (in
    # Float32 the direct form loses 0.7 % of β at 512 cells per wavelength)
    n = order(shape) + 1
    d = one_minus_sinc(x)
    s = 1 - d
    return d * sum(j -> s^j, 0:(n - 1)) / (s^n * 4 * sin(x)^2)
end

# 1 − sin(x)/x = x²/3! − x⁴/5! + x⁶/7! − …, summed from its first term (alternating, decreasing
# for x < π/2, so without cancellation); 14 terms reach Float64's roundoff at x = π/2.
function one_minus_sinc(x)
    x2 = x * x
    t = x2 / 6
    acc = t
    for k in 2:14
        t = -t * x2 / ((2k) * (2k + 1))
        acc += t
    end
    return acc
end

"""
    gather_transfer(gather, shape, dx, k) -> factor

The mean factor by which a particle of `shape` reads a field mode of wavenumber `k` along axis 1
(cell size `dx`), averaged over its offset in the cell: `sinc(kΔx/2)^(p+1)` without compensation
(`gather = nothing`), times the stencil's `1 + 4β sin²(kΔx/2)` with a [`CompensatedGather`](@ref).
One at `k = gather.k`; below it slightly above one, by O((kΔx)⁴).
"""
function gather_transfer(::Nothing, shape::Shape, dx, k)
    x = k * dx / 2
    return iszero(x) ? one(x) : (sin(x) / x)^(order(shape) + 1)
end
function gather_transfer(c::CompensatedGather, shape::Shape, dx, k)
    x = k * dx / 2
    return gather_transfer(nothing, shape, dx, k) * (1 + 4 * sharpening_beta(c, shape, dx) * sin(x)^2)
end

# One sharpened entry per work item over the whole array, guards included: the neighbours along
# axis 1 of the first and the last column are their periodic images (the column n1 further in), so
# that with the guards of `a` periodic images the copy's guards are those of the sharpened
# interior, bit for bit. (2c − l − r) is symmetric in l and r: the stencil keeps a mirror symmetry
# of the data and maps a constant to itself exactly.
@kernel function sharpen_x_kernel!(out, @Const(a), β, n1::Int)
    I = @index(Global, Cartesian)
    i = I[1]
    rest = Base.tail(Tuple(I))
    il = ifelse(i == 1, i - 1 + n1, i - 1)
    ir = ifelse(i == size(a, 1), i + 1 - n1, i + 1)
    @inbounds c = a[I]
    @inbounds out[I] = c + β * (2 * c - a[CartesianIndex(il, rest...)] - a[CartesianIndex(ir, rest...)])
end

"""
    sharpen_x!(out, a, β, grid; workgroupsize = nothing) -> out

The copy `a + β(2a_i − a_{i−1} − a_{i+1})` along axis 1 of the field `a` (guards filled) into
`out`, every entry guards included, the periodic images standing in for the neighbours beyond the
array's first and last column. Deterministic: one work item per entry from fixed inputs.
"""
function sharpen_x!(out, a, β, grid::PeriodicGrid; workgroupsize = nothing)
    size(out) == size(a) || throw(DimensionMismatch("copy of size $(size(out)) for a field of size $(size(a))"))
    launcher(sharpen_x_kernel!, get_backend(a), workgroupsize)(out, a, convert(eltype(a), β), Int(ncells(grid)[1]); ndrange = size(a))
    return out
end

# The scratch of the copy: nothing without compensation, E and B arrays like the state's otherwise.
gather_scratch(g::GalerkinGather, fields::NamedTuple) = gather_scratch(g.compensate, fields)
gather_scratch(::Nothing, fields::NamedTuple) = nothing
gather_scratch(::CompensatedGather, fields::NamedTuple) =
    (Ex = similar(fields.Ex), Ey = similar(fields.Ey), Ez = similar(fields.Ez), Bx = similar(fields.Bx), By = similar(fields.By),
     Bz = similar(fields.Bz))

# A state's check that its gather's wavenumber is resolved (β is finite and the stencil sharpens).
check_gather(::Nothing, shape, grid) = nothing
check_gather(c::CompensatedGather, shape, grid) = (sharpening_beta(c, shape, grid.dx[1]); nothing)
check_gather(g::GalerkinGather, shape, grid) = check_gather(g.compensate, shape, grid)

"""
    particle_fields(state) -> NamedTuple

The fields the particles of `state` read (the push, the ionization, the initial kick): the
state's `fields` without compensation, and with a [`CompensatedGather`](@ref) the same tuple with
E and B replaced by the sharpened copy that [`sharpen_gather!`](@ref) last wrote.
"""
particle_fields(st::EMState) = particle_fields(st.sharpened, st.fields)
particle_fields(::Nothing, fields::NamedTuple) = fields
particle_fields(copy::NamedTuple, fields::NamedTuple) = merge(fields, copy)

"""
    sharpen_gather!(state) -> state

Write the sharpened copy of the state's E and B (guards filled) for its [`CompensatedGather`](@ref)
(alone or in a [`GalerkinGather`](@ref)): each component through [`sharpen_x!`](@ref) with the β
of the shape it is read with along axis 1, then, with a moving window, its x-guards set to zero
as the grid's are. Nothing without compensation.
"""
sharpen_gather!(st::EMState) = (sharpen_gather!(st.numerics.gather, st); st)
sharpen_gather!(::Nothing, st::EMState) = nothing
sharpen_gather!(g::GalerkinGather, st::EMState) = sharpen_gather!(g.compensate, st)
function sharpen_gather!(c::CompensatedGather, st::EMState{N}) where {N}
    (; numerics, grid, fields) = st
    wgs = numerics.workgroupsize
    read = read_shape(numerics)
    L = field_layout(Val(N))
    window = window_frame(st)
    for name in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        β = sharpening_beta(c, axis1_shape(read, L[name]), grid.dx[1])
        out = st.sharpened[name]
        sharpen_x!(out, fields[name], β, grid; workgroupsize = wgs)
        zero_x_edges!(window, out; workgroupsize = wgs)
    end
    return nothing
end
