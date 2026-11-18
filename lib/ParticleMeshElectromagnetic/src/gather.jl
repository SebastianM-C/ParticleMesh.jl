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
    s = sin(x)
    return (inv((s / x)^(order(shape) + 1)) - 1) / (4 * s * s)
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
gather_scratch(::Nothing, fields::NamedTuple) = nothing
gather_scratch(::CompensatedGather, fields::NamedTuple) =
    (Ex = similar(fields.Ex), Ey = similar(fields.Ey), Ez = similar(fields.Ez), Bx = similar(fields.Bx), By = similar(fields.By),
     Bz = similar(fields.Bz))

# A state's check that its gather's wavenumber is resolved (β is finite and the stencil sharpens).
check_gather(::Nothing, shape, grid) = nothing
check_gather(c::CompensatedGather, shape, grid) = (sharpening_beta(c, shape, grid.dx[1]); nothing)

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

Write the sharpened copy of the state's E and B (guards filled) for its [`CompensatedGather`](@ref):
each component through [`sharpen_x!`](@ref), then, with a moving window, its x-guards set to
zero as the grid's are. Nothing without compensation.
"""
sharpen_gather!(st::EMState) = (sharpen_gather!(st.numerics.gather, st); st)
sharpen_gather!(::Nothing, st::EMState) = nothing
function sharpen_gather!(c::CompensatedGather, st::EMState)
    (; numerics, grid, fields) = st
    wgs = numerics.workgroupsize
    β = sharpening_beta(c, numerics.shape, grid.dx[1])
    window = window_frame(st)
    for name in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        out = st.sharpened[name]
        sharpen_x!(out, fields[name], β, grid; workgroupsize = wgs)
        zero_x_edges!(window, out; workgroupsize = wgs)
    end
    return nothing
end
