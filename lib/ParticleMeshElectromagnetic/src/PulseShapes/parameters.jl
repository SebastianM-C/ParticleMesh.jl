# The differentiable numbers of a shape, read as a tuple and written back: what an optimizer or a
# vector of tunables sees. Configuration (a super-Gaussian's order, the samples of a quadrature,
# its span) is kept as it is.

"""
    parameter_names(x) -> Tuple of Symbols

The names of the differentiable numbers of an envelope, a profile or a pupil amplitude, in the
order of [`parameter_values`](@ref): `(:duration, :chirp)` for a [`GaussianEnvelope`](@ref),
`(:duration, :gdd, :tod, :fod)` for a [`SpectralEnvelope`](@ref), `(:waist,)` for a
[`GaussianBeam`](@ref), and for a [`PupilBeam`](@ref) `:na`, its amplitude's (`:fill` of a
[`GaussianPupil`](@ref), `:radius` of a [`SuperGaussianPupil`](@ref), none of a
[`UniformPupil`](@ref)) and `:aberration1` to `:aberration<M>`, and for a [`ZernikeBeam`](@ref)
the same, `:aberration<j>` the coefficient of the OSA/ANSI index j.
"""
function parameter_names end

"""
    parameter_values(x) -> Tuple

The differentiable numbers of `x`, in the order of [`parameter_names`](@ref).
"""
function parameter_values end

"""
    with_parameter_values(x, v::Tuple) -> x′

`x` with its differentiable numbers replaced by `v` (in the order of [`parameter_names`](@ref)),
of their promoted element type; its configuration kept.
"""
function with_parameter_values end

parameter_names(::GaussianEnvelope) = (:duration, :chirp)
parameter_values(e::GaussianEnvelope) = (e.duration, e.chirp)
with_parameter_values(::GaussianEnvelope, v::Tuple) = GaussianEnvelope(; duration = v[1], chirp = v[2])

parameter_names(::SpectralEnvelope) = (:duration, :gdd, :tod, :fod)
parameter_values(e::SpectralEnvelope) = (e.duration, e.gdd, e.tod, e.fod)
function with_parameter_values(e::SpectralEnvelope, v::Tuple)
    T = shape_eltype(v...)
    return check_envelope(SpectralEnvelope{T}(map(T, v)..., T(e.span), e.samples))
end

parameter_names(::GaussianPupil) = (:fill,)
parameter_values(p::GaussianPupil) = (p.fill,)
with_parameter_values(::GaussianPupil, v::Tuple) = GaussianPupil(; fill = v[1])

parameter_names(::SuperGaussianPupil) = (:radius,)
parameter_values(p::SuperGaussianPupil) = (p.radius,)
with_parameter_values(p::SuperGaussianPupil, v::Tuple) = SuperGaussianPupil(; radius = v[1], order = p.order)

parameter_names(::UniformPupil) = ()
parameter_values(::UniformPupil) = ()
with_parameter_values(p::UniformPupil, ::Tuple{}) = p

parameter_names(::GaussianBeam) = (:waist,)
parameter_values(b::GaussianBeam) = (b.waist,)
with_parameter_values(::GaussianBeam, v::Tuple) = GaussianBeam(; waist = v[1])

parameter_names(b::PupilBeam) = (:na, parameter_names(b.amplitude)..., ntuple(n -> Symbol(:aberration, n), length(b.aberrations))...)
parameter_values(b::PupilBeam) = (b.na, parameter_values(b.amplitude)..., b.aberrations...)
function with_parameter_values(b::PupilBeam, v::Tuple)
    na = length(parameter_values(b.amplitude))
    amplitude = with_parameter_values(b.amplitude, v[2:(1 + na)])
    return PupilBeam(; na = v[1], amplitude, aberrations = v[(2 + na):end], samples = b.samples)
end

"""
    convert_eltype(T, x) -> x′

`x` with its differentiable numbers converted to `T` ([`with_parameter_values`](@ref)).
"""
convert_eltype(::Type{T}, x) where {T} = with_parameter_values(x, map(T, parameter_values(x)))
convert_eltype(::Type{T}, p::UniformPupil) where {T} = p

parameter_names(b::ZernikeBeam) = (:na, parameter_names(b.amplitude)..., ntuple(n -> Symbol(:aberration, n), length(b.aberrations))...)
parameter_values(b::ZernikeBeam) = (b.na, parameter_values(b.amplitude)..., b.aberrations...)
function with_parameter_values(b::ZernikeBeam, v::Tuple)
    na = length(parameter_values(b.amplitude))
    amplitude = with_parameter_values(b.amplitude, v[2:(1 + na)])
    return ZernikeBeam(; na = v[1], amplitude, aberrations = v[(2 + na):end], radial = b.radial, angular = b.angular)
end
