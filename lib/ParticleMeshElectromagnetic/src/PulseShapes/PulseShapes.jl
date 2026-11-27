"""
    PulseShapes

Laser pulses as a temporal envelope times a transverse profile, in two dimensions and as the
round beam of three, for an antenna or any other code that needs the incident field of a focused
pulse. Self-contained: it uses
nothing but `Base`, names nothing of ParticleMesh, and can be included into any module on its own
(T-900), so that it can move to a package of its own.

Units: normalized to the carrier, ω₀ = 1 and c = 1, so times are in 1/ω₀, lengths in c/ω₀ and the
carrier's wavenumber is 1; phases of the wavefront are in waves of the carrier. The pulse travels
along +x; `ξ` is the distance from the focal plane along x, `y` the transverse distance from the
axis, `s` the time after the envelope's reference time at that x (retarded time). The field is

    E(ξ, y, s) = Re[U(ξ, y) · A(s) · e^{−is}]

with `U` the transverse profile ([`profile`](@ref), a solution of the paraxial equation
2i·∂U/∂ξ + ∂²U/∂y² = 0) and `A` the complex envelope ([`envelope`](@ref)). Everything is generic
in the element type: every parameter may be a `ForwardDiff.Dual`, and the profiles are isbits
device functions with no branch on the data, so they run inside GPU kernels.

Envelopes: [`GaussianEnvelope`](@ref) (closed form, with a temporal chirp) and
[`SpectralEnvelope`](@ref) (a Gaussian spectrum with a polynomial spectral phase: GDD, TOD, FOD).

Profiles: [`GaussianBeam`](@ref) (closed form) and [`PupilBeam`](@ref), the focus of a lens of a
given numerical aperture fed by a near field ([`GaussianPupil`](@ref),
[`SuperGaussianPupil`](@ref), [`UniformPupil`](@ref)) with a wavefront of Legendre aberration
modes ([`wavefront`](@ref)); in three dimensions [`ZernikeBeam`](@ref), the focus of a round
pupil with a wavefront of Zernike modes ([`zernike`](@ref)), `profile(b, ξ, y, z)` a solution of
2i·∂U/∂ξ + ∂²U/∂y² + ∂²U/∂z² = 0 with its second derivatives across the beam.

Parameters: [`parameter_names`](@ref), [`parameter_values`](@ref) and
[`with_parameter_values`](@ref) read and write the differentiable numbers of a shape;
[`convert_eltype`](@ref) converts them.
"""
module PulseShapes

export TemporalEnvelope, GaussianEnvelope, SpectralEnvelope, envelope, envelope_moments, envelope_extent, envelope_energy
export TransverseProfile, GaussianBeam, PupilBeam, PupilAmplitude, GaussianPupil, SuperGaussianPupil, UniformPupil
export profile, pupil_amplitude, wavefront, legendre, profile_moments, aliasing_period
export ZernikeBeam, zernike, zernike_index, zernike_radial, transverse_dimensions, quadrature_radius, pupil_sample, pupil_terms, radial_node
export parameter_names, parameter_values, with_parameter_values, convert_eltype

include("envelopes.jl")
include("beams.jl")
include("zernike.jl")
include("parameters.jl")

end
