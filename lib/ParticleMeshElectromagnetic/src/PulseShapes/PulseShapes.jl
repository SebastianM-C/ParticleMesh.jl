"""
    PulseShapes

Laser pulses in two dimensions as a temporal envelope times a transverse profile, for an antenna
or any other code that needs the incident field of a focused pulse. Self-contained: it uses
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
modes ([`wavefront`](@ref)).

Parameters: [`parameter_names`](@ref), [`parameter_values`](@ref) and
[`with_parameter_values`](@ref) read and write the differentiable numbers of a shape;
[`convert_eltype`](@ref) converts them.
"""
module PulseShapes

export TemporalEnvelope, GaussianEnvelope, SpectralEnvelope, envelope, envelope_moments, envelope_extent, envelope_energy
export TransverseProfile, GaussianBeam, PupilBeam, PupilAmplitude, GaussianPupil, SuperGaussianPupil, UniformPupil
export profile, pupil_amplitude, wavefront, legendre, profile_moments, aliasing_period
export parameter_names, parameter_values, with_parameter_values, convert_eltype

include("envelopes.jl")
include("beams.jl")
include("parameters.jl")

end
