"""
    ParticleMeshElectromagneticForwardDiffExt

`sin_cos` of a `ForwardDiff.Dual` (WP-083): the value and the derivative from the separate `sin`
and `cos` of the value, as for a plain number. ForwardDiff's own `sin` and `cos` of a dual call
`sincos`, which inside a kernel differs from the separate calls in the last bits, so without this
method a dual run of a laser would not keep the bits of the plain run. Nested duals recurse.
"""
module ParticleMeshElectromagneticForwardDiffExt

using ForwardDiff: Dual, value, partials
import ParticleMeshElectromagnetic.PulseShapes: sin_cos

@inline function sin_cos(d::Dual{T}) where {T}
    s, c = sin_cos(value(d))
    return Dual{T}(s, c * partials(d)), Dual{T}(c, -s * partials(d))
end

end
