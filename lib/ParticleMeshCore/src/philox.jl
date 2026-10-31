# Philox4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as 1, 2, 3",
# SC'11), written with Base integer operations only so that it runs inside any kernel.
# Checked against the known-answer vectors of the Random123 distribution (test T-139).

const PHILOX_M0 = 0xD2511F53
const PHILOX_M1 = 0xCD9E8D57
const PHILOX_W0 = 0x9E3779B9
const PHILOX_W1 = 0xBB67AE85

@inline function mulhilo(a::UInt32, b::UInt32)
    p = UInt64(a) * UInt64(b)
    return ((p >> 32) % UInt32, p % UInt32)
end

@inline function philox_round(c::NTuple{4,UInt32}, k::NTuple{2,UInt32})
    hi0, lo0 = mulhilo(PHILOX_M0, c[1])
    hi1, lo1 = mulhilo(PHILOX_M1, c[3])
    return (hi1 ⊻ c[2] ⊻ k[1], lo1, hi0 ⊻ c[4] ⊻ k[2], lo0)
end

@inline bump(k::NTuple{2,UInt32}) = (k[1] + PHILOX_W0, k[2] + PHILOX_W1)

"""
    philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}, ::Val{R} = Val(10))

The Philox4x32 bijection with `R` rounds: four 32-bit outputs per counter.
"""
@inline function philox4x32(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}, ::Val{R} = Val(10)) where {R}
    c, k = ctr, key
    for r in 1:R
        c = philox_round(c, k)
        r < R && (k = bump(k))
    end
    return c
end

"""
    uniform_open(T, x::UInt32)

A number in the open interval (0, 1) of type `T` from 32 random bits: `(x + 1/2) / 2³²`.
It is never zero; in Float32 it may round up to one, which the callers tolerate.
"""
@inline uniform_open(::Type{T}, x::UInt32) where {T} = (T(x) + one(T) / 2) / T(4294967296)

"""
    box_muller(T, x1, x2) -> (z1, z2)

Two independent standard normal numbers from two 32-bit draws, with Base functions only.
"""
@inline function box_muller(::Type{T}, x1::UInt32, x2::UInt32) where {T}
    u1 = uniform_open(T, x1)
    u2 = uniform_open(T, x2)
    r = sqrt(-2 * log(u1))
    θ = 2 * T(π) * u2
    return (r * cos(θ), r * sin(θ))
end
