# Momentum distributions of the quiet start (WP-033): isbits objects that turn the Philox draws
# of one particle into its three velocity or momentum components. They are sampling, like
# `box_muller`, and name no physics (A9).

"""
    GaussianVelocity(vdrift, vth)

The quiet start's velocity draw: component `d` is `vdrift[d] + vth[d]·z_d`, `z` standard normal
from Box–Muller over one Philox4x32-10 call (draw index 0), in velocity. It writes velocities
(`draws_momentum` is false), so a module that keeps momenta converts them, and the Gaussian tail
can reach |v| ≥ 1. `load_quiet!` with it is the keyword form `load_quiet!(…; vdrift, vth)` bit
for bit (T-530).
"""
struct GaussianVelocity{T}
    vdrift::NTuple{3,T}
    vth::NTuple{3,T}
end

"""
    GaussianMomentum(udrift, uth)

The same draw as [`GaussianVelocity`](@ref) (one Philox call, draw index 0), read as the momentum
per unit mass `u = γv` (in c): component `d` is `udrift[d] + uth[d]·z_d`. Every γ = √(1 + u²) is
finite, and the spread may differ per component.
"""
struct GaussianMomentum{T}
    udrift::NTuple{3,T}
    uth::NTuple{3,T}
end

"""
    MaxwellJuttner(θ, β; tries = 32)

The Maxwell–Jüttner distribution of rest-frame temperature `θ = T/(mc²)` (θ > 0, finite) in a
plasma drifting with the velocity `β` (|β| < 1, in c), drawn in momentum `u = γv`, so every γ is
finite. The rest-frame speed comes from a rejection loop of at most `tries` tries (1 to 32767): for
θ < 1 the kinetic energy ε = γ − 1, whose density is ∝ (1 + ε)√(ε(ε + 2)) e^(−ε/θ), is proposed
from the envelope √ε (1 + 5ε/4 + ε²/4) e^(−ε/θ), a mixture of Gamma(3/2, θ), Gamma(5/2, θ) and
Gamma(7/2, θ) with weights 1 : 15θ/8 : 15θ²/16 (from √(ε + 2) ≤ √2 (1 + ε/4)), and accepted with
probability √(1 + ε/2)/(1 + ε/4), so |u| = √(ε(ε + 2)) has no cancellation as θ → 0; for θ ≥ 1
by Sobol's method as Zenitani gives it (Phys. Plasmas 22, 042116, 2015; the attribution there is
not checked against Sobol's text): u = −θ log(U₁U₂U₃), η = −θ log(U₁U₂U₃U₄), accepted when
η² − u² > 1. Try `j` takes the Philox draw indices `2j − 1` and `2j`. The direction is isotropic
from a call at draw index 65535, which also gives the uniform of Zenitani's flipping method: the
component along β is reversed when −|β|·v′∥ > U, which supplies the flux factor (1 + β·v′), and
the momentum is then boosted along β, u∥ = Γ(u′∥ + |β|γ′). A particle whose tries all fail keeps
its last proposal (finite) and is counted (`load_quiet!`'s `exhausted`). The acceptance stays
above about 0.8 at every θ (lowest, 0.815, just above θ = 1), so with 32 tries that has a
probability near 1e−22 per particle. In Float32 the boost factor Γ = 1/√(1 − |β|²) carries the
rounding of 1 − |β|², which at |β| ≥ 0.999 and θ ~ 1e−5 shifts ⟨u∥⟩ by about 1e−5 relative (the
rounding of β itself to Float32 shifts it more).
"""
struct MaxwellJuttner{T}
    θ::T
    β::NTuple{3,T}
    tries::Int32
    function MaxwellJuttner{T}(θ, β; tries::Integer = 32) where {T}
        θT, βT = T(θ), T.(Tuple(β))
        θT > 0 && isfinite(θT) || throw(ArgumentError("MaxwellJuttner: θ = $θ must be positive and finite"))
        b2 = βT[1]^2 + βT[2]^2 + βT[3]^2
        b2 < 1 || throw(ArgumentError("MaxwellJuttner: |β| = $(sqrt(b2)) must be below 1"))
        0 < tries < 2^15 || throw(ArgumentError("MaxwellJuttner: tries = $tries must be in 1:32767 (two draw indices per try)"))
        return new{T}(θT, βT, Int32(tries))
    end
end
MaxwellJuttner(θ::T, β; tries::Integer = 32) where {T} = MaxwellJuttner{T}(θ, β; tries)

"""
    draws_momentum(dist) -> Bool

Whether the distribution writes the momentum `u = γv` (true for [`GaussianMomentum`](@ref) and
[`MaxwellJuttner`](@ref)) or the velocity (false for [`GaussianVelocity`](@ref)). A trait of the
type, so a module branches on it on the host, not on device data (§0.4).
"""
draws_momentum(::GaussianVelocity) = false
draws_momentum(::GaussianMomentum) = true
draws_momentum(::MaxwellJuttner) = true

# The counter of a particle's draw `d`: the low 16 bits of the last word (`k << 16 | d`).
@inline with_draw(ctr::NTuple{4,UInt32}, d) = (ctr[1], ctr[2], ctr[3], ctr[4] | UInt32(d))

@inline function gaussian3(::Type{T}, ctr, key) where {T}
    r = philox4x32(ctr, key)
    z1, z2 = box_muller(T, r[1], r[2])
    z3, _ = box_muller(T, r[3], r[4])
    return (z1, z2, z3)
end

"""
    draw(dist, T, ctr, key) -> (u::NTuple{3,T}, accepted::Bool)

The three components a distribution gives the particle whose Philox counter is `ctr` (draw index
0 in its low 16 bits) under `key`, and whether its draw was accepted (always, except a
[`MaxwellJuttner`](@ref) particle that used up its tries). A device function.
"""
@inline function draw(d::GaussianVelocity, ::Type{T}, ctr, key) where {T}
    z = gaussian3(T, ctr, key)
    return (T(d.vdrift[1]) + T(d.vth[1]) * z[1], T(d.vdrift[2]) + T(d.vth[2]) * z[2], T(d.vdrift[3]) + T(d.vth[3]) * z[3]), true
end

@inline function draw(d::GaussianMomentum, ::Type{T}, ctr, key) where {T}
    z = gaussian3(T, ctr, key)
    return (T(d.udrift[1]) + T(d.uth[1]) * z[1], T(d.udrift[2]) + T(d.uth[2]) * z[2], T(d.udrift[3]) + T(d.uth[3]) * z[3]), true
end

# One try of the gamma mixture (θ < 1): the proposed kinetic energy ε and whether it is accepted.
@inline function juttner_mixture_try(θ::T, r1, r2) where {T}
    w2 = T(15) / T(8) * θ
    w3 = T(15) / T(16) * θ * θ
    pick = uniform_open(T, r1[1]) * (one(T) + w2 + w3)
    z, _ = box_muller(T, r1[2], r1[3])
    e = -log(uniform_open(T, r1[4]))
    e = e - ifelse(pick >= one(T), log(uniform_open(T, r2[1])), zero(T))
    e = e - ifelse(pick >= one(T) + w2, log(uniform_open(T, r2[2])), zero(T))
    ε = θ * (e + z * z / 2)
    ok = uniform_open(T, r2[3]) * (one(T) + ε / 4) < sqrt(one(T) + ε / 2)
    return ε, ok
end

# One try of Sobol's method (θ ≥ 1): the proposed |u| and whether it is accepted. The logs are
# summed rather than the uniforms multiplied, so the product cannot underflow in Float32.
@inline function juttner_sobol_try(θ::T, r1) where {T}
    l3 = log(uniform_open(T, r1[1])) + log(uniform_open(T, r1[2])) + log(uniform_open(T, r1[3]))
    u = -θ * l3
    η = u - θ * log(uniform_open(T, r1[4]))
    return u, (η - u) * (η + u) > one(T)
end

# The rest-frame speed |u′| and γ′ by rejection; the last proposal if every try fails.
@inline function juttner_speed(θ::T, tries::Int32, ctr, key) where {T}
    u = zero(T)
    γ = one(T)
    ok = false
    for j in Int32(1):tries
        r1 = philox4x32(with_draw(ctr, Int32(2) * j - Int32(1)), key)
        if θ < one(T)
            r2 = philox4x32(with_draw(ctr, Int32(2) * j), key)
            ε, ok = juttner_mixture_try(θ, r1, r2)
            u = sqrt(ε * (ε + 2))
            γ = one(T) + ε
        else
            u, ok = juttner_sobol_try(θ, r1)
            γ = sqrt(one(T) + u * u)
        end
        ok && break
    end
    return u, γ, ok
end

@inline function draw(d::MaxwellJuttner, ::Type{T}, ctr, key) where {T}
    θ = T(d.θ)
    um, γr, ok = juttner_speed(θ, d.tries, ctr, key)
    r = philox4x32(with_draw(ctr, 0xffff), key)
    μ = 2 * uniform_open(T, r[1]) - one(T)
    φ = 2 * T(π) * uniform_open(T, r[2])
    sθ = sqrt(max(zero(T), (one(T) - μ) * (one(T) + μ)))
    u1, u2, u3 = um * sθ * cos(φ), um * sθ * sin(φ), um * μ
    # Zenitani's flip and the boost along β; with β = 0 the unit vector is zero and nothing moves
    β1, β2, β3 = T(d.β[1]), T(d.β[2]), T(d.β[3])
    b2 = β1 * β1 + β2 * β2 + β3 * β3
    b = sqrt(ifelse(b2 > zero(T), b2, one(T)))
    n1, n2, n3 = β1 / b, β2 / b, β3 / b
    b = ifelse(b2 > zero(T), b, zero(T))
    up = u1 * n1 + u2 * n2 + u3 * n3
    flip = -b * up / γr > uniform_open(T, r[3])
    up2 = ifelse(flip, -up, up)
    Γ = one(T) / sqrt(one(T) - b2)
    # u∥ = Γ(u′∥ + bγ′): add (Γ − 1)u′∥ + Γbγ′ along n, with Γ − 1 = Γ²b²/(Γ + 1) (no cancellation)
    a = (up2 - up) + Γ * Γ * b2 / (Γ + one(T)) * up2 + Γ * b * γr
    return (u1 + a * n1, u2 + a * n2, u3 + a * n3), ok
end

@inline count_exhausted!(::Nothing, ok) = nothing
@inline function count_exhausted!(c, ok)
    ok || Atomix.@atomic c[1] += Int32(1)
    return nothing
end

@kernel function load_dist_kernel!(s, density, p, dx, ppc::Int32, first::Int32, dist, key, at, exhausted)
    i = @index(Global, Linear)
    T = eltype(s.w)
    c = Int32((i - 1) ÷ ppc + 1)
    k = Int32((i - 1) % ppc + 1)
    slot = first - Int32(1) + Int32(i)
    δ = (T(k) - one(T) / 2) / T(ppc)
    x = (T(c - Int32(1)) + density_offset(at, δ)) * dx
    u, ok = draw(dist, T, (UInt32(c), UInt32(0), UInt32(0), UInt32(k) << 16), key)
    count_exhausted!(exhausted, ok)
    @inbounds begin
        s.cell[1][slot] = c
        s.offset[1][slot] = δ
        s.u[1][slot] = u[1]
        s.u[2][slot] = u[2]
        s.u[3][slot] = u[3]
        s.w[slot] = density(x, p) / T(ppc)
        s.status[slot] = ALIVE
    end
end

@kernel function load_dist_2d_kernel!(s, density, p, dx, n1::Int32, ppc::NTuple{2,Int32}, first::Int32, dist, key, at, exhausted)
    i = @index(Global, Linear)
    T = eltype(s.w)
    K = ppc[1] * ppc[2]
    L = Int32((i - 1) ÷ K + 1)
    k = Int32((i - 1) % K + 1)
    c1 = (L - Int32(1)) % n1 + Int32(1)
    c2 = (L - Int32(1)) ÷ n1 + Int32(1)
    k1 = (k - Int32(1)) % ppc[1] + Int32(1)
    k2 = (k - Int32(1)) ÷ ppc[1] + Int32(1)
    slot = first - Int32(1) + Int32(i)
    δ1 = (T(k1) - one(T) / 2) / T(ppc[1])
    δ2 = (T(k2) - one(T) / 2) / T(ppc[2])
    x = ((T(c1 - Int32(1)) + density_offset(at, δ1)) * dx[1], (T(c2 - Int32(1)) + density_offset(at, δ2)) * dx[2])
    u, ok = draw(dist, T, (UInt32(c1), UInt32(c2), UInt32(0), UInt32(k) << 16), key)
    count_exhausted!(exhausted, ok)
    @inbounds begin
        s.cell[1][slot] = c1
        s.cell[2][slot] = c2
        s.offset[1][slot] = δ1
        s.offset[2][slot] = δ2
        s.u[1][slot] = u[1]
        s.u[2][slot] = u[2]
        s.u[3][slot] = u[3]
        s.w[slot] = density(x, p) / T(K)
        s.status[slot] = ALIVE
    end
end

"""
    load_quiet!(species, grid, density, p, ppc, dist; seed, stream, first = 1, density_at = :particle,
                exhausted = nothing, workgroupsize = nothing)

The quiet start of the keyword form (`load_quiet!(…; vdrift, vth)`, the same lattice, slots,
weights, `density_at` and Philox key and counter, at N = 1 and N = 2) with the components drawn
from the distribution `dist`: [`GaussianVelocity`](@ref) (that form bit for bit, velocities),
[`GaussianMomentum`](@ref) or [`MaxwellJuttner`](@ref) (momenta `u = γv`); [`draws_momentum`](@ref)
says which. Each particle's draws depend only on `(seed, stream)`, its cell and its in-cell index,
so a seed means the same load on every backend, thread count and launch shape (A4). `exhausted`,
a one-element `Int32` device array, gets the number of Maxwell–Jüttner particles whose tries all
failed added to it (with `nothing`, the default, nothing is counted).
"""
function load_quiet!(s::Species{1,T}, grid::PeriodicGrid{1}, density, p, ppc::Integer, dist;
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     exhausted = nothing, workgroupsize = nothing) where {T}
    check_density_at(density_at)
    n = ncells(grid)
    0 < ppc < 2^16 || throw(ArgumentError("ppc must be in 1:65535 for the counter layout"))
    last = first - 1 + n * ppc
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    launcher(load_dist_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32(ppc), Int32(first),
                                                                dist, key, density_val(density_at), exhausted; ndrange = n * ppc)
    fill!(s.count, Int32(last))
    return s
end

function load_quiet!(s::Species{2,T}, grid::PeriodicGrid{2}, density, p, ppc::NTuple{2,Integer}, dist;
                     seed::Integer, stream::Integer, first::Integer = 1, density_at::Symbol = :particle,
                     exhausted = nothing, workgroupsize = nothing) where {T}
    check_density_at(density_at)
    n = ncells(grid)
    all(>(0), ppc) && prod(ppc) < 2^16 || throw(ArgumentError("ppc = $ppc: each ≥ 1 and prod(ppc) < 2^16 for the counter layout"))
    total = prod(n) * prod(ppc)
    last = first - 1 + total
    last <= capacity(s) || throw(ArgumentError("capacity $(capacity(s)) < $last particles"))
    key = (UInt32(seed), UInt32(stream))
    launcher(load_dist_2d_kernel!, get_backend(s), workgroupsize)(s, density, p, cellsize(grid), Int32(n[1]), Int32.(ppc),
                                                                   Int32(first), dist, key, density_val(density_at), exhausted;
                                                                   ndrange = total)
    fill!(s.count, Int32(last))
    return s
end
