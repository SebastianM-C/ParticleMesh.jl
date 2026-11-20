# WP-083: the laser's device functions call sin and cos, not sincos (#547). T-1141: the two give the
# same bits on the CPU, so the CPU's runs are unchanged. T-1140 (run by gpu.jl on CUDA): Enzyme's
# reverse mode through a KernelAbstractions kernel that evaluates the antenna's incident field per
# row, from the laser's parameters in a device array, against ForwardDiff on the CPU.

using Enzyme: Enzyme
using ParticleMeshElectromagnetic: GaussianLaser, AxisTime, WavefrontTime, laser_field
using ParticleMeshElectromagnetic.PulseShapes: GaussianBeam, PupilBeam, GaussianPupil, profile

@testset "T-1141 sin and cos give the bits of sincos for Float32, Float64 and Dual{Float32,3} [A4, A8, WP-083]" begin
    rng = Xoshiro(1141)
    for T in (Float32, Float64)
        # the phases of the antenna (|φ| up to the box's time, about 10³) and of the pupil's sum,
        # wider ranges for the argument reduction, and the edges
        xs = vcat(T(2000) .* rand(rng, T, 20_000) .- T(1000), T(1e6) .* randn(rng, T, 2_000), T(1e-3) .* randn(rng, T, 2_000),
                  T[0, -0.0, floatmin(T), -floatmin(T), eps(T), T(π) / 2, T(π), 2 * T(π), floatmax(T), T(1e30)])
        @test all(x -> (sin(x), cos(x)) === sincos(x), xs)
    end
    D = ForwardDiff.Dual{Nothing,Float32,3}
    ds = [D(v, ForwardDiff.Partials(Tuple(randn(rng, Float32, 3)))) for v in Float32(2000) .* rand(rng, Float32, 20_000) .- 1000]
    @test all(d -> (sin(d), cos(d)) === sincos(d), ds)
end

# The cases of T-1140: what the antenna's kernels evaluate per row, as functions of a tuple of
# parameters `P` and of the row's transverse position(s) `dj`, at the plane x and the time t.
struct Laser2D{E}
    envelope::E
end
struct Laser3D{E}
    envelope::E
end
struct GaussianProfile end
struct PupilProfile end
nparams(::Laser2D) = 8
nparams(::Laser3D) = 8
nparams(::GaussianProfile) = 1
nparams(::PupilProfile) = 4
ndims_row(::Laser2D) = 1
ndims_row(::Laser3D) = 2
ndims_row(::Union{GaussianProfile,PupilProfile}) = 2
nouts(::Laser2D) = 3
nouts(::Laser3D) = 4
nouts(::Union{GaussianProfile,PupilProfile}) = 4

# a0, waist, duration, delay, focus, chirp, polarization, phase; the plane at x, the axis at 0
@inline laser_of(P, env, center) = GaussianLaser{eltype(P),eltype(P),typeof(env),typeof(center)}(P..., zero(eltype(P)), center, env)

# antenna_b_kernel! and antenna_e_kernel!: Bz += c·e·cos θ, Ey += c·bp·cos θ, Ez += c·bs·sin θ
@inline function rows(c::Laser2D, P, dj, x, t)
    T = eltype(P)
    l = laser_of(P, c.envelope, zero(T))
    e, bp, bs = laser_field(l, x - l.focus, dj[1], t - l.delay)
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    return (e * cθ, bp * cθ, bs * sθ)
end
# antenna_b_kernel_3d! and antenna_e_kernel_3d!: Ey_inc, Ez_inc, Bz_inc, By_inc
@inline function rows(c::Laser3D, P, dj, x, t)
    T = eltype(P)
    l = laser_of(P, c.envelope, (zero(T), zero(T)))
    e, bp, bs, byz = laser_field(l, x - l.focus, dj[1], dj[2], t - l.delay)
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    return (e * cθ, e * sθ, bp * cθ - byz * sθ, byz * cθ - bs * sθ)
end
# the shaped antenna's profiles (shaped_field): U and U″ at ξ = dj[1], y = dj[2]
@inline profile_rows(u, u2) = (real(u), imag(u), real(u2), imag(u2))
@inline rows(::GaussianProfile, P, dj, x, t) = profile_rows(profile(GaussianBeam{eltype(P)}(P[1]), dj[1], dj[2])...)
@inline function rows(::PupilProfile, P, dj, x, t)
    T = eltype(P)
    b = PupilBeam{T,GaussianPupil{T},2}(P[1], GaussianPupil{T}(P[2]), (P[3], P[4]), Int32(48))
    return profile_rows(profile(b, dj[1], dj[2])...)
end

@kernel function rows_kernel!(out, @Const(p), @Const(d), c, x, t)
    j = @index(Global, Linear)
    @inbounds begin
        P = ntuple(k -> p[k], Val(nparams(c)))
        dj = ntuple(k -> d[k, j], Val(ndims_row(c)))
        o = rows(c, P, dj, x, t)
        ntuple(k -> (out[k, j] = o[k]; nothing), Val(nouts(c)))
    end
end

function rows!(out, p, d, c, x, t)
    rows_kernel!(get_backend(out))(out, p, d, c, x, t; ndrange = size(out, 2))
    return nothing
end

# The parameters and rows of each case: the laser of the trapped-beam studies' scale (waist 6,
# duration 8), 10 c/ω₀ before its focus, at a time on the pulse's rising edge, so that every
# derivative is far from zero; rows across ±2 waists.
function t1140_case(::Type{T}, c, nrows) where {T}
    rng = Xoshiro(1140 + nrows)
    if c isa Union{Laser2D,Laser3D}
        p = T[1.3, 6, 8, 48, 28, 1 // 50, 3 // 10, 1 // 5]
        d = T(12) .* (rand(rng, T, ndims_row(c), nrows) .- T(1 // 2)) .* 2
        return p, d, T(18), T(41)
    end
    p = c isa GaussianProfile ? T[6] : T[1 // 4, 1 // 2, 1 // 10, -1 // 20]
    d = vcat(T(20) .* (rand(rng, T, 1, nrows) .- T(1 // 2)), T(16) .* (rand(rng, T, 1, nrows) .- T(1 // 2)))
    return p, d, zero(T), zero(T)
end

"""
    t1140_check(be, T, c; nrows = 37) -> (; ratio_p, ratio_d)

Enzyme's reverse mode through `rows_kernel!` on `be`, seeded with a random adjoint w̄ of the output,
against ForwardDiff on the CPU: the adjoints of the parameters p̄ = Σ_j J_jᵀ w̄_j and of the rows'
positions d̄_j = J_jᵀ w̄_j, with J_j the row's Jacobian. Each ratio is the largest
|reverse − forward| / (eps(T)·Σ|J_j|ᵀ|w̄_j|) over the entries (the sum of the magnitudes of the
terms, so that a cancellation in the sum over rows does not inflate it).
"""
function t1140_check(be, ::Type{T}, c; nrows = 37) where {T}
    p, d, x, t = t1140_case(T, c, nrows)
    w̄ = randn(Xoshiro(41), T, nouts(c), nrows)
    dev(a) = (b = KernelAbstractions.allocate(be, eltype(a), size(a)); copyto!(b, a); b)
    out = dev(zeros(T, nouts(c), nrows))
    pd, dd = dev(p), dev(d)
    p̄, d̄ = dev(zero(p)), dev(zero(d))
    Enzyme.autodiff(Enzyme.Reverse, rows!, Enzyme.Const, Enzyme.Duplicated(out, dev(w̄)), Enzyme.Duplicated(pd, p̄),
                    Enzyme.Duplicated(dd, d̄), Enzyme.Const(c), Enzyme.Const(x), Enzyme.Const(t))
    KernelAbstractions.synchronize(be)
    np, nd = nparams(c), ndims_row(c)
    gp, sp = zeros(Float64, np), zeros(Float64, np)
    gd, sd = zeros(Float64, nd, nrows), zeros(Float64, nd, nrows)
    for j in 1:nrows
        f(z) = collect(rows(c, Tuple(z[1:np]), Tuple(z[np+1:end]), x, t))
        J = ForwardDiff.jacobian(f, vcat(p, d[:, j]))
        g, s = J' * Float64.(w̄[:, j]), abs.(J)' * abs.(Float64.(w̄[:, j]))
        gp .+= g[1:np]; sp .+= s[1:np]
        gd[:, j] .= g[np+1:end]; sd[:, j] .= s[np+1:end]
    end
    ratio(a, g, s) = maximum(abs.(Float64.(Array(a)) .- g) ./ (eps(T) .* max.(s, floatmin(Float64))))
    return (; ratio_p = ratio(p̄, gp, sp), ratio_d = ratio(d̄, gd, sd), out = Array(out))
end

const T1140_CASES = (Laser2D(AxisTime()), Laser2D(WavefrontTime()), Laser3D(AxisTime()), Laser3D(WavefrontTime()),
                     GaussianProfile(), PupilProfile())

function body_T1140(be)
    C = emC("T-1140")
    for T in (Float32, Float64), c in T1140_CASES
        r = t1140_check(be, T, c)
        rp = record!("T-1140 $(nameof(typeof(c))) p", T, r.ratio_p, be)
        rd = record!("T-1140 $(nameof(typeof(c))) d", T, r.ratio_d, be)
        @info "T-1140 Enzyme reverse against ForwardDiff" T case = c ratio_p = rp ratio_d = rd
        @test rp <= C
        @test rd <= C
    end
end
