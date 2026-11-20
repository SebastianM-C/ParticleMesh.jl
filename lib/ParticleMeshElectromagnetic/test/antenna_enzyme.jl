# WP-083: the laser's device functions call sin and cos, not sincos (#547). T-1141: what that does
# to the bits inside a kernel. T-1140 (run by gpu.jl on CUDA): Enzyme's
# reverse mode through what the antenna's kernels add per row, with the laser's parameters in a
# device array, against ForwardDiff on the CPU.

using Enzyme: Enzyme
using ParticleMeshElectromagnetic: GaussianLaser, ShapedLaser, AxisTime, WavefrontTime, Antenna, AntennaPlane, antenna_b_rows, antenna_e_rows
using ParticleMeshElectromagnetic.PulseShapes: GaussianBeam, PupilBeam, GaussianPupil, GaussianEnvelope

# Inside a kernel the backend's own sin, cos and sincos run (OpenCL's builtins on KernelAbstractions'
# CPU backend, libdevice on CUDA), and its sincos need not give the bits of its sin and cos: on the
# CPU backend they differ in about a fifth (Float32) and a tenth (Float64) of the arguments, by up
# to 2 ulp, and the separate calls are the closer to the exact values. So the antenna's fields move
# in their last bits with WP-083; its tests' tolerances hold them.
@kernel function sincos_pair_kernel!(s1, c1, s2, c2, @Const(x))
    i = @index(Global, Linear)
    @inbounds begin
        v = x[i]
        s1[i], c1[i] = sincos(v)
        s2[i] = sin(v)
        c2[i] = cos(v)
    end
end

"Distance of `a` from the exact value `r` in ulps of `r` (of the type of `a`)."
ulps(a, r) = Float64(abs(big(a) - r) / eps(max(typeof(a)(abs(r)), floatmin(typeof(a)))))

@testset "T-1141 in a kernel, sin and cos are within 2 ulp of sincos and within 1 ulp of the exact values, Float32 and Float64 [A4, A7, WP-083]" begin
    for T in (Float32, Float64), (k, x) in enumerate((T(2000) .* rand(Xoshiro(1141), T, 20_000) .- T(1000),   # the antenna's phases
                                                      T(2e6) .* rand(Xoshiro(1142), T, 4_000) .- T(1e6),       # long, chirped runs
                                                      T(1 // 1000) .* randn(Xoshiro(1143), T, 4_000)))
        xd = KernelAbstractions.allocate(backend, T, length(x)); copyto!(xd, x)
        s1, c1, s2, c2 = (similar(xd) for _ in 1:4)
        sincos_pair_kernel!(backend)(s1, c1, s2, c2, xd; ndrange = length(x))
        KernelAbstractions.synchronize(backend)
        s1, c1, s2, c2 = Array.((s1, c1, s2, c2))
        apart, err_pair, err_sincos = setprecision(BigFloat, 256) do
            (maximum(i -> max(ulps(s2[i], big(s1[i])), ulps(c2[i], big(c1[i]))), eachindex(x)),
             maximum(i -> max(ulps(s2[i], sin(big(x[i]))), ulps(c2[i], cos(big(x[i])))), eachindex(x)),
             maximum(i -> max(ulps(s1[i], sin(big(x[i]))), ulps(c1[i], cos(big(x[i])))), eachindex(x)))
        end
        @info "T-1141 sin and cos against sincos in a kernel" T range = k differ = count(s1 .!== s2) + count(c1 .!== c2) apart err_pair err_sincos
        @test record!("T-1141 apart", T, apart) <= emC("T-1141", "C_apart")
        @test record!("T-1141 sin, cos", T, err_pair) <= emC("T-1141")
    end
end

# The cases of T-1140: the antenna's own per-row functions (`antenna_b_rows`, `antenna_e_rows`,
# what its kernels add before the factor c), on an antenna whose laser is built inside the kernel
# from a tuple of parameters `P` read from a device array, on a fixed plane.
struct Gauss2D{E,T}
    envelope::E
    center::T
end
struct Gauss3D{E,T}
    envelope::E
    center::NTuple{2,T}
end
struct Shaped{L}
    laser::L        # its a0, delay and phase stay (they reach the kernels through the time factor only)
end
# a0, waist, duration, delay, focus, chirp, polarization, phase; the shaped lasers' focus,
# polarization and profile numbers (GaussianBeam: waist; PupilBeam: na, fill, two aberrations)
nparams(::Union{Gauss2D,Gauss3D}) = 8
nparams(c::Shaped) = 2 + (c.laser.profile isa GaussianBeam ? 1 : 4)

laser_of(c::Union{Gauss2D,Gauss3D}, P, ::Type{S}) where {S} =
    GaussianLaser{eltype(P),S,typeof(c.envelope),typeof(c.center)}(P..., zero(S), c.center, c.envelope)
@inline shaped_profile(b::GaussianBeam, P) = GaussianBeam{eltype(P)}(P[3])
@inline function shaped_profile(b::PupilBeam, P)
    D = eltype(P)
    return PupilBeam{D,GaussianPupil{D},2}(P[3], GaussianPupil{D}(P[4]), (P[5], P[6]), b.samples)
end
function laser_of(c::Shaped, P, ::Type{S}) where {S}
    l, D = c.laser, eltype(P)
    b = shaped_profile(l.profile, P)
    return ShapedLaser{D,S,typeof(b),typeof(l.envelope)}(D(l.a0), D(l.delay), P[1], P[2], D(l.phase), b, l.envelope, l.position, l.center)
end

# the four values the kernels add at the row q (at N = 3 the q-th (row, layer) of the plane, rows
# first): Bz and By after a B half step to time t, Ey and Ez after the E step
@inline function antenna_rows(a, ::Union{Gauss2D,Shaped}, q::Int32, nj::Int32, t)
    T = typeof(a.xa)
    return (antenna_b_rows(a, q, t, T)..., antenna_e_rows(a, q, t, T)...)
end
@inline function antenna_rows(a, ::Gauss3D, q::Int32, nj::Int32, t)
    T = typeof(a.xa)
    j, k = (q - Int32(1)) % nj + Int32(1), (q - Int32(1)) ÷ nj + Int32(1)
    return (antenna_b_rows(a, j, k, t, T)..., antenna_e_rows(a, j, k, t, T)...)
end

@kernel function rows_kernel!(out, @Const(p), c, plane, t, nj)
    q = Int32(@index(Global, Linear))
    @inbounds begin
        P = ntuple(k -> p[k], Val(nparams(c)))
        a = Antenna(laser_of(c, P, typeof(plane.xa)), plane)
        o = antenna_rows(a, c, q, nj, t)
        ntuple(k -> (out[k, q] = o[k]; nothing), Val(4))
    end
end

function rows!(out, p, c, plane, t, nj)
    rows_kernel!(get_backend(out))(out, p, c, plane, t, nj; ndrange = size(out, 2))
    return nothing
end

# A laser of the trapped-beam studies' scale (waist 6, duration 8), 26 c/ω₀ before its focus, on
# its rising edge, so that every derivative is far from zero; every row of the plane, guards included.
function t1140_case(::Type{T}, kind) where {T}
    p = T[13 // 10, 6, 8, 48, 28, 1 // 50, 3 // 10, 1 // 5]
    t = T(41)
    if kind === :gauss3d
        grid = PeriodicGrid((16, 12, 10), (T(1 // 5), T(3 // 4), T(3 // 4)), 3)
        plane = AntennaPlane(T(2), grid, T(1 // 10))
        nj, nk = grid.n[2] + 2 * guards(grid), grid.n[3] + 2 * guards(grid)
        return Gauss3D(AxisTime(), (T(9 // 2), T(15 // 4))), p, plane, t, Int32(nj), nj * nk
    end
    grid = PeriodicGrid((32, 24), (T(1 // 5), T(3 // 4)), 3)
    plane = AntennaPlane(T(2), grid, T(1 // 10))
    nj = grid.n[2] + 2 * guards(grid)
    kind === :gauss2d && return Gauss2D(AxisTime(), T(9)), p, plane, t, Int32(nj), nj
    kind === :wave2d && return Gauss2D(WavefrontTime(), T(9)), p, plane, t, Int32(nj), nj
    kind === :wave3d && return (c = t1140_case(T, :gauss3d); (Gauss3D(WavefrontTime(), c[1].center), c[2:end]...))
    profile, q = kind === :gaussbeam ? (GaussianBeam(; waist = T(6)), T[6]) :
                 (PupilBeam(; na = T(1 // 4), amplitude = GaussianPupil(; fill = T(1 // 2)), aberrations = (T(1 // 10), -T(1 // 20)), samples = 48),
                  T[1 // 4, 1 // 2, 1 // 10, -1 // 20])
    l = ShapedLaser(; a0 = T(13 // 10), delay = T(48), focus = T(28), position = T(2), center = T(9), profile,
                    envelope = GaussianEnvelope(; duration = T(8), chirp = T(1 // 50)), polarization = T(3 // 10), phase = T(1 // 5))
    g = ParticleMeshElectromagnetic.time_factor(l, t - l.delay)
    return Shaped(l), vcat(T[28, 3 // 10], q), plane, g, Int32(nj), nj
end

"""
    t1140_check(be, T, kind) -> ratio

Enzyme's reverse mode through `rows_kernel!` on `be`, seeded with a random adjoint w̄ of the four
values of every row, against ForwardDiff on the CPU: the adjoint of the parameters
p̄ = Σ_q J_qᵀ w̄_q, J_q the row's Jacobian in p. The ratio is the largest |reverse − forward| /
(eps(T)·Σ_q |J_q|ᵀ|w̄_q|) over the parameters (the sum of the terms' magnitudes, so that a
cancellation in the sum over rows does not inflate it).
"""
function t1140_check(be, ::Type{T}, kind) where {T}
    c, p, plane, t, nj, nrows = t1140_case(T, kind)
    w̄ = randn(Xoshiro(41), T, 4, nrows)
    dev(a) = (b = KernelAbstractions.allocate(be, eltype(a), size(a)); copyto!(b, a); b)
    out, pd, p̄ = dev(zeros(T, 4, nrows)), dev(p), dev(zero(p))
    Enzyme.autodiff(Enzyme.Reverse, rows!, Enzyme.Const, Enzyme.Duplicated(out, dev(w̄)), Enzyme.Duplicated(pd, p̄),
                    Enzyme.Const(c), Enzyme.Const(plane), Enzyme.Const(t), Enzyme.Const(nj))
    KernelAbstractions.synchronize(be)
    g, s = zeros(Float64, length(p)), zeros(Float64, length(p))
    for q in 1:nrows
        f(z) = collect(antenna_rows(Antenna(laser_of(c, Tuple(z), T), plane), c, Int32(q), nj, t))
        J = ForwardDiff.jacobian(f, p)
        g .+= J' * Float64.(w̄[:, q])
        s .+= abs.(J)' * abs.(Float64.(w̄[:, q]))
    end
    return maximum(abs.(Float64.(Array(p̄)) .- g) ./ (eps(T) .* max.(s, floatmin(Float64))))
end

const T1140_KINDS = (:gauss2d, :wave2d, :gauss3d, :wave3d, :gaussbeam, :pupil)

function body_T1140(be)
    C = emC("T-1140")
    for T in (Float32, Float64), kind in T1140_KINDS
        r = record!("T-1140 $kind", T, t1140_check(be, T, kind), be)
        @info "T-1140 Enzyme reverse against ForwardDiff" T kind ratio = r
        @test r <= C
    end
end
