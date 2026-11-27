# WP-086: what the 3D trapped-beam step calls that Enzyme's reverse mode could not take on CUDA, made
# so in the engine instead of by script-side patches: the density ramp's sin²(πt/2) through `sin`
# (the device's `sinpi` has no reverse pass, #545) and the integer counts through `add_count!`, which
# Enzyme takes as inactive (its reverse pass of `sum!` on a device array fails, #551). T-1173 on the
# CPU; T-1174's body runs on CUDA (gpu.jl), since Enzyme does not differentiate the CPU backend's
# kernels (#509). The third change, `current!`'s species loops as `map`s (#384), has no test of its
# own: it keeps every launch and its order, so the step's bits are those that the suite checks, and
# Enzyme's reverse pass through the 3D step on CUDA is the evidence (scripts/enzyme/trapped3d.jl).

using Enzyme: Enzyme
using ParticleMeshCore: add_count!
using ParticleMeshElectromagnetic: ramp_factor, RampedCosine

@testset "T-1173 the density ramp's sin²(πt/2) through sin: within a few ulps of the exact value, 0 and 1 at the ends and clamped outside, and its derivative under Enzyme and ForwardDiff is π/(2L)·sin(πt) [A7, A8, WP-086]" begin
    C, Cd = emC("T-1173"), emC("T-1173", "C_derivative")
    for T in (Float32, Float64)
        v, L = T(5), T(4)
        err, derr = 0.0, 0.0
        for t in range(0, 1; length = 2001)
            x = v + L * T(t)
            exact, dexact = setprecision(BigFloat, 256) do
                τ = clamp((big(x) - big(v)) / big(L), 0, 1)
                (sinpi(τ / 2)^2, big(π) / (2 * big(L)) * sinpi(τ))
            end
            r = ramp_factor(x, v, L)
            @test r isa T
            err = max(err, Float64(abs(big(r) - exact) / eps(max(T(exact), floatmin(T)))))
            0 < x - v < L || continue                                     # the derivative inside the ramp
            d = Enzyme.autodiff(Enzyme.Reverse, ramp_factor, Enzyme.Active, Enzyme.Active(x), Enzyme.Const(v), Enzyme.Const(L))[1][1]
            @test d isa T
            @test d == ForwardDiff.derivative(z -> ramp_factor(z, v, L), x)
            derr = max(derr, Float64(abs(big(d) - dexact) / (eps(T) * big(π) / (2 * big(L)))))
        end
        @info "T-1173 the ramp through sin" T ulps = err derivative = derr
        @test record!("T-1173", T, err) <= C
        @test record!("T-1173 derivative", T, derr) <= Cd
        @test ramp_factor(v, v, L) === zero(T) && ramp_factor(v + L, v, L) === one(T)
        @test ramp_factor(v - 3, v, L) === zero(T) && ramp_factor(v + L + 3, v, L) === one(T)
    end
end

# --- T-1174's body ------------------------------------------------------------------------------

# The weights of particles at positions (x₁, x₂, x₃) from the 3D window's refill profile, its seven
# parameters read from a device array (density, modulation, three wave numbers, vacuum, ramp).
@kernel function ramped_weights_kernel!(w, @Const(x1), @Const(x2), @Const(x3), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        P = ntuple(k -> p[k], Val(7))
        w[i] = RampedCosine()((x1[i], x2[i], x3[i]), P)
    end
end
ramped_weights!(w, x1, x2, x3, p) = (ramped_weights_kernel!(get_backend(w))(w, x1, x2, x3, p; ndrange = length(w)); nothing)

# Squares of `x` and their flags (x > 0), then the count of all flags but the last into `acc`
# through `add_count!` on a view, as the window's refill counts its fast particles.
@kernel function square_flag_kernel!(y, flags, @Const(x))
    i = @index(Global, Linear)
    @inbounds begin
        y[i] = x[i] * x[i]
        flags[i] = Int32(x[i] > zero(eltype(x)))
    end
end
function counted_squares!(y, flags, acc, x)
    square_flag_kernel!(get_backend(y))(y, flags, x; ndrange = length(x))
    add_count!(acc, view(flags, 1:(length(x) - 1)))
    return nothing
end

"""
    body_T1174(be)

On `be` (CUDA), Enzyme's reverse mode with runtime activity, as the trapped-beam driver runs it:
through a kernel that evaluates the 3D refill profile `RampedCosine` (the ramp's `sin`) against
ForwardDiff on the CPU, with T-1140's ratio |reverse − forward| / (eps(T)·scale) per input, its
scale widened by the Hessian's terms (below); and through a kernel and an `add_count!` of its integer flags, whose count is the primal's,
whose float gradient is 2x·ȳ exactly, and whose arrays are usable after the pass (#551 freed them).
"""
function body_T1174(be)
    dev(a) = (b = KernelAbstractions.allocate(be, eltype(a), size(a)); copyto!(b, a); b)
    rmode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    C = emC("T-1174")
    for T in (Float32, Float64)
        n = 4096
        rng = Xoshiro(1174)
        x = (T(12) .* rand(rng, T, n), T(8) .* rand(rng, T, n) .- T(4), T(8) .* rand(rng, T, n) .- T(4))   # vacuum, ramp and plateau in x₁
        p = T[3 // 2, 2 // 5, 3 // 10, -1 // 7, 1 // 3, 5, 4]
        w̄ = randn(rng, T, n)
        xd, x̄ = map(dev, x), map(a -> dev(zero(a)), x)
        w, pd, p̄ = dev(zeros(T, n)), dev(p), dev(zero(p))
        Enzyme.autodiff(rmode, ramped_weights!, Enzyme.Const, Enzyme.Duplicated(w, dev(w̄)), map(Enzyme.Duplicated, xd, x̄)..., Enzyme.Duplicated(pd, p̄))
        KernelAbstractions.synchronize(be)
        wcpu = [RampedCosine()((x[1][i], x[2][i], x[3][i]), Tuple(p)) for i in 1:n]
        @test maximum(abs.(Array(w) .- wcpu)) <= 8 * eps(T) * maximum(abs, wcpu)   # the reverse pass's primal; the device's sin, cos to an ulp or two
        # the scale per input k of point i: |w̄_i|·(|J_ik| + Σ_l |H_ikl|·|z_il|), the first-order error
        # of J_ik with every input rounded by eps (H the Hessian, in Float64): at a point where J_ik
        # nearly vanishes, as the cosine's slope at its crests, the rounding of the argument alone
        # moves it by many of its own ulps (the CPU's Float32 against its Float64 reaches 1080 of
        # them over these points, and 0.37 in this scale), summed over the points for a parameter
        f(z) = RampedCosine()((z[1], z[2], z[3]), Tuple(z[4:10]))
        gp, sp = zeros(10), zeros(10)
        gx, sx = zeros(n, 3), zeros(n, 3)
        for i in 1:n
            z = vcat(T[x[1][i], x[2][i], x[3][i]], p)
            J = Float64.(ForwardDiff.gradient(f, z))
            H = ForwardDiff.hessian(f, Float64.(z))
            s = (abs.(J) .+ abs.(H) * abs.(Float64.(z))) .* abs(Float64(w̄[i]))
            gx[i, :] .= J[1:3] .* Float64(w̄[i]); sx[i, :] .= s[1:3]
            gp .+= J .* Float64(w̄[i]); sp .+= s
        end
        rx = maximum(d -> maximum(abs.(Float64.(Array(x̄[d])) .- gx[:, d]) ./ (eps(T) .* max.(sx[:, d], floatmin(Float64)))), 1:3)
        rp = maximum(abs.(Float64.(Array(p̄)) .- gp[4:10]) ./ (eps(T) .* max.(sp[4:10], floatmin(Float64))))
        r = record!("T-1174", T, max(rx, rp), be)
        @info "T-1174 the ramped profile under Enzyme against ForwardDiff" T positions = rx parameters = rp
        @test r <= C
        @test count(i -> 0 < x[1][i] - 5 < 4, 1:n) > n ÷ 4                     # points on the ramp
        @test all(a -> any(!iszero, Array(a)), x̄) && any(!iszero, Array(p̄))
        # the count: the primal's, the gradient exact, the arrays usable after the pass
        xh = randn(rng, T, n)
        xc, dxc = dev(xh), dev(zeros(T, n))
        y, ȳ = dev(zeros(T, n)), dev(ones(T, n))
        flags, dflags = dev(zeros(Int32, n)), dev(zeros(Int32, n))
        acc, dacc = dev(Int32[3]), dev(Int32[0])
        Enzyme.autodiff(rmode, counted_squares!, Enzyme.Const, Enzyme.Duplicated(y, ȳ), Enzyme.Duplicated(flags, dflags),
                        Enzyme.Duplicated(acc, dacc), Enzyme.Duplicated(xc, dxc))
        KernelAbstractions.synchronize(be)
        @test Array(acc) == Int32[3 + count(>(0), xh[1:(end - 1)])]
        @test Array(dxc) == 2 .* xh
        @test Array(flags) == Int32.(xh .> 0)
        copyto!(flags, zeros(Int32, n))                                         # usable after the pass
        counted_squares!(y, flags, acc, xc)
        KernelAbstractions.synchronize(be)
        @test Array(flags) == Int32.(xh .> 0) && Array(acc) == Int32[3 + 2 * count(>(0), xh[1:(end - 1)])]
    end
end
