# T-300, T-302 and T-840 of ParticleMeshCore with the engine's work on a backend, through the
# suite's own cases (flux2d_cases.jl, which take `be`), plus the Reference bits per launch shape:
#
#   julia +1.13 -t 4 --project=<repo>/lib/ParticleMeshCore/test gpu_flux_tests.jl <repo> [cpu|cuda]
#
# T-300 and T-302 are the bodies of test/flux2d.jl with `be` passed on. T-840 calls `flux_stencil`
# per particle on the host, so here its comparison runs in a kernel, one work item per particle,
# and the host adds up what the particles report; on the CPU backend that is the test itself.
using Test
const REPO = ARGS[1]
const TESTDIR = joinpath(REPO, "lib", "ParticleMeshCore", "test")
for f in ("setup.jl", "helpers.jl", "digest.jl", "engine_cases.jl", "flux_cases.jl", "n1_digests.jl", "engine2d_cases.jl",
          "flux2d_cases.jl")
    include(joinpath(TESTDIR, f))
end
const MODES = (Reference(), Fast())
const NAME = length(ARGS) >= 2 ? ARGS[2] : "cpu"
const BE = NAME == "cuda" ? CUDABackend() : CPU()
NAME == "cuda" && (CUDACore.allowscalar(false); println("device: ", CUDACore.name(CUDACore.device())))
const PMC = ParticleMeshCore

# The value and the partials as a tuple, in a form a device compiles (setup.jl's `components` collects).
@inline parts(x::AbstractFloat) = (x,)
@inline parts(d::Dual) = (value(d), partials(d).values...)
@inline amax(x) = maximum(abs, parts(x))
@inline bits_equal(a::T, b::T) where {T<:AbstractFloat} = reinterpret(Unsigned, a) == reinterpret(Unsigned, b)
@inline bits_equal(a::T, b::T) where {T<:Dual} = all(map(bits_equal, parts(a), parts(b)))

# One component of T-840 for one particle: (the shared nodes hold the same bits, the other nodes
# are zeros, the count of tail nodes, the largest tail over the scale).
@inline function frame_check(::Val{c}, JC, JP, lo, f, M, ::Val{P}, tol) where {c,P}
    R = (P + 2 - (c == 1), P + 2 - (c == 2))
    shared, zeros_ok, tails, worst = true, true, Int32(0), zero(tol)
    for k2 in 1:M[2], k1 in 1:M[1]
        @inbounds v = JC[k1 + (k2 - 1) * M[1]]
        m = (k1 + lo[1] - f[1], k2 + lo[2] - f[2])
        in1, in2 = 1 <= m[1] <= R[1], 1 <= m[2] <= R[2]
        if in1 && in2
            @inbounds shared &= bits_equal(v, JP[m[1] + (m[2] - 1) * R[1]])
        elseif c != 3 && m[c] > R[c] && (c == 1 ? in2 : in1)
            kk = c == 1 ? (M[1], k2) : (k1, M[2])
            @inbounds shared &= bits_equal(v, JC[kk[1] + (kk[2] - 1) * M[1]])
            tails += Int32(1)
            worst = max(worst, amax(v) / tol)
        else
            zeros_ok &= amax(v) == 0
        end
    end
    return shared, zeros_ok, tails, worst
end

@kernel function frames_kernel!(flags, tails, worst, s, shape::Shape{P}, st, n, sx, qz) where {P}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == PMC.ALIVE
        lo, M, JC = PMC.flux_stencil(s, i, shape, st, n, sx, qz)
        f, MP, JP = PMC.flux_stencil(s, i, shape, st, n, sx, qz, PMC.ParticleFrame())
        sizes = MP == (P + 2, P + 2) && map(length, JP) == ((P + 1) * (P + 2), (P + 2) * (P + 1), (P + 2)^2)
        sizes &= lo[1] <= f[1] && f[1] + P + 1 <= lo[1] + M[1] - 1 && lo[2] <= f[2] && f[2] + P + 1 <= lo[2] + M[2] - 1
        e = eps(typeof(worst[i]))
        qw = s.q * s.w[i]
        a = frame_check(Val(1), JC[1], JP[1], lo, f, M, Val(P), e * amax(qw * sx[1]))
        b = frame_check(Val(2), JC[2], JP[2], lo, f, M, Val(P), e * amax(qw * sx[2]))
        c = frame_check(Val(3), JC[3], JP[3], lo, f, M, Val(P), e * amax(qz(s, i)))
        flags[i] = UInt8(sizes) | UInt8(a[1] & b[1] & c[1]) << 1 | UInt8(a[2] & b[2] & c[2]) << 2 | 0x08
        tails[i] = a[3] + b[3]
        worst[i] = max(a[4], b[4])
    end
end

label = backend_label(BE)
@testset "the 2D flux tests on $label" begin
    @testset "T-300 discrete continuity at N = 2, on $label" begin
        C = tolC("T-300")
        for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS, mode in MODES
            c = folded_2d(continuity_case_2d(T, mode, shape, st; be = BE); be = BE)
            @test record!("T-300", T, continuity_residual_2d(c), BE) <= C
        end
    end

    @testset "T-302 Reference and Fast agree per node at N = 2, on $label" begin
        C = tolC("T-302")
        for T in (Float32, Float64), shape in SHAPES, st in STAGGER_PAIRS
            cR = continuity_case_2d(T, Reference(), shape, st; np = 192 * 20, seed = 11, be = BE)
            cF = continuity_case_2d(T, Fast(), shape, st; np = 192 * 20, seed = 11, be = BE)
            A = flux_abs_2d(cR.s, shape, st, cR.grid, cR.dt)
            worst = 0.0
            for (d, (JR, JF)) in enumerate(((cR.Jx, cF.Jx), (cR.Jy, cF.Jy), (cR.Jz, cF.Jz))), a in eachindex(JR)
                A[d][a] == 0 && (@test JR[a] == 0 && JF[a] == 0; continue)
                worst = max(worst, abs(JR[a] - JF[a]) / (eps(T) * A[d][a]))
            end
            @test record!("T-302", T, worst, BE) <= C
        end
    end

    @testset "T-303's bits on $label: the 2D Reference flux across reruns and launch shapes" begin
        wgs = (nothing, 64, 256)
        lines = flux2d_digests(BE; wgs)
        groups = Dict{String,Vector{String}}()
        for l in lines
            f = split(l, '|')
            push!(get!(groups, join(f[1:3], '|'), String[]), f[5])
        end
        @test length(groups) == length(TYPES) * length(SHAPES) * length(FLUX2D_DIGEST_PAIRS)
        @test all(v -> length(v) == length(wgs) && allequal(v), values(groups))
        @test flux2d_digests(BE; wgs) == lines
        if !(BE isa CPU)       # not a requirement (A4 is per backend): which cases give the CPU's bits
            cpu = Dict(join(split(l, '|')[1:3], '|') => split(l, '|')[5] for l in flux2d_digests(CPU(); wgs = (nothing,)))
            for k in sort(collect(keys(groups)))
                println("  ", rpad(k, 44), groups[k][1] == cpu[k] ? "the CPU's bits" : "other bits than the CPU")
            end
        end
    end

    @testset "T-840 the particle frame is the cell frame on its nodes, in a kernel on $label" begin
        C = tolC("T-840")
        for T in TYPES, shape in SHAPES, st in STAGGER_PAIRS
            V = basetype(T)
            grid = grid2(T, flux_g2(shape, st))
            s = flux_species_2d(T, grid, 400)
            n = Int32.(ncells(grid))
            sx = map(h -> h / T(V(1 // 4)), cellsize(grid))
            sd = BE isa CPU ? s : todev(BE, s)
            flags = KernelAbstractions.zeros(BE, UInt8, capacity(s))
            tails = KernelAbstractions.zeros(BE, Int32, capacity(s))
            worst = KernelAbstractions.zeros(BE, V, capacity(s))
            frames_kernel!(BE)(flags, tails, worst, sd, shape, st, n, sx, QZ2; ndrange = capacity(s))
            KernelAbstractions.synchronize(BE)
            flags, tails, worst = Array(flags), Array(tails), Array(worst)
            alive = [i for i in 1:capacity(s) if s.status[i] == PMC.ALIVE]
            @test flags[alive] == fill(0x0f, length(alive))
            @test all(iszero, flags[setdiff(1:capacity(s), alive)])
            @test sum(tails[alive]) > 0
            @test record!("T-840", T, maximum(worst[alive]), BE) <= C
        end
    end
end
println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
for id in sort(collect(keys(MEASURED)))
    println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
end
