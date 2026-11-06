# WP-033, the momentum distributions of the quiet start: T-530 to T-536 and T-538. The loads,
# references and digests are in momenta_cases.jl. Statistical checks run at fixed seeds, so they
# are deterministic; their bounds are in tolerances.toml.

const MJ_THETAS = (1e-5, 1e-3, 0.1, 0.5, 0.99, 1.0, 3.0, 100.0)

function body_T530(be = CPU())
    for T in TYPES
        vdrift, vth = (T(1 // 10), zero(T), T(-1 // 10)), (T(1 // 20), T(1 // 30), T(1 // 40))
        grid = PeriodicGrid(16, basetype(T)(1 // 8), 2)
        p = (seeded(T, 1), seeded(T, 1 // 20), T(2π / length(grid)))
        g2 = PeriodicGrid((8, 6), (basetype(T)(1 // 4), basetype(T)(1 // 3)), 2)
        p2 = oblique_params(T, g2)
        for at in (:particle, :cell_centre), first in (1, 4)
            a = Species(backend, T, 16 * 8 + 3; q = -1, m = 1)
            b = Species(backend, T, 16 * 8 + 3; q = -1, m = 1)
            on(be, load_quiet!, a, grid, CosineDensity(), p, 8; vdrift, vth, seed = 3, stream = 2, first, density_at = at)
            on(be, load_quiet!, b, grid, CosineDensity(), p, 8, GaussianVelocity(vdrift, vth); seed = 3, stream = 2, first,
               density_at = at)
            @test all(f -> same_bits(f(a), f(b)), (s -> s.cell[1], s -> s.offset[1], s -> s.u[1], s -> s.u[2], s -> s.u[3],
                                                   s -> s.w, s -> s.status, s -> s.count))
            a2 = Species(backend, T, 48 * 6 + 3; q = -1, m = 1, N = 2)
            b2 = Species(backend, T, 48 * 6 + 3; q = -1, m = 1, N = 2)
            on(be, load_quiet!, a2, g2, ObliqueDensity(), p2, (3, 2); vdrift, vth, seed = 3, stream = 2, first, density_at = at)
            on(be, load_quiet!, b2, g2, ObliqueDensity(), p2, (3, 2), GaussianVelocity(vdrift, vth); seed = 3, stream = 2, first,
               density_at = at)
            @test all(f -> same_bits(f(a2), f(b2)), (s -> s.cell[1], s -> s.cell[2], s -> s.offset[1], s -> s.offset[2], s -> s.u[1],
                                                     s -> s.u[2], s -> s.u[3], s -> s.w, s -> s.status, s -> s.count))
        end
        # the momentum Gaussian is the same draw, read as u
        a = Species(backend, T, 16 * 8; q = -1, m = 1)
        b = Species(backend, T, 16 * 8; q = -1, m = 1)
        on(be, load_quiet!, a, grid, CosineDensity(), p, 8, GaussianVelocity(vdrift, vth); seed = 3, stream = 2)
        on(be, load_quiet!, b, grid, CosineDensity(), p, 8, GaussianMomentum(vdrift, vth); seed = 3, stream = 2)
        @test all(d -> same_bits(a.u[d], b.u[d]), 1:3)
    end
    @test !draws_momentum(GaussianVelocity((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)))
    @test draws_momentum(GaussianMomentum((0.0, 0.0, 0.0), (0.1, 0.1, 0.1))) && draws_momentum(MaxwellJuttner(0.1, (0.0, 0.0, 0.0)))
end

"Every value and partial of every loaded momentum is finite."
all_finite(s) = all(i -> all(d -> all(isfinite, components(s.u[d][i])), 1:3), 1:alive_count(s))

function body_T531(be = CPU())
    for T in TYPES, θ in MJ_THETAS, β in ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.99, 0.0), (0.4, -0.6, 0.66))
        exhausted = KernelAbstractions.zeros(backend, Int32, 1)
        s = momenta_sample(T, MaxwellJuttner(T(θ), T.(β)), 4096; be, exhausted = be isa CPU ? exhausted : nothing)
        @test all_finite(s)
        be isa CPU && @test only(Array(exhausted)) == 0
    end
    # a sampler with one try: the rejected particles are counted, keep a finite proposal, and the count is the host's
    for T in (Float32, Float64), θ in (0.5, 2.0)
        exhausted = KernelAbstractions.zeros(backend, Int32, 1)
        dist = MaxwellJuttner(T(θ), T.((0.3, 0.0, 0.0)); tries = 1)
        s = momenta_sample(T, dist, 4096; exhausted)
        host = count(1:4096) do i
            c, k = (i - 1) ÷ 128 + 1, (i - 1) % 128 + 1
            !ParticleMeshCore.draw(dist, T, (UInt32(c), UInt32(0), UInt32(0), UInt32(k) << 16), (UInt32(11), UInt32(3)))[2]
        end
        @test all_finite(s)
        @test 0 < host == only(Array(exhausted))
        # the same at N = 2, where each slot's momentum is the device function at its own counter
        e2 = KernelAbstractions.zeros(backend, Int32, 1)
        g = PeriodicGrid((8, 6), (basetype(T)(1 // 4), basetype(T)(1 // 3)), 2)
        s2 = Species(backend, T, 48 * 6; q = -1, m = 1, N = 2)
        load_quiet!(s2, g, ObliqueDensity(), oblique_params(T, g), (3, 2), dist; seed = 5, stream = 2, exhausted = e2)
        misses = 0
        for i in 1:(48 * 6)
            L, k = (i - 1) ÷ 6 + 1, (i - 1) % 6 + 1
            c1, c2 = (L - 1) % 8 + 1, (L - 1) ÷ 8 + 1
            u, ok = ParticleMeshCore.draw(dist, T, (UInt32(c1), UInt32(c2), UInt32(0), UInt32(k) << 16), (UInt32(5), UInt32(2)))
            misses += !ok
            # within a few ulp: the kernel may contract to FMAs where the host does not (as in T-520)
            @test (s2.cell[1][i], s2.cell[2][i]) == (c1, c2) &&
                  all(d -> abs(s2.u[d][i] - u[d]) <= 8 * eps(T) * max(abs(u[d]), one(T)), 1:3)
        end
        @test 0 < misses == only(Array(e2))
    end
    # every particle of a load draws its own momentum: no two share one
    for T in (Float32, Float64)
        s = momenta_sample(T, MaxwellJuttner(T(0.3), T.((0.2, 0.0, 0.0))), 4096)
        @test allunique(momenta64(s))
    end
end

function body_T532()
    zmax = tolC("T-532", "z")
    for T in (Float32, Float64, D3), θ in MJ_THETAS
        q = juttner_quadrature(θ)
        u = momenta64(momenta_sample(T, MaxwellJuttner(T(θ), (zero(T), zero(T), zero(T))), 2^17))
        z = max(abs(zscore(kinetic.(u), q.mean_ε)), abs(zscore(sum.(abs2, u), q.mean_u2)))
        record!("T-532", T, z)
        @test z <= zmax
    end
end

function body_T533()
    cmax = tolC("T-533", "ks")
    for T in (Float32, Float64), θ in (0.01, 0.3, 0.99, 1.0, 3.0)
        q = juttner_quadrature(θ)
        u = momenta64(momenta_sample(T, MaxwellJuttner(T(θ), (zero(T), zero(T), zero(T))), 2^16; seed = 21))
        c = ks_statistic(kinetic.(u), q.ε, q.cdf)
        record!("T-533", T, c)
        @test c <= cmax
        # the direction is isotropic: the cosine to the z axis is uniform on (−1, 1)
        μ = [v[3] / sqrt(sum(abs2, v)) for v in u]
        cμ = ks_statistic(μ, [-1.0, 1.0], [0.0, 1.0])
        record!("T-533 direction", T, cμ)
        @test cμ <= cmax
    end
end

function body_T534()
    zmax = tolC("T-534", "z")
    for T in (Float32, Float64), θ in (0.01, 0.2, 3.0), β in ((0.5, 0.0, 0.0), (0.0, 0.99, 0.0), (0.3, -0.5, 0.6))
        ref = boosted_quadrature(θ, β)
        u = momenta64(momenta_sample(T, MaxwellJuttner(T(θ), T.(β)), 2^17; seed = 31))
        b = sqrt(sum(abs2, β))
        n = β ./ b
        upar = [sum(v .* n) for v in u]
        uperp2 = [sum(abs2, v) - a^2 for (v, a) in zip(u, upar)]
        γ = [sqrt(1 + sum(abs2, v)) for v in u]
        z = maximum(abs, (zscore(upar, ref.upar), zscore(γ, ref.γ), zscore(uperp2, ref.uperp2)))
        record!("T-534", T, z)
        @test z <= zmax
        # no mean momentum across the drift
        e1 = abs(n[3]) < 0.9 ? (n[2], -n[1], 0.0) : (0.0, n[3], -n[2])
        e1 = e1 ./ sqrt(sum(abs2, e1))
        zt = abs(zscore([sum(v .* e1) for v in u], 0.0))
        record!("T-534 transverse", T, zt)
        @test zt <= zmax
    end
end

function body_T535()
    cmax = tolC("T-535", "ks")
    θ = 1e-4
    for T in (Float32, Float64)
        mj = momenta64(momenta_sample(T, MaxwellJuttner(T(θ), (zero(T), zero(T), zero(T))), 2^16; seed = 41))
        gm = momenta64(momenta_sample(T, GaussianMomentum((zero(T), zero(T), zero(T)), ntuple(_ -> T(sqrt(θ)), 3)), 2^16; seed = 43))
        for d in 1:3
            c = ks_two_sample(getindex.(mj, d), getindex.(gm, d))
            record!("T-535", T, c)
            @test c <= cmax
        end
    end
end

function body_T536(be = CPU())
    runs = [momenta_digests(be) for _ in 1:2]
    @test runs[1] == runs[2]
    groups = Dict{String,Set{String}}()
    for l in runs[1]
        f = split(l, '|')
        push!(get!(groups, join(f[1:3], '|'), Set{String}()), f[5])
    end
    @test all(g -> length(g) == 1, values(groups))          # the launch shape does not change a bit
    be isa CPU && @test Set(filter(l -> split(l, '|')[4] == "default", runs[1])) == MOMENTA_PINNED   # and the recorded bits
    return runs[1]
end

@testset "T-530 GaussianVelocity is the keyword load bit for bit (N = 1 and 2, density_at, first), GaussianMomentum the same draw read as u [A4, A8, WP-033]" begin
    body_T530()
end

@testset "T-531 every Maxwell–Jüttner γ is finite for θ from 1e-5 to 100 and drifts up to 0.99, no try runs out; a one-try sampler counts its misses at N = 1 and 2, each slot holding its own counter's draw [A4, A8, WP-033]" begin
    body_T531()
end

@testset "T-532 rest-frame Maxwell–Jüttner moments ⟨γ − 1⟩ and ⟨u²⟩ against quadrature, θ from 1e-5 to 100 [A7, A8, WP-033]" begin
    body_T532()
end

@testset "T-533 Kolmogorov–Smirnov: the rest-frame kinetic energy against the quadrature cdf on both sides of the switch at θ = 1, and an isotropic direction [A7, WP-033]" begin
    body_T533()
end

@testset "T-534 the boosted Maxwell–Jüttner: ⟨u∥⟩, ⟨γ⟩ and ⟨u⊥²⟩ against 2D quadrature of the lab-frame density exp(−Γ(γ − β·u)/θ) [A7, WP-033]" begin
    body_T534()
end

@testset "T-535 at θ = 1e-4 Maxwell–Jüttner and GaussianMomentum with u_th = √θ have the same component distributions (two-sample Kolmogorov–Smirnov) [A7, WP-033]" begin
    body_T535()
end

@testset "T-536 the momentum loads are bitwise across reruns, launch shapes and thread counts, and keep their recorded digests [A4, A8, WP-033]" begin
    inproc = body_T536()
    w1 = [chopprefix(l, "MOM ") for l in worker_lines(1) if startswith(l, "MOM ")]
    w4 = [chopprefix(l, "MOM ") for l in worker_lines(4) if startswith(l, "MOM ")]
    @test !isempty(w1) && w1 == w4 == inproc
end

@testset "T-538 MaxwellJuttner refuses θ ≤ 0, a non-finite θ, |β| ≥ 1 and tries outside 1:32767 [WP-033]" begin
    for θ in (0.0, -0.1, Inf, NaN)
        @test_throws ArgumentError MaxwellJuttner(θ, (0.0, 0.0, 0.0))
    end
    @test_throws ArgumentError MaxwellJuttner(0.1, (0.6, 0.8, 0.0))
    @test_throws ArgumentError MaxwellJuttner(0.1, (1.2, 0.0, 0.0))
    @test_throws ArgumentError MaxwellJuttner(0.1, (0.0, 0.0, 0.0); tries = 0)
    @test_throws ArgumentError MaxwellJuttner(0.1, (0.0, 0.0, 0.0); tries = 2^15)
    @test MaxwellJuttner(0.1f0, (0.0, 0.5, 0.0)) isa MaxwellJuttner{Float32}
    @test MaxwellJuttner(mkdual(0.1, 1, 0, 0), (0.0, 0.0, 0.0)).θ isa D3
end
