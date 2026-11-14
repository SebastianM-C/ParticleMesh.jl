# WP-029: the binomial current filter, T-440 to T-446. One pass of (1, 2, 1)/4 along axis d
# multiplies the Fourier mode m_d of n_d cells by cos²(π m_d/n_d) on any stagger, so p passes over
# every axis transfer H_p(m) = Π_d cos^{2p}(π m_d/n_d). Applied to the folded current before the
# mesh solve, it commutes with the backward differences of the continuity equation, so the
# filtered Gauss law D⁻·Eⁿ − Fρⁿ = D⁻·E⁰ − Fρ⁰ holds to roundoff (T-441), and the plain residual
# D⁻·Eⁿ − (ρⁿ − mean ρ) is (F − 1)(ρⁿ − ρ⁰), which T-441 measures. Host references: the
# `circshift` filter `binomial_host` and the DFT below, neither of which uses the module's kernels.

foreach(start_determinism_worker, DETERMINISM_THREADS)   # the worker runs (determinism_cases.jl) beside this file's in-process runs

using ParticleMeshElectromagnetic: RHO_STAGGER_2D, FIELD_LAYOUT_2D

"A random value of `T` in [−1, 1) (each partial of a dual drawn too)."
rand_T(rng, ::Type{T}) where {T<:AbstractFloat} = T(2rand(rng) - 1)
rand_T(rng, ::Type{Dual{Nothing,V,N}}) where {V,N} = Dual{Nothing}(V(2rand(rng) - 1), ntuple(_ -> V(2rand(rng) - 1), N)...)

"The value and each partial of an array of `T`, as BigFloat arrays (exact conversions)."
component_arrays(A::AbstractArray{<:AbstractFloat}) = [BigFloat.(A)]
component_arrays(A::AbstractArray{<:Dual}) = [BigFloat.(map(x -> components(x)[c], A)) for c in 1:length(components(first(A)))]

"""
The normalized DFT of a real array in one or two dimensions, Â[m] = Σ_j A[j]·exp(−2πi m·j/n)/Π n,
in BigFloat, so that its own rounding stays far below eps(Float64) at any test size (in Float64
it reached 1.4 of T-440's scale at n = 33, measured by the WP-029 physics breaker).
"""
function dft(A::AbstractArray{BigFloat,N}) where {N}
    W(n) = [cispi(-2 * big((k - 1) * (j - 1)) / n) for k in 1:n, j in 1:n]
    N == 1 && return W(length(A)) * A ./ length(A)
    return W(size(A, 1)) * A * transpose(W(size(A, 2))) ./ length(A)
end

"The closed form H_p of the DFT index `m` (1-based) on `n` cells: Π_d cos^{2p}(π(m_d − 1)/n_d)."
transfer_closed(m::CartesianIndex, n, p) = prod(d -> cospi(big(m[d] - 1) / n[d])^(2p), 1:length(n))

"A field of `grid` with the interior `A` and every guard set to `junk`, and a scratch field filled with `junk2`."
function guarded(grid, A::AbstractArray{T}, junk, junk2) where {T}
    a = field(CPU(), T, grid)
    fill!(a, junk)
    interior_view(a, grid) .= A
    s = field(CPU(), T, grid)
    fill!(s, junk2)
    return a, s
end

const FILTER_GRIDS = (PeriodicGrid(16, 1 // 4, 2), PeriodicGrid((12, 10), (1 // 4, 1 // 5), 3))
grid_T(g::PeriodicGrid{N}, ::Type{T}) where {N,T} = PeriodicGrid(g.n, map(d -> T(d), g.dx), guards(g))

# --- T-440 ----------------------------------------------------------------------------------

# Mode by mode: a random field (value and partials) through `binomial!`, whose guards start at
# junk and whose scratch starts at NaN (a pass that read a guard before filling it, or the
# scratch before writing it, shows), then the DFT of input and output: every coefficient must be
# H_p times the input's, to roundoff of p·N passes of the stencil (two roundings each) relative to
# the largest input value. The same for a single mode sampled on each of the four 2D staggers.
# Exact identities, bit for bit: a constant stays the constant, each axis' Nyquist mode becomes
# +0.0, a field mirror-symmetric along an axis stays so, zero passes touch nothing.
function body_T440(be)
    C = emC("T-440")
    rng = Xoshiro(440)
    for T in TYPES, g0 in FILTER_GRIDS, p in (1, 2, 3, 5)
        grid = grid_T(g0, T)
        n, N = grid.n, length(grid.n)
        A = [rand_T(rng, T) for _ in CartesianIndices(n)]
        a, s = guarded(grid, A, T(1000), T(NaN))
        a, s = todev(be, a), todev(be, s)
        binomial!(a, s, grid, p)
        out = Array(a)
        Bout = collect(interior_view(out, grid))
        worst = 0.0
        for (ci, co) in zip(component_arrays(A), component_arrays(Bout))
            Âi, Âo = dft(ci), dft(co)
            err = maximum(m -> abs(Âo[m] - transfer_closed(m, n, p) * Âi[m]), CartesianIndices(Âi))
            worst = max(worst, Float64(err / (base_eps(T) * maximum(abs, ci) * p * N)))
        end
        # the guards hold the periodic images on return
        ref = copy(out)
        fill_guards!(ref, grid)
        @test same_bits(ref, out)
        # the exported closed form is the same function of k = 2π m/L
        L = map(*, n, grid.dx)
        m = ntuple(d -> d, N)
        k = N == 1 ? 2π * m[1] / Float64(value_of(L[1])) : map((md, Ld) -> 2π * md / Float64(value_of(Ld)), m, L)
        dx = N == 1 ? Float64(value_of(grid.dx[1])) : map(d -> Float64(value_of(d)), grid.dx)
        @test binomial_transfer(k, dx, p) ≈ transfer_closed(CartesianIndex(map(+, m, ntuple(_ -> 1, N))), n, p) rtol = 64 * base_eps(T)   # k and Δ in T
        record!("T-440", T, worst, be)
        @test worst <= C
    end
    # one mode at each of the four staggers of the 2D layout: H_p times the mode, to roundoff
    for T in TYPES
        g = grid_T(FILTER_GRIDS[2], T)
        L = map(*, g.n, g.dx)
        m = (3, -2)
        p = 2
        H = T(transfer_closed(CartesianIndex(m[1] + 1, mod(m[2], g.n[2]) + 1), g.n, p))
        worst = 0.0
        for st in ((Center(), Node()), (Node(), Center()), (Node(), Node()), (Center(), Center()))
            a = field(CPU(), T, g)
            set_mode!(a, g, st, m, L, T(3) / 4, T(1) / 3)
            want = collect(interior_view(a, g))
            s = field(CPU(), T, g)
            a, s = todev(be, a), todev(be, s)
            binomial!(a, s, g, p)
            got = collect(interior_view(Array(a), g))
            worst = max(worst, maximum(absmax, got .- H .* want) / (base_eps(T) * p * 2))
        end
        record!("T-440 staggers", T, worst, be)
        @test worst <= C
    end
    # exact identities
    for T in TYPES, g0 in FILTER_GRIDS
        grid = grid_T(g0, T)
        n, N = grid.n, length(grid.n)
        c = seeded(T, 3 / 7)
        a, s = guarded(grid, fill(c, n), zero(T), T(NaN))
        a, s = todev(be, a), todev(be, s)
        binomial!(a, s, grid, 3)
        @test all(x -> same_bits([x], [c]), collect(interior_view(Array(a), grid)))
        for d in 1:N
            iseven(n[d]) || continue
            A = [isodd(I[d]) ? c : -c for I in CartesianIndices(n)]
            a, s = guarded(grid, A, zero(T), T(NaN))
            a, s = todev(be, a), todev(be, s)
            binomial!(a, s, grid, 1)
            @test all_plus_zero(collect(interior_view(Array(a), grid)))
            # mirror symmetry along d: j − 1 ↦ −(j − 1) mod n_d
            mirror(I) = CartesianIndex(ntuple(k -> k == d ? mod(-(I[k] - 1), n[k]) + 1 : I[k], N))
            B = [rand_T(rng, T) for _ in CartesianIndices(n)]
            S = [B[I] + B[mirror(I)] for I in CartesianIndices(n)]
            a, s = guarded(grid, S, zero(T), T(NaN))
            a, s = todev(be, a), todev(be, s)
            binomial!(a, s, grid, 3)
            O = collect(interior_view(Array(a), grid))
            @test all(I -> same_bits([O[I]], [O[mirror(I)]]), CartesianIndices(n))
            @test !all(I -> same_bits([O[I]], [O[CartesianIndices(n)[1]]]), CartesianIndices(n))   # not trivially constant
        end
        A = [rand_T(rng, T) for _ in CartesianIndices(n)]
        a, s = guarded(grid, A, T(7), T(NaN))
        a0, s0 = copy(a), copy(s)
        a, s = todev(be, a), todev(be, s)
        @test binomial!(a, s, grid, 0) === a
        @test same_bits(Array(a), a0) && same_bits(Array(s), s0)
    end
end

@testset "T-440 the binomial filter's transfer function is Π cos^(2p)(kΔ/2) mode by mode, on every stagger, with its exact identities [A2, A7, A8, WP-029]" begin
    body_T440(CPU())
end

# --- T-441 ----------------------------------------------------------------------------------

"""
    gauss_parts(state) -> (; divE, ρ)

The backward-difference divergence of E and the charge deposited now in the state's mode, on the
host, as interior arrays (the species are sorted by their current cells after every step).
"""
function gauss_parts(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    N = ndims(st)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, N == 1 ? RHO_STAGGER : RHO_STAGGER_2D, grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), grid.n, grid.dx
    ρ = collect(interior_view(Array(ρd), grid))
    Ex = Array(st.fields.Ex)
    if N == 1
        divE = [(Ex[g + j] - Ex[g + j - 1]) / h[1] for j in 1:n[1]]
    else
        Ey = Array(st.fields.Ey)
        divE = [(Ex[g + i, g + j] - Ex[g + i - 1, g + j]) / h[1] + (Ey[g + i, g + j] - Ey[g + i, g + j - 1]) / h[2]
                for i in 1:n[1], j in 1:n[2]]
    end
    return (; divE, ρ)
end

# Each step changes D⁻·E by −Δt·D⁻·(FJ) = −Δt·F(D⁻·J) = F(ρⁿ⁺¹ − ρⁿ) (the filter is a periodic
# convolution, so it commutes with D⁻, and deposit_flux! makes the continuity equation hold to
# roundoff, T-300), hence Qⁿ = D⁻·Eⁿ − Fρⁿ is constant, to the roundoff of T-212 and T-360 (same
# scales, √(k + 1) random walk). The plain residual rⁿ = D⁻·Eⁿ − (ρⁿ − mean ρ) is then
# (F − 1)(ρⁿ − ρ⁰): it is measured (largest |rⁿ| over the run, relative to the largest
# |ρⁿ − mean ρ|) and must stand far above roundoff, so the filter acts; its mean stays zero, since
# F keeps the mean. 1D: the Landau case with 1 and 3 passes; 2D: the two-species plasma of T-360
# with 1 and 2 passes; both modes, every element type.
function body_T441(be)
    C = emC("T-441")
    runs = Tuple{String,Int,Any}[]
    for T in TYPES, mode in (Reference(), Fast()), N in (1, 2), passes in (N == 1 ? (1, 3) : (1, 2))
        if N == 1
            st, prob = landau_filtered(T, passes; mode, be)
        else
            prob = plasma_step_2d(T; amplitude = seeded(T, 1 / 10))
            st = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes)); backend = be)
            initialize!(st, prob)
        end
        p = params(prob)
        n = st.grid.n
        gp = gauss_parts(st)
        Q0 = gp.divE .- binomial_host(gp.ρ, passes)
        ρ0 = gp.ρ
        worst = 0.0
        effect = 0.0
        meanr = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            gp = gauss_parts(st)
            Q = gp.divE .- binomial_host(gp.ρ, passes)
            scale = sum(n) * maximum(absmax, gp.ρ)
            worst = max(worst, maximum(absmax, Q .- Q0) / (base_eps(T) * scale * sqrt(k + 1)))
            ρ̄ = sum(gp.ρ) / prod(n)
            r = gp.divE .- (gp.ρ .- ρ̄)
            effect = max(effect, Float64(value_of(maximum(x -> abs(value_of(x)), r) / maximum(x -> abs(value_of(x)), gp.ρ .- ρ̄))))
            meanr = max(meanr, Float64(absmax(sum(r))) / (base_eps(T) * scale * prod(n) * sqrt(k + 1)))
        end
        # the plain residual is (F − 1)(ρⁿ − ρ⁰), and the run has moved the charge
        @test maximum(absmax, gp.ρ .- ρ0) > 0
        label = "$(N)D p=$passes $(nameof(typeof(mode)))"
        record!("T-441 $label", T, worst, be)
        record!("T-441 mean $label", T, meanr, be)
        d = get!(MEASURED, "T-441 max|D⁻·E − (ρ − mean ρ)| / max|ρ − mean ρ|", Dict{String,Float64}())
        d["$label $(type_label(T))$(be isa CPU ? "" : " on GPU")"] = effect
        @test worst <= C
        @test meanr <= C
        @test effect > 1e4 * base_eps(T)
    end
end

@testset "T-441 with p passes Gauss's law holds in its filtered form D⁻·E − Fρ = const to roundoff, and the plain residual (F − 1)(ρⁿ − ρ⁰) is measured, both modes [A2, A7, A8, WP-029]" begin
    body_T441(CPU())
end

# --- T-442 ----------------------------------------------------------------------------------

# The wiring: one step from the same initialized state with no passes and with two. The push and
# the deposition are the same, so the filtered state's Jx, Jy, Jz must be the host filter of the
# unfiltered ones (all three components, after the fold), and since the first B half step is the
# same, its E must differ by −Δt times the filtered minus the unfiltered current (the filtered
# current is the one the mesh solve takes). The guards of J hold their periodic images. 1D: the
# beams of T-361 (currents in all three components); 2D: the plasma of T-360. Reference is
# deterministic, so both states see the same unfiltered current (J matches the host filter bit
# for bit); in Fast mode the atomics make the two states differ by T-362's roundoff.
function body_T442(be)
    C = emC("T-442")
    for T in TYPES, mode in (Reference(), Fast()), N in (1, 2)
        prob = N == 1 ? beams_1d(T; amplitude = seeded(T, 1 / 20)) : plasma_step_2d(T; amplitude = seeded(T, 1 / 10))
        passes = 2
        s0 = EMState(prob, EMNumerics(; mode); backend = be)
        s1 = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes)); backend = be)
        initialize!(s0, prob)
        initialize!(s1, prob)
        step!(s0, params(prob), 1)
        step!(s1, params(prob), 1)
        grid = s0.grid
        int(a) = collect(interior_view(Array(a), grid))
        worstJ = worstE = 0.0
        for (j, e) in ((:Jx, :Ex), (:Jy, :Ey), (:Jz, :Ez))
            J0, J1 = int(getfield(s0.fields, j)), int(getfield(s1.fields, j))
            @test maximum(absmax, J0) > 0
            @test maximum(absmax, J1 .- J0) > 0                  # the filter changes every component
            worstJ = max(worstJ, maximum(absmax, J1 .- binomial_host(J0, passes)) / (base_eps(T) * maximum(absmax, J0) * passes * N))
            E0, E1 = int(getfield(s0.fields, e)), int(getfield(s1.fields, e))
            scale = maximum(absmax, E0) + Float64(value_of(prob.dt)) * maximum(absmax, J0)
            worstE = max(worstE, maximum(absmax, (E1 .- E0) .+ prob.dt .* (J1 .- J0)) / (base_eps(T) * scale))
            full = Array(getfield(s1.fields, j))
            ref = copy(full)
            fill_guards!(ref, grid)
            @test same_bits(ref, full)
        end
        # Fast: the two states deposit (ρ for the initial field, then J) with atomics in different
        # association orders at more than one thread and on CUDA, so they agree only as the two
        # modes do (A4 rule 3): T-362's scale √(prod ppc) and constant
        if mode isa Fast
            f = sqrt(prod(prob.species[1].ppc))
            worstJ, worstE = worstJ / f, worstE / f
        end
        CJ = mode isa Fast ? emC("T-362") : C
        record!("T-442 J $(N)D $(nameof(typeof(mode)))", T, worstJ, be)
        record!("T-442 E $(N)D $(nameof(typeof(mode)))", T, worstE, be)
        @test worstJ <= CJ
        @test worstE <= CJ
    end
end

@testset "T-442 the step filters the folded Jx, Jy, Jz and the mesh solve takes the filtered current, at N = 1 and 2, both modes [A2, A7, A8, WP-029]" begin
    body_T442(CPU())
end

# --- T-443 ----------------------------------------------------------------------------------

@testset "T-443 with the filter the Reference step is bitwise across reruns, launch shapes and 1 and 4 threads, and Fast agrees with it to roundoff [A4, A7, WP-029]" begin
    ref = filter_digests()
    @test ref == filter_digests()
    for wgs in (32, 256)
        @test filter_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        lines = filter(startswith("EMFILTER "), split(worker_output(nt), '\n'))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
    # A4 rule 3 with the filter: the two deposition modes agree, measured as T-362
    C = emC("T-362")
    for T in (Float32, Float64)
        prob = plasma_step_2d(T; nsteps = 20)
        sr = EMState(prob, EMNumerics(; mode = Reference(), filter = BinomialFilter(2)))
        sf = EMState(prob, EMNumerics(; mode = Fast(), filter = BinomialFilter(2)))
        initialize!(sr, prob)
        initialize!(sf, prob)
        worst = 0.0
        for k in 1:prob.nsteps
            step!(sr, params(prob), k)
            step!(sf, params(prob), k)
            for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz, :Jx, :Jy, :Jz)
                a, b = Array(getfield(sr.fields, c)), Array(getfield(sf.fields, c))
                worst = max(worst, maximum(abs, a .- b) / (eps(T) * maximum(abs, a) * k * sqrt(prod(prob.species[1].ppc))))
            end
        end
        record!("T-443 Reference against Fast", T, worst)
        @test worst <= C
    end
end

# --- T-444 ----------------------------------------------------------------------------------

@testset "T-444 the case files' filter_passes, and zero passes leave the step's bits alone [A2, A4, A10, WP-029]" begin
    # the loader takes the pass count from the case, and refuses anything but a non-negative integer
    for name in ("landau", "two-stream", "weibel", "landau-oblique-2d", "weibel-2d")
        case = load_case(name)
        @test numerics_from_case(case).filter == BinomialFilter(case["numerics"]["filter_passes"]) == BinomialFilter(0)
    end
    lo = load_case("landau-oblique-2d")
    for passes in (1, 4)
        lo["numerics"]["filter_passes"] = passes
        @test numerics_from_case(lo).filter.passes == passes
    end
    for bad in (-1, 1.5, 2.0, "2", true)
        lo["numerics"]["filter_passes"] = bad
        @test_throws ArgumentError numerics_from_case(lo)
    end
    delete!(lo["numerics"], "filter_passes")
    @test numerics_from_case(lo).filter.passes == 0
    @test_throws ArgumentError BinomialFilter(-1)
    @test EMNumerics().filter == BinomialFilter(0)
    # zero passes: filter_current! touches neither the fields (guards included) nor the scratch
    for T in TYPES
        grid = grid_T(FILTER_GRIDS[2], T)
        rng = Xoshiro(444)
        f = em_fields(CPU(), T, grid)
        foreach(a -> (a .= [rand_T(rng, T) for _ in CartesianIndices(a)]), f)
        s = field(CPU(), T, grid)
        s .= [rand_T(rng, T) for _ in CartesianIndices(s)]
        f0, s0 = map(copy, f), copy(s)
        filter_current!(BinomialFilter(0), f, grid, s)
        @test all(c -> same_bits(getfield(f, c), getfield(f0, c)), keys(f)) && same_bits(s, s0)
    end
    # and the zero-pass step leaves the folded current's guards at +0.0 and the scratch ρ at the
    # charge of initialize!, at N = 1 and 2 (a filter run would fill both), while with passes it fills them
    for (prob, N) in ((beams_1d(Float64), 1), (plasma_step_2d(Float64), 2)), passes in (0, 1)
        st = EMState(prob, EMNumerics(; filter = BinomialFilter(passes)))
        initialize!(st, prob)
        ρ0 = copy(st.ρ)
        run!(st, params(prob), 3)
        for c in (:Jx, :Jy, :Jz)
            b = Array(getfield(st.fields, c))
            @test maximum(abs, b) > 0
            interior_view(b, st.grid) .= 0
            @test all_plus_zero(b) == (passes == 0)
        end
        @test same_bits(st.ρ, ρ0) == (passes == 0)
    end
end

# --- T-445 ----------------------------------------------------------------------------------

function step_bytes_filter(N, mode; ppc, cells, passes = 2)
    prob = N == 1 ? beams_1d(Float64; n = cells[1], ppc = ppc[1], nsteps = 4) : plasma_step_2d(Float64; ppc, cells, nsteps = 4)
    st = EMState(prob, EMNumerics(; mode, filter = BinomialFilter(passes)))
    initialize!(st, prob)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-445 allocation per step with the filter does not grow with particle count, grid size or passes [A7, WP-029]" begin
    for N in (1, 2), mode in (Reference(), Fast())
        small = step_bytes_filter(N, mode; ppc = (2, 2), cells = (16, 12))
        more_particles = step_bytes_filter(N, mode; ppc = (8, 8), cells = (16, 12))
        larger_grid = step_bytes_filter(N, mode; ppc = (2, 2), cells = (64, 48))
        unfiltered = step_bytes_filter(N, mode; ppc = (2, 2), cells = (16, 12), passes = 0)
        more_passes = step_bytes_filter(N, mode; ppc = (2, 2), cells = (16, 12), passes = 8)
        @info "T-445 bytes per step with two filter passes" N mode small more_particles larger_grid unfiltered more_passes threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-367); CI runs one thread
        if Threads.nthreads() == 1 || mode isa Fast
            @test_alloc more_particles <= small
            @test_alloc larger_grid <= small
            @test_alloc more_passes <= small       # the passes reuse the scratch and launch with fixed arguments
        end
    end
end
