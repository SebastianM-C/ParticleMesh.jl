# WP-070: the Galerkin gather, T-991 to T-993 and T-996 to T-998 (T-990 is the engine's per-axis
# gather, in ParticleMeshCore's transfers.jl; T-994 and T-995 are the 1D cases, galerkin_physics.jl;
# T-999 reruns bodies of this file on the GPU, gpu.jl). Each field component is read one shape
# order lower along every axis on which it sits at `Center()`: E's weights are then those of the
# charge-conserving current for a vanishing move (the gather is the deposition's transpose, the
# energy condition), B's make the gathered B follow Faraday's law with the gathered E.
# Uses gather.jl's helpers (gather_state, mode_field!, gather_lattice_mean, particle_bits,
# snap_state) and ionization_cases.jl's problems.

using ParticleMeshCore: gather_at
using ParticleMeshElectromagnetic: GalerkinRead, component_shape, read_shape, field_layout, em_fields, em_guard_width,
                                   ionization_frame, ionize_kernel!, push_kernel!

"The module's layout at N with a bare stagger at N = 1, as `fields_at` reads it."
galerkin_layout(N) = N == 1 ? FIELD_LAYOUT : field_layout(Val(N))

"The engine's gather of `a` at cell `c` and offset `δ` (tuples of N) with `shape` (a `Shape` or a per-axis tuple), 1D on a bare stagger."
galerkin_read(a, c, δ, shape, st, g) = length(c) == 1 ? gather_at(a, c[1], δ[1], shape, st, g) : gather_at(a, c, δ, shape, st, g)

"An interior of random values in [−1/3, 2/3) and its periodic guards, from `rng`."
function random_field!(a, grid, rng, ::Type{T}) where {T}
    A = zeros(T, size(a))
    g, n = guards(grid), ncells(grid)
    for I in CartesianIndices(A)
        all(d -> g < I[d] <= g + n[d], 1:ndims(A)) && (A[I] = T(rand(rng)) - T(1) / 3)
    end
    copyto!(a, A)
    fill_guards!(a, grid)
    return a
end

"An offset in a cell away from the knots of the shapes (0 and 1/2) by more than `m`: in (m, 1/2 − m) or (1/2 + m, 1 − m)."
away_from_knots(rng, ::Type{T}, m) where {T} = T(rand(rng) < 1 / 2 ? m + (1 / 2 - 2m) * rand(rng) : 1 / 2 + m + (1 / 2 - 2m) * rand(rng))

"The zero quantity of a transverse current that T-991 does not read (an isbits callable, for the kernels)."
struct NoTransverse end
@inline (::NoTransverse)(s, i) = zero(eltype(s.w))

# --- T-991 ----------------------------------------------------------------------------------

# The energy condition. One particle moves along axis d from x − Δ to x + Δ (Δ = 1/32 cell, no
# shape knot crossed, the other axes fixed). The quadratic (linear) B-spline is a polynomial of
# degree 2 (1) between knots, so the difference of its weights at the two ends is exactly 2Δ times
# its derivative at x, and Esirkepov's current along d is q·w·2Δ·Δx_d/Δt times the derivative's
# prefix sum, the spline one order lower at the faces, times the unchanged weights of the other
# axes. So ⟨J_d, E⟩ = ΣJ_d · (E_d read at x with E_d's Galerkin weights), for any E, to roundoff:
# the Galerkin gather is the transpose of the current deposition. The error is measured against
# |q w| Δx_d/Δt · max|E| and not against ΣJ_d, which is proportional to the move: Esirkepov's
# weight differences cancel at the scale of the weights, so the error relative to ΣJ_d grows as
# 1/Δ for small moves (found by the physics breaker; 1.0e5 eps at Δ = 5·10⁻⁷ in Float64) while
# this one does not. Δ = 1/32 and 1/1024 of a cell, N = 1, 2 and 3, both shapes, Float32 and
# Float64, 20 random positions and fields per axis; the full-shape gather of today's step misses
# ⟨J_d, E⟩/ΣJ_d by more than 2 % of the field's size.
function body_T991()
    C = emC("T-991")
    worst = 0.0
    for T in (Float32, Float64), N in 1:3, P in 1:2
        rng = Xoshiro(991 + 10N + P)
        shape = Shape{P}()
        n = ntuple(_ -> 8, N)
        grid = PeriodicGrid(n, ntuple(d -> T(1) / 4 + T(d) / 8, N), em_guard_width(shape))
        L = galerkin_layout(N)
        stρ = N == 1 ? Node() : ntuple(_ -> Node(), N)
        g = Int32(guards(grid))
        miss_full = 0.0
        for Δ in (T(1) / 32, T(1) / 1024), d in 1:N, trial in 1:20
            s = Species(CPU(), T, 1; q = -1, m = 1, N, extras = flux_extras(T, N))
            c = ntuple(_ -> Int32(rand(rng, 2:7)), N)
            x = ntuple(_ -> away_from_knots(rng, T, 1 / 16), N)
            s.w[1] = T(7) / 10
            s.status[1] = ParticleMeshCore.ALIVE
            fill!(s.count, Int32(1))
            for k in 1:N
                s.cell[k][1] = c[k]
                s.offset[k][1] = k == d ? x[k] + Δ : x[k]
            end
            old = ntuple(k -> k == d ? x[k] - Δ : x[k], N)
            if N == 1
                s.extras.old_cell[1], s.extras.old_offset[1] = c[1], old[1]
            else
                for k in 1:N
                    getproperty(s.extras, Symbol("old_cell_$k"))[1] = c[k]
                    getproperty(s.extras, Symbol("old_offset_$k"))[1] = old[k]
                end
            end
            ws = N == 1 ? Workspace(s, grid) : Workspace(s, grid; flux = (shape, stρ))
            J = ntuple(_ -> field(CPU(), T, grid), 3)
            kw = N == 1 ? (; quantity_y = NoTransverse(), quantity_z = NoTransverse()) : N == 2 ? (; quantity_z = NoTransverse()) : (;)
            deposit_flux!(Fast(), J..., s, shape, stρ, grid, ws; dt = T(1) / 4, kw...)
            E = random_field!(field(CPU(), T, grid), grid, rng, T)
            Jd = interior_view(fold_guards!(J[d], grid), grid)
            Ei = interior_view(E, grid)
            work = sum(Jd .* Ei)
            st = L[(:Ex, :Ey, :Ez)[d]]
            gal = galerkin_read(E, c, x, component_shape(GalerkinRead{P}(), st), st, g)
            full = galerkin_read(E, c, x, shape, st, g)
            scale = abs(s.w[1]) * grid.dx[d] / (T(1) / 4) * maximum(abs, Ei)     # |q w| Δx_d/Δt · max|E|, q = -1
            err = abs(work - sum(Jd) * gal) / (eps(T) * scale)
            worst = max(worst, err)
            @test err <= C
            Δ == T(1) / 32 && (miss_full = max(miss_full, abs(work / sum(Jd) - full)))
            @test all(k -> k == d || all(iszero, J[k]), 1:N)      # a move along d puts no current on the other resolved axes
        end
        @test miss_full > 1 / 50        # today's full-shape gather is not the transpose
    end
    record!("T-991", Float64, worst)      # the largest over Float32 and Float64
    @info "T-991 transpose err/eps" worst
end

@testset "T-991 the Galerkin gather of E is the transpose of the charge-conserving current: ⟨J_d, E⟩ for a symmetric move along d is ΣJ_d times E_d read with the Galerkin weights at the move's centre, to roundoff of the current's scale, moves of 1/32 and 1/1024 cell, N = 1 to 3, both shapes [A2, A7, WP-070]" begin
    body_T991()
end

# --- T-992 ----------------------------------------------------------------------------------

# B's weights. With E random and B zero, the module's half-step B ← B − h∇×E changes the gathered
# B at a point by exactly −h times the curl of the gathered E at that point, where the curl is
# taken of the gathered E as a function of the position (ForwardDiff in the offsets, over the
# cell sizes), for the Galerkin weights: the derivative of the order-p spline along an axis is the
# difference of order-(p − 1) splines, so the gathered Galerkin fields satisfy Faraday's law
# exactly, and their B is divergence free where the grid's is. N = 1, 2 and 3, both shapes,
# Float32 and Float64, 30 points away from the knots; with today's full-shape gather the change
# misses the curl by more than 10 % of its size.
function body_T992()
    C = emC("T-992")
    worst = 0.0
    for T in (Float32, Float64), N in 1:3, P in 1:2, galerkin in (true, false)
        rng = Xoshiro(992 + 10N + P)
        shape = Shape{P}()
        n = ntuple(_ -> 8, N)
        h = ntuple(d -> T(1) / 4 + T(d) / 8, N)
        grid = PeriodicGrid(n, h, em_guard_width(shape))
        f = em_fields(CPU(), T, grid)
        foreach(k -> random_field!(f[k], grid, rng, T), (:Ex, :Ey, :Ez))
        L = galerkin_layout(N)
        rd = galerkin ? GalerkinRead{P}() : shape
        g = Int32(guards(grid))
        rd_at(k, c, δ) = galerkin_read(f[k], c, δ, component_shape(rd, L[k]), L[k], g)
        pts = [(ntuple(_ -> Int32(rand(rng, 2:7)), N), ntuple(_ -> away_from_knots(rng, T, 1 / 20), N)) for _ in 1:30]
        curl = map(pts) do (c, δ)
            D = [j <= N ? ForwardDiff.derivative(t -> rd_at(k, c, ntuple(q -> δ[q] + (q == j ? t : zero(t)), N)), zero(T)) / h[j] : zero(T)
                 for k in (:Ex, :Ey, :Ez), j in 1:3]
            (D[3, 2] - D[2, 3], D[1, 3] - D[3, 1], D[2, 1] - D[1, 2])
        end
        B0 = [map(k -> rd_at(k, c, δ), (:Bx, :By, :Bz)) for (c, δ) in pts]
        hb = T(1) / 10
        yee_b_half!(f, grid, hb)
        B1 = [map(k -> rd_at(k, c, δ), (:Bx, :By, :Bz)) for (c, δ) in pts]
        scale = maximum(i -> maximum(abs, curl[i]), eachindex(pts))
        miss = maximum(i -> maximum(abs.((B1[i] .- B0[i]) ./ hb .+ curl[i])), eachindex(pts)) / scale
        if galerkin
            worst = max(worst, miss / eps(T))
            @test miss <= C * eps(T)
        else
            @test miss > 1 / 10
        end
    end
    record!("T-992", Float64, worst)      # the largest over Float32 and Float64
    @info "T-992 Faraday err/(eps·scale)" worst
end

@testset "T-992 the Galerkin gather of B follows Faraday's law with the gathered E: the Yee half step changes the gathered B by −h times the curl of the gathered E as a function of position, to roundoff, N = 1 to 3, both shapes [A2, A7, WP-070]" begin
    body_T992()
end

# --- T-993 ----------------------------------------------------------------------------------

# The option. The shapes per component are the rule's (one order lower along each `Center()`
# axis, the linear shape's lower order the nearest grid point) for N = 1, 2 and 3; a state with
# `gather = nothing` or a `CompensatedGather` passes its shape itself to the kernels; the
# construction refuses what is not a gather, a 3D state takes the Galerkin gather, plain and with a
# compensation (WP-076). What reads the Galerkin weights, each bit for bit as the same
# kernel launched by hand with `GalerkinRead` and not as with the plain shape: the push (a 2D
# Fast-mode step of a plasma in random fields, which moves no particle between slots), the
# ionization's rates (T-962's run from its step-5 snapshot, a step without births), and
# `initialize!`'s half-step kick. And the module's gather `fields_at`, which they all call, reads
# every component with the table's shapes bit for bit at N = 1, 2 and 3 (the mutation breaker
# found that a wrong rule inside `fields_at` was otherwise caught by T-996 alone).
"""
`gather_state`'s 2D problem (24 × 6 cells, one cold species) in Fast mode with `gather`, and its
parameters: a step that moves no particle between slots for Float32 and Float64.
"""
function galerkin_fast_state(::Type{T}, gather; be = CPU()) where {T}
    h = 2 * T(π) / 12
    sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), (1, 0), (2, 2), zero3(T), zero3(T), 1),)
    prob = EMProblem("gather-2d", (24h, 6 * T(7) / 10), (24, 6), 2h / 5, 4, sp, 1, (1, 0))
    st = EMState(prob, EMNumerics(; mode = Fast(), gather); backend = be)
    initialize!(st, prob)
    return st, params(prob)
end

function body_T993(be)
    for T in (Float32, Float64)
        # the push: one step of the state against the push kernel launched by hand on a copy of its particles
        for (gather, shape) in ((GalerkinGather(), GalerkinRead{2}()), (nothing, Quadratic))
            st, p = galerkin_fast_state(T, gather; be)
            rng = Xoshiro(993)
            foreach(k -> random_field!(st.fields[k], st.grid, rng, T), GATHER_NAMES)
            s = st.species[1]
            foreach(a -> fill!(a, T(1) / 5), s.u)
            hand = deepcopy(s)
            n, g = Int32.(st.grid.n), Int32(guards(st.grid))
            ParticleMeshCore.launcher(push_kernel!, get_backend(hand), nothing)(hand, st.fields, st.numerics.pusher, shape, g,
                                                                                hand.q / hand.m * st.dt / 2, map(h -> st.dt / h, st.grid.dx), n;
                                                                                ndrange = capacity(hand))
            other = deepcopy(s)
            ParticleMeshCore.launcher(push_kernel!, get_backend(other), nothing)(other, st.fields, st.numerics.pusher,
                                                                                 shape isa Shape ? GalerkinRead{2}() : Quadratic, g,
                                                                                 other.q / other.m * st.dt / 2, map(h -> st.dt / h, st.grid.dx), n;
                                                                                 ndrange = capacity(other))
            step!(st, p, 1)
            @test map(bits, s.u) == map(bits, hand.u) && map(bits, s.offset) == map(bits, hand.offset)
            @test map(bits, s.u) != map(bits, other.u)
        end
    end
    # the ionization's rates and the initial kick, T-962's run
    for T in TYPES
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1), every = 2, nsteps = 8, plasma = true, rows = (1.1, 2.9))
        p = params(prob)
        gg = GalerkinGather()
        st = ion_state(prob; be, gather = gg)
        run!(st, p, 4)
        snap = snapshot(st, 4)
        step!(st, p, 5)                            # odd: no birth, the electrons keep their slots
        pl = snap_state(prob, snap, be)
        step!(pl, p, 5)
        atoms(x) = (fr = ionization_frame(x); (bits(fr.atoms.status), map(bits, values(fr.atoms.extras))...))
        for (state, shape) in ((st, GalerkinRead{2}()), (pl, Quadratic))
            hand = snap_state(prob, snap, be)
            fr = ionization_frame(hand)
            e = fr.electrons
            ParticleMeshCore.launcher(ionize_kernel!, get_backend(fr.atoms), nothing)(fr.atoms, e, fr.flags, e.count, hand.fields, p.ionization.levels,
                                                                                     hand.numerics.pusher, shape, Int32(guards(hand.grid)), hand.dt,
                                                                                     -e.q / e.m * hand.dt / 4, false, Int32(capacity(e));
                                                                                     ndrange = capacity(fr.atoms))
            @test atoms(state) == atoms(hand)
        end
        @test atoms(st) != atoms(pl)
        # the initial kick
        ic = EMState(prob, EMNumerics(; gather = gg); backend = be)
        initialize!(ic, prob)
        ip = EMState(prob, EMNumerics(); backend = be)
        initialize!(ip, prob; momenta_at = :t_minus_half)
        for s in ip.species
            ParticleMeshCore.launcher(kick_kernel!, get_backend(s), nothing)(s, ip.fields, ip.numerics.pusher, GalerkinRead{2}(), Int32(guards(ip.grid)),
                                                                           -s.q / s.m * ip.dt / 4; ndrange = capacity(s))
        end
        @test map(bits, ic.species[1].u) == map(bits, ip.species[1].u)
        i0 = EMState(prob, EMNumerics(); backend = be)
        initialize!(i0, prob)
        @test map(bits, ic.species[1].u) != map(bits, i0.species[1].u)
    end
end

@testset "T-993 the Galerkin gather as an option: each component's shapes by the rule, the plain and compensated states' shape untouched, the refusals, and the push, the ionization's rates and the initial kick read the Galerkin weights bit for bit [A2, A8, A10, WP-070]" begin
    lo(P) = Shape{P - 1}()
    hi(P) = Shape{P}()
    for P in 1:2
        r = GalerkinRead{P}()
        want1 = (Ex = lo(P), Ey = hi(P), Ez = hi(P), Bx = hi(P), By = lo(P), Bz = lo(P))
        want2 = (Ex = (lo(P), hi(P)), Ey = (hi(P), lo(P)), Ez = (hi(P), hi(P)), Bx = (hi(P), lo(P)), By = (lo(P), hi(P)), Bz = (lo(P), lo(P)))
        want3 = (Ex = (lo(P), hi(P), hi(P)), Ey = (hi(P), lo(P), hi(P)), Ez = (hi(P), hi(P), lo(P)),
                 Bx = (hi(P), lo(P), lo(P)), By = (lo(P), hi(P), lo(P)), Bz = (lo(P), lo(P), hi(P)))
        for (N, want) in ((1, want1), (2, want2), (3, want3)), k in GATHER_NAMES
            @test component_shape(r, galerkin_layout(N)[k]) === want[k]
        end
        # the module's gather (`fields_at`, what the push, the kick and the ionization call) reads
        # each component with the shapes of the table, bit for bit, random fields and positions
        for (N, want) in ((1, want1), (2, want2), (3, want3)), T in (Float32, Float64)
            rng = Xoshiro(993 + 10N + P)
            grid = PeriodicGrid(ntuple(_ -> 8, N), ntuple(d -> T(1) / 4 + T(d) / 8, N), em_guard_width(Shape{P}()))
            g = Int32(guards(grid))
            f = NamedTuple{GATHER_NAMES}(ntuple(_ -> random_field!(field(CPU(), T, grid), grid, rng, T), 6))
            L = galerkin_layout(N)
            same = true
            for trial in 1:20
                c = ntuple(_ -> Int32(rand(rng, 1:8)), N)
                δ = ntuple(_ -> T(rand(rng)), N)
                E, B = ParticleMeshElectromagnetic.fields_at(f, c, δ, r, g)
                got = (E..., B...)
                for (j, k) in enumerate(GATHER_NAMES)
                    same &= got[j] === galerkin_read(f[k], c, δ, want[k], L[k], g)
                end
            end
            @test same
        end
        @test read_shape(GalerkinGather(), Shape{P}()) === r
        @test read_shape(nothing, Shape{P}()) === Shape{P}() && read_shape(CompensatedGather(), Shape{P}()) === Shape{P}()
        @test component_shape(Shape{P}(), galerkin_layout(2).Ex) === Shape{P}()
    end
    @test EMNumerics(; gather = GalerkinGather()).gather === GalerkinGather()
    @test GalerkinGather(; compensate = CompensatedGather(; k = 2)).compensate === CompensatedGather(; k = 2)
    @test_throws ArgumentError GalerkinGather(; compensate = 1)
    @test_throws ArgumentError GalerkinGather(; compensate = GalerkinGather())
    @test_throws ArgumentError gather_state(Float64, 1, Quadratic, GalerkinGather(; compensate = CompensatedGather(; k = 13)))   # unresolved, as T-960
    # a 3D state takes the Galerkin gather, plain and with a compensation (WP-076; T-996 at N = 3)
    p3 = EMProblem("g3", (3.0, 3.0, 3.0), (6, 6, 6), 0.1, 1, (), 1, (1, 0, 0))
    @test EMState(p3, EMNumerics(; gather = GalerkinGather())).numerics.gather === GalerkinGather()
    @test EMState(p3, EMNumerics(; gather = GalerkinGather(; compensate = CompensatedGather()))).sharpened isa NamedTuple
    @test EMState(p3, EMNumerics(; gather = CompensatedGather())).sharpened isa NamedTuple
    body_T993(CPU())
end

# --- T-996 ----------------------------------------------------------------------------------

# The two gathers together. With `GalerkinGather(; compensate = CompensatedGather())` each
# component's copy is sharpened with the β of the order it is read with along axis 1 (the lower
# order where it sits at `Center()` along x), so that through the module's gather, at 8 offsets
# per cell along x of every cell, the projection of each of the six components of a sampled mode
# of wavenumber k is that stencil's transfer times the mean of that order over the offsets
# (`gather_lattice_mean`, the nearest grid point's sin x/(M sin(x/M)) for order 0), to roundoff:
# one at k = 1 for every component and both shapes but for the offsets' aliases. Without the
# compensation the particles read the grid's fields themselves. All three types, N = 1, 2 and 3
# (WP-076: the copy is sharpened along x on every transverse line).
function body_T996(be)
    C = emC("T-960")
    worst = 0.0
    for T in TYPES, N in (1, 2, 3), shape in (Linear, Quadratic), k in (1, 2), comp in (true, false)
        cg = CompensatedGather()
        gg = GalerkinGather(; compensate = comp ? cg : nothing)
        st = gather_state(T, N, shape, gg; be)
        grid = st.grid
        h = Float64(value_of(cellsize(grid)[1]))
        rd = GalerkinRead{order(shape)}()
        L = galerkin_layout(N)
        for (j, name) in enumerate(GATHER_NAMES)
            mode_field!(st.fields[name], grid, k, j / 3, T)
        end
        sharpen_gather!(st)
        pf = particle_fields(st)
        comp || @test pf === st.fields && st.sharpened === nothing
        x = k * h / 2
        M = 8
        hf = map(Array, pf)
        g = Int32(guards(grid))
        n1 = ncells(grid)[1]
        for (j, name) in enumerate(GATHER_NAMES)
            s1 = N == 1 ? component_shape(rd, L[name]) : component_shape(rd, L[name])[1]
            Ts = comp ? 1 + 4 * big(sharpening_beta(cg, s1, h)) * sin(big(x))^2 : big(1)
            mean = Ts * gather_lattice_mean(order(s1), x, M)
            nc = length(components(zero(T)))
            Cs, Ss = zeros(BigFloat, nc), zeros(BigFloat, nc)
            for c in 1:n1, m in 0:(M - 1)
                δ = T(m) / M
                E, B = N == 1 ? fields_at(hf, (Int32(c),), (δ,), rd, g) :
                       N == 2 ? fields_at(hf, (Int32(c), Int32(3)), (δ, T(37) / 100), rd, g) :
                                fields_at(hf, (Int32(c), Int32(3), Int32(4)), (δ, T(37) / 100, T(61) / 100), rd, g)
                v = (E..., B...)[j]
                ξ = k * (c - 1 + m / M) * big(h)
                Cs .+= BigFloat.(components(v)) .* cos(ξ)
                Ss .+= BigFloat.(components(v)) .* sin(ξ)
            end
            amp = sqrt.(Cs .^ 2 .+ Ss .^ 2) .* 2 ./ (n1 * M)
            ref = T <: Dual ? (1, 1, big(1) / 2, 1) : (1,)
            err = maximum(abs.(amp .- mean .* collect(ref)))
            worst = max(worst, Float64(err / base_eps(T)))
            @test err <= C * base_eps(T)
            comp && k == 1 && @test abs(mean - 1) < 2.0e-3
        end
    end
    record!("T-996", Float64, worst, be)
    @info "T-996 err/eps" worst
end

@testset "T-996 the Galerkin and the compensated gather together: each component's copy sharpened for the order it is read with along x, its mean transfer one at the laser's wavenumber, all three types [A2, A7, A8, WP-070]" begin
    @test gather_lattice_mean(0, big(0.3), 8) ≈ sin(big(0.3)) / (8 * sin(big(0.3) / 8)) rtol = 1e-30
    body_T996(CPU())
end

# --- T-997 ----------------------------------------------------------------------------------

# The current and the mesh solve are untouched: T-961's run (a thermal plasma crossing both seams
# in Ex = 1/2, a dopant of two levels ionized every 2 steps) with the Galerkin gather, alone and
# with the compensation, all three types, both deposition modes, holds Gauss's law with the ions'
# charge to T-773's bound after every step; the momenta and the born electrons differ from the
# plain run's; in Reference mode the run is bitwise across launch shapes.
function body_T997(be)
    C = emC("T-773")
    for T in TYPES, mode in (Reference(), Fast()), comp in (false, true)
        gg = GalerkinGather(; compensate = comp ? CompensatedGather() : nothing)
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1), every = 2, nsteps = 30, plasma = true, rows = (1.1, 2.9))
        st = ion_state(prob; mode, be, gather = gg)
        p = params(prob)
        r = ion_gauss(st, p, 0)
        worst = maximum(r.res) / (base_eps(T) * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        @info "T-997" T mode comp worst born = born(st).n
        record!("T-997 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test ionization_lost(st) == 0 && born(st).n == 15 * 128
        plain = ion_state(prob; mode, be)
        run!(plain, p, prob.nsteps)
        @test Array(st.species[1].u[1]) != Array(plain.species[1].u[1])
        @test born(st).u[1] != born(plain).u[1]
        if mode isa Reference
            again = ion_state(prob; mode, be, gather = gg, workgroupsize = 64)
            run!(again, p, prob.nsteps)
            @test ion_digest(again) == ion_digest(st)
        end
    end
end

@testset "T-997 with the Galerkin gather, alone and compensated, Gauss's law holds to roundoff with the ions' charge after every step of an ionizing plasma, all three types, both modes, bitwise across launch shapes [A2, A4, A7, A8, WP-070]" begin
    body_T997(CPU())
end

# --- T-998 ----------------------------------------------------------------------------------

# The gradient through the Galerkin gather (A7): T-963's small ionization-injection run with the
# Galerkin gather, ForwardDiff of the born charge and the born electrons' energy in the laser's a₀
# and the dopant's density against the finite-difference floor with T-778's bounds. Along the
# staggered axes the lower order makes the gathered field a less smooth function of the position
# (continuous with kinks for the quadratic shape), which this run's gradient does not notice.
@testset "T-998 ForwardDiff through the Galerkin gather against the finite-difference floor in a small ionization-injection run [A0, A3, A7, WP-070]" begin
    x0 = collect(ION_X0)
    gg = GalerkinGather()
    J0 = ion_objectives(ION_X0; gather = gg)
    Jp = ion_objectives(ION_X0)
    @info "T-998 objectives" galerkin = J0 plain = Jp ratio = J0 ./ Jp
    MEASURED["T-998 born charge, galerkin/plain"] = Dict("Float64" => J0[1] / Jp[1])
    @test all(isfinite, J0) && all(>(0), J0)
    ad = ForwardDiff.jacobian(x -> ion_objectives(x; gather = gg), x0)
    for (o, name) in enumerate(("charge", "energy")), i in 1:2
        f(xi, mode, seed) = ion_objectives(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode; gather = gg)[o]
        hs = x0[i] .* 2.0 .^ (-4:-2:-20)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[o, i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-998 $name, $(ION_NAMES[i])" ad = ad[o, i] fd = r.fd[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-998 rel/floor, $name", Dict{String,Float64}())[ION_NAMES[i]] = ratio
        @test isfinite(ad[o, i]) && ad[o, i] != 0
        @test r.rel[r.istar] <= (r.floor_rel > 0 ? emC("T-778") * r.floor_rel : 1.0e-10)
        @test r.floor_rel <= emC("T-778", "floor_max")
    end
end
