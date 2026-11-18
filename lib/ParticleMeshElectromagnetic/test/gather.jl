# WP-067: the shape-compensated gather, T-960 to T-963 (T-969 is their rerun on the GPU, gpu.jl).
# The particles read a copy of E and B sharpened along axis 1 by E + β(2E_i − E_{i−1} − E_{i+1}),
# whose transfer 1 + 4β sin²(kΔx/2) undoes the shape's mean transfer sinc(kΔx/2)^(p+1) at the
# laser's wavenumber; the mesh solve, the current and the diagnostics keep the grid's fields.
# The problems with a dopant are ionization_cases.jl's.

using ParticleMeshElectromagnetic: fields_at, kick_kernel!, FIELD_LAYOUT

const GATHER_NAMES = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)

"""
The mean of a B-spline gather of order `p` over `M` evenly spaced offsets per cell (M even), of a
sampled mode of wavenumber `k` (x = kΔx/2): the shape's transfer at k and at its aliases
k + 2πlM/Δx, which those offsets cannot tell from k, Σ_l (sin x/(x + πlM))^(p+1), in closed form
with Σ_l (z + πl)^−2 = csc²z and Σ_l (z + πl)^−3 = csc²z·cot z (z = x/M). In BigFloat.
"""
function gather_lattice_mean(p, x, M)
    x, z = big(x), big(x) / M
    p == 1 && return sin(x)^2 / M^2 / sin(z)^2
    p == 2 && return sin(x)^3 / M^3 * cos(z) / sin(z)^3
    error("order $p")
end

"""
A state of `N` = 1 or 2 resolved axes, 12 cells per laser wavelength along x over two wavelengths
(24 cells; at N = 2 six rows of 0.7), a cold species that the test does not push, the `gather`.
"""
function gather_state(::Type{T}, N, shape, gather; be = CPU()) where {T}
    h = 2 * T(π) / 12
    if N == 1
        sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), 1, 4, zero3(T), zero3(T), 1),)
        prob = EMProblem("gather-1d", 24h, 24, 2h / 5, 4, sp, 1, 1)
    else
        sp = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), (1, 0), (2, 2), zero3(T), zero3(T), 1),)
        prob = EMProblem("gather-2d", (24h, 6 * T(7) / 10), (24, 6), 2h / 5, 4, sp, 1, (1, 0))
    end
    return EMState(prob, EMNumerics(; shape, gather); backend = be)
end

"The mode cos(k·x_i + φ) along axis 1 of a field of `grid`, sampled at the interior array index i (x_i = (i − 1)Δx), each partial of a dual its own phase; guards the periodic images."
function mode_field!(a, grid, k, φ, ::Type{T}) where {T}
    g, n, h = guards(grid), ncells(grid), Float64(value_of(cellsize(grid)[1]))
    A = zeros(T, size(a))
    for I in CartesianIndices(A)
        i = I[1]
        g < i <= g + n[1] || continue
        all(d -> g < I[d] <= g + n[d], 2:ndims(A)) || continue
        θ = k * (i - g - 1) * h + φ
        A[I] = T <: Dual ? mkdual(cos(θ), sin(θ), cos(θ + 1) / 2, -sin(θ + 2)) : T(cos(θ))
    end
    copyto!(a, A)
    fill_guards!(a, grid)
    return a
end

# --- T-960 ----------------------------------------------------------------------------------

# The transfer, from the definitions. β makes sinc(x)^(p+1)·(1 + 4β sin²x) one at x = kΔx/2 to
# roundoff in every type, tends to (p + 1)/24 for small cells, and a wavenumber the grid does not
# resolve, k ≤ 0, or a gather that is not one is refused. On a state of N = 1 and N = 2 with each
# of E and B a sampled mode of wavenumber 1 or 2 (its own phase, each partial of a dual its own),
# `sharpen_gather!`'s copy is the mode times 1 + 4β sin²(kΔx/2) at every entry, guards included,
# and its guards are the periodic images of its interior bit for bit; a constant field is copied
# bit for bit. Through the module's gather (`fields_at` on `particle_fields`) at 8 offsets per cell
# along x of every cell (and a fixed row offset at N = 2), the projection of each of the six
# components onto the mode is the stencil's transfer times the shape's mean over those offsets
# (`gather_lattice_mean`), to roundoff: one at k = 1 for both orders but for the aliases of the
# offsets' lattice, and `gather_transfer` at k = 2. Without compensation the particles read the
# grid's fields themselves (`particle_fields` is `fields`), and the same projection is the
# shape's alone.
function body_T960(be)
    C = emC("T-960")
    worst_s, worst_g = 0.0, 0.0
    for T in TYPES, N in (1, 2), shape in (Linear, Quadratic), k in (1, 2), comp in (true, false)
        cg = comp ? CompensatedGather() : nothing
        st = gather_state(T, N, shape, cg; be)
        grid = st.grid
        h = Float64(value_of(cellsize(grid)[1]))
        p = order(shape)
        for (j, name) in enumerate(GATHER_NAMES)
            mode_field!(st.fields[name], grid, k, j / 3, T)
        end
        sharpen_gather!(st)
        pf = particle_fields(st)
        if !comp
            @test pf === st.fields && st.sharpened === nothing
        end
        x = k * h / 2
        Ts = comp ? 1 + 4 * big(sharpening_beta(cg, shape, h)) * sin(big(x))^2 : big(1)
        for name in GATHER_NAMES
            a, b = Array(st.fields[name]), Array(pf[name])
            err = maximum(I -> maximum(abs.(BigFloat.(components(b[I])) .- Ts .* BigFloat.(components(a[I])))), CartesianIndices(a))
            worst_s = max(worst_s, Float64(err / base_eps(T)))
            @test err <= C * base_eps(T)
            img = copy(pf[name])
            fill_guards!(img, grid)
            @test bits(img) == bits(pf[name])
        end
        # the gather: 8 offsets per cell along x, every cell of a row
        M = 8
        hf = map(Array, pf)
        g = Int32(guards(grid))
        n1 = ncells(grid)[1]
        mean = Ts * gather_lattice_mean(p, x, M)
        for (j, name) in enumerate(GATHER_NAMES)
            nc = length(components(zero(T)))
            Cs, Ss = zeros(BigFloat, nc), zeros(BigFloat, nc)
            for c in 1:n1, m in 0:(M - 1)
                δ = T(m) / M
                E, B = N == 1 ? fields_at(hf, (Int32(c),), (δ,), shape, g) : fields_at(hf, (Int32(c), Int32(3)), (δ, T(37) / 100), shape, g)
                v = (E..., B...)[j]
                ξ = k * (c - 1 + m / M) * big(h)
                Cs .+= BigFloat.(components(v)) .* cos(ξ)
                Ss .+= BigFloat.(components(v)) .* sin(ξ)
            end
            amp = sqrt.(Cs .^ 2 .+ Ss .^ 2) .* 2 ./ (n1 * M)
            # each component of a dual is a mode of amplitude 1 (value), 1, 1/2 and 1
            ref = T <: Dual ? (1, 1, big(1) / 2, 1) : (1,)
            err = maximum(abs.(amp .- mean .* collect(ref)))
            worst_g = max(worst_g, Float64(err / base_eps(T)))
            @test err <= C * base_eps(T)
        end
        comp && k == 1 && @test abs(mean - 1) < 2.0e-3            # one, but for the 8 offsets' aliases
        comp && @test Float64(Ts) * Float64(gather_lattice_mean(p, x, 4096)) ≈ gather_transfer(cg, shape, h, k) rtol = 1.0e-6
        # a constant is copied bit for bit
        for name in GATHER_NAMES
            fill!(st.fields[name], one(T) / 3)
        end
        sharpen_gather!(st)
        @test all(name -> bits(particle_fields(st)[name]) == bits(st.fields[name]), GATHER_NAMES)
    end
    record!("T-960 stencil", Float64, worst_s, be)
    record!("T-960 gather", Float64, worst_g, be)
    @info "T-960 err/eps" stencil = worst_s gather = worst_g
end

@testset "T-960 the compensated gather's transfer: β from its closed form, the copy of a sampled mode times 1 + 4β sin²(kΔx/2) with its guards the interior's images, the particles' gather of every component one at the laser's wavenumber [A2, A7, A8, WP-067]" begin
    for T in (Float32, Float64), shape in (Linear, Quadratic), ppl in (8, 12, 16, 32, 128)
        h = 2 * T(π) / ppl
        c = CompensatedGather()
        β = sharpening_beta(c, shape, h)
        x = h / 2
        @test β isa T
        @test (sin(x) / x)^(order(shape) + 1) * (1 + 4β * sin(x)^2) ≈ 1 atol = 4 * eps(T)
        @test gather_transfer(c, shape, h, one(T)) ≈ 1 atol = 4 * eps(T)
        @test gather_transfer(nothing, shape, h, one(T)) ≈ (sin(x) / x)^(order(shape) + 1) rtol = 2 * eps(T)
        @test abs(β - (order(shape) + 1) / 24) <= x^2 / 10              # the small-cell limit, O(x²)
        @test -4 * eps(T) < gather_transfer(c, shape, h, T(1) / 2) - 1 <= x^4     # below k: 1 + O((kΔx)⁴), a slight overshoot
    end
    # β to the precision of its type down to the finest cells (no cancellation), against BigFloat
    for T in (Float32, Float64), shape in (Linear, Quadratic), ppl in (4, 6, 64, 256, 1024, 4096, 2^16)
        h = 2 * T(π) / ppl
        x = big(h) / 2
        βbig = (inv((sin(x) / x)^(order(shape) + 1)) - 1) / (4 * sin(x)^2)
        @test abs(sharpening_beta(CompensatedGather(), shape, h) / βbig - 1) <= 16 * eps(T)
    end
    @test sharpening_beta(CompensatedGather(), Quadratic, mkdual(0.5, 1, 0, 0)) isa D3
    @test CompensatedGather(; k = 2).k === 2.0
    @test_throws ArgumentError CompensatedGather(; k = 0)
    @test_throws ArgumentError CompensatedGather(; k = -1)
    @test_throws ArgumentError CompensatedGather(; k = NaN)
    @test_throws ArgumentError sharpening_beta(CompensatedGather(; k = 7), Quadratic, 0.5)      # kΔx = 3.5 > π
    @test_throws ArgumentError EMNumerics(; gather = 1)
    @test_throws ArgumentError gather_state(Float64, 1, Quadratic, CompensatedGather(; k = 13))  # 12 cells per wavelength 2π/k₀
    @test_throws ArgumentError gather_state(Float64, 2, Quadratic, CompensatedGather(; k = 13))
    @test EMNumerics().gather === nothing && EMNumerics(; gather = CompensatedGather()).gather === CompensatedGather()
    body_T960(CPU())
end

# --- T-961 ----------------------------------------------------------------------------------

# The current and the mesh solve keep the grid's fields, so Gauss's law is untouched: T-773's run
# (a thermal plasma that crosses both seams, in Ex = 1/2, with a dopant of two levels ionized every
# 2 steps, at Δx = 1/2 so that the laser's wavenumber is resolved) with the compensated gather, all
# three types, both deposition modes, holds D⁻·E − ρ (the ions' charge included) at roundoff after
# every step with T-773's bound. The option is active: the momenta and the born electrons differ
# from the plain run's. In Reference mode the run is bitwise across launch shapes.
function body_T961(be)
    C = emC("T-773")
    for T in TYPES, mode in (Reference(), Fast())
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1), every = 2, nsteps = 30, plasma = true, rows = (1.1, 2.9))
        st = ion_state(prob; mode, be, gather = CompensatedGather())
        p = params(prob)
        r = ion_gauss(st, p, 0)
        worst = maximum(r.res) / (base_eps(T) * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        @info "T-961" T mode worst born = born(st).n
        record!("T-961 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test ionization_lost(st) == 0 && born(st).n == 15 * 128
        plain = ion_state(prob; mode, be)
        run!(plain, p, prob.nsteps)
        @test Array(st.species[1].u[1]) != Array(plain.species[1].u[1])
        @test born(st).u[1] != born(plain).u[1]
        if mode isa Reference
            again = ion_state(prob; mode, be, gather = CompensatedGather(), workgroupsize = 64)
            run!(again, p, prob.nsteps)
            @test ion_digest(again) == ion_digest(st)
        end
    end
end

@testset "T-961 with the compensated gather Gauss's law holds to roundoff with the ions' charge after every step of a plasma with a dopant being ionized, all three types, both modes, bitwise across launch shapes [A2, A4, A7, A8, WP-067]" begin
    body_T961(CPU())
end

# --- T-962 ----------------------------------------------------------------------------------

"The bits of every particle array of a state: its species and, with a dopant, the atoms (their populations included) and the born electrons."
function particle_bits(st)
    sb(s) = (map(bits, s.cell)..., map(bits, s.offset)..., map(bits, s.u)..., bits(s.w), bits(s.status), bits(s.count),
             map(bits, values(s.extras))...)
    fr = ionization_frame(st)
    return (map(sb, st.species)..., (fr === nothing ? () : (sb(fr.atoms), sb(fr.electrons), bits(fr.lost)))...)
end

# What reads the copy. The compensated step is the plain step with the particles' fields replaced
# by the sharpened copy: T-961's run with the compensated gather to step 5, restored into a plain
# state whose E and B are then the compensated state's copy, and both take step 6 (a birth step):
# every particle array (the push, the atoms' populations from the rates, the births' positions,
# weights and kicked momenta) has the same bits, and the steps' fields differ (the plain state's
# solve starts from the copy). The same for `initialize!`'s half-step kick: the compensated load
# is the plain load at t − Δt/2 kicked by the module's kernel in the sharpened copy, bit for bit,
# and not the plain kick. The step writes the copy itself before anything reads it (it starts
# from NaN), and the observer reads the grid's fields: step 6's history entries are those of the
# plain step from the same state. A compensated run restarted at step 3, and a view of it over
# copies of its arrays (`state_view`, the harness' path), continue with the uninterrupted run's
# bits (the copy is scratch, written before it is read). In the moving window the copy's x-guards
# are zero as the grid's are, and its interior is the stencil's.
function body_T962(be)
    for T in TYPES
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), ADKLevel(one(T) / 2, T(4), T(3) / 2))
        prob = ion_in_E(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1), every = 2, nsteps = 8, plasma = true, rows = (1.1, 2.9))
        p = params(prob)
        cg = CompensatedGather()
        st = ion_state(prob; be, gather = cg)
        run!(st, p, 5)
        snap = snapshot(st, 5)
        plain = snap_state(prob, snap, be)          # to step in the sharpened fields
        diag = snap_state(prob, snap, be)           # to step in the grid's own fields
        β = sharpening_beta(cg, st.numerics.shape, st.grid.dx[1])
        for name in GATHER_NAMES
            sharpen_x!(plain.fields[name], st.fields[name], β, st.grid)
        end
        foreach(a -> fill!(a, T(NaN)), values(st.sharpened))     # the step writes the copy before anything reads it
        step!(st, p, 6)
        step!(plain, p, 6)
        step!(diag, p, 6)
        @test particle_bits(plain) == particle_bits(st)
        @test bits(plain.fields.Ey) != bits(st.fields.Ey)
        @test born(st).n > born(snap_state(prob, snap, be)).n           # step 6 gave birth
        # the observer reads the grid's fields: step 6's history entries are those of the plain step from the same state
        @test [bits(Array(h)[6:6]) for h in values(st.recorder.hist)] == [bits(Array(h)[6:6]) for h in values(diag.recorder.hist)]
        # the initial kick
        ic = EMState(prob, EMNumerics(; gather = cg); backend = be)
        initialize!(ic, prob)
        ip = EMState(prob, EMNumerics(); backend = be)
        initialize!(ip, prob; momenta_at = :t_minus_half)
        copyf = map(similar, (; Ex = ip.fields.Ex, Ey = ip.fields.Ey, Ez = ip.fields.Ez, Bx = ip.fields.Bx, By = ip.fields.By, Bz = ip.fields.Bz))
        β = sharpening_beta(cg, ip.numerics.shape, ip.grid.dx[1])
        for name in GATHER_NAMES
            sharpen_x!(copyf[name], ip.fields[name], β, ip.grid)
        end
        pf = merge(ip.fields, copyf)
        for s in ip.species
            ParticleMeshCore.launcher(kick_kernel!, get_backend(s), nothing)(s, pf, ip.numerics.pusher, ip.numerics.shape, Int32(guards(ip.grid)),
                                                                           -s.q / s.m * ip.dt / 4; ndrange = capacity(s))
        end
        @test map(bits, ic.species[1].u) == map(bits, ip.species[1].u)
        it0 = EMState(prob, EMNumerics(); backend = be)
        initialize!(it0, prob)
        @test map(bits, ic.species[1].u) != map(bits, it0.species[1].u)
        # a restart
        full = ion_state(prob; be, gather = cg)
        run!(full, p, 3)
        s3 = snapshot(full, 3)
        run!(full, p, prob.nsteps - 3; first = 4)
        fresh = EMState(prob, EMNumerics(; gather = cg); backend = be)
        restore!(fresh, s3)
        run!(fresh, p, prob.nsteps - 3; first = 4)
        @test map(bits, evolved(fresh)) == map(bits, evolved(full))
        # a view of the state over other arrays (the harness' path) keeps the copy
        ran = ion_state(prob; be, gather = cg)
        run!(ran, p, 3)
        sv = state_view(ran, map(copy, evolved(ran)))
        @test sv.sharpened === ran.sharpened
        run!(sv, p, prob.nsteps - 3; first = 4)
        @test map(bits, evolved(sv)) == map(bits, evolved(full))
        # the moving window: the copy's x-guards are zero as the grid's are, its interior the stencil's
        wp = ion_window(T; nsteps = 70)
        ws = ion_state(wp; be, gather = cg)
        run!(ws, params(wp), 70)
        sharpen_gather!(ws)
        g, n1 = guards(ws.grid), ncells(ws.grid)[1]
        βw = sharpening_beta(cg, ws.numerics.shape, ws.grid.dx[1])
        for name in GATHER_NAMES
            c = Array(ws.sharpened[name])
            ref = similar(ws.fields[name])
            sharpen_x!(ref, ws.fields[name], βw, ws.grid)
            r = Array(ref)
            @test all(iszero, c[1:g, :]) && all(iszero, c[(end - g + 1):end, :])
            @test bits(c[(g + 1):(g + n1), :]) == bits(r[(g + 1):(g + n1), :])
        end
        @test maximum(absmax, Array(ws.sharpened.Ey)) > 1 / 10
    end
end

"A plain state holding the snapshot `snap` of `prob`."
snap_state(prob, snap, be) = restore!(EMState(prob, EMNumerics(); backend = be), snap)

@testset "T-962 what reads the compensated copy: the push, the ionization's rates and births, and the initial kick, bit for bit as the plain step in the sharpened fields; the observer the grid's; a restart and a view continue with the run's bits; the window's edges [A2, A4, A8, WP-067]" begin
    body_T962(CPU())
end

# --- T-963 ----------------------------------------------------------------------------------

# The gradient through the compensated gather (A7), and its effect where it matters: T-778's small
# ionization-injection run (8 cells per laser wavelength, where the quadratic shape reads 7.5 %
# less of the laser's field) with the compensated gather, ForwardDiff of the born charge and the
# born electrons' energy in the laser's a₀ and the dopant's density against the finite-difference
# floor with T-778's bounds. The particles no longer lose the shape's share of the laser's field
# (the grid's own dispersion of the pulse stays), so more charge is born than without
# compensation (8.7 % more charge, 18 % more energy, measured).
@testset "T-963 ForwardDiff through the compensated gather against the finite-difference floor in a small ionization-injection run, which then births more charge [A0, A3, A7, WP-067]" begin
    x0 = collect(ION_X0)
    cg = CompensatedGather()
    J0 = ion_objectives(ION_X0; gather = cg)
    Jp = ion_objectives(ION_X0)
    @info "T-963 objectives" compensated = J0 plain = Jp rise = J0 ./ Jp .- 1
    MEASURED["T-963 born charge, compensated/plain"] = Dict("Float64" => J0[1] / Jp[1])
    @test J0[1] > 1.05 * Jp[1]
    ad = ForwardDiff.jacobian(x -> ion_objectives(x; gather = cg), x0)
    for (o, name) in enumerate(("charge", "energy")), i in 1:2
        f(xi, mode, seed) = ion_objectives(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode; gather = cg)[o]
        hs = x0[i] .* 2.0 .^ (-4:-2:-20)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[o, i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-963 $name, $(ION_NAMES[i])" ad = ad[o, i] fd = r.fd[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-963 rel/floor, $name", Dict{String,Float64}())[ION_NAMES[i]] = ratio
        @test isfinite(ad[o, i]) && ad[o, i] != 0
        # the born charge is linear in the density (up to the electrons' own weak fields), so three
        # neighbouring quotients can agree to the bit: a floor of zero, where the AD must then agree
        # with them to a few hundred eps of the run (measured 5.3e-12)
        @test r.rel[r.istar] <= (r.floor_rel > 0 ? emC("T-778") * r.floor_rel : 1.0e-10)
        @test r.floor_rel <= emC("T-778", "floor_max")
    end
end
