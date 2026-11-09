# WP-047: the witness bunch and its mean energy gain, T-640 to T-645. Test electrons injected at a
# fixed time on a fixed lattice of the box (the window's phases), averaged with Gaussian weights
# whose centre, length and width are the bunch's parameters (A3). After lwfa_gradient.jl, whose
# case and helpers T-642 to T-644 reuse.

using ParticleMeshElectromagnetic: witness_frame

"The lattice of a bunch on a grid of cell sizes `h`, computed here from the definition: the points `((c − 1 + (k − ½)/p)·h)` of every cell whose centre lies in the region, axis 1 fastest."
function lattice(b, h, n)
    cells = map(1:2) do d
        lo, hi = b.region[d]
        filter(c -> lo <= (c - 0.5) * h[d] <= hi, 1:n[d])
    end
    pts = Tuple{Float64,Float64}[]
    for c2 in cells[2], c1 in cells[1], k2 in 1:b.ppc[2], k1 in 1:b.ppc[1]
        push!(pts, ((c1 - 1 + (k1 - 0.5) / b.ppc[1]) * h[1], (c2 - 1 + (k2 - 0.5) / b.ppc[2]) * h[2]))
    end
    return pts, cells
end

"The first step whose start time (k − 1)·Δt reaches `inject`."
injection_step(inject, dt) = findfirst(k -> (k - 1) * dt >= inject, 1:10^6)

# --- T-640 ----------------------------------------------------------------------------------

"""
    witness_in_E(T; nsteps, Ex, center, len, wid) -> EMProblem{2}

A periodic 2D box (16 × 8 cells of 1/2, Δt at 0.9 of the Courant limit) with no plasma (one
species of zero density) and a uniform background field `E0 = (Ex, 0, 0)`, and a witness bunch of
2 × 1 points per cell on the cells with centres in ((2.1, 5.9), (1, 3)) (bounds off the cell edges,
where a selection by the cell's centre and one by its whole extent differ), injected at t = 5.5·Δt
(step 7) with u₀ = (3, 1/2, 1/4), of charge −2 and mass 4 (q/m = −1/2). `boundaries` and
`absorber` make axis 2 open.
"""
function witness_in_E(::Type{T}; nsteps = 40, Ex = -1 // 20, center = (T(4), T(2)), len = T(1), wid = T(3) / 2,
                      momentum = (T(3), T(1) / 2, T(1) / 4), region = ((2.1, 5.9), (1, 3)), boundaries = (:periodic, :periodic)) where {T}
    cells = (16, 8)
    L = (T(8), T(4))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0), (1, 1), ZERO3(T), ZERO3(T), 1)
    wb = WitnessBunch(; inject = 5.5 * Float64(value_of(dt)), region, ppc = (2, 1), momentum, center, length = len, width = wid,
                      charge = -2, mass = 4)
    return EMProblem("witness-E", L, cells, T(dt), nsteps, (e,), 1, (1, 0); E0 = (T(Ex), zero(T), zero(T)), witness = wb, boundaries)
end

# The witness's work in a prescribed field, an exact discrete identity: in a uniform E along x and
# no B, Boris adds (q/m)·Ex·Δt to u_x in every push (two half kicks, no rotation) and leaves u_y,
# u_z as they are, so after the injection at step k₀ and n − k₀ + 1 pushes every witness has
# u_x = u₀ₓ + (n − k₀ + 1)·(q/m)·Ex·Δt and the gain m·(γ − γ₀) of that momentum (to roundoff over
# the pushes; q/m = −1/2 and m = 4, so the bunch's own charge and mass are the ones used); the
# bunch is empty before step k₀ and whole from it, and empty again after a second `initialize!`;
# the injection step is the first whose start time reaches the injection time, a start time on it
# included; the weighted mean counts the alive witnesses only; the weights are the Gaussian of
# the lattice positions, computed here from the definition, and with the centre and the length
# duals their partials are the closed forms ∂w/∂c = w·(x − c)/ℓ², ∂w/∂ℓ = w·(x − c)²/ℓ³, while
# the positions carry none (A3).
function body_T640(be)
    for T in TYPES
        center = (seeded(T, 4), T(2))
        len = seeded(T, 1) / 2 + one(T) / 2           # ℓ = 1, ∂ℓ = (1, 2, 3)/16
        prob = witness_in_E(T; center, len)
        st = EMState(prob, EMNumerics(; current = NoCurrent()); backend = be)
        initialize!(st, prob)
        dt = Float64(value_of(prob.dt))
        k0 = injection_step(5.5 * dt, prob.dt)
        @test k0 == 7
        @test witness_frame(st).kinj == k0
        p = params(prob)
        for k in 1:(k0 - 1)
            step!(st, p, k)
        end
        @test witness_alive(st) == 0
        @test_throws ArgumentError witness_energy_gain(st)
        for k in k0:prob.nsteps
            step!(st, p, k)
        end
        pts, _ = lattice(prob.witness, Float64.(value_of.(cellsize(st.grid))), ncells(st.grid))
        @test witness_alive(st) == length(pts) == 8 * 4 * 2      # 8 cells × 4 rows × 2 points
        g = witness_gains(st)
        npush = prob.nsteps - k0 + 1
        ux = 3 + npush * (-1 / 2) * (-1 / 20) * dt
        gain = 4 * (sqrt(1 + ux^2 + 1 / 4 + 1 / 16) - sqrt(1 + 9 + 1 / 4 + 1 / 16))
        s = witness_frame(st).species
        err = maximum(abs, value_of.(Array(s.u[1])) .- ux) / (npush * base_eps(T) * ux)
        record!("T-640", T, err)
        @test err <= emC("T-640")
        @test all(==(T(1) / 2), Array(s.u[2])) && all(==(T(1) / 4), Array(s.u[3]))
        @test maximum(x -> abs(value_of(x) - gain), g.gain) <= emC("T-640") * 4 * npush * base_eps(T) * ux
        @test abs(value_of(witness_energy_gain(st)) - gain) <= emC("T-640") * 4 * npush * base_eps(T) * ux
        # the weights from the definition, at the lattice (the lattice order is the slot order)
        c, ℓ, wd = (4.0, 2.0), 1.0, 1.5
        for (i, (x, y)) in enumerate(pts)
            w = exp(-((x - c[1]) / ℓ)^2 / 2 - ((y - c[2]) / wd)^2 / 2)
            @test value_of(g.w[i]) ≈ w rtol = 8 * base_eps(T)
            if T === D3
                dw = w .* ((x - c[1]) .* [1, 2, 3] ./ 8 ./ ℓ^2 .+ (x - c[1])^2 .* [1, 2, 3] ./ 16 ./ ℓ^3)
                @test collect(Float64.(Tuple(partials(g.w[i])))) ≈ dw atol = 1e-6
            end
        end
        if T === D3
            @test all(x -> all(iszero, partials(x)), Array(s.offset[1]))
            @test all(x -> all(iszero, partials(x)), Array(s.u[1]))   # the field carries no partials
        end
        # the mean over the alive witnesses only: retire every third one and change one alive gain
        st_ = Array(s.status)
        st_[1:3:end] .= ParticleMeshCore.FREE
        copyto!(s.status, st_)
        u1 = Array(s.u[1])
        u1[2] += one(T)
        copyto!(s.u[1], u1)
        g = witness_gains(st)
        a = findall(g.alive)
        @test length(a) == length(pts) - length(1:3:length(pts))
        mean = sum(value_of(g.w[i]) * value_of(g.gain[i]) for i in a) / sum(value_of(g.w[i]) for i in a)
        @test_throws ArgumentError witness_energy_gain(st)            # a retired witness is refused by default
        @test value_of(witness_energy_gain(st; allow_lost = true)) ≈ mean rtol = 16 * base_eps(T)
        @test !(value_of(witness_energy_gain(st; allow_lost = true)) ≈ sum(value_of.(g.w) .* value_of.(g.gain)) / sum(value_of.(g.w)))
        # weights that underflow on every alive witness (the centre far outside the region) are refused, not a NaN
        copyto!(s.w, zero(Array(s.w)))
        @test_throws ArgumentError witness_energy_gain(st; allow_lost = true)
        initialize!(st, prob)
        @test witness_alive(st) == 0
    end
    # the injection step, on the host: a start time equal to the injection time is the injection
    @test ParticleMeshElectromagnetic.witness_step(1.5, 0.25) == 7
    @test ParticleMeshElectromagnetic.witness_step(1.6, 0.25) == 8
    @test ParticleMeshElectromagnetic.witness_step(0.0, 0.25) == 1
    # an open axis 2: witnesses of u₀ = (0, 3, 0) leave through the top edge and are retired, the
    # ones whose straight orbit y₀ + n·v·Δt has passed L₂ after n pushes and no others
    for T in TYPES
        prob = witness_in_E(T; Ex = 0, momentum = (zero(T), T(3), zero(T)), region = ((2, 6), (2.6, 3.9)),
                            boundaries = (:periodic, :open), nsteps = 10)
        st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = AbsorbingLayer(; cells = 3, strength = 1.0)); backend = be)
        initialize!(st, prob)
        k0 = witness_frame(st).kinj
        pts, _ = lattice(prob.witness, Float64.(value_of.(cellsize(st.grid))), ncells(st.grid))
        v = 3 / sqrt(10)
        dt = Float64(value_of(prob.dt))
        p = params(prob)
        counts, expected = Int[], Int[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            push!(expected, count(q -> q[2] + (k - k0 + 1) * v * dt < 4, pts))
        end
        @test counts == expected
        @test 0 < last(counts) < first(counts) < length(pts)       # the top row leaves in the first push, the bottom one stays
    end
end

@testset "T-640 a witness in a uniform field gains the field's work, its weights a Gaussian of the fixed lattice [A3, A7, A8, WP-047]" begin
    body_T640(CPU())
end

# --- T-641 ----------------------------------------------------------------------------------

"A window in vacuum (24 × 6 cells of 1/2, no plasma, moving from t = 0) with a witness bunch of `momentum` on the cells with centres in `region`, injected at t = 3."
function witness_in_window(::Type{T}; momentum, region, nsteps = 60) where {T}
    cells = (24, 6)
    L = (T(12), T(3))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0), (1, 1), ZERO3(T), ZERO3(T), 1; momentum = :gaussian_velocity)
    wb = WitnessBunch(; inject = 3, region, ppc = (2, 2), momentum, center = (T(6), T(3) / 2), length = T(2))
    return EMProblem("witness-window", L, cells, T(dt), nsteps, (e,), 1, (1, 0); boundaries = (:window, :periodic),
                     window = MovingWindow(; start = zero(T)), witness = wb)
end

# The bunch in window coordinates (WP-028's moving window, empty): a fast witness (u₀ = 10 along x)
# drifts back by (1 − v)·Δt per step against a window that moves by whole cells, so its window
# position after step n is its lattice point plus (n − k₀ + 1)·v·Δt minus the cells the window
# moved since the injection (to roundoff over the steps), and it gains nothing (the fields stay
# exactly zero); a bunch at rest moves one cell back with every shift and is retired when it falls
# behind the back edge, so the alive count after each step is the number of lattice points whose
# cell lies behind no more than the shifts since the injection, and once none is left the
# objective is refused; a bunch moving backwards leaves through the back edge between shifts.
function body_T641(be)
    for T in TYPES
        prob = witness_in_window(T; momentum = T(10), region = ((4, 8), (1, 2)))
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        h = Float64.(value_of.(cellsize(st.grid)))
        pts, _ = lattice(prob.witness, h, ncells(st.grid))
        moved = window_shifts(w, prob.nsteps) - window_shifts(w, k0 - 1)
        @test moved > 10
        v = 10 / sqrt(101)
        npush = prob.nsteps - k0 + 1
        g = witness_gains(st)
        @test witness_alive(st) == length(pts)
        @test all(iszero, g.gain)
        x = [p[1] + npush * v * Float64(value_of(prob.dt)) - moved * h[1] for p in pts]
        err = maximum(abs, value_of.(g.x) .- x) / (npush * base_eps(T) * 12)
        record!("T-641", T, err)
        @test err <= emC("T-641")
        @test value_of.(g.y) ≈ last.(pts) rtol = 4 * base_eps(T)
        # the bunch drifted back into the window's back band, where an observable is not to be read (#370): refused
        edge = window_back_band(st) * h[1]
        @test any(<(edge), value_of.(g.x))
        @test_throws ArgumentError witness_energy_gain(st)

        prob = witness_in_window(T; momentum = zero(T), region = ((1, 4), (0, 3)))
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        pts, cells = lattice(prob.witness, h, ncells(st.grid))
        cell1 = [floor(Int, p[1] / h[1]) + 1 for p in pts]
        p = params(prob)
        counts = Int[]
        expected = Int[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            push!(expected, count(>(window_shifts(w, k) - window_shifts(w, k0 - 1)), cell1))
        end
        @test counts == expected
        @test first(counts) == length(pts) && last(counts) == 0
        @test_throws ArgumentError witness_energy_gain(st)

        # a bunch moving backwards (u₀ = −3) crosses the back edge by its own motion between the
        # shifts, and is retired there (not wrapped to the front): alive after step k are the
        # witnesses whose window position, lattice point + (k − k₀ + 1)·v·Δt − cells moved, is ≥ 0
        prob = witness_in_window(T; momentum = -3 * one(T), region = ((1, 3), (1, 2)), nsteps = 16)
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        pts, _ = lattice(prob.witness, h, ncells(st.grid))
        v = -3 / sqrt(10)
        dt = Float64(value_of(prob.dt))
        p = params(prob)
        counts, expected = Int[], Int[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            moved = window_shifts(w, k) - window_shifts(w, k0 - 1)
            push!(expected, count(q -> q[1] + (k - k0 + 1) * v * dt - moved * h[1] >= 0, pts))
        end
        @test counts == expected
        @test last(counts) == 0 && length(unique(counts)) > 2
        @test all(<=(ncells(st.grid)[1] ÷ 2), Array(witness_frame(st).species.cell[1])[Array(witness_frame(st).species.status) .== 0x01])
    end
end

@testset "T-641 the witness bunch moves with the window and is retired behind its back edge [A3, WP-028, WP-047]" begin
    body_T641(CPU())
end

# --- T-642 ----------------------------------------------------------------------------------

"""
The witness bunch of the WP-043 case: test electrons of u₀ = 10 along x injected at t = 80 (the
window has moved for 8 c/ω₀) on 2 × 2 points per cell of the cells with centres in window
coordinates (14, 34) × (11.5, 13.6), the two rows about the axis, a Gaussian of rms length `len`
centred at `center` on the axis, uniform across the rows; at the end of the 350 steps the
accelerating phase of the wake lies about 22 to 31 c/ω₀ from the back edge, the pulse about 33 to
40, and no witness has left the window.
"""
function lwfa_witness(center, len)
    T = promote_type(typeof(center), typeof(len))
    return WitnessBunch(; inject = 80, region = ((14, 34), (11.5, 13.6)), ppc = (2, 2), momentum = T(10),
                        center = (T(center), T(8 * 4 * 2π / 16)), length = T(len))
end

"The parameters of T-642: WP-043's five and the bunch's centre and length (c/ω₀)."
const WITNESS_X0 = (LWFA_GRADIENT_X0..., 26.0, 3.0)
const WITNESS_NAMES = (LWFA_GRADIENT_NAMES..., "center", "length")

"The WP-047 objective at `x = (a0, duration, focus, density, ramp, center, length)`: the witness bunch's mean energy gain in the WP-043 case."
function lwfa_witness_gain(x, mode = Reference(); kw...)
    st, _ = lwfa_gradient_run(x[1:5], mode; witness = lwfa_witness(x[6], x[7]), kw...)
    return witness_energy_gain(st)                   # refuses a run in which a witness left the window
end

# One ForwardDiff gradient in all seven parameters (chunk 7, Float64, Reference) against A7's
# finite-difference floor on each partial, as T-581 does for the wake amplitude (the scan carried
# on to 2⁻²⁰, since this objective is smooth enough that the quotients at 2⁻¹⁴ still converge): the laser and
# plasma parameters reach the objective through the fields along the witnesses' orbits, the
# bunch's through its weights alone.
const T642_AD = Ref{Any}(nothing)
@testset "T-642 ForwardDiff of the witness bunch's mean energy gain through the small LWFA run against the finite-difference floor, seven parameters [A3, A7, WP-047]" begin
    x0 = collect(WITNESS_X0)
    st, _ = lwfa_gradient_run(LWFA_GRADIENT_X0, Reference(); witness = lwfa_witness(x0[6], x0[7]))
    g = witness_gains(st)
    @test witness_alive(st) == length(g.gain)
    J0 = witness_energy_gain(st)
    @info "T-642 objective" J0 maxgain = maximum(g.gain) mingain = minimum(g.gain)
    @test J0 > 0.1                                   # the bunch sits in the accelerating phase
    ad = ForwardDiff.gradient(x -> lwfa_witness_gain(x), x0)
    T642_AD[] = ad
    for i in eachindex(x0)
        f(xi, mode, seed) = lwfa_witness_gain(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode)
        hs = x0[i] .* 2.0 .^ (-4:-2:-20)                # to 2⁻²⁰, past where truncation gives way to roundoff
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-642 $(WITNESS_NAMES[i])" ad = ad[i] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-642 rel/floor", Dict{String,Float64}())[WITNESS_NAMES[i]] = ratio
        @test isfinite(ad[i]) && ad[i] != 0
        @test r.rel[r.istar] <= emC("T-642") * r.floor_rel
        @test r.floor_rel <= emC("T-642", "floor_max")
    end
end

# --- T-643 ----------------------------------------------------------------------------------

# The production element type: Dual{Float32,3} seeded in a₀, the density and the bunch's centre,
# against T-642's Float64 gradient, measured as in T-582.
@testset "T-643 Dual{Float32,3} partials of the witness bunch's mean energy gain against the Float64 gradient [A7, A8, WP-047]" begin
    x0 = WITNESS_X0
    ad = T642_AD[] === nothing ? ForwardDiff.gradient(x -> lwfa_witness_gain(x), collect(x0)) : T642_AD[]
    seeds = Dict(1 => (1, 0, 0), 4 => (0, 1, 0), 6 => (0, 0, 1))
    xd = ntuple(i -> mkdual(x0[i], get(seeds, i, (0, 0, 0))...), length(x0))
    J = lwfa_witness_gain(xd)
    p = Tuple(partials(J))
    ref = (ad[1], ad[4], ad[6])
    rel = maximum(abs.(p .- ref) ./ abs.(ref))
    primal = abs(value(J) - lwfa_witness_gain(x0)) / abs(lwfa_witness_gain(x0))
    @info "T-643" partials = p ref rel primal
    get!(MEASURED, "T-643 rel", Dict{String,Float64}())["Dual{Float32,3}"] = rel
    @test J isa D3
    @test all(isfinite, p) && all(!iszero, p)
    @test rel <= emC("T-643")
    @test primal <= emC("T-643")
end

# --- T-644 ----------------------------------------------------------------------------------

# Test particles: the bunch deposits nothing, so the run with it is the run without it, bit for bit
# (every field, the plasma and the histories), and the witness's objective depends on the bunch's
# parameters only through the weights: the gains of two bunches with different centres and lengths
# are the same bits.
@testset "T-644 the witness bunch changes nothing else in the run [A3, A4, WP-047]" begin
    st0, _ = lwfa_gradient_run(LWFA_GRADIENT_X0, Reference())
    st1, _ = lwfa_gradient_run(LWFA_GRADIENT_X0, Reference(); witness = lwfa_witness(26.0, 3.0))
    st2, _ = lwfa_gradient_run(LWFA_GRADIENT_X0, Reference(); witness = lwfa_witness(20.0, 1.5))
    for k in keys(st0.fields)
        @test same_bits(Array(st0.fields[k]), Array(st1.fields[k]))
    end
    for k in keys(st0.hist)
        @test same_bits(Array(st0.hist[k]), Array(st1.hist[k]))
    end
    @test same_bits(Array(st0.species[1].u[1]), Array(st1.species[1].u[1]))
    g1, g2 = witness_gains(st1), witness_gains(st2)
    @test same_bits(g1.gain, g2.gain)
    @test !same_bits(g1.w, g2.w)
end
