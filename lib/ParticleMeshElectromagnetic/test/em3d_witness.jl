# WP-093: the witness bunch at N = 3, T-1260 to T-1264 (T-1265 is their rerun on CUDA, in gpu.jl).
# The bunch of witness.jl with a third axis: a lattice, a centre and a range per axis, the length
# along axis 1 and the width along both transverse axes. Reuses no other test file: the problems
# are em3d_cases.jl's or built here.

using ParticleMeshElectromagnetic: witness_frame, witness_placement, check_setup

"""
    witness_lattice(b, h, n) -> (points, cells)

The lattice of a bunch of any N on a grid of cell sizes `h` and `n` cells, from the definition: the
points `(c_d − 1 + (k_d − ½)/p_d)·h_d` of every cell whose centre lies in the region, in slot
order (cells along axis 1 fastest, then axis 2, then axis 3; within a cell the same), and the
cells per axis.
"""
function witness_lattice(b, h, n)
    N = length(b.region)
    cells = map(1:N) do d
        lo, hi = b.region[d]
        filter(c -> lo <= (c - 0.5) * h[d] <= hi, 1:n[d])
    end
    pts = NTuple{N,Float64}[]
    for c in Iterators.product(cells...), k in Iterators.product(map(p -> 1:p, b.ppc)...)   # the first index runs fastest
        push!(pts, ntuple(d -> (c[d] - 1 + (k[d] - 0.5) / b.ppc[d]) * h[d], N))
    end
    return pts, cells
end

"`problem` with the witness bunch `wb`."
with_witness(p::EMProblem, wb) =
    EMProblem(p.name, p.length, p.cells, p.dt, p.nsteps, p.species, p.seed, p.kmode; E0 = p.E0, B0 = p.B0, lasers = p.lasers,
              boundaries = p.boundaries, window = p.window, profile = p.profile, witness = wb, ionization = p.ionization)

# --- T-1260 ---------------------------------------------------------------------------------

"""
    witness_in_E_3d(T; nsteps, Ex, center, len, wid, momentum, region, boundaries, cells, h) -> EMProblem{3}

`witness_in_E` of witness.jl in 3D: a periodic box (`cells`, 16 × 8 × 6 unless given, of the sizes
`h`, 1/2 on every axis unless given, Δt at 0.9 of the Courant limit) with no plasma (one species of
zero density) and a uniform background field `E0 = (Ex, 0, 0)`, and a witness bunch of 2 × 1 × 2
points per cell on the cells with centres in ((2.1, 5.9), (1, 3), (0.6, 2.4)) (8 × 4 × 4 cells of
1/2; 8 × 3 × 2 cells of 1/2 × 5/8 × 3/4), injected at t = 5.5·Δt (step 7) with u₀ = (3, 1/2, 1/4),
of charge −2 and mass 4. `boundaries` and an absorber make an axis open.
"""
function witness_in_E_3d(::Type{T}; nsteps = 40, Ex = -1 // 20, center = (T(4), T(2), T(3) / 2), len = one(T), wid = T(3) / 2,
                         momentum = (T(3), T(1) / 2, T(1) / 4), region = ((2.1, 5.9), (1, 3), (0.6, 2.4)),
                         boundaries = (:periodic, :periodic, :periodic), cells = (16, 8, 6), h = (1 // 2, 1 // 2, 1 // 2)) where {T}
    L = ntuple(d -> T(cells[d]) * T(h[d]), 3)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0, 0), (1, 1, 1), ZERO3(T), ZERO3(T), 1)
    wb = WitnessBunch(; inject = 5.5 * Float64(value_of(dt)), region, ppc = (2, 1, 2), momentum, center, length = len, width = wid,
                      charge = -2, mass = 4)
    return EMProblem("witness-E-3d", L, cells, T(dt), nsteps, (e,), 1, (1, 0, 0); E0 = (T(Ex), zero(T), zero(T)), witness = wb, boundaries)
end

# T-640 at N = 3. In a uniform E along x and no B every push adds (q/m)·Ex·Δt to u_x and leaves
# u_y, u_z as they are, so after the injection at step k₀ and n − k₀ + 1 pushes every witness has
# u_x = u₀ₓ + (n − k₀ + 1)·(q/m)·Ex·Δt and the gain m·(γ − γ₀) of that momentum; the bunch is empty
# before step k₀ and whole from it. The weights are the Gaussian of the lattice positions, computed
# here from the definition in slot order: the length along x, the width along y and along z; with
# the centre's x and z and the width duals (one partial each) their partials are the closed forms
# ∂w/∂c_x = w·(x − c_x)/ℓ², ∂w/∂c_z = w·(z − c_z)/σ² and ∂w/∂σ = w·((y − c_y)² + (z − c_z)²)/σ³,
# while the positions carry none (A3). The cells are 1/2 × 5/8 × 3/4, a different size on every
# axis, so a weight or a lattice taken with another axis' cell size is another number. Across an
# open z the witnesses of u₀ = (0, 0, 3) whose straight orbit has passed the edge are retired, and
# no others.
function body_T1260(be)
    for T in TYPES
        D = T === D3
        # a centre off the lattice's planes of symmetry along every axis, so that no two points of a line share a weight
        center = (D ? mkdual(65 // 16, 1, 0, 0) : T(65) / 16, T(17) / 8, D ? mkdual(23 // 16, 0, 1, 0) : T(23) / 16)
        wid = D ? mkdual(3 // 2, 0, 0, 1) : T(3) / 2
        prob = witness_in_E_3d(T; center, wid, h = (1 // 2, 5 // 8, 3 // 4))
        @test prob.witness isa WitnessBunch{T,3}
        st = EMState(prob, EMNumerics(; current = NoCurrent()); backend = be)
        initialize!(st, prob)
        dt = Float64(value_of(prob.dt))
        k0 = witness_frame(st).kinj
        @test k0 == 7
        p = params(prob)
        @test p.witness isa WitnessParams{T,3}
        for k in 1:(k0 - 1)
            step!(st, p, k)
        end
        @test witness_alive(st) == 0
        @test_throws ArgumentError witness_energy_gain(st, p)
        step!(st, p, k0)
        h = Float64.(value_of.(cellsize(st.grid)))
        pts, cells = witness_lattice(prob.witness, h, ncells(st.grid))
        @test h == (1 / 2, 5 / 8, 3 / 4)
        @test map(length, cells) == [8, 3, 2] && map(first, cells) == [5, 3, 2]
        @test witness_alive(st) == length(pts) == 8 * 3 * 2 * 4
        s = witness_frame(st).species
        @test capacity(s) == length(pts)
        for k in (k0 + 1):prob.nsteps
            step!(st, p, k)
        end
        g = witness_gains(st, p)
        @test keys(g) == (:x, :y, :z, :w, :gain, :alive)
        npush = prob.nsteps - k0 + 1
        ux = 3 + npush * (-1 / 2) * (-1 / 20) * dt
        gain = 4 * (sqrt(1 + ux^2 + 1 / 4 + 1 / 16) - sqrt(1 + 9 + 1 / 4 + 1 / 16))
        err = maximum(abs, value_of.(Array(s.u[1])) .- ux) / (npush * base_eps(T) * ux)
        record!("T-1260", T, err, be)
        @test err <= emC("T-1260")
        @test all(==(T(1) / 2), Array(s.u[2])) && all(==(T(1) / 4), Array(s.u[3]))
        @test maximum(x -> abs(value_of(x) - gain), g.gain) <= emC("T-1260") * 4 * npush * base_eps(T) * ux
        @test abs(value_of(witness_energy_gain(st, p)) - gain) <= emC("T-1260") * 4 * npush * base_eps(T) * ux
        # the weights from the definition at the lattice, in slot order; u_y and u_z are constants
        # of the motion, so y and z have moved by a known amount and x stays the lattice's in order
        c, ℓ, σ = (65 / 16, 17 / 8, 23 / 16), 1.0, 1.5
        worst = 0.0
        for (i, (x, y, z)) in enumerate(pts)
            w = exp(-((x - c[1]) / ℓ)^2 / 2 - ((y - c[2]) / σ)^2 / 2 - ((z - c[3]) / σ)^2 / 2)
            worst = max(worst, abs(value_of(g.w[i]) - w) / w)
            if D
                dw = [w * (x - c[1]) / ℓ^2, w * (z - c[3]) / σ^2, w * ((y - c[2])^2 + (z - c[3])^2) / σ^3]
                @test collect(Float64.(Tuple(partials(g.w[i])))) ≈ dw atol = 1e-6
            end
        end
        record!("T-1260 weights", T, worst / base_eps(T), be)
        @test worst <= emC("T-1260", "weights") * base_eps(T)
        @test length(unique(value_of.(g.w))) > length(pts) ÷ 2          # the weights vary along all three axes
        if D
            @test all(x -> all(iszero, partials(x)), Array(s.offset[1])) && all(x -> all(iszero, partials(x)), Array(s.offset[3]))
            @test all(x -> all(iszero, partials(x)), Array(s.u[1]))
        end
        initialize!(st, prob)
        @test witness_alive(st) == 0
    end
    # an open axis 3: the witnesses of u₀ = (0, 0, 3) leave through the top face and are retired,
    # the ones whose straight orbit z₀ + n·v·Δt has passed L₃ after n pushes and no others
    for T in TYPES
        prob = witness_in_E_3d(T; Ex = 0, momentum = (zero(T), zero(T), T(3)), region = ((2, 6), (1, 3), (4.6, 5.9)),
                               boundaries = (:periodic, :periodic, :open), nsteps = 10, cells = (16, 8, 12))
        st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = AbsorbingLayer(; cells = 3, strength = 1.0)); backend = be)
        initialize!(st, prob)
        k0 = witness_frame(st).kinj
        pts, _ = witness_lattice(prob.witness, Float64.(value_of.(cellsize(st.grid))), ncells(st.grid))
        v = 3 / sqrt(10)
        dt = Float64(value_of(prob.dt))
        p = params(prob)
        counts, expected = Int[], Int[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            push!(expected, count(q -> q[3] + (k - k0 + 1) * v * dt < 6, pts))
        end
        @test counts == expected
        @test 0 < last(counts) < first(counts) < length(pts)       # the top layer leaves in the first push, the bottom one stays
    end
end

@testset "T-1260 a 3D witness in a uniform field gains the field's work, its weights a Gaussian of the fixed lattice along three axes, and it is retired at an open face [A3, A7, A8, WP-093]" begin
    body_T1260(CPU())
end

# --- T-1261 ---------------------------------------------------------------------------------

"A 3D window in vacuum (24 × 6 × 6 cells of 1/2 × 5/8 × 3/4, no plasma, moving from t = 0) with a witness bunch of `momentum` on the cells with centres in `region`, 2 × 1 × 2 points per cell, injected at t = 3."
function witness_in_window_3d(::Type{T}; momentum, region, nsteps = 60) where {T}
    cells = (24, 6, 6)
    L = (T(12), T(15) / 4, T(9) / 2)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:3))) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), zero(T), zero(T), (1, 0, 0), (1, 1, 1), ZERO3(T), ZERO3(T), 1; momentum = :gaussian_velocity)
    wb = WitnessBunch(; inject = 3, region, ppc = (2, 1, 2), momentum, center = (T(6), T(3) / 2, T(3) / 2), length = T(2))
    return EMProblem("witness-window-3d", L, cells, T(dt), nsteps, (e,), 1, (1, 0, 0); boundaries = (:window, :periodic, :periodic),
                     window = MovingWindow(; start = zero(T)), witness = wb)
end

# T-641 at N = 3: the bunch in the coordinates of the 3D moving window (WP-074's, empty). A fast
# witness (u₀ = 10 along x) is at its lattice point plus (n − k₀ + 1)·v·Δt minus the cells the
# window moved since the injection, its y and z the lattice's (on cells of another size along
# each axis), and gains nothing; a bunch at rest
# moves one cell back with every shift and is retired behind the back edge, so the alive count
# after each step is the number of lattice points whose cell lies behind no more than the shifts
# since the injection; a bunch moving backwards leaves through the back edge between shifts and
# is not wrapped to the front.
function body_T1261(be)
    for T in TYPES
        prob = witness_in_window_3d(T; momentum = T(10), region = ((4, 8), (1, 2), (0.6, 2.4)))
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        h = Float64.(value_of.(cellsize(st.grid)))
        pts, _ = witness_lattice(prob.witness, h, ncells(st.grid))
        moved = window_shifts(w, prob.nsteps) - window_shifts(w, k0 - 1)
        @test moved > 10
        v = 10 / sqrt(101)
        npush = prob.nsteps - k0 + 1
        g = witness_gains(st, params(prob))
        @test witness_alive(st) == length(pts) > 0
        @test all(iszero, g.gain)
        x = [q[1] + npush * v * Float64(value_of(prob.dt)) - moved * h[1] for q in pts]
        err = maximum(abs, value_of.(g.x) .- x) / (npush * base_eps(T) * 12)
        record!("T-1261", T, err, be)
        @test err <= emC("T-1261")
        @test h == (1 / 2, 5 / 8, 3 / 4) && length(unique(q -> q[2], pts)) == 1 && length(unique(q -> q[3], pts)) == 4
        @test value_of.(g.y) ≈ [q[2] for q in pts] rtol = 4 * base_eps(T)
        @test value_of.(g.z) ≈ [q[3] for q in pts] rtol = 4 * base_eps(T)

        prob = witness_in_window_3d(T; momentum = zero(T), region = ((1, 4), (0, 3), (0, 3)))
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        pts, _ = witness_lattice(prob.witness, h, ncells(st.grid))
        cell1 = [floor(Int, q[1] / h[1]) + 1 for q in pts]
        p = params(prob)
        counts, expected = Int[], Int[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            push!(expected, count(>(window_shifts(w, k) - window_shifts(w, k0 - 1)), cell1))
        end
        @test counts == expected
        @test first(counts) == length(pts) && last(counts) == 0
        @test_throws ArgumentError witness_energy_gain(st, params(prob))

        prob = witness_in_window_3d(T; momentum = -3 * one(T), region = ((1, 3), (1, 2), (1, 2)), nsteps = 22)
        st = EMState(prob, EMNumerics(); backend = be)
        initialize!(st, prob)
        w = st.aux.window
        k0 = witness_frame(st).kinj
        pts, _ = witness_lattice(prob.witness, h, ncells(st.grid))
        v = -3 / sqrt(10)
        dt = Float64(value_of(prob.dt))
        p = params(prob)
        counts, expected = Int[], Int[]
        front = true
        s = witness_frame(st).species
        for k in 1:prob.nsteps
            step!(st, p, k)
            k >= k0 || continue
            push!(counts, witness_alive(st))
            moved = window_shifts(w, k) - window_shifts(w, k0 - 1)
            push!(expected, count(q -> q[1] + (k - k0 + 1) * v * dt - moved * h[1] >= 0, pts))
            front &= all(<=(ncells(st.grid)[1] ÷ 2), Array(s.cell[1])[Array(s.status) .== 0x01])
        end
        @test counts == expected
        @test last(counts) == 0 && length(unique(counts)) > 2
        @test front                                                # none was wrapped to the front
    end
end

@testset "T-1261 the 3D witness bunch moves with the window and is retired behind its back edge [A3, WP-074, WP-093]" begin
    body_T1261(CPU())
end

# --- T-1262 ---------------------------------------------------------------------------------

# A 3D run that does not vary along one axis is the 2D run (T-953: two cold beams, one perturbed),
# and a bunch whose lattice is the 2D bunch's on the plane's axes, with any range along the
# invariant one, gains what the 2D bunch gains: every 3D witness against the 2D witness of its
# place in the plane, along each of the three axes (the plane's axes on the 3D ones by the cyclic
# map `inv3_axes`, the momentum's components likewise). This is the test of the fields a 3D
# witness sees at its place, on each axis: the two runs' fields agree to roundoff (T-953), so the
# gains do, relative to the largest 2D gain, scaled as T-953 by k·(n₁ + n₂)·√ppc.
function body_T1262(be)
    C = emC("T-1262")
    u2 = (3 // 10, -1 // 5, 1 // 10)
    for T in TYPES, axis in (3, 1, 2)
        σ = inv3_axes(axis)
        put(x2, x3) = (y = Vector{Any}(undef, 3); y[σ[1]] = x2[1]; y[σ[2]] = x2[2]; y[σ[3]] = x3; Tuple(y))
        amplitude = seeded(T, 1 / 20)
        p2, p3 = beams_plane_2d(T; amplitude), beams_3d(T, axis; amplitude)
        inject = 2.5 * Float64(value_of(p2.dt))
        region = ((1.0, 4.5), (0.8, 3.5))
        b2 = WitnessBunch(; inject, region, ppc = (2, 1), momentum = T.(u2), center = (T(3), T(2)), length = T(1), width = T(2))
        b3 = WitnessBunch(; inject, region = put(region, (0.0, Float64(value_of(p3.length[σ[3]])))), ppc = put((2, 1), 2),
                          momentum = T.(put((u2[1], u2[2]), u2[3])), center = put((T(3), T(2)), T(1)), length = T(1), width = T(2))
        p2, p3 = with_witness(p2, b2), with_witness(p3, b3)
        s2 = EMState(p2, EMNumerics(); backend = be)
        s3 = EMState(p3, EMNumerics(); backend = be)
        initialize!(s2, p2)
        initialize!(s3, p3)
        run!(s2, params(p2), p2.nsteps)
        run!(s3, params(p3), p3.nsteps)
        g2, g3 = witness_gains(s2, params(p2)), witness_gains(s3, params(p3))
        @test all(g2.alive) && all(g3.alive)
        h2 = Float64.(value_of.(cellsize(s2.grid)))
        h3 = Float64.(value_of.(cellsize(s3.grid)))
        pts2, _ = witness_lattice(p2.witness, h2, ncells(s2.grid))
        pts3, _ = witness_lattice(p3.witness, h3, ncells(s3.grid))
        @test length(pts3) == 6 * 2 * length(pts2)                 # six cells of two points along the invariant axis
        key(x, y) = (round(Int, 4 * x / h2[1]), round(Int, 4 * y / h2[2]))     # a lattice point of the plane, as integers
        slot2 = Dict(key(q...) => i for (i, q) in enumerate(pts2))
        @test length(slot2) == length(pts2)
        top = maximum(absmax, g2.gain)
        @test top > 0
        worst = 0.0
        for (j, q) in enumerate(pts3)
            i = slot2[key(q[σ[1]], q[σ[2]])]
            worst = max(worst, Float64(absmax(g3.gain[j] - g2.gain[i]) / top))
        end
        scale = p2.nsteps * sum(ncells(s2.grid)) * sqrt(prod(p2.species[1].ppc))
        worst /= base_eps(T) * scale
        record!("T-1262 axis $axis", T, worst, be)
        @test worst <= C
    end
end

@testset "T-1262 the witnesses of a 3D run invariant along one axis gain what the 2D bunch gains, along each of the three axes [A3, A7, A8, WP-093]" begin
    body_T1262(CPU())
end

# --- T-1263 ---------------------------------------------------------------------------------

"The plasma of T-950 (`plasma_step_3d`, 12 steps) with a witness bunch of centre `c` and width `wid`, injected at step 4."
function plasma_witness_3d(::Type{T}; c = (2.0, 1.5, 1.0), len = 1.0, wid = 1.0) where {T}
    p = plasma_step_3d(T; nsteps = 12)
    L = Float64.(value_of.(p.length))
    wb = WitnessBunch(; inject = 2.5 * Float64(value_of(p.dt)), region = ntuple(d -> (L[d] / 8, 7 * L[d] / 8), 3), ppc = (1, 2, 1),
                      momentum = (T(2), T(1) / 4, -T(1) / 2), center = T.(c), length = T(len), width = T(wid))
    return with_witness(p, wb)
end

# Test particles at N = 3: the bunch deposits nothing, so the run with it is the run without it,
# bit for bit (every particle array, field and history: one digest), the gains do not depend on
# the bunch's centre, length and width, which enter the weights alone, and a restart from a
# snapshot taken after the injection continues the bunch's bits (its arrays are in the evolved
# state), into the state that ran on and into a new one.
function body_T1263(be)
    for T in TYPES
        plain = plasma_step_3d(T; nsteps = 12)
        p1, p2 = plasma_witness_3d(T), plasma_witness_3d(T; c = (3.0, 2.0, 2.5), len = 0.5, wid = 2.0)
        finished(prob) = (st = EMState(prob, EMNumerics(); backend = be); initialize!(st, prob); run!(st, params(prob), prob.nsteps); st)
        s0, s1, s2 = finished(plain), finished(p1), finished(p2)
        @test state_digest_3d(s1) == state_digest_3d(s0)
        g1, g2 = witness_gains(s1, params(p1)), witness_gains(s2, params(p2))
        @test all(g1.alive) && any(!iszero, g1.gain)
        @test same_bits(g1.gain, g2.gain) && same_bits(g1.x, g2.x) && same_bits(g1.z, g2.z)
        @test !same_bits(g1.w, g2.w)
        # a restart after the injection
        st = EMState(p1, EMNumerics(); backend = be)
        initialize!(st, p1)
        pp = params(p1)
        foreach(k -> step!(st, pp, k), 1:7)
        snap = snapshot(st, 7)
        foreach(k -> step!(st, pp, k), 8:12)
        want = map(a -> bits(Array(a)), evolved(st))
        @test want == map(a -> bits(Array(a)), evolved(s1))
        for target in (st, EMState(p1, EMNumerics(); backend = be))
            restore!(target, snap)
            @test witness_alive(target) == capacity(witness_frame(target).species)
            foreach(k -> step!(target, pp, k), 8:12)
            @test map(a -> bits(Array(a)), evolved(target)) == want
        end
    end
end

@testset "T-1263 the 3D witness bunch changes nothing else in the run, its parameters reach the weights alone, and a restart continues its bits [A3, A4, WP-093]" begin
    body_T1263(CPU())
end

# --- T-1264 ---------------------------------------------------------------------------------

@testset "T-1264 what a witness bunch of each N takes, and its tunables at N = 3 [A3, A10, WP-093]" begin
    T = Float64
    b2 = WitnessBunch(; inject = 1, region = ((1, 2), (1, 2)), momentum = 1, center = (1, 1), length = 1)
    b3 = WitnessBunch(; inject = 1, region = ((1, 2), (1, 2), (0.5, 2)), momentum = 1, center = (1, 1, 1), length = 1)
    @test b2 isa WitnessBunch{Float64,2} && b2.ppc == (1, 1)
    @test b3 isa WitnessBunch{Float64,3} && b3.ppc == (1, 1, 1) && b3.width == Inf
    # one centre and one ppc per axis of the region; two or three axes
    @test_throws "a centre per axis" WitnessBunch(; inject = 1, region = ((1, 2), (1, 2), (1, 2)), momentum = 1, center = (1, 1), length = 1)
    @test_throws "ppc per axis" WitnessBunch(; inject = 1, region = ((1, 2), (1, 2), (1, 2)), momentum = 1, center = (1, 1, 1), length = 1, ppc = (1, 1))
    @test_throws "2D or 3D box" WitnessBunch(; inject = 1, region = ((1, 2),), momentum = 1, center = (1,), length = 1)
    @test_throws "lo < hi" WitnessBunch(; inject = 1, region = ((1, 2), (1, 2), (2, 1)), momentum = 1, center = (1, 1, 1), length = 1)
    # a problem takes the bunch of its N
    p3 = plasma_step_3d(T; nsteps = 12)
    p2 = plasma_step_2d(T)
    @test with_witness(p3, b3).witness isa WitnessBunch{T,3}
    @test_throws "of 3 axes" with_witness(p3, b2)
    @test_throws "of 2 axes" with_witness(p2, b3)
    @test_throws "N = 2 and 3 only" with_witness(beams_1d(T), b2)
    # the placement on the grid, host integers per axis. The box has 8 × 7 × 6 cells of about
    # 0.785 × 0.673 × 0.654, so the cells with centres in the region differ from axis to axis:
    # centres 1.18 and 1.96 along x, 1.01 to 3.03 along y, 0.98 to 2.95 along z
    bunch(; region = ((1, 2), (0.9, 3.2), (0.5, 3.4)), kw...) = WitnessBunch(; inject = 1, region, momentum = 1, center = (1, 1, 1), length = 1, kw...)
    pb = with_witness(p3, bunch())
    st = EMState(pb, EMNumerics())
    f = witness_frame(st)
    pts, cells = witness_lattice(pb.witness, Float64.(cellsize(st.grid)), ncells(st.grid))
    @test f.kinj == 4 && f.c0 == (2, 2, 2) && f.nc == (2, 4, 4) && f.ppc == (1, 1, 1)
    @test f.c0 == Tuple(map(first, cells)) && f.nc == Tuple(map(length, cells))
    @test capacity(f.species) == length(pts) == 2 * 4 * 4
    # a region that reaches one cell past the box along the third axis (centre 4.25 of 6 cells to 3.93) is refused by the state
    @test_throws "axis 3" EMState(with_witness(p3, bunch(; region = ((1, 2), (0.9, 3.2), (0.5, 4.3)))), EMNumerics())
    @test_throws "axis 2" EMState(with_witness(p3, bunch(; region = ((1, 2), (0.9, 5.1), (0.5, 3.4)))), EMNumerics())
    # a state takes a problem whose bunch differs in its parameters, and refuses another lattice, injection step, charge or mass
    @test check_setup(st, with_witness(p3, bunch(; momentum = 2, center = (2, 1, 3), length = 3, width = 2))) isa EMProblem
    for other in (bunch(; region = ((1, 2), (0.9, 3.2), (0.5, 2.7))), bunch(; region = ((1, 2), (0.9, 2.6), (0.5, 3.4))), bunch(; ppc = (1, 1, 2)),
                  bunch(; inject = 2), bunch(; charge = 1), bunch(; mass = 2))
        @test_throws ArgumentError check_setup(st, with_witness(p3, other))
        @test_throws ArgumentError initialize!(st, with_witness(p3, other))
    end
    # the tunables: the centre's third coordinate after the second, then the length and the width, in
    # the description and the parameters (every value another number, so that each has its place)
    prob = plasma_witness_3d(T; c = (2.0, 1.5, 1.25), len = 0.75, wid = 1.75)
    names = tunable_names(prob)
    at = findfirst(==(:witness_center_x), names)
    @test names[(at - 3):(at + 4)] == [:witness_momentum_x, :witness_momentum_y, :witness_momentum_z, :witness_center_x, :witness_center_y,
                                       :witness_center_z, :witness_length, :witness_width]
    @test tunable_names(params(prob)) == names
    v = tunables(prob)
    @test v[(at - 3):(at + 4)] == [2.0, 0.25, -0.5, 2.0, 1.5, 1.25, 0.75, 1.75]
    w = copy(v)
    w[(at + 2):(at + 4)] = [2.25, 0.5, 3.0]
    q = with_tunables(prob, w).witness
    @test q isa WitnessBunch{T,3} && q.center == (2.0, 1.5, 2.25) && q.length == 0.5 && q.width == 3.0 && q.region == prob.witness.region
    qp = with_tunables(params(prob), w).witness
    @test qp isa ParticleMeshElectromagnetic.WitnessParams{T,3} && qp.momentum == (2.0, 0.25, -0.5) && qp.center == (2.0, 1.5, 2.25) && qp.length == 0.5 && qp.width == 3.0
    @test with_tunables(prob, v).witness.length == 0.75 && with_tunables(params(prob), v).witness.width == 1.75
    @test tunables(with_tunables(prob, v)) == v
    # the 2D bunch keeps its seven
    n2 = tunable_names(with_witness(p2, b2))
    @test count(n -> startswith(String(n), "witness_"), n2) == 7 && !(:witness_center_z in n2)
end
