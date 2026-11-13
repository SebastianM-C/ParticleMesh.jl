# WP-057: the one step body over N, T-755 to T-757. `initialize!` and `step!` are one method each for
# N = 1 and N = 2, with the 1D case as the case of one resolved axis: the push and the kick are one
# kernel each, held here against the 1D and the 2D push written out with the engine's calls, particle
# by particle. After em2d_cases.jl (`plasma_step_2d`, `state_2d`), modifiers.jl (`random_fields`) and
# params.jl.

using ParticleMeshElectromagnetic: FIELD_LAYOUT_2D, check_aux_params, check_aux_setup, current!, kinetic!, initial_field!, refresh_caches!, window_frame, zero_x_edges!
using ParticleMeshCore: gather_at, advance, launcher, ALIVE
using KernelAbstractions: KernelAbstractions, @kernel, @index

# --- T-755 -----------------------------------------------------------------------------------------

"A 1D and a 2D state of plasma with their parameters and problems."
function states_1d_2d()
    st1, p1, prob1, _ = case_state("weibel", Float64; ppc = 4, t_end = 1)
    prob2 = plasma_step_2d(Float64)
    return (st1, p1, prob1), (state_2d(prob2; current = Esirkepov()), params(prob2), prob2)
end

@testset "T-755 step! and initialize! are one method each for N = 1 and N = 2, and N is a dispatch in the kinetic energy and the initial field [A2, A9, WP-057]" begin
    (st1, p1, prob1), (st2, p2, prob2) = states_1d_2d()
    @test st1 isa EMState{1} && st2 isa EMState{2}
    one_method(f, a1, a2) = which(f, map(typeof, a1)) === which(f, map(typeof, a2))
    @test one_method(step!, (st1, p1, 1), (st2, p2, 1))
    @test length(methods(step!)) == 1
    @test one_method(initialize!, (st1, p1), (st2, p2)) && one_method(initialize!, (st1, prob1), (st2, prob2))
    @test one_method(run!, (st1, p1, 1), (st2, p2, 1))
    @test one_method(current!, (Esirkepov(), st1), (Esirkepov(), st2))
    @test one_method(check_params, (st1, p1), (st2, p2)) && one_method(check_aux_params, (st1, p1), (st2, p2))
    @test one_method(check_setup, (st1, prob1), (st2, prob2)) && one_method(check_aux_setup, (st1, prob1, identity), (st2, prob2, identity))
    @test one_method(reset_convolution!, (st1,), (st2,)) && one_method(evolved, (st1,), (st2,))
    @test one_method(solve_fields!, (Yee(), st1.fields, st1.grid, st1.dt, st1.modifiers, p1, 1), (Yee(), st2.fields, st2.grid, st2.dt, st2.modifiers, p2, 1))
    # the two that stay per N, because each gives other bits than its counterpart
    @test !one_method(kinetic!, (st1,), (st2,)) && !one_method(initial_field!, (st1,), (st2,))
    @test ndims(st1.K) == 1 && size(st1.K) == size(st1.ρ)                                   # a deposited density
    @test length(st2.K) == sum(s -> cld(capacity(s), 256), st2.species)                      # chunk sums
    @test !one_method(refresh_caches!, (st1,), (st2,))                                       # K's: the 1D deposit reads the sort
    # the one step is the run: both dimensions step, and what an N = 1 problem cannot have is refused there too
    run!(st1, p1, 3)
    run!(st2, p2, 3)
    @test all(isfinite, Array(st1.fields.Ex)) && all(isfinite, Array(st2.fields.Ex))
    # the observer comes first in the one step: it sees the particles where the last step left them
    # (the default recorder reads the fields and `K`, which the push leaves, so it cannot tell)
    for (st, p) in ((st1, p1), (st2, p2))
        s = first(st.species)
        before = (Array(s.offset[1]), Array(s.u[1]))
        seen = Ref{Any}(nothing)
        step!(st, p, 4; observer = (x, k) -> (seen[] = (k, Array(first(x.species).offset[1]), Array(first(x.species).u[1]))))
        @test seen[][1] == 4 && same_bits(seen[][2], before[1]) && same_bits(seen[][3], before[2])
        @test !same_bits(Array(s.offset[1]), before[1])
    end
    laser = GaussianLaser(; a0 = 1.0, waist = 4.0, duration = 2.0, delay = 12.0, focus = 12.0, position = 1.0, center = 1.0)
    @test_throws "implemented at N = 2 only" check_params(st1, EMParams(p1; lasers = (laser,)))
    @test_throws "0 antenna planes in the state" check_params(st2, EMParams(p2; lasers = (laser,)))
    @test_throws ArgumentError step!(st1, EMParams(p1; lasers = (laser,)), 5)
    @test_throws "witness bunch" check_params(st1, EMParams(p1; witness = WitnessParams{Float64}((1.0, 0.0, 0.0), (1.0, 1.0), 1.0, 1.0)))
    @test_throws "moving window only" check_params(st1, EMParams(p1; profile = DensityRamp(; vacuum = 1.0, ramp = 1.0)))
    @test_throws ArgumentError initialize!(st1, prob2)
    @test_throws ArgumentError initialize!(st2, prob1)
    # a Float64 description with a mass that is not exact in the problem's element type is the state's own
    # problem through the one `initialize!` and `step!`, at N = 1 as at N = 2 (T-733 holds N = 2)
    for S in (Float32, D3)
        ion = SpeciesDescription{Float64}(:ion, 0.3, 1836.15267343, 0.9, 0.05, 1, 16, (0.1, 0.0, 0.0), (0.05, 0.0, 0.0), 1)
        heavy = EMProblem("inexact-mass-1d", 2 * S(π), 32, S(15) / 100, 3, (ion,), 1, one(S))
        s = EMState(heavy, EMNumerics())
        @test s.species[1].m == S(1836.15267343) && value_of(s.species[1].m) != 1836.15267343
        @test check_setup(s, heavy) === heavy
        initialize!(s, heavy)
        run!(s, params(heavy), heavy.nsteps)
        @test all(isfinite, value_of.(Array(s.fields.Ex))) && any(!iszero, value_of.(Array(s.fields.Ex)))
    end
end

# --- T-756 -----------------------------------------------------------------------------------------

host(s) = (cell = map(Array, s.cell), offset = map(Array, s.offset), u = map(Array, s.u), status = Array(s.status), extras = map(Array, s.extras))

"The fields of `st` set to random values, guards filled, and their host copies."
function randomize_fields!(st, seed)
    T = eltype(st)
    r = random_fields(get_backend(st.ρ), T, st.grid, seed)
    foreach(copyto!, values(st.fields), values(r))
    return map(Array, st.fields)
end

"""
One particle's momentum update and move at N = 1 written out with the engine's scalar calls (the
1D kernel before WP-057): the gathers on the bare staggers of `FIELD_LAYOUT`, the scalar `advance`.
`move = false` is the kick.
"""
function pushed_1d(f, c::Int32, δ, u, pusher, shape, g, qm_h, dt_dx, n; move = true)
    L = FIELD_LAYOUT
    E = (gather_at(f.Ex, c, δ, shape, L.Ex, g), gather_at(f.Ey, c, δ, shape, L.Ey, g), gather_at(f.Ez, c, δ, shape, L.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, L.Bx, g), gather_at(f.By, c, δ, shape, L.By, g), gather_at(f.Bz, c, δ, shape, L.Bz, g))
    v = push_momentum(pusher, u, E, B, qm_h)
    move || return v, (c,), (δ,)
    c′, δ′ = advance(c, δ, v[1] / lorentz_factor(v) * dt_dx, n)
    return v, (c′,), (δ′,)
end

"The same at N = 2 (the 2D kernel before WP-057): the gathers on `FIELD_LAYOUT_2D`, `advance` per axis."
function pushed_2d(f, c::NTuple{2,Int32}, δ, u, pusher, shape, g, qm_h, dt_dx, n; move = true)
    L = FIELD_LAYOUT_2D
    E = (gather_at(f.Ex, c, δ, shape, L.Ex, g), gather_at(f.Ey, c, δ, shape, L.Ey, g), gather_at(f.Ez, c, δ, shape, L.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, L.Bx, g), gather_at(f.By, c, δ, shape, L.By, g), gather_at(f.Bz, c, δ, shape, L.Bz, g))
    v = push_momentum(pusher, u, E, B, qm_h)
    move || return v, c, δ
    γ = lorentz_factor(v)
    c′, δ′ = advance(c, δ, (v[1] / γ * dt_dx[1], v[2] / γ * dt_dx[2]), n)
    return v, c′, δ′
end

same_value(a::T, b::T) where {T} = a === b
same_value(a, b) = false

function body_T756(be)
    for T in TYPES, N in (1, 2), pusher in (Boris(), HigueraCary())
        if N == 1
            prob = problem_from_case(load_case("weibel"), T; ppc = 8, t_end = 1)
            st = EMState(prob, EMNumerics(; pusher); backend = be)
            initialize!(st, prob)
        else
            prob = plasma_step_2d(T)
            st = state_2d(prob; current = Esirkepov(), pusher, be)
        end
        (; grid, numerics, dt) = st
        shape = numerics.shape
        g = Int32(guards(grid))
        n = Int32.(grid.n)
        dt_dx = map(h -> dt / h, grid.dx)
        for move in (true, false)
            f = randomize_fields!(st, 90 + N)
            s = first(st.species)
            before = host(s)
            qm_h = move ? s.q / s.m * dt / 2 : -s.q / s.m * dt / 4
            if move
                launcher(push_kernel!, get_backend(s), nothing)(s, st.fields, pusher, shape, g, qm_h, dt_dx, n; ndrange = capacity(s))
            else
                launcher(kick_kernel!, get_backend(s), nothing)(s, st.fields, pusher, shape, g, qm_h; ndrange = capacity(s))
            end
            KernelAbstractions.synchronize(get_backend(s))
            after = host(s)
            alive = findall(==(ALIVE), before.status)
            @test length(alive) > 100 && after.status == before.status
            ok = true
            moved = 0
            for i in alive
                u = (before.u[1][i], before.u[2][i], before.u[3][i])
                c = ntuple(d -> before.cell[d][i], N)
                δ = ntuple(d -> before.offset[d][i], N)
                v, c′, δ′ = N == 1 ? pushed_1d(f, c[1], δ[1], u, pusher, shape, g, qm_h, dt_dx[1], n[1]; move) :
                                     pushed_2d(f, c, δ, u, pusher, shape, g, qm_h, dt_dx, n; move)
                ok &= all(d -> same_value(after.u[d][i], v[d]), 1:3)
                ok &= all(d -> after.cell[d][i] === c′[d] && same_value(after.offset[d][i], δ′[d]), 1:N)
                moved += δ′ != δ
                if move                                   # the old position, where deposit_flux! reads it
                    x = after.extras
                    ok &= N == 1 ? (x.old_cell[i] === c[1] && same_value(x.old_offset[i], δ[1])) :
                                   (x.old_cell_1[i] === c[1] && x.old_cell_2[i] === c[2] && same_value(x.old_offset_1[i], δ[1]) && same_value(x.old_offset_2[i], δ[2]))
                end
            end
            @test ok
            @test move ? moved > 0 : moved == 0
            @test any(i -> !same_value(after.u[1][i], before.u[1][i]), alive)
        end
    end
end

# The 1D push and kick kernels as they were before WP-057 (commit 0eeca89, `step.jl`), for the backend
# the package's kernels run on: a device compiler may contract a multiplication and an addition
# into one rounding, so the host's arithmetic above is no reference there, and the kernels are.
@inline function fields_1d(f, c::Int32, δ, shape, g::Int32)
    L = FIELD_LAYOUT
    E = (gather_at(f.Ex, c, δ, shape, L.Ex, g), gather_at(f.Ey, c, δ, shape, L.Ey, g), gather_at(f.Ez, c, δ, shape, L.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, L.Bx, g), gather_at(f.By, c, δ, shape, L.By, g), gather_at(f.Bz, c, δ, shape, L.Bz, g))
    return E, B
end
@kernel function push_1d_kernel!(s, f, pusher, shape, g::Int32, qm_h, dt_dx, n::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = s.cell[1][i]
        δ = s.offset[1][i]
        s.extras.old_cell[i] = c
        s.extras.old_offset[i] = δ
        E, B = fields_1d(f, c, δ, shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
        c′, δ′ = advance(c, δ, u[1] / lorentz_factor(u) * dt_dx, n)
        s.cell[1][i] = c′
        s.offset[1][i] = δ′
    end
end
@kernel function kick_1d_kernel!(s, f, pusher, shape, g::Int32, qm_h)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        E, B = fields_1d(f, s.cell[1][i], s.offset[1][i], shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
    end
end

"""
On the backend `be`: the package's push and kick at N = 1 against the 1D kernels of before WP-057,
on two states of the same load and the same fields, over several pushes (so that the momenta and
the offsets leave their loaded values), bit for bit in every array, for each element type, both
shapes and two pushers.
"""
function body_T756_kernels(be)
    for T in TYPES, shape in (Quadratic, Linear), pusher in (Boris(), HigueraCary())
        mk() = (prob = problem_from_case(load_case("landau"), T; ppc = 16, t_end = 1, amplitude = seeded(T, 1 // 100));
                st = EMState(prob, EMNumerics(; pusher, shape); backend = be); initialize!(st, prob); st)
        a, b = mk(), mk()
        f = random_fields(be, T, a.grid, 95)
        foreach(st -> foreach(copyto!, values(st.fields), values(f)), (a, b))
        g, n, dt = Int32(guards(a.grid)), Int32(ncells(a.grid)), a.dt
        dt_dx = dt / cellsize(a.grid)
        sa, sb = first(a.species), first(b.species)
        start = host(sa)
        qk = -sa.q / sa.m * dt / 4
        launcher(kick_kernel!, be, nothing)(sa, a.fields, pusher, shape, g, qk; ndrange = capacity(sa))
        launcher(kick_1d_kernel!, be, nothing)(sb, b.fields, pusher, shape, g, qk; ndrange = capacity(sb))
        KernelAbstractions.synchronize(be)
        same_particles(x, y) = all(map(same_bits, (x.cell..., x.offset..., x.u..., values(x.extras)...), (y.cell..., y.offset..., y.u..., values(y.extras)...)))
        @test same_particles(host(sa), host(sb))
        @test !same_bits(host(sa).u[1], start.u[1])
        qp = sa.q / sa.m * dt / 2
        for _ in 1:8
            launcher(push_kernel!, be, nothing)(sa, a.fields, pusher, shape, g, qp, (dt_dx,), (n,); ndrange = capacity(sa))
            launcher(push_1d_kernel!, be, nothing)(sb, b.fields, pusher, shape, g, qp, dt_dx, n; ndrange = capacity(sb))
        end
        KernelAbstractions.synchronize(be)
        @test same_particles(host(sa), host(sb))
        @test !same_bits(host(sa).offset[1], start.offset[1]) && !same_bits(host(sa).u[2], start.u[2])
    end
end

@testset "T-756 the push and the kick, one kernel each over N, are the 1D and the 2D push written out with the engine's calls, bit for bit, and at N = 1 the 1D kernels of before on their backend [A2, A4, A8, WP-057]" begin
    body_T756(CPU())
    body_T756_kernels(CPU())
end

# --- T-757 -----------------------------------------------------------------------------------------

"""
One step of a periodic state written out from the pieces, in the order of the 1D step before
WP-057 (`N = 1`) or of the periodic 2D step: push, sort, current, kinetic energy, the three Yee
sub-steps. What the one `step!` has more is `nothing` for these states.
"""
function step_written_out!(st::EMState{N}, k) where {N}
    (; numerics, grid, fields, dt) = st
    T = eltype(st)
    g = Int32(guards(grid))
    observe!(st.recorder, st, k)
    for s in st.species
        launcher(push_kernel!, get_backend(s), nothing)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2, map(h -> dt / h, grid.dx),
                                                        Int32.(grid.n); ndrange = capacity(s))
    end
    for (s, ws) in zip(st.species, st.ws)
        ParticleMeshElectromagnetic.sorts(st, s) && sort_by_cell!(s, grid, ws)
    end
    foreach(J -> fill!(J, zero(T)), (fields.Jx, fields.Jy, fields.Jz))
    for (s, ws) in zip(st.species, st.ws)
        mode = ParticleMeshElectromagnetic.deposit_mode(st, s)
        if N == 1
            deposit_flux!(mode, fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER, grid, ws; dt, quantity_y = CurrentY(), quantity_z = CurrentZ())
        else
            deposit_flux!(mode, fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, (RHO_STAGGER, RHO_STAGGER), grid, ws; dt, quantity_z = CurrentZ())
        end
    end
    foreach(J -> fold_guards!(J, grid), (fields.Jx, fields.Jy, fields.Jz))
    filter_current!(numerics.filter, fields, grid, st.ρ)
    kinetic!(st)
    yee_b_half!(fields, grid, dt / 2)
    yee_e!(fields, grid, dt)
    yee_b_half!(fields, grid, dt / 2)
    return st
end

function body_T757(be)
    noise = Float64[]
    for T in TYPES, mode in (Reference(), Fast()), (N, passes) in ((1, 0), (1, 2), (2, 0), (2, 1))
        mk() = if N == 1
            case = load_case("weibel")
            case["numerics"]["filter_passes"] = passes
            prob = problem_from_case(case, T; ppc = 8, t_end = 1)
            st = EMState(prob, numerics_from_case(case; mode); backend = be)
            initialize!(st, prob)
            st, params(prob)
        else
            prob = plasma_step_2d(T)
            st = EMState(prob, EMNumerics(; mode, current = Esirkepov(), filter = BinomialFilter(passes)); backend = be)
            initialize!(st, prob)
            st, params(prob)
        end
        (a, p), (b, _) = mk(), mk()
        for k in 1:4
            step!(a, p, k)
            step_written_out!(b, k)
        end
        # Fast's atomic deposition is ordered on one CPU thread, and not on a device or with more threads
        exact = mode isa Reference || (be isa CPU && Threads.nthreads() == 1)
        if exact
            @test isempty(differences(a, b, 0))
        else
            # the two runs then differ by the order of the atomic sums: roundoff of the plasma's
            # own scales, not of the fields, which a case may leave at roundoff themselves (the
            # Weibel case's beams cancel in the charge and in the current, so its first steps'
            # fields are the deposition's noise and nothing else)
            # (measured: 5 eps on the CPU at 4 threads, 4 eps on CUDA, over 12 runs each)
            worst = step_distance(a, b)
            push!(noise, worst / eps(basetype(T)))
            @test worst <= 256 * eps(basetype(T))
        end
    end
    isempty(noise) || @info "T-757 unordered Fast deposition: largest difference of two runs of one step order over the plasma's scales, in eps" backend = nameof(typeof(be)) worst = maximum(noise) runs = length(noise)
    return nothing
end

"""
The largest difference between two states of the same slot order, over the plasma's own scales:
the momenta over the largest momentum, the positions in cells, the current over ρ₀, the summed
charge density of the species taken as positive (ρ₀·c is the current of the whole plasma at c),
and E and B over ρ₀·Δ, the field that ρ₀ makes over the smallest cell. `Inf` if the alive slots
differ.
"""
function step_distance(a::EMState{N}, b::EMState{N}) where {N}
    hv(x) = Float64.(value_of.(Array(x)))
    alive(s) = Array(s.status) .== ALIVE
    ρ0 = sum(s -> abs(Float64(value_of(s.q))) * sum(hv(s.w)[alive(s)]), a.species) / prod(a.grid.n)
    Δ = Float64(value_of(minimum(a.grid.dx)))
    umax = maximum(s -> maximum(c -> maximum(abs, hv(c)[alive(s)]), s.u), a.species)
    worst = 0.0
    for (names, scale) in (((:Ex, :Ey, :Ez), ρ0 * Δ), ((:Bx, :By, :Bz), ρ0 * Δ), ((:Jx, :Jy, :Jz), ρ0)), n in names
        worst = max(worst, maximum(abs, hv(getfield(a.fields, n)) .- hv(getfield(b.fields, n))) / scale)
    end
    for (sa, sb) in zip(a.species, b.species)
        m = alive(sa)
        m == alive(sb) || return Inf
        for d in 1:3
            worst = max(worst, maximum(abs, (hv(sa.u[d]) .- hv(sb.u[d]))[m]) / umax)
        end
        for d in 1:N
            δ = abs.((Array(sa.cell[d]) .- Array(sb.cell[d]) .+ hv(sa.offset[d]) .- hv(sb.offset[d]))[m])
            worst = max(worst, maximum(min.(δ, a.grid.n[d] .- δ)))          # the nearer way round the periodic axis
        end
    end
    return worst
end

"""
The current of a moving-window state with filter passes, against its pieces written out: the
x-guards of the deposited current dropped before the fold, and the filter with the window's edges
(zero x-guards in every pass). A filter that wraps along x gives another current.
"""
function body_T757_window(be)
    for T in TYPES
        tenth = one(T) / 10
        prob = window_plasma(T; cells = (24, 8), nsteps = 30, thermal = (tenth, tenth, tenth), drift = (2tenth, zero(T), -tenth))
        # twelve steps in, then the push and the retirement at the x edges of the next step: the state as `current!` finds it
        function mk()
            st = EMState(prob, EMNumerics(; filter = BinomialFilter(2), headroom = 1 / 2); backend = be)
            initialize!(st, prob)
            run!(st, params(prob), 12; observer = nothing)
            (; numerics, grid, dt) = st
            for s in st.species
                launcher(push_kernel!, get_backend(s), nothing)(s, st.fields, numerics.pusher, numerics.shape, Int32(guards(grid)), s.q / s.m * dt / 2,
                                                                map(h -> dt / h, grid.dx), Int32.(grid.n); ndrange = capacity(s))
            end
            ParticleMeshElectromagnetic.retire_x!(window_frame(st), st)
            return st
        end
        function written_out!(st; edges = window_frame(st))
            (; numerics, grid, fields) = st
            w = window_frame(st)
            for (s, ws) in zip(st.species, st.ws)
                sort_by_cell!(s, grid, ws)
            end
            foreach(J -> fill!(J, zero(T)), (fields.Jx, fields.Jy, fields.Jz))
            for (s, ws) in zip(st.species, st.ws)
                deposit_flux!(ParticleMeshElectromagnetic.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, (RHO_STAGGER, RHO_STAGGER), grid, ws;
                              dt = st.dt, quantity_z = CurrentZ())
            end
            for J in (fields.Jx, fields.Jy, fields.Jz)
                zero_x_edges!(w, J)
                fold_guards!(J, grid)
            end
            filter_current!(numerics.filter, fields, grid, st.ρ; edges)
            return map(J -> bits(Array(J)), (fields.Jx, fields.Jy, fields.Jz))
        end
        a, b, c = mk(), mk(), mk()
        current!(Esirkepov(), a)
        ja = map(J -> bits(Array(J)), (a.fields.Jx, a.fields.Jy, a.fields.Jz))
        @test ja == written_out!(b)
        @test ja != written_out!(c; edges = nothing)          # a filter that wraps along x gives another current
    end
end

@testset "T-757 the one step of a periodic state is the 1D step and the 2D step written out from their pieces, bit for bit in Reference mode, and a window's current its pieces [A2, A4, WP-057]" begin
    body_T757(CPU())
    body_T757_window(CPU())
end
