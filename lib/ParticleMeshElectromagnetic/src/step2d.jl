# The module at N = 2: the state, the initial field, the push and the kinetic energy (WP-024),
# and the charge-conserving current of the periodic 2D3V step (WP-023); the order of step.jl.

"The module's 2D field layout, [`field_layout`](@ref)`(Val(2))`."
const FIELD_LAYOUT_2D = field_layout(Val(2))
"The stagger of ρ at N = 2, the primal nodes on both axes."
const RHO_STAGGER_2D = (RHO_STAGGER, RHO_STAGGER)
"Particle slots per chunk of the 2D kinetic-energy reduction (the chunks fix its association order)."
const KINETIC_CHUNK = 256

"E and B of the 2D layout at offset `δ` of cell `c` (guards filled), each on its own stagger."
@inline function fields_at(f, c::NTuple{2,Int32}, δ::NTuple{2,T}, shape, g::Int32) where {T}
    L = FIELD_LAYOUT_2D
    E = (gather_at(f.Ex, c, δ, shape, L.Ex, g), gather_at(f.Ey, c, δ, shape, L.Ey, g), gather_at(f.Ez, c, δ, shape, L.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, L.Bx, g), gather_at(f.By, c, δ, shape, L.By, g), gather_at(f.Bz, c, δ, shape, L.Bz, g))
    return E, B
end

# Store the old position per axis (the extras of flux_extras(T, 2), for a current that needs it),
# push the momentum in Eⁿ, Bⁿ at xⁿ, and move the particle by Δt·u_d/γ along each axis.
@kernel function push_2d_kernel!(s, f, pusher, shape, g::Int32, qm_h, dt_dx::NTuple{2}, n::NTuple{2,Int32})
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = (s.cell[1][i], s.cell[2][i])
        δ = (s.offset[1][i], s.offset[2][i])
        s.extras.old_cell_1[i] = c[1]
        s.extras.old_cell_2[i] = c[2]
        s.extras.old_offset_1[i] = δ[1]
        s.extras.old_offset_2[i] = δ[2]
        E, B = fields_at(f, c, δ, shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
        γ = lorentz_factor(u)
        c′, δ′ = advance(c, δ, (u[1] / γ * dt_dx[1], u[2] / γ * dt_dx[2]), n)
        s.cell[1][i] = c′[1]
        s.cell[2][i] = c′[2]
        s.offset[1][i] = δ′[1]
        s.offset[2][i] = δ′[2]
    end
end

# The momentum update alone (no move): the half step back of initialize!, with qm_h = −(q/m)Δt/4.
@kernel function kick_2d_kernel!(s, f, pusher, shape, g::Int32, qm_h)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        E, B = fields_at(f, (s.cell[1][i], s.cell[2][i]), (s.offset[1][i], s.offset[2][i]), shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
    end
end

# One work item per chunk of `chunk` slots: the kinetic energy m·w·(γ − 1) of its alive
# particles, summed serially in slot order from zero, written to K[first + c − 1].
@kernel function kinetic_chunks_kernel!(K, s, first::Int32, cap::Int32, chunk::Int32)
    c = Int32(@index(Global, Linear))
    T = eltype(K)
    acc = zero(T)
    lo = (c - Int32(1)) * chunk + Int32(1)
    hi = min(c * chunk, cap)
    @inbounds for i in lo:hi
        if s.status[i] == ALIVE
            acc += KineticEnergy()(s, i)
        end
    end
    @inbounds K[first + c - Int32(1)] = acc
end

# Per-axis phase tables of the diagnostic mode m on n nodes: columns cos and sin of 2π·m·x/n at
# the Node() positions x = j − 1 and at the Center() positions x = j − 1/2. The angle over π,
# m·(2j − 2)/n and m·(2j − 1)/n, is reduced exactly in integers modulo 2 before the division.
@kernel function phase_table_kernel!(t, m::Int, n::Int)
    j = @index(Global, Linear)
    T = eltype(t)
    @inbounds begin
        sn, cs = sincospi(T(mod(m * (2j - 2), 2n)) / T(n))
        t[j, 1] = cs
        t[j, 2] = sn
        sn, cs = sincospi(T(mod(m * (2j - 1), 2n)) / T(n))
        t[j, 3] = cs
        t[j, 4] = sn
    end
end

"""
    EMState(problem::EMProblem{2}, numerics; backend = CPU())

The state of a 2D problem: the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, one species per [`SpeciesDescription`](@ref) at capacity
`prod(cells)·prod(ppc)` with the extras of `flux_extras(T, 2)` (the old position per axis, which
the push stores), the workspaces, the fields of [`field_layout`](@ref)`(Val(2))`, `ρ`, the
kinetic-energy chunks `K` (one entry per [`KINETIC_CHUNK`](@ref) slots of each species), the
histories for `problem.nsteps` steps (the keys of `HISTORY_KEYS_2D`) and in `aux`: the spectral workspace and the potential `φ` of the initial solve, the
diagnostic mode `kmode`, the unit vector `khat` along its wave vector, the phase tables of each
axis (written once), the diagnostics' per-row partial sums, the [`Antenna`](@ref)s of the
problem's lasers, placed on the grid, the absorbing layer's [`LayerFactors`](@ref) (`layer`,
`nothing` for a periodic box) and the time step `dt` that the antennas and the layer were made for.

With the current [`Esirkepov`](@ref)`()` each workspace also holds the per-cell patches of the
`Reference()` flux deposition (`Workspace(species, grid; flux = (shape, ρ's stagger))`, also
used by `Fast()` on an element type without atomics). Throws an `ArgumentError` unless the
current is `Esirkepov()` or [`NoCurrent`](@ref)`()` and unless `problem.dt` is below the
Courant limit of the numerics' solver ([`check_courant`](@ref)), and unless the numerics'
`absorber` is an [`AbsorbingLayer`](@ref) exactly when the problem's axis 2 is `:open` (the layer
also checks its fit on the grid).
"""
function EMState(problem::EMProblem{2,T}, numerics::EMNumerics; backend = CPU()) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 2D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    open_y = problem.boundaries[2] === :open
    open_y && !(numerics.absorber isa AbsorbingLayer) &&
        throw(ArgumentError("axis 2 of the problem is open: EMNumerics needs absorber = AbsorbingLayer(...), got $(numerics.absorber)"))
    !open_y && numerics.absorber !== nothing &&
        throw(ArgumentError("absorber = $(numerics.absorber) needs an open axis 2 in the problem (boundaries = (:periodic, :open))"))
    layer = open_y ? LayerFactors(numerics.absorber, grid, problem.dt; backend, workgroupsize = numerics.workgroupsize) : nothing
    species = map(problem.species) do d
        Species(backend, T, prod(n) * prod(d.ppc); q = d.charge, m = d.mass, N = 2, extras = flux_extras(T, 2))
    end
    flux = numerics.current isa Esirkepov ? (numerics.shape, RHO_STAGGER_2D) : nothing
    ws = map(s -> Workspace(s, grid; flux), species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), species; init = 0)
    hist = NamedTuple{HISTORY_KEYS_2D}(ntuple(_ -> KernelAbstractions.zeros(backend, T, problem.nsteps), length(HISTORY_KEYS_2D)))
    tables = ntuple(Val(2)) do d
        t = KernelAbstractions.allocate(backend, T, n[d], 4)
        launcher(phase_table_kernel!, backend, wgs)(t, problem.kmode[d], n[d]; ndrange = n[d])
        t
    end
    k = ntuple(d -> 2 * T(π) * problem.kmode[d] / problem.length[d], Val(2))
    knorm = sqrt(k[1] * k[1] + k[2] * k[2])
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           kmode = problem.kmode, khat = (k[1] / knorm, k[2] / knorm), tables = tables,
           partials = KernelAbstractions.zeros(backend, T, DIAG_ROWS, n[1]),
           antennas = map(l -> Antenna(l, grid, problem.dt), problem.lasers), layer = layer, dt = problem.dt)
    return EMState(numerics, grid, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), hist, aux)
end

"The kinetic energy of every chunk of particle slots at the current momenta, into `st.K` (deterministic, no deposition)."
function kinetic!(st::EMState{2})
    wgs = st.numerics.workgroupsize
    first = 1
    for s in st.species
        nc = cld(capacity(s), KINETIC_CHUNK)
        launcher(kinetic_chunks_kernel!, get_backend(s), wgs)(st.K, s, Int32(first), Int32(capacity(s)), Int32(KINETIC_CHUNK);
                                                             ndrange = nc)
        first += nc
    end
    return st
end

"""
    initialize!(state::EMState{2}, problem::EMProblem{2}; momenta_at = :t0) -> state

The 2D state at t = 0, as in 1D: each species loaded by the engine's 2D quiet start (weights
`density·(1 + amplitude·cos(k·x))/prod(ppc)` with `k_d = 2π·mode_d/L_d`, so the parameters enter
through the weights, A3), the velocities converted to momenta; the fields zero; Ex and Ey from
the deposited charge by the engine's periodic solve, `φ` from `inverse_laplacian_periodic!` of ρ
and `E = G` from `gradient_periodic!`, so that the backward-difference divergence of E is
ρ − mean(ρ) to roundoff (the mean is the immobile background); the background fields `E0`, `B0`
added where nonzero; then the momenta at t = −Δt/2 (half-step centring, A2) and the kinetic
energy there: with `momenta_at = :t0` (the default) the loaded momenta are u⁰ and are moved
half a step back by the pusher with −Δt/2 in E⁰, B⁰; with `:t_minus_half` they are taken as
u⁻¹ᐟ², the convention of Smilei and PIConGPU, and kept as loaded.
"""
function initialize!(st::EMState{2}, problem::EMProblem{2}; momenta_at::Symbol = :t0)
    check_momenta_at(momenta_at)
    (; numerics, grid, aux) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, d) in zip(st.species, problem.species)
        fill!(s.status, ParticleMeshCore.FREE)
        k = ntuple(a -> 2 * T(π) * d.mode[a] / problem.length[a], Val(2))
        load_quiet!(s, grid, CosineProfile(), (d.density, d.amplitude, k[1], k[2]), d.ppc; vdrift = d.drift,
                    vth = d.thermal, seed = problem.seed, stream = d.stream, workgroupsize = wgs)
        launcher(velocity_to_momentum_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    foreach(a -> fill!(a, zero(T)), st.fields)
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(st.ρ, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(deposit_mode(st, s), st.ρ, s, numerics.shape, RHO_STAGGER_2D, grid, ws; workgroupsize = wgs)
    end
    fold_guards!(st.ρ, grid; workgroupsize = wgs)
    inverse_laplacian_periodic!(aux.φ, st.ρ, grid, aux.spectral; workgroupsize = wgs)
    gradient_periodic!((st.fields.Ex, st.fields.Ey), aux.φ, grid; workgroupsize = wgs)
    add_background!(st.fields, problem.E0, problem.B0)
    if momenta_at === :t0
        g = Int32(guards(grid))
        for s in st.species
            launcher(kick_2d_kernel!, get_backend(s), wgs)(s, st.fields, numerics.pusher, numerics.shape, g,
                                                           -s.q / s.m * problem.dt / 4; ndrange = capacity(s))
        end
    end
    kinetic!(st)
    return st
end

# The current of the step, by the numerics' choice: none, or the charge-conserving one.
current!(::NoCurrent, st::EMState{2}, p::EMParams) = st

# Sort where the deposition mode needs it (the old positions in the extras follow the
# particles), zero Jx, Jy, Jz, add every species' Esirkepov current from its old and new
# positions (`q·w·u_z/γ` for Jz), fold the guards and apply the numerics' binomial filter
# (through the scratch field ρ, which only initialize! reads otherwise).
function current!(::Esirkepov, st::EMState{2}, p::EMParams)
    (; numerics, grid, fields) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(fields.Jx, zero(T))
    fill!(fields.Jy, zero(T))
    fill!(fields.Jz, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER_2D, grid, ws;
                      dt = p.dt, quantity_z = CurrentZ(), workgroupsize = wgs)
    end
    fold_guards!(fields.Jx, grid; workgroupsize = wgs)
    fold_guards!(fields.Jy, grid; workgroupsize = wgs)
    fold_guards!(fields.Jz, grid; workgroupsize = wgs)
    filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs)
    return st
end

"""
    step!(state::EMState{2}, params, k) -> state

Step `k` at N = 2, the order of the 1D step: the diagnostics of Eⁿ, Bⁿ and of the kinetic
energy at n − 1/2 (parallel reductions in a fixed order); per species, store the old position,
push the momentum in Eⁿ, Bⁿ gathered at xⁿ on each component's stagger and move the particle
along both axes; the current of the numerics' choice (for [`Esirkepov`](@ref)`()`: sort where
the mode needs it, zero J, the engine's `deposit_flux!` per species, fold the guards, then the
numerics' [`BinomialFilter`](@ref); for `NoCurrent()` J stays zero); the kinetic energy at
n + 1/2; the mesh solve (Yee: B half step, E full step with J, B half step), with the antennas'
total-field/scattered-field corrections between the sub-steps (time `(k − 1)·Δt` at the start of
the step). With an open axis 2 the particles that crossed the box edge are retired after the
current of their last move is deposited, and the absorbing layer damps the fields around each
sub-step (WP-027). With `Esirkepov()` and no filter passes, Gauss's law then holds to roundoff
after every step (T-360), off the antenna columns with lasers (T-406), and off the layer rows
with an open boundary (T-451); with passes its filtered form does (T-441). `k` must lie within
the allocated steps, `params.kmode` must be the state's diagnostic mode and `params.dt` the
problem's time step, for which the antennas and the layer were made (an `ArgumentError`
otherwise).
"""
function step!(st::EMState{2}, p::EMParams, k::Integer)
    (; numerics, grid, fields) = st
    1 <= k <= length(st.hist.K) || throw(ArgumentError("step $k outside the $(length(st.hist.K)) steps the state was allocated for"))
    p.kmode == st.aux.kmode || throw(ArgumentError("params.kmode = $(p.kmode) is not the state's diagnostic mode $(st.aux.kmode)"))
    wgs = numerics.workgroupsize
    n = Int32.(ncells(grid))
    g = Int32(guards(grid))
    dx = cellsize(grid)
    p.dt == st.aux.dt || throw(ArgumentError("params.dt = $(p.dt) is not the time step $(st.aux.dt) the state's antennas and layer were made for"))
    record_diagnostics!(st, k)
    for s in st.species
        launcher(push_2d_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * p.dt / 2,
                                                       (p.dt / dx[1], p.dt / dx[2]), n; ndrange = capacity(s))
    end
    current!(numerics.current, st, p)
    retire!(st.aux.layer, st)
    kinetic!(st)
    solve_fields!(numerics.solver, fields, grid, p.dt, st.aux.antennas, (k - 1) * p.dt, st.aux.layer; workgroupsize = wgs)
    return st
end
