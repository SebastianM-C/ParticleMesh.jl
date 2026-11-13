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

"""
    EMState(problem::EMProblem{2}, numerics; backend = CPU())

The state of a 2D problem ([`EMState`](@ref)): the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, the time step `problem.dt`, one species per
[`SpeciesDescription`](@ref) at capacity `prod(cells)·prod(ppc)` with the extras of
`flux_extras(T, 2)` (the old position per axis, which the push stores), the workspaces, the fields
of [`field_layout`](@ref)`(Val(2))`, `ρ`, the kinetic-energy chunks `K` (one entry per
[`KINETIC_CHUNK`](@ref) slots of each species; none without species), the default [`HistoryRecorder`](@ref) for
`problem.kmode` and `problem.nsteps` steps, the field modifiers of the mesh solve
([`field_modifiers`](@ref): the absorber placed on the grid, the absorbing layer's
[`LayerFactors`](@ref) or the PML's [`PMLFields`](@ref); the back edge's [`BackPMLFields`](@ref);
the [`AntennaPlanes`](@ref) of the problem's lasers, where each plane lies on the grid, not the
laser; the window's [`WindowEdges`](@ref)), and in `aux`: the spectral workspace and the potential
`φ` of the initial solve, the moving window's [`WindowFrame`](@ref) (`window`, `nothing` without one; its
species buffers then hold `(1 + numerics.headroom)` times the window's particles) and the
[`WitnessFrame`](@ref) of a witness bunch (`witness`, or `nothing`).

With the current [`Esirkepov`](@ref)`()` each workspace also holds the per-cell patches of the
`Reference()` flux deposition (`Workspace(species, grid; flux = (shape, ρ's stagger))`, also
used by `Fast()` on an element type without atomics). Throws an `ArgumentError` unless the
current is `Esirkepov()` or [`NoCurrent`](@ref)`()` and unless `problem.dt` is below the
Courant limit of the numerics' solver ([`check_courant`](@ref)), and unless the numerics'
`absorber` is an [`AbsorbingLayer`](@ref) or a [`PML`](@ref) exactly when the problem's axis 2 is
`:open` (the absorber also checks its fit on the grid), and unless the numerics' `filter` has zero
passes with an open axis 2 (the filter wraps the current across the open edge, and no test checks
the two together yet), and unless the problem's own parameters fit the planes and the window
([`check_params`](@ref)'s conditions: with a window, every pulse over before the window passes its
antenna).
"""
function EMState(problem::EMProblem{2,T}, numerics::EMNumerics; backend = CPU()) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 2D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    open_y = problem.boundaries[2] === :open
    open_y && !(numerics.absorber isa Union{AbsorbingLayer,PML}) &&
        throw(ArgumentError("axis 2 of the problem is open: EMNumerics needs absorber = AbsorbingLayer(...) or PML(...), got $(numerics.absorber)"))
    !open_y && numerics.absorber !== nothing &&
        throw(ArgumentError("absorber = $(numerics.absorber) needs an open axis 2 in the problem (boundaries = (:periodic, :open))"))
    open_y && numerics.filter.passes > 0 &&
        throw(ArgumentError("filter = $(numerics.filter) with absorber = $(numerics.absorber): the binomial filter is not yet tested with an open boundary (it wraps J across the open edge), so an open axis 2 takes BinomialFilter(0)"))
    layer = open_y ? place_absorber(numerics.absorber, grid, problem.dt; backend, workgroupsize = numerics.workgroupsize) : nothing
    numerics.back_absorber === nothing || numerics.back_absorber isa PML ||
        throw(ArgumentError("back_absorber = $(numerics.back_absorber): the window's back edge takes a PML(...) or nothing"))
    problem.window === nothing && numerics.back_absorber !== nothing &&
        throw(ArgumentError("back_absorber = $(numerics.back_absorber) needs a moving window (boundaries = (:window, ·))"))
    window = problem.window === nothing ? nothing : WindowFrame(problem, grid, backend, numerics.shape, numerics.filter.passes)
    back = numerics.back_absorber === nothing ? nothing : BackPMLFields(numerics.back_absorber, grid, problem.dt; backend, workgroupsize = wgs)
    planes = map(l -> AntennaPlane(l.position, grid, problem.dt), problem.lasers)
    window === nothing || check_window_antennas(window, back, map(Antenna, params(problem).lasers, planes))
    witness = problem.witness === nothing ? nothing : WitnessFrame(problem, grid, backend)
    species = map(problem.species) do d
        cap = prod(n) * prod(d.ppc)
        window === nothing || (cap = ceil(Int, (1 + numerics.headroom) * cap))
        Species(backend, T, cap; q = d.charge, m = d.mass, N = 2, extras = flux_extras(T, 2))
    end
    flux = numerics.current isa Esirkepov ? (numerics.shape, RHO_STAGGER_2D) : nothing
    ws = map(s -> Workspace(s, grid; flux), species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), species; init = 0)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = wgs)
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           window = window, witness = witness)
    return EMState(numerics, grid, problem.dt, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), recorder, map(species_setup, problem.species),
                   field_modifiers(layer, back, planes, window), aux)
end

"""
    field_modifiers(layer, back, planes, window) -> Tuple

The [`FieldModifier`](@ref)s of a 2D state, in the order the mesh solve walks them
([`solve_fields!`](@ref)), from what the state placed on its grid (each `nothing` or empty where
the problem has none): the open axis' absorber `layer`, the back edge's PML `back`, the antenna
`planes` and the moving window's frame. Around each Yee sub-step that is

    before:  the absorbing layer's half damping
    the Yee update
    after:   the open axis' PML correction, the back edge's PML correction, the antennas'
             corrections, the absorbing layer's half damping, the window's zero x-guards

so a PML corrects the curl the update has just taken before a source is added to it, the
absorbing layer damps what the sub-step and its source left (symmetric splitting), and the
window's edges are set last, over whatever the others wrote into the x-guards. The order is the
tuple's: a PML first, an absorbing layer after the antennas (an open axis has one or the other),
the edges last.
"""
field_modifiers(layer, back, planes::Tuple, window) =
    (modifier_first(layer)..., present(back)..., antenna_modifier(planes, window)..., modifier_late(layer)..., present(edges_of(window))...)

present(::Nothing) = ()
present(m::FieldModifier) = (m,)
modifier_first(::Any) = ()
modifier_late(::Any) = ()
antenna_modifier(::Tuple{}, window) = ()
antenna_modifier(planes::Tuple, window) = (AntennaPlanes(planes, WindowClock(window)),)
edges_of(::Nothing) = nothing

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

function check_aux_setup(st::EMState{2}, problem, wrong)
    window = st.aux.window
    (problem.boundaries[2] === :open) == (open_absorber(st) !== nothing) || wrong("axis 2 is $(problem.boundaries[2])")
    (problem.window === nothing ? window === nothing : window !== nothing && window.window == problem.window) ||
        wrong("the moving window $(problem.window)")
    w = witness_frame(st)
    b = problem.witness
    (b === nothing) == (w === nothing) || wrong("the witness bunch")
    b === nothing || (witness_placement(b, st.grid, st.dt) == (w.kinj, w.c0, w.nc, w.ppc) && w.species.q == oftype(w.species.q, b.charge) && w.species.m == oftype(w.species.m, b.mass)) ||
        wrong("the witness bunch's injection step, lattice, charge or mass")
    return nothing
end

function check_aux_params(st::EMState{2}, p::EMParams)
    antennas, window = antenna_planes(st), st.aux.window
    length(p.lasers) == length(antennas) ||
        throw(ArgumentError("$(length(p.lasers)) lasers in the parameters, $(length(antennas)) antenna planes in the state"))
    foreach((l, a) -> check_plane(a, l), p.lasers, antennas)
    window === nothing || check_window_antennas(window, back_absorber(st), map(Antenna, p.lasers, antennas))
    (p.witness === nothing) == (witness_frame(st) === nothing) ||
        throw(ArgumentError("the parameters $(p.witness === nothing ? "have no" : "have a") witness bunch, the state $(witness_frame(st) === nothing ? "has none" : "has one")"))
    p.profile === nothing || window !== nothing ||
        throw(ArgumentError("a density profile is implemented with the moving window only"))
    window === nothing || check_window_params(window, p)
    return nothing
end

# What `EMProblem` requires of a moving-window plasma, of the parameters handed to the state of
# the `WindowFrame` `w`.
function check_window_params(w, p::EMParams)
    all(ps -> iszero(ps.amplitude), p.species) ||
        throw(ArgumentError("a moving-window plasma starts neutral against its immobile background, so every species' amplitude must be zero"))
    all(iszero, p.E0) && all(iszero, p.B0) ||
        throw(ArgumentError("a moving-window run takes no background fields (the outside of the window is field-free)"))
    w.window.start > 0 && !all(ps -> all(iszero, ps.drift) && all(iszero, ps.thermal), p.species) &&
        throw(ArgumentError("before a moving window starts (start = $(w.window.start)) its front is a fixed edge that a plasma with a drift or a thermal speed reaches: start the window at 0 or keep the species cold"))
    return nothing
end

"""
    initialize!(state::EMState{2}, params; momenta_at = :t0, density_at = :particle) -> state

The 2D state at t = 0, as in 1D: each species loaded by the engine's 2D quiet start (weights
`density·(1 + amplitude·cos(k·x))/prod(ppc)` with the parameters' wave vector, so the parameters
enter through the weights, A3), the momenta from the species' distribution as in 1D
([`load_species!`](@ref)); the fields zero, and with them the convolution fields of a PML (they
are field state: ψ of a field that starts at zero is zero, and a state that ran before must not
carry the last run's into this one); the witness bunch's buffer empty; the window's counters zero; Ex and Ey from
the deposited charge by the engine's periodic solve, `φ` from `inverse_laplacian_periodic!` of ρ
and `E = G` from `gradient_periodic!`, so that the backward-difference divergence of E is
ρ − mean(ρ) to roundoff (the mean is the immobile background); the background fields `E0`, `B0`
added where nonzero (with a moving window the plasma starts neutral against its immobile
background, so E⁰ = 0 and no solve is made; the charge in the x-guards is dropped); then the
momenta at t = −Δt/2 (half-step centring, A2) and the kinetic
energy there: with `momenta_at = :t0` (the default) the loaded momenta are u⁰ and are moved
half a step back by the pusher with −Δt/2 in E⁰, B⁰; with `:t_minus_half` they are taken as
u⁻¹ᐟ², the convention of Smilei and PIConGPU, and kept as loaded. `density_at` is the 1D
method's: `:particle` (the default) evaluates the profile at each particle, `:cell_centre` at
the centre of its cell, as the reference codes do.
"""
function initialize!(st::EMState{2}, p::EMParams; momenta_at::Symbol = :t0, density_at::Symbol = :particle)
    check_momenta_at(momenta_at)
    ParticleMeshCore.check_density_at(density_at)
    st.aux.window === nothing || density_at === :particle ||
        throw(ArgumentError("density_at = :$density_at: the moving window's loads and refills evaluate the density at each particle (:particle) only"))
    check_params(st, p)
    (; numerics, grid, aux) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    window = aux.window
    for (i, (s, ps, setup)) in enumerate(zip(st.species, p.species, st.setup))
        fill!(s.status, ParticleMeshCore.FREE)
        window === nothing || (load_window_species!(window, i, s, grid, wgs, window_load(setup, ps, p), setup.name); continue)
        load_species!(s, grid, ps, setup, p.seed; density_at, workgroupsize = wgs)
    end
    clear_witness!(witness_frame(st))
    foreach(a -> fill!(a, zero(T)), st.fields)
    reset_convolution!(st)
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(st.ρ, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(deposit_mode(st, s), st.ρ, s, numerics.shape, RHO_STAGGER_2D, grid, ws; workgroupsize = wgs)
    end
    zero_x_edges!(window, st.ρ; workgroupsize = wgs)
    fold_guards!(st.ρ, grid; workgroupsize = wgs)
    if window === nothing
        inverse_laplacian_periodic!(aux.φ, st.ρ, grid, aux.spectral; workgroupsize = wgs)
        gradient_periodic!((st.fields.Ex, st.fields.Ey), aux.φ, grid; workgroupsize = wgs)
    end
    add_background!(st.fields, p.E0, p.B0)
    if momenta_at === :t0
        g = Int32(guards(grid))
        for s in st.species
            launcher(kick_2d_kernel!, get_backend(s), wgs)(s, st.fields, numerics.pusher, numerics.shape, g,
                                                           -s.q / s.m * st.dt / 4; ndrange = capacity(s))
        end
    end
    kinetic!(st)
    return st
end

# The current of the step, by the numerics' choice: none, or the charge-conserving one.
current!(::NoCurrent, st::EMState{2}) = st

# Sort where the deposition mode needs it (the old positions in the extras follow the
# particles), zero Jx, Jy, Jz, add every species' Esirkepov current from its old and new
# positions (`q·w·u_z/γ` for Jz), fold the guards (with a moving window, the x-guards dropped
# first) and apply the numerics' binomial filter (through the scratch field ρ, which only
# initialize! and the window shift read otherwise).
function current!(::Esirkepov, st::EMState{2})
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
                      dt = st.dt, quantity_z = CurrentZ(), workgroupsize = wgs)
    end
    window = st.aux.window
    for J in (fields.Jx, fields.Jy, fields.Jz)
        zero_x_edges!(window, J; workgroupsize = wgs)
        fold_guards!(J, grid; workgroupsize = wgs)
    end
    filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
    return st
end

"""
    step!(state::EMState{2}, params, k; observer = state.recorder) -> state

Step `k` at N = 2, a function of the state and the parameters ([`EMParams`](@ref)) in the order
of the 1D step: the `observer` sees Eⁿ, Bⁿ and the kinetic energy at n − 1/2
([`observe!`](@ref)); per species, store the old position,
push the momentum in Eⁿ, Bⁿ gathered at xⁿ on each component's stagger and move the particle
along both axes; the current of the numerics' choice (for [`Esirkepov`](@ref)`()`: sort where
the mode needs it, zero J, the engine's `deposit_flux!` per species, fold the guards, then the
numerics' [`BinomialFilter`](@ref); for `NoCurrent()` J stays zero); the kinetic energy at
n + 1/2; the mesh solve (Yee: B half step, E full step with J, B half step), with the antennas'
total-field/scattered-field corrections between the sub-steps (time `(k − 1)·Δt` at the start of
the step), each antenna the parameters' laser on the state's plane ([`Antenna`](@ref)). With an open axis 2 the particles that crossed the box edge are retired after the
current of their last move is deposited, and the absorbing layer damps the fields around each
sub-step (WP-027). With a moving window (WP-028) the particles that crossed an x edge are
retired before the current (their current would land at the opposite edge), the x-guards of the fields are zero after every sub-step, the antenna planes stay
in the lab frame (their window columns move back with the shifts, and they stop once the window
has passed them), and after the field solve the window shifts by the cells it moved in this
step ([`window_shifts`](@ref)): E, B and the particles one column back, the back column dropped,
the fresh plasma of the injection column loaded with the parameters' densities, profile, drifts,
thermal speeds and seed. A witness bunch (WP-047, [`WitnessBunch`](@ref)) is
injected at the start of its step with the parameters' momentum and weights, pushed with the species and moved and retired at the edges as
they are, and deposits nothing. With `Esirkepov()` and no filter passes, Gauss's law then holds to roundoff
after every step (T-360), off the antenna columns with lasers (T-406), and off the layer rows
with an open boundary (T-451); with passes its filtered form does (T-441). Parameters that do
not fit the state are refused first ([`check_params`](@ref): a laser off its plane, a pulse the
window would pass while it emits, another element type); the default observer refuses a `k`
outside the steps it was allocated for. The step reads and writes the state's evolved arrays
([`evolved`](@ref)) and nothing else that outlives it but the cache `K`, and `k` enters only
through the source's time, the window's position and the witness bunch's injection step: no step
is special, so a run continues from any step of a saved state ([`restore!`](@ref)).
"""
function step!(st::EMState{2}, p::EMParams, k::Integer; observer = st.recorder)
    check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(ncells(grid))
    g = Int32(guards(grid))
    dx = cellsize(grid)
    observe!(observer, st, k)
    inject_witness!(witness_frame(st), p.witness, grid, k, wgs)
    for s in (st.species..., witness_species(st)...)
        launcher(push_2d_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2,
                                                       (dt / dx[1], dt / dx[2]), n; ndrange = capacity(s))
    end
    retire_x!(st.aux.window, st)
    current!(numerics.current, st)
    retire!(open_absorber(st), st)
    kinetic!(st)
    solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    advance_window!(st.aux.window, st, p, k)
    return st
end
