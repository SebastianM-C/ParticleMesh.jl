# The problem description (A10): what is simulated, in A1 units, with no kernel, solver or pusher
# named. The numerics are chosen separately, by type, in EMNumerics.

"""
    CosineProfile()

The density profile `n(x, p) = p[1]·(1 + p[2]·cos(p[3]·x))`: background density (n_c), relative
perturbation amplitude and wavenumber (ω₀/c); at N = 2, for a position `x::NTuple{2}`,
`n(x, p) = p[1]·(1 + p[2]·cos(p[3]·x[1] + p[4]·x[2]))` with the wave vector `(p[3], p[4])`. An
isbits callable for the engine's loader, so the parameters enter through the weights (A3).
"""
struct CosineProfile end
@inline (::CosineProfile)(x, p) = p[1] * (1 + p[2] * cos(p[3] * x))
@inline (::CosineProfile)(x::NTuple{2}, p) = p[1] * (1 + p[2] * cos(p[3] * x[1] + p[4] * x[2]))

"""
    SpeciesDescription{T}

One mobile species of the problem, in A1 units: `charge` (e), `mass` (mₑ), `density` (n_c),
`amplitude` and `mode` of the cosine perturbation, `ppc` particles per cell (quiet start), the
drift velocity `drift` (β, in c) and the thermal speeds `thermal` per component (v_th = √(T/m),
in c), the Philox `stream` of its momentum draws, and `momentum`, the distribution they are
drawn from (WP-033, [`momentum_distribution`](@ref)):

- `:maxwell_juttner`: Maxwell–Jüttner at θ = T/(mc²) = v_th², drifting with β = `drift`; it needs
  an isotropic, nonzero `thermal`;
- `:gaussian_momentum`: `u = Γβ + thermal·z` per component, Γ = 1/√(1 − β²), `z` standard normal;
- `:gaussian_velocity`: `v = drift + thermal·z` per component, converted to `u = γv` after loading
  (a load in which a drawn speed reaches c is refused, [`check_subluminal`](@ref)).

The keyword `momentum = :default` (of the 10-argument constructor) picks `:maxwell_juttner` for an
isotropic nonzero spread, `:gaussian_momentum` for any other nonzero spread, and
`:gaussian_velocity` for a cold species, where all three load `u = Γβ` and that one keeps the
bits of the cold loads; the first two always give a finite γ. Anything else, or
`:maxwell_juttner` with a spread that is not isotropic and nonzero, is an `ArgumentError`. At N = 1 `mode` and `ppc` are integers and
n(x) = density·(1 + amplitude·cos(2π·mode·x/L)); at N = 2 both are `NTuple{2,Int}`, one integer
per axis, and n(x, y) = density·(1 + amplitude·cos(2π(mode₁·x/L₁ + mode₂·y/L₂))) on a lattice of
`ppc₁ × ppc₂` particles per cell. A species of zero density is a set of test particles: zero
weight, so it carries no charge and no current.
"""
struct SpeciesDescription{T,M}
    name::Symbol
    charge::T
    mass::T
    density::T
    amplitude::T
    mode::M
    ppc::M
    drift::NTuple{3,T}
    thermal::NTuple{3,T}
    stream::Int
    momentum::Symbol
end

const MOMENTUM_DISTRIBUTIONS = (:maxwell_juttner, :gaussian_momentum, :gaussian_velocity)

function SpeciesDescription{T}(name, charge, mass, density, amplitude, mode::M, ppc::M, drift, thermal, stream;
                               momentum::Symbol = :default) where {T,M}
    th = T.(Tuple(thermal))
    isotropic = th[1] == th[2] == th[3] && th[1] > 0
    cold = all(iszero, th)
    m = momentum === :default ? (cold ? :gaussian_velocity : isotropic ? :maxwell_juttner : :gaussian_momentum) : momentum
    m in MOMENTUM_DISTRIBUTIONS ||
        throw(ArgumentError("species $name: momentum = :$momentum, one of :default, :maxwell_juttner, :gaussian_momentum and :gaussian_velocity"))
    m === :maxwell_juttner && !isotropic &&
        throw(ArgumentError("species $name: a Maxwell–Jüttner load is isotropic and hot, so its thermal spread must be the same nonzero speed in every component, got $thermal"))
    return SpeciesDescription{T,M}(name, charge, mass, density, amplitude, mode, ppc, drift, th, stream, m)
end

"""
    momentum_distribution(d::SpeciesDescription{T}) -> GaussianVelocity, GaussianMomentum or MaxwellJuttner

The engine's distribution object (`load_quiet!`'s `dist`) of a species' `momentum`:
`MaxwellJuttner(thermal[1]², drift)`, `GaussianMomentum(Γ·drift, thermal)` or
`GaussianVelocity(drift, thermal)`. For the two momentum draws a drift of |β| ≥ 1 (or NaN) or a
thermal speed that is not finite is an `ArgumentError` (the velocity draw leaves both to
[`check_subluminal`](@ref)).
"""
function momentum_distribution(d::SpeciesDescription{T}) where {T}
    d.momentum === :gaussian_velocity && return GaussianVelocity(d.drift, d.thermal)
    all(isfinite, d.thermal) || throw(ArgumentError("species $(d.name): the thermal speeds $(d.thermal) must be finite"))
    b2 = d.drift[1]^2 + d.drift[2]^2 + d.drift[3]^2
    b2 < 1 || throw(ArgumentError("species $(d.name): the drift speed $(sqrt(b2)) must be below c"))
    d.momentum === :maxwell_juttner && return MaxwellJuttner{T}(d.thermal[1]^2, d.drift)
    Γ = one(T) / sqrt(one(T) - b2)
    return GaussianMomentum(map(b -> Γ * b, d.drift), d.thermal)
end

"""
    EMProblem{N,T}

The problem description of a periodic electromagnetic run in `N = 1` or `N = 2` dimensions, in
A1 units (time 1/ω₀, length c/ω₀, density n_c, velocity c, fields mₑcω₀/e): the box `length`
and number of `cells` (a number at N = 1, one per axis at N = 2), the time step `dt` and the
number of steps `nsteps`, the mobile `species` (a tuple of [`SpeciesDescription`](@ref);
immobile species of uniform density are the neutralizing background and carry no particles,
A2), the `seed` of the velocity draws, the diagnostic `kmode`, the uniform background fields
`E0` and `B0` (three components each, zero by default), which [`initialize!`](@ref) adds to the
fields at t = 0 (a magnetized plasma, or the prescribed fields of test particles; they are
not differentiable parameters, A3: the addition tests them for zero on the host), and the
`lasers`, a tuple of [`GaussianLaser`](@ref) descriptions launched by antenna planes during
the steps (N = 2 only, empty by default; their parameters may be differentiable, A3), and the
`boundaries`, one `Symbol` per axis: `:periodic` (the default on every axis), or `:open` on axis 2
at N = 2, the open transverse boundary (WP-027), where fields leave through the numerics'
[`AbsorbingLayer`](@ref) or [`PML`](@ref) and particles that cross the box edge are retired, or `:window` on
axis 1 at N = 2 with a [`MovingWindow`](@ref) as `window` (WP-028): the box moves at c along
axis 1, its x edges are not periodic, and the plasma, of the longitudinal `profile` (a
[`DensityRamp`](@ref), or `nothing` for a uniform one), starts neutral against an immobile
background of its own profile (so every species' `amplitude` is zero and `E0`, `B0` are zero) and
is refilled behind the front, every species with `momentum = :gaussian_velocity` (the window's
loads draw in velocity); a window that starts later than t = 0 takes only cold species at
rest (before it moves, its front is a fixed edge that a moving plasma reaches); a laser's antenna plane stays at its place in the lab frame and must have
finished its pulse before the window passes it (checked by the state). At N = 2 a `witness`, a
[`WitnessBunch`](@ref) (WP-047), adds test particles injected during the run whose mean energy
gain is the optimization objective; they change nothing else in the run. At N = 1
`kmode` is the diagnostic wavenumber (ω₀/c); at N = 2 it is an integer mode per axis, the wave
vector `k_d = 2π·kmode[d]/length[d]`, nonzero. It names no kernel: the same description runs with
any [`EMNumerics`](@ref).

    EMProblem(name, length, cells, dt, nsteps, species, seed, kmode; E0 = zeros, B0 = zeros, lasers = (), boundaries = periodic,
              window = nothing, profile = nothing, witness = nothing)

checks that every species has the problem's dimension (an integer `mode` and `ppc` at N = 1, two
of each at N = 2), that the cells and the particles per cell are positive, at N = 2 that
`kmode` is not the zero mode, that lasers are given only at N = 2, and that the boundaries are
`:periodic` or `:open`, one per axis, with axis 1 periodic, or `:window` on axis 1 together with
a `window` and the conditions above (an `ArgumentError` otherwise).
"""
struct EMProblem{N,T,S<:Tuple,L,C,K,LZ<:Tuple,BD,W,PR,WB}
    name::String
    length::L
    cells::C
    dt::T
    nsteps::Int
    species::S
    seed::Int
    kmode::K
    E0::NTuple{3,T}
    B0::NTuple{3,T}
    lasers::LZ
    boundaries::BD
    window::W
    profile::PR
    witness::WB
end

function EMProblem(name, box, cells, dt::T, nsteps::Integer, species::Tuple, seed::Integer, kmode;
                   E0 = ntuple(_ -> zero(T), 3), B0 = ntuple(_ -> zero(T), 3), lasers::Tuple = (),
                   boundaries = nothing, window = nothing, profile = nothing, witness = nothing) where {T}
    N = cells isa Integer ? 1 : length(cells)
    N in (1, 2) || throw(ArgumentError("cells = $cells: N = 1 (an integer) or N = 2 (two integers) are implemented"))
    M = N == 1 ? Int : NTuple{2,Int}
    if N == 1
        box isa Number || throw(ArgumentError("a 1D problem takes one box length, got $box"))
        kmode isa Number || throw(ArgumentError("a 1D problem takes the diagnostic wavenumber as a number, got $kmode"))
    else
        box isa NTuple{2} || throw(ArgumentError("a 2D problem takes two box lengths, got $box"))
        kmode isa NTuple{2,Int} || throw(ArgumentError("a 2D problem takes the diagnostic mode as two integers, got $kmode"))
        kmode == (0, 0) && throw(ArgumentError("the diagnostic mode must not be (0, 0)"))
    end
    all(>(0), cells) || throw(ArgumentError("cells = $cells must be positive"))
    for d in species
        d.mode isa M && d.ppc isa M ||
            throw(ArgumentError("species $(d.name): mode $(d.mode) and ppc $(d.ppc) must be $(N == 1 ? "integers" : "two integers each") for N = $N"))
        all(>(0), d.ppc) || throw(ArgumentError("species $(d.name): ppc = $(d.ppc) must be positive"))
    end
    isempty(lasers) || N == 2 || throw(ArgumentError("laser antennas are implemented at N = 2 only"))
    all(l -> l isa GaussianLaser, lasers) || throw(ArgumentError("lasers must be GaussianLaser descriptions, got $(map(typeof, lasers))"))
    boundaries isa Union{Symbol,AbstractString} &&
        throw(ArgumentError("boundaries = $(repr(boundaries)): one of :periodic, :open and :window per axis, a tuple of $N"))
    bd = boundaries === nothing ? ntuple(_ -> :periodic, N) : Tuple(Symbol.(boundaries))
    length(bd) == N && all(in((:periodic, :open, :window)), bd) ||
        throw(ArgumentError("boundaries = $boundaries: one of :periodic, :open and :window per axis, $N of them"))
    first(bd) in (:periodic, :window) && last(bd) !== :window ||
        throw(ArgumentError("boundaries = $bd: axis 1 is :periodic or :window (the moving window), axis 2 :periodic or :open (the open transverse boundary)"))
    moving = first(bd) === :window
    moving && N != 2 && throw(ArgumentError("the moving window is implemented at N = 2 only"))
    moving == (window !== nothing) ||
        throw(ArgumentError("boundaries = $bd with window = $window: a :window axis 1 and a MovingWindow come together"))
    window === nothing || window isa MovingWindow || throw(ArgumentError("window must be a MovingWindow, got $window"))
    profile === nothing || profile isa DensityRamp || throw(ArgumentError("profile must be a DensityRamp, got $profile"))
    profile !== nothing && !moving && throw(ArgumentError("a density profile is implemented with the moving window only"))
    if moving
        for d in species
            iszero(d.amplitude) ||
                throw(ArgumentError("species $(d.name): a moving-window plasma starts neutral against its immobile background, so its amplitude must be zero, got $(d.amplitude)"))
            d.momentum === :gaussian_velocity ||
                throw(ArgumentError("species $(d.name): the moving window loads and refills with the Gaussian velocity draw only, got momentum = :$(d.momentum); pass momentum = :gaussian_velocity (a thermal spread in momentum in the window is #350)"))
        end
        all(iszero, E0) && all(iszero, B0) ||
            throw(ArgumentError("a moving-window problem takes no background fields (the outside of the window is field-free)"))
        if window isa MovingWindow && window.start > 0
            for d in species
                all(iszero, d.drift) && all(iszero, d.thermal) ||
                    throw(ArgumentError("species $(d.name): before a moving window starts (start = $(window.start)) its front is a fixed edge, and a plasma with a drift or a thermal speed reaches it and leaves a residual of Gauss's law that the shifts carry into the box; start the window at 0 or load the species cold, got drift $(d.drift) and thermal $(d.thermal)"))
            end
        end
    end
    witness === nothing || witness isa WitnessBunch || throw(ArgumentError("witness must be a WitnessBunch, got $witness"))
    witness === nothing || N == 2 || throw(ArgumentError("a witness bunch is implemented at N = 2 only"))
    wb = witness === nothing ? nothing : witness_as(T, witness)
    L = N == 1 ? T(box) : T.(box)
    c = N == 1 ? Int(cells) : Int.(Tuple(cells))
    k = N == 1 ? T(kmode) : kmode
    win = window === nothing ? nothing : MovingWindow(T(window.start))
    return EMProblem{N,T,typeof(species),typeof(L),typeof(c),typeof(k),typeof(lasers),typeof(bd),typeof(win),typeof(profile),typeof(wb)}(
        String(name), L, c, dt, Int(nsteps), species, Int(seed), k, T.(Tuple(E0)), T.(Tuple(B0)), lasers, bd, win, profile, wb)
end

Base.eltype(::EMProblem{N,T}) where {N,T} = T
Base.ndims(::EMProblem{N}) where {N} = N

"""
    Esirkepov()

The charge-conserving current of an [`EMNumerics`](@ref), the default at N = 1 and N = 2: the
engine's `deposit_flux!` from each particle's position before and after the move (Esirkepov's
decomposition for the in-plane components; at N = 1 `q·w·u_{y,z}/γ` for Jy and Jz, at N = 2
`q·w·u_z/γ` for Jz), so that with ρ deposited by the same shape the discrete continuity
equation, and with it Gauss's law, holds to roundoff after every step (T-212, T-360).
"""
struct Esirkepov end

"""
    NoCurrent()

The current choice of an [`EMNumerics`](@ref) that deposits no current: the step pushes the
particles through the fields and solves Maxwell's equations with J = 0. For prescribed-field
runs and test particles at N = 2; the 1D step always deposits the current and refuses it.
"""
struct NoCurrent end

"""
    EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing,
               current = Esirkepov(), filter = BinomialFilter(0), absorber = nothing, back_absorber = nothing, headroom = 1/4)

The numerical implementation, chosen by type (A10): the shape order, the deposition mode
(`Reference()`, sorted and bitwise deterministic, or `Fast()`, atomics), the pusher, the mesh
solver, the current and its filter. `workgroupsize` is passed to every engine launch (it selects
the launch shape only). A fixed size has to fit every kernel of the step: the largest work-group
a kernel accepts depends on the backend, the kernel and the element type (on the CPU (PoCL)
backend the 2D flux deposition with `Dual{Float32,3}` and the quadratic shape takes at most 983
work-items, as PoCL reported it, #297), and a launch it does not fit stops the step with an
`ArgumentError` that names the size and the kernel; `nothing` fits every kernel. `current` is [`Esirkepov`](@ref)`()`, the charge-conserving current, at
N = 1 and N = 2, or [`NoCurrent`](@ref)`()` at N = 2 only (prescribed fields); a state refuses
any other. `filter` is the [`BinomialFilter`](@ref) the step applies to the folded current
before the mesh solve, with no passes by default. `absorber` realizes an open boundary of the
problem: an [`AbsorbingLayer`](@ref) or a [`PML`](@ref) at N = 2 when the problem's axis 2 is `:open` (required
then, refused otherwise), `nothing` for a periodic box; a 2D state refuses filter passes together
with either absorber until a test checks the two together. `back_absorber` is a [`PML`](@ref) on
the moving window's back edge (WP-034: its first `cells` columns, the profile graded towards the
back), or `nothing`, the window's field-free outside alone; a state refuses it without a
window, and refuses a laser antenna that the layer reaches before its pulse is over. `headroom` sizes the particle buffers of
a moving-window state, `(1 + headroom)·prod(cells)·prod(ppc)` slots per species, for particles
that stay in the window longer than the plasma at rest (a refill that finds no free slot is
counted, [`window_lost`](@ref)); other states ignore it.
"""
struct EMNumerics{S<:Shape,M,P<:Pusher,F,W,C,A,K}
    shape::S
    mode::M
    pusher::P
    solver::F
    workgroupsize::W
    current::C
    filter::BinomialFilter
    absorber::A
    back_absorber::K
    headroom::Float64
end

function EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing,
                    current = Esirkepov(), filter::BinomialFilter = BinomialFilter(0), absorber = nothing, back_absorber = nothing,
                    headroom::Real = 1 / 4)
    headroom >= 0 && isfinite(headroom) || throw(ArgumentError("headroom must be finite and ≥ 0, got $headroom"))
    return EMNumerics(shape, mode, pusher, solver, workgroupsize, current, filter, absorber, back_absorber, Float64(headroom))
end

"""
    EMParams(dt, kmode)

Run parameters of the step: the time step and the diagnostic `kmode` of the problem (the
wavenumber at N = 1; the integer mode per axis at N = 2, which must be the state's, since the
2D state holds its phase tables).
"""
struct EMParams{T,K}
    dt::T
    kmode::K
end

params(p::EMProblem) = EMParams(p.dt, p.kmode)

getnum(d, key, default) = haskey(d, key) ? d[key] : default

"""
    case_seed(case) -> Int

The default seed of a case file's velocity draws: the first entry of its `seeds` list (a seed
ensemble, WP-012, WP-017) or its single `random_seed`. A case with both keys, or neither, is an
`ArgumentError`, as for the cross-code generator (`scripts/crosscode/smilei.jl`).
"""
function case_seed(case::AbstractDict)
    num = case["numerics"]
    a, b = haskey(num, "seeds"), haskey(num, "random_seed")
    a && b && throw(ArgumentError("case $(case["name"]): give either seeds or random_seed, not both"))
    a || b || throw(ArgumentError("case $(case["name"]): no seeds and no random_seed"))
    if a
        seeds = num["seeds"]
        seeds isa AbstractVector && !isempty(seeds) || throw(ArgumentError("case $(case["name"]): seeds must be a non-empty list"))
        return Int(first(seeds))
    end
    return Int(num["random_seed"])
end

"""
    problem_from_case(case, T; ppc = nothing, cells = nothing, t_end = nothing, dt = nothing, amplitude = nothing,
                      seed = nothing, transverse_thermal = false, momentum = nothing) -> EMProblem{1,T} or EMProblem{2,T}

The problem description of a parsed case file (`scripts/crosscode/cases/<case>.toml`, in A1
units): schema 1 gives an `EMProblem{1}` (with the per-species `transverse_drift = [v_y, v_z]`
extension), schema 2 an `EMProblem{2}`. Mobile species become [`SpeciesDescription`](@ref)s;
immobile species must be uniform (they are the neutralizing background, not particles).

Schema 1: the velocity is the `drift` along x plus the `transverse_drift`, with a thermal spread
`thermal_velocity` along x only, unless `transverse_thermal = true` (the electrostatic edge
cases have no transverse momenta); the diagnostic wavenumber is that of the first perturbed
species' mode. Schema 2: two box lengths, cells, particles per cell and mode integers per axis,
the drift `[v_x, v_y, v_z]`, an isotropic thermal spread `thermal_velocity` in all three
components (Smilei's scalar temperature), the diagnostic mode the case's `wavevector.mode`, and
`grid.boundaries`, `"periodic"` or `"open"` per axis (axis 2 may be open, WP-027; a schema-1
case is periodic).

The default seed is [`case_seed`](@ref)`(case)`: the first of a `seeds` list or the
`random_seed`. The keywords override the case for tests and scans: `ppc` for every mobile species
(an integer at schema 1, two at schema 2), `cells` (likewise), `t_end` (the run takes
`round(t_end/dt)` steps), `dt` (with `cells`, so that a finer grid keeps Δt below the Courant
limit, which `EMState` requires), `amplitude`, which replaces the magnitude of every nonzero
perturbation amplitude and keeps its sign (the parameter of the A3 gradient check, so it may be a
`Dual`), and `seed`, the seed of the velocity draws (common random numbers on both sides of a
gradient check, A7; one member of a seed ensemble), and `momentum`, the distribution of every mobile
species' momentum draws (a `Symbol` of [`SpeciesDescription`](@ref)'s `momentum`), which replaces
each species' optional `momentum_distribution` key (`"maxwell_juttner"`, `"gaussian_momentum"` or
`"gaussian_velocity"`; without it the species takes `SpeciesDescription`'s default, so an
isotropic thermal species loads Maxwell–Jüttner and a schema-1 spread along x only a Gaussian in
momentum).
"""
function problem_from_case(case::AbstractDict, ::Type{T}; ppc = nothing, cells = nothing, t_end = nothing,
                           dt = nothing, amplitude = nothing, seed = nothing, transverse_thermal::Bool = false,
                           momentum = nothing) where {T}
    schema = getnum(case, "schema", 1)
    schema in (1, 2) || throw(ArgumentError("case schema $schema: 1 and 2 are implemented"))
    grid = case["grid"]
    bounds = getnum(grid, "boundaries", "periodic")
    bounds = bounds isa AbstractVector ? bounds : [bounds]
    all(in(("periodic", "open")), bounds) || throw(ArgumentError("boundaries $bounds: \"periodic\" or \"open\" per axis"))
    schema == 2 || all(==("periodic"), bounds) || throw(ArgumentError("a schema-1 (1D) case is periodic, got $bounds"))
    dt_run = dt === nothing ? case["time"]["dt"] : dt
    dtT = T(dt_run)
    tend = t_end === nothing ? case["time"]["t_end"] : t_end
    nsteps = round(Int, tend / dt_run)
    seed = seed === nothing ? case_seed(case) : Int(seed)
    schema == 2 && return problem_from_case_2d(case, T, ppc, cells, dtT, nsteps, amplitude, seed, momentum)
    L = T(grid["length"])
    n = cells === nothing ? Int(grid["cells"]) : Int(cells)
    species = SpeciesDescription{T,Int}[]
    for (k, s) in enumerate(case["species"])
        pert = getnum(s, "perturbation", Dict("amplitude" => 0.0, "mode" => 1))
        a0 = pert["amplitude"]
        if !getnum(s, "mobile", true)
            iszero(a0) || throw(ArgumentError("immobile species $(s["name"]) must be uniform: it is the background"))
            continue
        end
        a = amplitude === nothing || iszero(a0) ? T(a0) : T(sign(a0)) * T(amplitude)
        vth = T(getnum(s, "thermal_velocity", 0.0))
        getnum(s, "velocity", "cold") == "cold" && (vth = zero(T))
        vt = getnum(s, "transverse_drift", [0.0, 0.0])
        push!(species, SpeciesDescription{T}(Symbol(s["name"]), T(s["charge"]), T(s["mass"]), T(s["density"]), a,
                                             Int(pert["mode"]),
                                             ppc === nothing ? Int(s["particles_per_cell"]) : Int(ppc),
                                             (T(getnum(s, "drift", 0.0)), T(vt[1]), T(vt[2])),
                                             transverse_thermal ? (vth, vth, vth) : (vth, zero(T), zero(T)), k;
                                             momentum = case_momentum(s, momentum)))
    end
    isempty(species) && throw(ArgumentError("the case has no mobile species"))
    i = findfirst(s -> !iszero(s.amplitude), species)
    m = i === nothing ? 1 : species[i].mode
    kmode = 2 * T(π) * m / L
    return EMProblem(String(case["name"]), L, n, dtT, nsteps, Tuple(species), seed, kmode)
end

pair(v) = (Int(v[1]), Int(v[2]))

# A species' momentum distribution: the keyword of `problem_from_case` for every species, else the
# species' `momentum_distribution` key ("maxwell_juttner", "gaussian_momentum" or
# "gaussian_velocity"), else the default of `SpeciesDescription`.
function case_momentum(s, momentum)
    momentum === nothing || return Symbol(momentum)
    m = getnum(s, "momentum_distribution", "default")
    m isa AbstractString || throw(ArgumentError("species $(s["name"]): momentum_distribution must be a string, got $(repr(m))"))
    return Symbol(m)
end

# Schema 2 (WP-019's 2D cases): every per-axis entry is a pair, the drift has three components,
# the thermal spread is isotropic.
function problem_from_case_2d(case, ::Type{T}, ppc, cells, dt, nsteps, amplitude, seed, momentum) where {T}
    grid = case["grid"]
    L = (T(grid["length"][1]), T(grid["length"][2]))
    n = cells === nothing ? pair(grid["cells"]) : pair(cells)
    species = SpeciesDescription{T,NTuple{2,Int}}[]
    for (k, s) in enumerate(case["species"])
        pert = getnum(s, "perturbation", Dict("amplitude" => 0.0, "mode" => [0, 0]))
        a0 = pert["amplitude"]
        if !getnum(s, "mobile", true)
            iszero(a0) || throw(ArgumentError("immobile species $(s["name"]) must be uniform: it is the background"))
            continue
        end
        a = amplitude === nothing || iszero(a0) ? T(a0) : T(sign(a0)) * T(amplitude)
        cold = getnum(s, "velocity", "cold") == "cold"
        cold || haskey(s, "thermal_velocity") ||
            throw(ArgumentError("species $(s["name"]): a thermal schema-2 species needs a thermal_velocity"))
        vth = cold ? zero(T) : T(s["thermal_velocity"])
        v = getnum(s, "drift", [0.0, 0.0, 0.0])
        length(v) == 3 || throw(ArgumentError("species $(s["name"]): a schema-2 drift has three components, got $v"))
        push!(species, SpeciesDescription{T}(Symbol(s["name"]), T(s["charge"]), T(s["mass"]), T(s["density"]), a,
                                             pair(pert["mode"]), ppc === nothing ? pair(s["particles_per_cell"]) : pair(ppc),
                                             (T(v[1]), T(v[2]), T(v[3])), (vth, vth, vth), k; momentum = case_momentum(s, momentum)))
    end
    isempty(species) && throw(ArgumentError("the case has no mobile species"))
    kmode = pair(case["wavevector"]["mode"])
    bounds = getnum(grid, "boundaries", ["periodic", "periodic"])
    boundaries = bounds isa AbstractVector ? Tuple(Symbol.(bounds)) : (Symbol(bounds), Symbol(bounds))
    return EMProblem(String(case["name"]), L, n, dt, nsteps, Tuple(species), seed, kmode; boundaries)
end

"""
    numerics_from_case(case; mode = Reference(), pusher = Boris(), workgroupsize = nothing) -> EMNumerics

The numerics a case file asks for (either schema): its `shape_order` (1 or 2); the deposition
must be `"esirkepov"` ([`Esirkepov`](@ref), the engine's `deposit_flux!`); `filter_passes`
(default 0), a non-negative integer, is the pass count of the [`BinomialFilter`](@ref) (A2), and
anything else is an `ArgumentError`. With an open boundary in `grid.boundaries`, the table
`[numerics.absorbing_layer]` gives the layer's `cells`, `strength` and optional `order`
([`AbsorbingLayer`](@ref), WP-027), or the table `[numerics.pml]` the PML's `cells` and optional
`sigma_max`, `order`, `kappa_max`, `kappa_order`, `alpha_max` and `alpha_order` ([`PML`](@ref),
WP-034), exactly one of the two; a periodic case must have neither. The mode, the pusher
and the launch shape are not in the case.
"""
function numerics_from_case(case::AbstractDict; mode = Reference(), pusher = Boris(), workgroupsize = nothing)
    num = case["numerics"]
    getnum(num, "current_deposition", "esirkepov") == "esirkepov" ||
        throw(ArgumentError("current_deposition must be \"esirkepov\""))
    passes = getnum(num, "filter_passes", 0)
    passes isa Integer && !(passes isa Bool) && passes >= 0 ||
        throw(ArgumentError("filter_passes must be a non-negative integer, got $(repr(passes))"))
    shape = Int(getnum(num, "shape_order", 2)) == 1 ? Linear : Quadratic
    bounds = getnum(case["grid"], "boundaries", "periodic")
    open_axis = any(==("open"), bounds isa AbstractVector ? bounds : [bounds])
    absorber = nothing
    has_al, has_pml = haskey(num, "absorbing_layer"), haskey(num, "pml")
    if open_axis
        has_al + has_pml == 1 ||
            throw(ArgumentError("an open boundary needs exactly one of [numerics.absorbing_layer] (cells, strength) and [numerics.pml] (cells)"))
        if has_al
            al = num["absorbing_layer"]
            absorber = AbsorbingLayer(; cells = Int(al["cells"]), strength = Float64(al["strength"]), order = Int(getnum(al, "order", 2)))
        else
            pl = num["pml"]
            order = Int(getnum(pl, "order", 4))
            sm = getnum(pl, "sigma_max", nothing)
            absorber = PML(; cells = Int(pl["cells"]), sigma_max = sm === nothing ? nothing : Float64(sm), order,
                           kappa_max = Float64(getnum(pl, "kappa_max", 1)), kappa_order = Int(getnum(pl, "kappa_order", order)),
                           alpha_max = Float64(getnum(pl, "alpha_max", 0)), alpha_order = Int(getnum(pl, "alpha_order", 1)))
        end
    else
        has_al && throw(ArgumentError("[numerics.absorbing_layer] needs an open boundary"))
        has_pml && throw(ArgumentError("[numerics.pml] needs an open boundary"))
    end
    return EMNumerics(; shape, mode, pusher, workgroupsize, current = Esirkepov(), filter = BinomialFilter(passes),
                      absorber)
end
