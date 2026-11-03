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
drift velocity `drift` and the thermal speeds `thermal` per component (v_th = √(T/m), in c; a
velocity-space Maxwellian, converted to the momentum `u = γv` after loading), and the Philox
`stream` of its velocity draws. At N = 1 `mode` and `ppc` are integers and
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
end

SpeciesDescription{T}(name, charge, mass, density, amplitude, mode::M, ppc::M, drift, thermal, stream) where {T,M} =
    SpeciesDescription{T,M}(name, charge, mass, density, amplitude, mode, ppc, drift, thermal, stream)

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
[`AbsorbingLayer`](@ref) and particles that cross the box edge are retired. At N = 1
`kmode` is the diagnostic wavenumber (ω₀/c); at N = 2 it is an integer mode per axis, the wave
vector `k_d = 2π·kmode[d]/length[d]`, nonzero. It names no kernel: the same description runs with
any [`EMNumerics`](@ref).

    EMProblem(name, length, cells, dt, nsteps, species, seed, kmode; E0 = zeros, B0 = zeros, lasers = (), boundaries = periodic)

checks that every species has the problem's dimension (an integer `mode` and `ppc` at N = 1, two
of each at N = 2), that the cells and the particles per cell are positive, at N = 2 that
`kmode` is not the zero mode, that lasers are given only at N = 2, and that the boundaries are
`:periodic` or `:open`, one per axis, with axis 1 periodic (an `ArgumentError` otherwise).
"""
struct EMProblem{N,T,S<:Tuple,L,C,K,LZ<:Tuple,BD}
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
end

function EMProblem(name, box, cells, dt::T, nsteps::Integer, species::Tuple, seed::Integer, kmode;
                   E0 = ntuple(_ -> zero(T), 3), B0 = ntuple(_ -> zero(T), 3), lasers::Tuple = (),
                   boundaries = nothing) where {T}
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
    bd = boundaries === nothing ? ntuple(_ -> :periodic, N) : Tuple(Symbol.(boundaries))
    length(bd) == N && all(in((:periodic, :open)), bd) ||
        throw(ArgumentError("boundaries = $boundaries: one of :periodic and :open per axis, $N of them"))
    first(bd) === :periodic ||
        throw(ArgumentError("boundaries = $bd: axis 1 is periodic (the open transverse boundary is axis 2; the moving window is WP-028)"))
    L = N == 1 ? T(box) : T.(box)
    c = N == 1 ? Int(cells) : Int.(Tuple(cells))
    k = N == 1 ? T(kmode) : kmode
    return EMProblem{N,T,typeof(species),typeof(L),typeof(c),typeof(k),typeof(lasers),typeof(bd)}(String(name), L, c, dt,
                                                                                                    Int(nsteps), species,
                                                                                                    Int(seed), k, T.(Tuple(E0)),
                                                                                                    T.(Tuple(B0)), lasers, bd)
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
    EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing, current = Esirkepov(),
               absorber = nothing)

The numerical implementation, chosen by type (A10): the shape order, the deposition mode
(`Reference()`, sorted and bitwise deterministic, or `Fast()`, atomics), the pusher, the mesh
solver and the current. `workgroupsize` is passed to every engine launch (it selects the launch
shape only). `current` is [`Esirkepov`](@ref)`()`, the charge-conserving current, at N = 1 and
N = 2, or [`NoCurrent`](@ref)`()` at N = 2 only (prescribed fields); a state refuses any other.
`absorber` realizes an open boundary of the problem: an [`AbsorbingLayer`](@ref) at N = 2 when
the problem's axis 2 is `:open` (required then, refused otherwise), `nothing` for a periodic box.
"""
struct EMNumerics{S<:Shape,M,P<:Pusher,F,W,C,A}
    shape::S
    mode::M
    pusher::P
    solver::F
    workgroupsize::W
    current::C
    absorber::A
end

EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing, current = Esirkepov(),
           absorber = nothing) =
    EMNumerics(shape, mode, pusher, solver, workgroupsize, current, absorber)

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
                      seed = nothing, transverse_thermal = false) -> EMProblem{1,T} or EMProblem{2,T}

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
gradient check, A7; one member of a seed ensemble).
"""
function problem_from_case(case::AbstractDict, ::Type{T}; ppc = nothing, cells = nothing, t_end = nothing,
                           dt = nothing, amplitude = nothing, seed = nothing, transverse_thermal::Bool = false) where {T}
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
    schema == 2 && return problem_from_case_2d(case, T, ppc, cells, dtT, nsteps, amplitude, seed)
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
                                             transverse_thermal ? (vth, vth, vth) : (vth, zero(T), zero(T)), k))
    end
    isempty(species) && throw(ArgumentError("the case has no mobile species"))
    i = findfirst(s -> !iszero(s.amplitude), species)
    m = i === nothing ? 1 : species[i].mode
    kmode = 2 * T(π) * m / L
    return EMProblem(String(case["name"]), L, n, dtT, nsteps, Tuple(species), seed, kmode)
end

pair(v) = (Int(v[1]), Int(v[2]))

# Schema 2 (WP-019's 2D cases): every per-axis entry is a pair, the drift has three components,
# the thermal spread is isotropic.
function problem_from_case_2d(case, ::Type{T}, ppc, cells, dt, nsteps, amplitude, seed) where {T}
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
                                             (T(v[1]), T(v[2]), T(v[3])), (vth, vth, vth), k))
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
must be `"esirkepov"` ([`Esirkepov`](@ref), the engine's `deposit_flux!`) and `filter_passes`
zero (the binomial filter is deferred to the 2D LWFA module); with an open boundary in
`grid.boundaries`, the table `[numerics.absorbing_layer]` gives the layer's `cells`, `strength`
and optional `order` ([`AbsorbingLayer`](@ref), WP-027), which a periodic case must not have. The
mode, the pusher and the launch shape are not in the case.
"""
function numerics_from_case(case::AbstractDict; mode = Reference(), pusher = Boris(), workgroupsize = nothing)
    num = case["numerics"]
    getnum(num, "current_deposition", "esirkepov") == "esirkepov" ||
        throw(ArgumentError("current_deposition must be \"esirkepov\""))
    iszero(getnum(num, "filter_passes", 0)) || throw(ArgumentError("the binomial filter is not implemented"))
    shape = Int(getnum(num, "shape_order", 2)) == 1 ? Linear : Quadratic
    bounds = getnum(case["grid"], "boundaries", "periodic")
    open_axis = any(==("open"), bounds isa AbstractVector ? bounds : [bounds])
    absorber = nothing
    if open_axis
        haskey(num, "absorbing_layer") ||
            throw(ArgumentError("an open boundary needs [numerics.absorbing_layer] with cells and strength"))
        al = num["absorbing_layer"]
        absorber = AbsorbingLayer(; cells = Int(al["cells"]), strength = Float64(al["strength"]), order = Int(getnum(al, "order", 2)))
    else
        haskey(num, "absorbing_layer") && throw(ArgumentError("[numerics.absorbing_layer] needs an open boundary"))
    end
    return EMNumerics(; shape, mode, pusher, workgroupsize, current = Esirkepov(), absorber)
end
