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
A2), the `seed` of the velocity draws, the diagnostic `kmode` and the uniform background fields
`E0` and `B0` (three components each, zero by default), which [`initialize!`](@ref) adds to the
fields at t = 0 (a magnetized plasma, or the prescribed fields of test particles; they are
not differentiable parameters, A3: the addition tests them for zero on the host). At N = 1
`kmode` is the diagnostic wavenumber (ω₀/c); at N = 2 it is an integer mode per axis, the wave
vector `k_d = 2π·kmode[d]/length[d]`, nonzero. It names no kernel: the same description runs with
any [`EMNumerics`](@ref).

    EMProblem(name, length, cells, dt, nsteps, species, seed, kmode; E0 = zeros, B0 = zeros)

checks that every species has the problem's dimension (an integer `mode` and `ppc` at N = 1, two
of each at N = 2), that the cells and the particles per cell are positive and, at N = 2, that
`kmode` is not the zero mode (an `ArgumentError` otherwise).
"""
struct EMProblem{N,T,S<:Tuple,L,C,K}
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
end

function EMProblem(name, box, cells, dt::T, nsteps::Integer, species::Tuple, seed::Integer, kmode;
                   E0 = ntuple(_ -> zero(T), 3), B0 = ntuple(_ -> zero(T), 3)) where {T}
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
    L = N == 1 ? T(box) : T.(box)
    c = N == 1 ? Int(cells) : Int.(Tuple(cells))
    k = N == 1 ? T(kmode) : kmode
    return EMProblem{N,T,typeof(species),typeof(L),typeof(c),typeof(k)}(String(name), L, c, dt, Int(nsteps), species,
                                                                         Int(seed), k, T.(Tuple(E0)), T.(Tuple(B0)))
end

Base.eltype(::EMProblem{N,T}) where {N,T} = T
Base.ndims(::EMProblem{N}) where {N} = N

"""
    NoCurrent()

The current choice of an [`EMNumerics`](@ref) that deposits no current: the step pushes the
particles through the fields and solves Maxwell's equations with J = 0. For prescribed-field
runs and test particles at N = 2, where the step is otherwise not complete; the 1D step always
deposits the current and refuses it.
"""
struct NoCurrent end

"""
    EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing, current = nothing)

The numerical implementation, chosen by type (A10): the shape order, the deposition mode
(`Reference()`, sorted and bitwise deterministic, or `Fast()`, atomics), the pusher, the mesh
solver and the current. `workgroupsize` is passed to every engine launch (it selects the launch
shape only). `current = nothing` is the 1D step's charge-conserving current (the engine's
`deposit_flux!`); at N = 2 the current must be chosen explicitly, and [`NoCurrent`](@ref) is the
only choice so far, so a 2D state refuses `nothing` rather than run a step that silently leaves
the current out, and a 1D state refuses `NoCurrent()`.
"""
struct EMNumerics{S<:Shape,M,P<:Pusher,F,W,C}
    shape::S
    mode::M
    pusher::P
    solver::F
    workgroupsize::W
    current::C
end

EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing, current = nothing) =
    EMNumerics(shape, mode, pusher, solver, workgroupsize, current)

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
    problem_from_case(case, T; ppc = nothing, cells = nothing, t_end = nothing, amplitude = nothing,
                      seed = nothing, transverse_thermal = false) -> EMProblem{1,T}

The problem description of a parsed case file (`scripts/crosscode/cases/<case>.toml`, schema 1
with the per-species `transverse_drift = [v_y, v_z]` extension, in A1 units). Mobile species
become [`SpeciesDescription`](@ref)s; immobile species must be uniform (they are the
neutralizing background, not particles). The velocity is the `drift` along x plus the
`transverse_drift`, with a thermal spread `thermal_velocity` along x only, unless
`transverse_thermal = true` (the electrostatic edge cases have no transverse momenta). The
diagnostic wavenumber is that of the first perturbed species' mode.

The keywords override the case for tests and scans: `ppc` for every mobile species, `cells`,
`t_end` (the run takes `round(t_end/dt)` steps), `dt` (with `cells`, so that a finer grid keeps
Δt below the Courant limit, which `EMState` requires), and `amplitude`, which replaces
the magnitude of every nonzero perturbation amplitude and keeps its sign (the parameter of the
A3 gradient check, so it may be a `Dual`), and `seed`, the seed of the velocity draws (common
random numbers on both sides of a gradient check, A7).
"""
function problem_from_case(case::AbstractDict, ::Type{T}; ppc = nothing, cells = nothing, t_end = nothing,
                           dt = nothing, amplitude = nothing, seed = nothing, transverse_thermal::Bool = false) where {T}
    grid = case["grid"]
    get(grid, "boundaries", "periodic") == "periodic" || throw(ArgumentError("only periodic boundaries are implemented"))
    L = T(grid["length"])
    n = cells === nothing ? Int(grid["cells"]) : Int(cells)
    dt_run = dt === nothing ? case["time"]["dt"] : dt
    dt = T(dt_run)
    tend = t_end === nothing ? case["time"]["t_end"] : t_end
    nsteps = round(Int, tend / dt_run)
    seed = seed === nothing ? Int(getnum(case["numerics"], "random_seed", 1)) : Int(seed)
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
    return EMProblem(String(case["name"]), L, n, dt, nsteps, Tuple(species), seed, kmode)
end

"""
    numerics_from_case(case; mode = Reference(), pusher = Boris(), workgroupsize = nothing) -> EMNumerics

The numerics a case file asks for: its `shape_order` (1 or 2); the deposition must be
`"esirkepov"` (the engine's `deposit_flux!`) and `filter_passes` zero (the binomial filter is
deferred to the 2D LWFA module). The mode, the pusher and the launch shape are not in the case.
"""
function numerics_from_case(case::AbstractDict; mode = Reference(), pusher = Boris(), workgroupsize = nothing)
    num = case["numerics"]
    getnum(num, "current_deposition", "esirkepov") == "esirkepov" ||
        throw(ArgumentError("current_deposition must be \"esirkepov\""))
    iszero(getnum(num, "filter_passes", 0)) || throw(ArgumentError("the binomial filter is not implemented in 1D"))
    shape = Int(getnum(num, "shape_order", 2)) == 1 ? Linear : Quadratic
    return EMNumerics(; shape, mode, pusher, workgroupsize)
end
