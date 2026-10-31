# The problem description (A10): what is simulated, in A1 units, with no kernel, solver or pusher
# named. The numerics are chosen separately, by type, in EMNumerics.

"""
    CosineProfile()

The density profile `n(x, p) = p[1]·(1 + p[2]·cos(p[3]·x))`: background density (n_c), relative
perturbation amplitude and wavenumber (ω₀/c). An isbits callable for the engine's loader, so
the parameters enter through the weights (A3).
"""
struct CosineProfile end
@inline (::CosineProfile)(x, p) = p[1] * (1 + p[2] * cos(p[3] * x))

"""
    SpeciesDescription{T}

One mobile species of the problem, in A1 units: `charge` (e), `mass` (mₑ), `density` (n_c),
`amplitude` and `mode` of the cosine perturbation n(x) = density·(1 + amplitude·cos(2π·mode·x/L)),
`ppc` particles per cell (quiet start), the drift velocity `drift` and the thermal speeds
`thermal` per component (v_th = √(T/m), in c; a velocity-space Maxwellian, converted to the
momentum `u = γv` after loading), and the Philox `stream` of its velocity draws.
"""
struct SpeciesDescription{T}
    name::Symbol
    charge::T
    mass::T
    density::T
    amplitude::T
    mode::Int
    ppc::Int
    drift::NTuple{3,T}
    thermal::NTuple{3,T}
    stream::Int
end

"""
    EMProblem{T}

The problem description of a 1D periodic electromagnetic run, in A1 units (time 1/ω₀, length
c/ω₀, density n_c, velocity c): the box `length` and number of `cells`, the time step `dt` and
the number of steps `nsteps`, the mobile `species` (a tuple of [`SpeciesDescription`](@ref);
immobile species of uniform density are the neutralizing background and carry no particles,
A2), the `seed` of the velocity draws and the diagnostic wavenumber `kmode`. It names no
kernel: the same description runs with any [`EMNumerics`](@ref).
"""
struct EMProblem{T,S<:Tuple}
    name::String
    length::T
    cells::Int
    dt::T
    nsteps::Int
    species::S
    seed::Int
    kmode::T
end

Base.eltype(::EMProblem{T}) where {T} = T

"""
    EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing)

The numerical implementation, chosen by type (A10): the shape order, the deposition mode
(`Reference()`, sorted and bitwise deterministic, or `Fast()`, atomics), the pusher and the
mesh solver. `workgroupsize` is passed to every engine launch (it selects the launch shape only).
"""
struct EMNumerics{S<:Shape,M,P<:Pusher,F,W}
    shape::S
    mode::M
    pusher::P
    solver::F
    workgroupsize::W
end

EMNumerics(; shape = Quadratic, mode = Reference(), pusher = Boris(), solver = Yee(), workgroupsize = nothing) =
    EMNumerics(shape, mode, pusher, solver, workgroupsize)

"Run parameters of the step: the time step and the diagnostic wavenumber, both of type `T`."
struct EMParams{T}
    dt::T
    kmode::T
end

params(p::EMProblem) = EMParams(p.dt, p.kmode)

getnum(d, key, default) = haskey(d, key) ? d[key] : default

"""
    problem_from_case(case, T; ppc = nothing, cells = nothing, t_end = nothing, amplitude = nothing,
                      seed = nothing, transverse_thermal = false) -> EMProblem{T}

The problem description of a parsed case file (`scripts/crosscode/cases/<case>.toml`, schema 1
with the per-species `transverse_drift = [v_y, v_z]` extension, in A1 units). Mobile species
become [`SpeciesDescription`](@ref)s; immobile species must be uniform (they are the
neutralizing background, not particles). The velocity is the `drift` along x plus the
`transverse_drift`, with a thermal spread `thermal_velocity` along x only, unless
`transverse_thermal = true` (the electrostatic edge cases have no transverse momenta). The
diagnostic wavenumber is that of the first perturbed species' mode.

The keywords override the case for tests and scans: `ppc` for every mobile species, `cells`,
`t_end` (keeping `dt`; the run takes `round(t_end/dt)` steps), and `amplitude`, which replaces
the magnitude of every nonzero perturbation amplitude and keeps its sign (the parameter of the
A3 gradient check, so it may be a `Dual`), and `seed`, the seed of the velocity draws (common
random numbers on both sides of a gradient check, A7).
"""
function problem_from_case(case::AbstractDict, ::Type{T}; ppc = nothing, cells = nothing, t_end = nothing,
                           amplitude = nothing, seed = nothing, transverse_thermal::Bool = false) where {T}
    grid = case["grid"]
    get(grid, "boundaries", "periodic") == "periodic" || throw(ArgumentError("only periodic boundaries are implemented"))
    L = T(grid["length"])
    n = cells === nothing ? Int(grid["cells"]) : Int(cells)
    dt = T(case["time"]["dt"])
    tend = t_end === nothing ? case["time"]["t_end"] : t_end
    nsteps = round(Int, tend / case["time"]["dt"])
    seed = seed === nothing ? Int(getnum(case["numerics"], "random_seed", 1)) : Int(seed)
    species = SpeciesDescription{T}[]
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
