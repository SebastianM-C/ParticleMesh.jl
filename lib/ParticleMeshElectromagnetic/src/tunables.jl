# The parameters of a run as one vector (WP-060): the numbers of a problem description, or of its
# `EMParams`, that A3 lets a derivative be taken with respect to, read into a vector and written
# back. Host code on host objects: nothing here reaches a kernel.

const LASER_TUNABLES = (:a0, :waist, :duration, :delay, :focus, :chirp, :polarization, :phase)
const SHAPED_TUNABLES = (:a0, :delay, :focus, :polarization, :phase)
const SPECIES_TUNABLES = (:density, :amplitude, :drift_x, :drift_y, :drift_z, :thermal_x, :thermal_y, :thermal_z)
const PROFILE_TUNABLES = (:vacuum, :ramp)
const WITNESS_TUNABLES = (:momentum_x, :momentum_y, :momentum_z, :center_x, :center_y, :length, :width)
const SECTION_TUNABLES = (:section_start, :section_stop, :section_ramp)

part_values(l::GaussianLaser) = (l.a0, l.waist, l.duration, l.delay, l.focus, l.chirp, l.polarization, l.phase)
part_values(d::Union{SpeciesDescription,SpeciesParams}) = (d.density, d.amplitude, d.drift..., d.thermal...)
part_values(r::DensityRamp) = (r.vacuum, r.ramp)
part_values(b::Union{WitnessBunch,WitnessParams}) = (b.momentum..., b.center..., b.length, b.width)
part_values(d::Union{FieldIonization,IonizationParams}) = (d.density, section_values(d.section)..., level_values(d.levels)...)
part_values(::Nothing) = ()

section_values(::Nothing) = ()
section_values(s::DopantSection) = (s.start, s.stop, s.ramp)
level_values(levels::Tuple) = foldl((xs, l) -> (xs..., values(level_numbers(l))...), levels; init = ())

part_names(::GaussianLaser) = LASER_TUNABLES
# A shaped laser's: its own five, then its profile's and its envelope's (PulseShapes.parameter_names).
part_values(l::ShapedLaser) = (l.a0, l.delay, l.focus, l.polarization, l.phase, parameter_values(l.profile)..., parameter_values(l.envelope)...)
part_names(l::ShapedLaser) = (SHAPED_TUNABLES..., parameter_names(l.profile)..., parameter_names(l.envelope)...)
part_names(::Union{SpeciesDescription,SpeciesParams}) = SPECIES_TUNABLES
part_names(::DensityRamp) = PROFILE_TUNABLES
part_names(::Union{WitnessBunch,WitnessParams}) = WITNESS_TUNABLES
part_names(::Nothing) = ()
# The dopant's names depend on the description: a section or none, and each level's numbers
# (those of its type: `level_numbers`).
function part_names(d::Union{FieldIonization,IonizationParams})
    section = d.section === nothing ? () : SECTION_TUNABLES
    levels = foldl((xs, j) -> (xs..., map(n -> Symbol(:level, j, :_, n), keys(level_numbers(d.levels[j])))...), eachindex(d.levels); init = ())
    return (:density, section..., levels...)
end

# The parts that hold tunables, in the order of the vector, each with the prefix of its names.
function tunable_parts(p::Union{EMProblem,EMParams})
    lasers = ntuple(i -> (Symbol(:laser, i), p.lasers[i]), length(p.lasers))
    species = ntuple(i -> (Symbol(:species, i), p.species[i]), length(p.species))
    return (lasers..., species..., (:profile, p.profile), (:witness, p.witness), (:ionization, p.ionization))
end

"""
    tunable_names(p::Union{EMProblem,EMParams}) -> Vector{Symbol}

The names of the entries of [`tunables`](@ref)`(p)`, in its order: for the `i`-th laser
`laser<i>_a0`, `_waist`, `_duration`, `_delay`, `_focus`, `_chirp`, `_polarization` and `_phase`
for a [`GaussianLaser`](@ref), and for a [`ShapedLaser`](@ref) `laser<i>_a0`, `_delay`, `_focus`,
`_polarization`, `_phase`, then its profile's and its envelope's names
([`PulseShapes.parameter_names`](@ref): `laser<i>_na`, `_aberration1`, …, `_duration`, `_gdd`, …);
for the `i`-th mobile species `species<i>_density`, `_amplitude`, `_drift_x`, `_drift_y`,
`_drift_z`, `_thermal_x`, `_thermal_y` and `_thermal_z`; with a density profile `profile_vacuum`
and `profile_ramp`; with a witness bunch `witness_momentum_x`, `_momentum_y`, `_momentum_z`,
`witness_center_x`, `_center_y`, `witness_length` and `witness_width`; with a field ionization
`ionization_density`, with a section `ionization_section_start`, `_section_stop` and
`_section_ramp`, and for its `j`-th level the numbers of the level's type
([`IonizationLevel`](@ref)): `ionization_level<j>_prefactor`, `_field` and `_power` for an
[`ADKLevel`](@ref), and after them `_suppression` for a [`TongLinLevel`](@ref).
`findfirst(==(:laser1_duration), tunable_names(p))` is the index of an entry.
"""
function tunable_names(p::Union{EMProblem,EMParams})
    names = Symbol[]
    for (prefix, part) in tunable_parts(p), name in part_names(part)
        push!(names, Symbol(prefix, :_, name))
    end
    return names
end

"""
    tunables(p::Union{EMProblem{N,T},EMParams{T}}) -> Vector{T}

The differentiable parameters of a problem description or of its [`EMParams`](@ref) as one
vector, named by [`tunable_names`](@ref): every number that A3 lets enter through the antenna's
source (a laser's amplitude, waist, duration, delay, focus, chirp, polarization and phase),
through the particle weights (a species' density and perturbation amplitude, the density
profile's vacuum and ramp lengths, a witness bunch's centre, length and width, a dopant's density
and its section's start, stop and ramp), through the loaded momenta (a species' drift and thermal
speeds, the witness's momentum) or through the ionization rates (each level's numbers: prefactor,
field and power, and a barrier-suppression level's suppression).
[`with_tunables`](@ref) writes a vector back, so a gradient with respect to any of them is the
derivative of `x -> J(with_tunables(p, x))`.

Not in the vector, because the run does not carry a derivative with respect to them
([`EMParams`](@ref)): the configuration (the box, the cells, the time step, a laser's `position`
and `center`, the window's start, the witness's injection time and region, charges and masses,
the dopant's lattice, rows, birth cadence and capacity and its levels' cutoffs),
the seed, and the background fields `E0` and `B0`. An entry may be a number at which the run is
not differentiable in practice (the `Inf` of a witness of unbounded width); the vector lists what
can be written, and the choice of what to vary is the caller's. A species keeps the kind of its
momentum distribution whatever its thermal speeds become, so the three thermal speeds of a
Maxwell–Jüttner species, which are one temperature, must stay equal, partials included; that
load is also only piecewise smooth in its drift and temperature (its rejection sampling), and
at a drift of exactly zero its derivative with respect to the drift is lost (#443).

A description may hold a part in another element type than its own (a Float64 species in a
Float32 problem, which the loader reads as it is): the vector is then in the common type of the
description and its parts, and written back it gives a description in that one type, no longer
the mixed one.
"""
function tunables(p::Union{EMProblem,EMParams})
    xs = Any[]
    for (_, part) in tunable_parts(p)
        append!(xs, part_values(part))
    end
    T = promote_type(eltype(p), map(typeof, xs)...)
    return convert(Vector{T}, xs)
end

"The number of entries of [`tunables`](@ref)`(p)`."
tunable_count(p::Union{EMProblem,EMParams}) = sum(q -> length(part_names(last(q))), tunable_parts(p); init = 0)

"""
    with_tunables(p::EMProblem, v::AbstractVector{S}) -> EMProblem{N,S}
    with_tunables(p::EMParams, v::AbstractVector{S}) -> EMParams{S}

`p` with the entries of [`tunables`](@ref) replaced by `v`, in the element type `S` of `v`: every
other number of `p` (the box, the time step, the charges, the background fields) is converted
to `S`, and what is not a number is kept, the kind of each species' momentum distribution
included; so is the configuration that a description holds in a type of its own (a laser's
`position` and `center`, the witness's injection time and region, the dopant's rows). A description whose numbers `S` cannot hold (a `Dual` description written back in a
plain float type, or in duals of another tag) is an `ArgumentError`, and so is an integer `S`. With `S` the element type of `p` and of all its parts,
`with_tunables(p, tunables(p)) === p`; with `S` a `Dual`, the result is the description of the
differentiated run, to build a state from (or, for an `EMParams`, to step an existing dual state
with). The description is built by its constructors, so a vector they refuse (a ramp of
non-positive length, a perturbed species in a moving window) is their `ArgumentError`. `v` must
have [`tunable_count`](@ref)`(p)` entries of a concrete real type.
"""
function with_tunables(p::EMProblem{N}, v::AbstractVector{S}) where {N,S}
    check_tunable_vector(p, v)
    i = Ref(0)
    lasers = map(l -> with_part(S, l, v, i), p.lasers)
    species = map(d -> with_part(S, d, v, i), p.species)
    profile = with_part(S, p.profile, v, i)
    witness = with_part(S, p.witness, v, i)
    ionization = with_part(S, p.ionization, v, i)
    box = N == 1 ? S(p.length) : S.(p.length)
    kmode = N == 1 ? S(p.kmode) : p.kmode
    window = p.window === nothing ? nothing : MovingWindow(S(p.window.start))
    return EMProblem(p.name, box, p.cells, S(p.dt), p.nsteps, species, p.seed, kmode; E0 = S.(p.E0), B0 = S.(p.B0), lasers,
                     boundaries = p.boundaries, window, profile, witness, ionization)
end

function with_tunables(p::EMParams, v::AbstractVector{S}) where {S}
    check_tunable_vector(p, v)
    i = Ref(0)
    lasers = map(l -> with_part(S, l, v, i), p.lasers)
    species = map(d -> with_part(S, d, v, i), p.species)
    profile = with_part(S, p.profile, v, i)
    witness = with_part(S, p.witness, v, i)
    ionization = with_part(S, p.ionization, v, i)
    return EMParams{S}(; lasers, species, profile, witness, ionization, seed = p.seed, E0 = S.(p.E0), B0 = S.(p.B0))
end

function check_tunable_vector(p, v::AbstractVector{S}) where {S}
    S <: Real && isconcretetype(S) && !(S <: Integer) ||
        throw(ArgumentError("with_tunables: the vector's element type $S must be a concrete real type that is not an integer (it becomes the element type of the result)"))
    # every other number of `p` is converted to `S`: one of them says whether `S` can hold them
    x = first(p.E0)
    try
        S(x)
    catch e
        e isa MethodError || e isa InexactError || rethrow()
        throw(ArgumentError("with_tunables: the numbers of the problem, of type $(typeof(x)), cannot be converted to the vector's element type $S"))
    end
    n = tunable_count(p)
    length(v) == n || throw(ArgumentError("with_tunables: $(length(v)) values for the $n tunables of the problem ($(join(tunable_names(p), ", ")))"))
    return nothing
end

# The next `n` entries of `v` as a tuple, `i` the number of entries taken so far.
function take_next!(v::AbstractVector, i::Base.RefValue{Int}, n::Int)
    first = i[] + firstindex(v)
    i[] += n
    return ntuple(k -> v[first + k - 1], n)
end

# A part of the description in the element type `S` with its tunables the next entries of `v`.
# A laser keeps its `position` and `center` as they are: they are configuration, of a type of
# their own.
function with_part(::Type{S}, l::GaussianLaser{<:Any,P,E}, v, i) where {S,P,E}
    x = take_next!(v, i, length(LASER_TUNABLES))
    return GaussianLaser{S,P,E}(x..., l.position, l.center, l.envelope)
end
function with_part(::Type{S}, l::ShapedLaser{<:Any,P}, v, i) where {S,P}
    x = take_next!(v, i, length(part_names(l)))
    nb = length(parameter_values(l.profile))
    b = with_parameter_values(l.profile, x[6:(5 + nb)])
    e = with_parameter_values(l.envelope, x[(6 + nb):end])
    return shaped_laser(S, P, x[1:5], b, e, l.position, l.center)
end
function with_part(::Type{S}, d::SpeciesDescription, v, i) where {S}
    x = take_next!(v, i, length(SPECIES_TUNABLES))
    return SpeciesDescription{S}(d.name, S(d.charge), S(d.mass), x[1], x[2], d.mode, d.ppc, x[3:5], x[6:8], d.stream; momentum = d.momentum)
end
function with_part(::Type{S}, d::SpeciesParams{<:Any,N}, v, i) where {S,N}
    x = take_next!(v, i, length(SPECIES_TUNABLES))
    return SpeciesParams{S,N}(x[1], x[2], d.wavevector, x[3:5], x[6:8])
end
function with_part(::Type{S}, ::DensityRamp, v, i) where {S}
    x = take_next!(v, i, length(PROFILE_TUNABLES))
    return DensityRamp(; vacuum = x[1], ramp = x[2])
end
function with_part(::Type{S}, b::WitnessBunch, v, i) where {S}
    x = take_next!(v, i, length(WITNESS_TUNABLES))
    return WitnessBunch{S}(b.inject, b.region, b.ppc, x[1:3], x[4:5], x[6], x[7], S(b.charge), S(b.mass))
end
function with_part(::Type{S}, ::WitnessParams, v, i) where {S}
    x = take_next!(v, i, length(WITNESS_TUNABLES))
    return WitnessParams{S}(x[1:3], x[4:5], x[6], x[7])
end
# The dopant's density, section and levels from the next entries, a level's cutoff kept.
function ionization_numbers(::Type{S}, d, v, i) where {S}
    density = only(take_next!(v, i, 1))
    section = d.section === nothing ? nothing : DopantSection(; NamedTuple{(:start, :stop, :ramp)}(take_next!(v, i, 3))...)
    levels = map(l -> with_level_numbers(l, S, take_next!(v, i, length(level_numbers(l)))), d.levels)
    return density, levels, section
end
function with_part(::Type{S}, d::FieldIonization, v, i) where {S}
    density, levels, section = ionization_numbers(S, d, v, i)
    return FieldIonization(; density, levels, section, capacity = d.capacity, ppc = d.ppc, every = d.every, rows = d.rows)
end
function with_part(::Type{S}, d::IonizationParams, v, i) where {S}
    density, levels, section = ionization_numbers(S, d, v, i)
    return IonizationParams{S}(density, levels, section)
end
with_part(::Type, ::Nothing, v, i) = nothing
