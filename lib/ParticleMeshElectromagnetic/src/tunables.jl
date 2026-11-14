# The parameters of a run as one vector (WP-059): the numbers of a problem description, or of its
# `EMParams`, that A3 lets a derivative be taken with respect to, read into a vector and written
# back. Host code on host objects: nothing here reaches a kernel.

const LASER_TUNABLES = (:a0, :waist, :duration, :delay, :focus, :chirp, :polarization, :phase)
const SPECIES_TUNABLES = (:density, :amplitude, :drift_x, :drift_y, :drift_z, :thermal_x, :thermal_y, :thermal_z)
const PROFILE_TUNABLES = (:vacuum, :ramp)
const WITNESS_TUNABLES = (:momentum_x, :momentum_y, :momentum_z, :center_x, :center_y, :length, :width)

part_values(l::GaussianLaser) = (l.a0, l.waist, l.duration, l.delay, l.focus, l.chirp, l.polarization, l.phase)
part_values(d::Union{SpeciesDescription,SpeciesParams}) = (d.density, d.amplitude, d.drift..., d.thermal...)
part_values(r::DensityRamp) = (r.vacuum, r.ramp)
part_values(b::Union{WitnessBunch,WitnessParams}) = (b.momentum..., b.center..., b.length, b.width)
part_values(::Nothing) = ()

part_names(::GaussianLaser) = LASER_TUNABLES
part_names(::Union{SpeciesDescription,SpeciesParams}) = SPECIES_TUNABLES
part_names(::DensityRamp) = PROFILE_TUNABLES
part_names(::Union{WitnessBunch,WitnessParams}) = WITNESS_TUNABLES
part_names(::Nothing) = ()

# The parts that hold tunables, in the order of the vector, each with the prefix of its names.
function tunable_parts(p::Union{EMProblem,EMParams})
    lasers = ntuple(i -> (Symbol(:laser, i), p.lasers[i]), length(p.lasers))
    species = ntuple(i -> (Symbol(:species, i), p.species[i]), length(p.species))
    return (lasers..., species..., (:profile, p.profile), (:witness, p.witness))
end

"""
    tunable_names(p::Union{EMProblem,EMParams}) -> Vector{Symbol}

The names of the entries of [`tunables`](@ref)`(p)`, in its order: for the `i`-th laser
`laser<i>_a0`, `_waist`, `_duration`, `_delay`, `_focus`, `_chirp`, `_polarization` and `_phase`;
for the `i`-th mobile species `species<i>_density`, `_amplitude`, `_drift_x`, `_drift_y`,
`_drift_z`, `_thermal_x`, `_thermal_y` and `_thermal_z`; with a density profile `profile_vacuum`
and `profile_ramp`; with a witness bunch `witness_momentum_x`, `_momentum_y`, `_momentum_z`,
`witness_center_x`, `_center_y`, `witness_length` and `witness_width`.
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
profile's vacuum and ramp lengths, a witness bunch's centre, length and width) or through the
loaded momenta (a species' drift and thermal speeds, the witness's momentum).
[`with_tunables`](@ref) writes a vector back, so a gradient with respect to any of them is the
derivative of `x -> J(with_tunables(p, x))`.

Not in the vector, because the run does not carry a derivative with respect to them
([`EMParams`](@ref)): the configuration (the box, the cells, the time step, a laser's `position`
and `center`, the window's start, the witness's injection time and region, charges and masses),
the seed, and the background fields `E0` and `B0`. An entry may be a number at which the run is
not differentiable in practice (the `Inf` of a witness of unbounded width, a zero thermal speed
that selects a cold load); the vector lists what can be written, and the choice of what to vary
is the caller's. The three thermal speeds of a Maxwell–Jüttner species are one temperature:
they must stay equal, partials included.

Every part must be in the element type `T` of `p`: a description that holds a species, a laser
or a profile of another type (a Float64 species in a Float32 problem) is an `ArgumentError`,
since a vector of one type could not give it back.
"""
function tunables(p::Union{EMProblem,EMParams})
    T = eltype(p)
    v = T[]
    for (prefix, part) in tunable_parts(p), (name, x) in zip(part_names(part), part_values(part))
        x isa T ||
            throw(ArgumentError("tunables: $(prefix)_$name is a $(typeof(x)) in a problem of element type $T; " *
                                "build every part of the description in one element type"))
        push!(v, x)
    end
    return v
end

"The number of entries of [`tunables`](@ref)`(p)`."
tunable_count(p::Union{EMProblem,EMParams}) = sum(q -> length(part_names(last(q))), tunable_parts(p); init = 0)

"""
    with_tunables(p::EMProblem, v::AbstractVector{S}) -> EMProblem{N,S}
    with_tunables(p::EMParams, v::AbstractVector{S}) -> EMParams{S}

`p` with the entries of [`tunables`](@ref) replaced by `v`, in the element type `S` of `v`: every
other number of `p` (the box, the time step, the charges, the background fields) is converted
to `S` (so a `Dual` description is not written back in a plain float type), and what is not a
number is kept. With `S` the element type of `p`,
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
    box = N == 1 ? S(p.length) : S.(p.length)
    kmode = N == 1 ? S(p.kmode) : p.kmode
    window = p.window === nothing ? nothing : MovingWindow(S(p.window.start))
    return EMProblem(p.name, box, p.cells, S(p.dt), p.nsteps, species, p.seed, kmode; E0 = S.(p.E0), B0 = S.(p.B0), lasers,
                     boundaries = p.boundaries, window, profile, witness)
end

function with_tunables(p::EMParams, v::AbstractVector{S}) where {S}
    check_tunable_vector(p, v)
    i = Ref(0)
    lasers = map(l -> with_part(S, l, v, i), p.lasers)
    species = map(d -> with_part(S, d, v, i), p.species)
    profile = with_part(S, p.profile, v, i)
    witness = with_part(S, p.witness, v, i)
    return EMParams{S}(; lasers, species, profile, witness, seed = p.seed, E0 = S.(p.E0), B0 = S.(p.B0))
end

function check_tunable_vector(p, v::AbstractVector{S}) where {S}
    S <: Real && isconcretetype(S) ||
        throw(ArgumentError("with_tunables: the vector's element type $S must be a concrete real type (it becomes the element type of the result)"))
    n = tunable_count(p)
    length(v) == n || throw(ArgumentError("with_tunables: $(length(v)) values for the $n tunables of the problem ($(join(tunable_names(p), ", ")))"))
    return nothing
end

# The next `n` entries of `v` as a tuple, `i` the number of entries taken so far.
function take!(v::AbstractVector, i::Base.RefValue{Int}, n::Int)
    first = i[] + firstindex(v)
    i[] += n
    return ntuple(k -> v[first + k - 1], n)
end

# A part of the description in the element type `S` with its tunables the next entries of `v`.
# A laser keeps its `position` and `center` as they are: they are configuration, of a type of
# their own.
function with_part(::Type{S}, l::GaussianLaser{<:Any,P,E}, v, i) where {S,P,E}
    x = take!(v, i, length(LASER_TUNABLES))
    return GaussianLaser{S,P,E}(x..., l.position, l.center, l.envelope)
end
function with_part(::Type{S}, d::SpeciesDescription, v, i) where {S}
    x = take!(v, i, length(SPECIES_TUNABLES))
    return SpeciesDescription{S}(d.name, S(d.charge), S(d.mass), x[1], x[2], d.mode, d.ppc, x[3:5], x[6:8], d.stream; momentum = d.momentum)
end
function with_part(::Type{S}, d::SpeciesParams{<:Any,N}, v, i) where {S,N}
    x = take!(v, i, length(SPECIES_TUNABLES))
    return SpeciesParams{S,N}(x[1], x[2], d.wavevector, x[3:5], x[6:8])
end
function with_part(::Type{S}, ::DensityRamp, v, i) where {S}
    x = take!(v, i, length(PROFILE_TUNABLES))
    return DensityRamp(; vacuum = x[1], ramp = x[2])
end
function with_part(::Type{S}, b::WitnessBunch, v, i) where {S}
    x = take!(v, i, length(WITNESS_TUNABLES))
    return WitnessBunch{S}(b.inject, b.region, b.ppc, x[1:3], x[4:5], x[6], x[7], S(b.charge), S(b.mass))
end
function with_part(::Type{S}, ::WitnessParams, v, i) where {S}
    x = take!(v, i, length(WITNESS_TUNABLES))
    return WitnessParams{S}(x[1:3], x[4:5], x[6], x[7])
end
with_part(::Type, ::Nothing, v, i) = nothing
