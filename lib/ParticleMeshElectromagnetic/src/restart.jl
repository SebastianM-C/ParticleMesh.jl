# The closed state (WP-056): what a run evolves as one flat tuple of arrays in a fixed order
# (`evolved`), a state over other arrays of that layout (`on_evolved`, the SciML harness's view of
# an integrator's state; `state_view`, the same with a cache of its own, for a state handed out),
# and the restart built on the two: `snapshot` copies the evolved arrays
# after a step, `restore!` writes them into a state of the same configuration, and the run
# continues from the next step with the bits of the uninterrupted run (Reference mode, A4).
#
# Everything else in an `EMState` is configuration (fixed when the state is built), scratch (written
# before it is read, within one call) or the one cache a step leaves for the next where the next
# step's observer reads it (WP-062; the host flag `fresh` says whether it did): the kinetic
# energy `K` at n − 1/2, a function of the evolved momenta, weights and status, with at N = 1 the sort's segment bounds its deposition reads
# (`refresh_caches!`). With `EMNumerics(; resort = true)` a step also leaves the order of its last
# sort in each deposited workspace (the segments, the alive count, the arrays sorted), which the next
# step's sort starts from (`resort_by_cell!`, WP-041); it belongs to the arrays it was taken on, so
# a state over other arrays sorts in full, and whatever writes the evolved arrays between two steps
# calls `forget_order!` on those workspaces (`refresh_after_write!` here, `initialize!`, the harness's
# first step). The step's inputs besides the state are the parameters and the step index:
# the window's position, the antennas' switch-off and the witness bunch's injection are functions
# of `k`.
#
# Equality is on alive slots: a free slot keeps the data of whatever held it last, and the
# particle extras are the push's scratch, so no result depends on either (T-742).

# --- the evolved arrays --------------------------------------------------------------------------

"The arrays a run evolves of a species: the cell and the offset per axis, `u[1:3]`, the weight, the status and the alive count."
species_arrays(s::Species) = (s.cell..., s.offset..., s.u..., s.w, s.status, s.count)
species_length(::Species{N}) where {N} = 2N + 6

flat_species() = ()
flat_species(s, rest...) = (species_arrays(s)..., flat_species(rest...)...)

window_arrays(::Nothing) = ()
window_arrays(w::WindowFrame) = (w.lost..., w.fast...)

witness_arrays(::Nothing) = ()
witness_arrays(w::WitnessFrame) = species_arrays(w.species)

# The atoms with their populations (extras that a run evolves, unlike the push's), the electrons
# and the counter of the births that found no slot.
ionization_arrays(::Nothing) = ()
ionization_arrays(fr::IonizationFrame) = (species_arrays(fr.atoms)..., values(fr.atoms.extras)..., species_arrays(fr.electrons)..., fr.lost)

# The evolved arrays of every modifier but the open axis' absorber, in the tuple's order, written
# out for the tuple's length (static at every length, as the solve's walk).
@generated function other_evolved(ms::NTuple{M,FieldModifier}) where {M}
    parts = [:(evolved_arrays(ms[$i])...) for i in 1:M if !(fieldtype(ms, i) <: OpenAbsorber)]
    return :(($(parts...),))
end

opens_evolved(::Tuple{}) = ()
opens_evolved(ls::Tuple) = (evolved_arrays(first(ls))..., opens_evolved(Base.tail(ls))...)

# What a run evolves besides the particles and the fields, in `evolved`'s order: the open axes'
# absorbers' arrays (in the tuple's order), the window's counters, the arrays of every other modifier in the tuple's
# order (`evolved_arrays`: the back edge's PML, and any modifier a state was given), the witness
# bunch, the ionization. The open axis' absorber comes before the counters whatever its place in the tuple: that
# is the order the list has had since it exists, and no configuration's list moves.
evolved_aux(st::EMState) = (opens_evolved(open_absorbers(st))..., window_arrays(window_frame(st))...,
                            other_evolved(st.modifiers)..., witness_arrays(witness_frame(st))...,
                            ionization_arrays(ionization_frame(st))...)

"""
    evolved(state::EMState) -> Tuple

What a run evolves, as a flat tuple of the state's own arrays (no copy), in this order. The tuple
is valid until the state's next sort: a state that owns its species' arrays swaps them with the
workspace's spare set when it sorts, so a tuple held across a `step!` then holds scratch that a
later sort overwrites; take a fresh `evolved(state)` after a step:

1. per species: the cells and the offsets per axis, the momenta `u[1:3]`, the weights, the status
   and the alive count;
2. the fields of [`field_layout`](@ref), in its order: E and B, and the current J of the last step
   (a step with `Esirkepov()` writes it before reading it; with `NoCurrent()` it is kept as it is);
3. at N = 2 and 3, where the state has them: the four convolution fields of each open axis' PML
   (`values(ψ)`: along y `ψ.Bx`, `ψ.Bz`, `ψ.Ex`, `ψ.Ez`; along z `ψ.Bx`, `ψ.By`, `ψ.Ex`, `ψ.Ey`;
   axis 2 before axis 3); the moving window's counters, per species `lost` and then per
   species `fast`; the evolved arrays of every other field modifier in the order of the state's
   tuple ([`evolved_arrays`](@ref): the four convolution fields of the window's back-edge PML,
   `ψ.By`, `ψ.Bz`, `ψ.Ey`, `ψ.Ez`, and those of a modifier the state was given with
   [`with_modifiers`](@ref)); the witness bunch's particles, as a species; of a
   [`FieldIonization`](@ref) the atoms as a species, their extras (`pending`, `emitted` and the
   bound population of each level), the electrons as a species and the counter `lost`.

The element types stay mixed (`Int32`, `UInt8` and `T`). With the step index `k`, the parameters
and the configuration the state was built for, these arrays are the whole run: [`snapshot`](@ref)
and [`restore!`](@ref) save and continue it, and [`state_partition`](@ref) is the same tuple as an
`ArrayPartition`. Of a slot that is not alive only the status means anything (a free slot keeps
stale data), and the particle extras, the workspaces, `ρ`, `K` and the recorder are not here:
scratch, a cache and an observer's.
"""
evolved(st::EMState) = (flat_species(st.species...)..., values(st.fields)..., evolved_aux(st)...)

# --- a state over other arrays of the same layout ------------------------------------------------

@inline take(a::Tuple, ::Val{n}) where {n} = ntuple(i -> a[i], Val(n))
@inline drop(a::Tuple, ::Val{n}) where {n} = ntuple(i -> a[n + i], Val(length(a) - n))

# A species of the same type as `s` over the first arrays of `a`: same charge, mass and extras.
function species_on(s::Species{N}, a::Tuple) where {N}
    cell = ntuple(i -> a[i], Val(N))
    offset = ntuple(i -> a[N + i], Val(N))
    u = (a[2N + 1], a[2N + 2], a[2N + 3])
    return Species(cell, offset, u, a[2N + 4], a[2N + 5], a[2N + 6], s.q, s.m, s.extras)
end

# Each `*_on(x, a)` returns `x` over the first arrays of `a` and the arrays left.
species_on(::Tuple{}, a::Tuple) = ((), a)
function species_on(sp::Tuple, a::Tuple)
    s = first(sp)
    n = Val(species_length(s))
    rest, a′ = species_on(Base.tail(sp), drop(a, n))
    return (species_on(s, take(a, n)), rest...), a′
end

modifier_on(::Nothing, a::Tuple) = (nothing, a)
function modifier_on(m::FieldModifier, a::Tuple)
    n = Val(length(evolved_arrays(m)))
    return on_arrays(m, take(a, n)), drop(a, n)
end

# The open axes' absorbers over the first arrays of `a`, in their order, and the arrays left.
opens_on(::Tuple{}, a::Tuple) = ((), a)
function opens_on(ls::Tuple, a::Tuple)
    l, a = modifier_on(first(ls), a)
    rest, a = opens_on(Base.tail(ls), a)
    return (l, rest...), a
end

# The tuple `ms` over the arrays `a`, consumed in `other_evolved`'s order; the open axes' absorbers,
# whose arrays come earlier in the list, are given (`opens`, in the tuple's order).
@generated function others_on(ms::NTuple{M,FieldModifier}, opens::Tuple, a::Tuple) where {M}
    body = Expr(:block)
    names = [Symbol(:m_, i) for i in 1:M]
    j = 0
    for i in 1:M
        push!(body.args, fieldtype(ms, i) <: OpenAbsorber ? (j += 1; :($(names[i]) = opens[$j])) : :(($(names[i]), a) = modifier_on(ms[$i], a)))
    end
    push!(body.args, :(return ($(names...),), a))
    return body
end

window_on(::Nothing, a::Tuple) = (nothing, a)
function window_on(w::WindowFrame, a::Tuple)
    ns = Val(length(w.lost))
    lost, a = take(a, ns), drop(a, ns)
    fast, a = take(a, ns), drop(a, ns)
    return WindowFrame(w.window, w.dt, w.dx1, w.n1, w.inj, w.g, lost, fast, w.flags), a
end

witness_on(::Nothing, a::Tuple) = (nothing, a)
function witness_on(w::WitnessFrame, a::Tuple)
    n = Val(species_length(w.species))
    return WitnessFrame(species_on(w.species, take(a, n)), w.kinj, w.c0, w.nc, w.ppc), drop(a, n)
end

ionization_on(::Nothing, a::Tuple) = (nothing, a)
function ionization_on(fr::IonizationFrame, a::Tuple)
    na = Val(species_length(fr.atoms))
    core, a = species_on(fr.atoms, take(a, na)), drop(a, na)
    nx = Val(length(fr.atoms.extras))
    extras, a = NamedTuple{keys(fr.atoms.extras)}(take(a, nx)), drop(a, nx)
    atoms = Species(core.cell, core.offset, core.u, core.w, core.status, core.count, core.q, core.m, extras)
    ne = Val(species_length(fr.electrons))
    electrons, a = species_on(fr.electrons, take(a, ne)), drop(a, ne)
    return IonizationFrame(atoms, fr.atoms_ws, electrons, fr.electrons_ws, fr.every, fr.c0, fr.nc, fr.ppc, a[1], fr.flags),
           drop(a, Val(1))
end

# The modifiers and `aux` of `st` over the arrays `a`, which are consumed in `evolved_aux`'s order:
# every modifier is the same modifier over its part of `a` (`on_arrays`).
function aux_on(st::EMState, a::Tuple)
    opens, a = opens_on(open_absorbers(st), a)
    window, a = window_on(window_frame(st), a)
    modifiers, a = others_on(st.modifiers, opens, a)
    witness, a = witness_on(witness_frame(st), a)
    ionization, a = ionization_on(ionization_frame(st), a)
    return modifiers, aux_with(st.aux, window, witness, ionization), a
end

aux_with(aux::NamedTuple{()}, window, witness, ionization) = aux
aux_with(aux::NamedTuple, window, witness, ionization) = merge(aux, (; window, witness, ionization))

"""
    on_evolved(state, arrays::Tuple, K, recorder) -> EMState

An `EMState` with the configuration and the scratch of `state`, whose evolved arrays are `arrays`:
as many arrays as [`evolved`](@ref)`(state)`, each of the type of its place there (an
`ArgumentError` otherwise; a comparison of types, so nothing is converted and nothing copied),
with the kinetic-energy cache `K`, the `recorder` and the cache's flag `fresh` given (a new
one, stale, by default). The cache belongs to the arrays: `K`
is valid for them once [`refresh_caches!`](@ref) or a step has written it through this state, so
a caller hands every set of arrays a buffer of its own (the harness: one for the integrator's
state, apart from its cache state's; [`state_view`](@ref) allocates one). The sizes are the
caller's to check where the arrays come from outside (`check_evolved`).
"""
function on_evolved(st::EMState, a::Tuple, K, recorder, fresh::Base.RefValue{Bool} = Ref(false))
    a isa typeof(evolved(st)) ||
        throw(ArgumentError("the arrays are not of the types of the state's evolved arrays (`evolved`): $(length(a)) arrays " *
                            "against $(length(evolved(st))), or another array or element type"))
    species, a = species_on(st.species, a)
    nf = Val(length(st.fields))
    fields = NamedTuple{keys(st.fields)}(take(a, nf))
    modifiers, aux, _ = aux_on(st, drop(a, nf))
    return EMState(st.numerics, st.grid, st.dt, species, st.ws, fields, st.sharpened, st.ρ, K, recorder, st.setup, modifiers, aux, fresh)
end

"""
    state_view(state, arrays::Tuple, recorder = empty_recorder(state.recorder)) -> EMState

A state to hand out over `arrays` (the layout of [`evolved`](@ref)`(state)`, checked: number,
array types and sizes), consistent on its own: its kinetic-energy cache `K` is a new buffer
computed from `arrays` ([`refresh_caches!`](@ref)), and its recorder is its own, empty and of
the state's mode and length, unless one is given. So an observation of it reads the kinetic
energy of its own particles, and a run continued from it records into its own histories. It
shares the configuration and the scratch (workspaces, extras, `ρ`) with `state`: use one of them
at a time. No copy of `arrays`; at N = 1 in a mode that sorts they are sorted, the identity for a
state saved from such a mode. The view borrows `arrays`: they stay valid only as long as the
tuple does, until the next sort of the state they were taken from (see [`evolved`](@ref)).
"""
function state_view(st::EMState, a::Tuple, recorder = empty_recorder(st.recorder))
    check_evolved(st, a)
    return refresh_after_write!(on_evolved(st, a, similar(st.K), recorder))
end

# --- the cache ------------------------------------------------------------------------------------

"""
    refresh_caches!(state) -> state

Rebuild what the state caches of its evolved arrays: the kinetic energy `K` at the current
momenta, which the next step's observer reads ([`observe!`](@ref)), and mark it current
(`state.fresh`). `initialize!` leaves it current, and a step does where its observer reads it at
the next step ([`reads_kinetic`](@ref): the default recorder, a function or `nothing` at every
step); whoever writes the evolved arrays otherwise calls this, through
[`refresh_after_write!`](@ref) ([`restore!`](@ref) does, and the harness at the first step of a
solve, whose `u0` its cache has not seen), and `observe!` calls
it for an observer that reads a cache a step left stale. At N ≥ 2 `K` is
the fixed chunks' sums over the slots as they are. At N = 1 it is a deposition, which in a mode
that sorts reads the sort's segment bounds, so the species are sorted first: the identity
permutation on the particles of a step's end or of `initialize!`, which are sorted, so the bits of
`K` are those the step left.
"""
refresh_caches!(st::EMState) = kinetic!(st)
function refresh_caches!(st::EMState{1})
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, st.grid, ws; workgroupsize = st.numerics.workgroupsize)
    end
    return kinetic!(st)
end

"""
    refresh_after_write!(state) -> state

What a caller that wrote the state's evolved arrays from elsewhere does before the next step
([`restore!`](@ref), [`state_view`](@ref), the harness at the first step of a solve): every
workspace forgets the order of its last sort ([`forget_order!`](@ref)), since it was taken on other
arrays, so the next step's sort is the full one, which gives the bits that a sort from the right
order would (T-854); then [`refresh_caches!`](@ref).
"""
function refresh_after_write!(st::EMState)
    foreach(forget_order!, deposited_workspaces(st))
    return refresh_caches!(st)
end

# --- snapshot and restore -------------------------------------------------------------------------

"What gives a snapshot's numbers their meaning: the time step (t = k·Δt) and the cell sizes (x = (cell − 1 + offset)·Δx), as the state holds them."
state_units(st::EMState) = (st.dt, Tuple(cellsize(st.grid))...)

"""
    EMSnapshot

A saved run: the step `k` after which it was taken (0: after `initialize!`), copies of the
[`evolved`](@ref) arrays in their order (`arrays`), and the time step and cell sizes of the state
they came from (`units`), which [`restore!`](@ref) compares. Made by [`snapshot`](@ref) or
[`load_snapshot`](@ref).
"""
struct EMSnapshot{U<:Tuple,A<:Tuple}
    k::Int
    units::U
    arrays::A
    function EMSnapshot(k::Integer, units::U, arrays::A) where {U<:Tuple,A<:Tuple}
        k >= 0 || throw(ArgumentError("a snapshot is taken after step k ≥ 0 (0: after initialize!), got k = $k"))
        return new{U,A}(Int(k), units, arrays)
    end
end

"""
    snapshot(state, k) -> EMSnapshot

Save the run of `state` after step `k` (`k = 0`: after `initialize!`): a copy of every evolved
array ([`evolved`](@ref)), on the state's backend, with `k`. A device copy, outside a step. With
the parameters and a state built for the same problem and numerics it is a complete restart
([`restore!`](@ref)); the recorder's histories are the observer's and are not saved
([`histories`](@ref) reads them).
"""
snapshot(st::EMState, k::Integer) = EMSnapshot(k, state_units(st), map(copy, evolved(st)))

# Whether `src` has the layout of the state's evolved arrays: as many arrays, each of the element
# type and size of its place (static shapes, a host check; the array types may differ, so host
# copies go back to a device state).
function check_evolved(st::EMState, src::Tuple)
    dst = evolved(st)
    length(src) == length(dst) && all(map((x, y) -> eltype(x) === eltype(y) && size(x) == size(y), src, dst)) ||
        throw(ArgumentError("the arrays do not have the layout of the state's evolved arrays (`evolved`): another number of arrays " *
                            "($(length(src)) against $(length(dst))), element type or size; build the state for the problem and numerics they were saved from"))
    return nothing
end

# copyto! over two tuples of as many arrays, one call per pair written out for the tuples' length
# (a foreach over a heterogeneous tuple dispatches dynamically and allocates, and so does a
# recursion over the tail past 32 arrays, where inference stops unrolling it; T-742)
@generated function copy_arrays!(dst::NTuple{N,Any}, src::NTuple{N,Any}) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> copyto!(dst[i], src[i])
        return nothing
    end
end

"""
    restore!(state, snapshot::EMSnapshot) -> state
    restore!(state, arrays::Tuple) -> state

Write a saved run into `state`: copy the snapshot's arrays into the state's evolved arrays
([`evolved`](@ref)) and rebuild the state's cache from them ([`refresh_caches!`](@ref)). The state
then is the saved one, and `run!(state, params, n; first = snapshot.k + 1)` continues it: for a
state built for the same problem and numerics, with the same parameters, in `Reference()` mode
the continuation has the bits of the uninterrupted run, per element type and backend (A4; T-740),
in every alive particle, the fields, a PML's convolution fields, the window's counters, the
witness bunch and every history entry from step `k + 1` on. The state may be new (no
`initialize!` is needed) or used; what it held is overwritten, and its recorder's entries up to
`k` are not touched.

Refused with an `ArgumentError`: arrays of another layout (number, element type or size: another
grid, capacity or element type, a species, a PML, a window or a witness bunch more or less), and
a snapshot taken with another time step or cell size. Nothing else is compared: the rest of the
configuration is the caller's to keep (the charges, the window's start, the witness bunch's
injection time, an absorber's profile, or an absorbing layer where the saved run had none, which
adds no array), the values in the arrays are taken as they are, and a restart with other
parameters or numerics is a run of its own from the saved state.

The second form takes the arrays alone, in `evolved`'s layout (those of another state, or of a
partition of the harness), with no step and no units to compare.
"""
function restore!(st::EMState, snap::EMSnapshot)
    check_evolved(st, snap.arrays)
    snap.units == state_units(st) ||
        throw(ArgumentError("the snapshot was taken with the time step and cell sizes $(snap.units), the state has $(state_units(st)): " *
                            "step $(snap.k) of the one is not a time of the other"))
    return write_evolved!(st, snap.arrays)
end

restore!(st::EMState, arrays::Tuple) = (check_evolved(st, arrays); write_evolved!(st, arrays))

write_evolved!(st::EMState, arrays::Tuple) = (copy_arrays!(evolved(st), arrays); refresh_after_write!(st))

# --- files ----------------------------------------------------------------------------------------

const SNAPSHOT_MAGIC = Tuple(codeunits("PMEMSNP1"))

snapshot_error(what) = throw(ArgumentError("not a snapshot of this state: $what"))

"""
    save_snapshot(path_or_io, snapshot::EMSnapshot)

Write a snapshot as raw bytes: a header (a tag, `k`, the units, and per array the element size and
the dimensions) and every array's memory in order, in the machine's byte order, so every bit
comes back, a dual's partials included. The file does not describe itself: the element types and
the configuration come from the state it is read for ([`load_snapshot`](@ref)). Device arrays are
copied to the host first.
"""
function save_snapshot(io::IO, snap::EMSnapshot)
    write(io, collect(SNAPSHOT_MAGIC))
    write(io, Int64(snap.k), Int64(length(snap.units)), Int64(sizeof(eltype(snap.units))))
    write(io, collect(snap.units))
    write(io, Int64(length(snap.arrays)))
    for a in snap.arrays
        h = Array(a)
        write(io, Int64(sizeof(eltype(h))), Int64(ndims(h)))
        foreach(d -> write(io, Int64(d)), size(h))
        write(io, h)
    end
    return nothing
end
save_snapshot(path::AbstractString, snap::EMSnapshot) = open(io -> save_snapshot(io, snap), path, "w")

"""
    load_snapshot(path_or_io, state) -> EMSnapshot

Read a snapshot written by [`save_snapshot`](@ref), for `state` (a state built for the problem and
numerics of the saved run, which gives the element types and sizes): host arrays, to hand to
[`restore!`](@ref). An `ArgumentError` when the file is not a snapshot, or not one of this state's
layout (another number of arrays, element size or dimensions), or ends early.
"""
function load_snapshot(io::IO, st::EMState)
    units = state_units(st)
    U = eltype(units)
    tag = read(io, length(SNAPSHOT_MAGIC))
    Tuple(tag) == SNAPSHOT_MAGIC || snapshot_error("the file does not start with a snapshot's tag")
    try
        k, nu, su = read(io, Int64), read(io, Int64), read(io, Int64)
        k >= 0 && (nu, su) == (length(units), sizeof(U)) ||
            snapshot_error("step $k, $nu units of $su bytes; the state has $(length(units)) of $(sizeof(U))")
        u = read!(io, Vector{U}(undef, nu))
        dst = evolved(st)
        n = read(io, Int64)
        n == length(dst) || snapshot_error("$n arrays, the state evolves $(length(dst))")
        arrays = map(dst) do a
            E = eltype(a)
            es, nd = read(io, Int64), read(io, Int64)
            (es, nd) == (sizeof(E), ndims(a)) || snapshot_error("an array of $nd dimensions and $es bytes per element where the state has $(ndims(a)) and $(sizeof(E))")
            dims = ntuple(_ -> Int(read(io, Int64)), ndims(a))
            dims == size(a) || snapshot_error("an array of size $dims where the state has $(size(a))")
            read!(io, Array{E}(undef, dims))
        end
        eof(io) || snapshot_error("bytes left after the last array")
        return EMSnapshot(Int(k), Tuple(u), arrays)
    catch err
        err isa EOFError && snapshot_error("the file ends early")
        rethrow()
    end
end
load_snapshot(path::AbstractString, st::EMState) = open(io -> load_snapshot(io, st), path, "r")
