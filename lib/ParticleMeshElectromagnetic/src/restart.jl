# The closed state (WP-056): what a run evolves as one flat tuple of arrays in a fixed order
# (`evolved`), a state over other arrays of that layout (`on_evolved`, the SciML harness's view of
# an integrator's state), and the restart built on the two: `snapshot` copies the evolved arrays
# after a step, `restore!` writes them into a state of the same configuration, and the run
# continues from the next step with the bits of the uninterrupted run (Reference mode, A4).
#
# Everything else in an `EMState` is configuration (fixed when the state is built), scratch (written
# before it is read, within one call) or the one cache a step leaves for the next: the kinetic
# energy `K` at n − 1/2, a function of the evolved momenta, weights and status that the next
# step's observer reads, with at N = 1 the sort's segment bounds its deposition reads
# (`refresh_caches!`). The step's inputs besides the state are the parameters and the step index:
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

"The convolution fields of a placed absorber: four for a PML (the open axis' or the back edge's), none otherwise."
convolution_fields(::Any) = ()
convolution_fields(L::PMLFields) = (L.ψBx, L.ψBz, L.ψEx, L.ψEz)
convolution_fields(L::BackPMLFields) = (L.ψBy, L.ψBz, L.ψEy, L.ψEz)

window_arrays(::Nothing) = ()
window_arrays(w::WindowFrame) = (w.lost..., w.fast..., convolution_fields(w.back)...)

witness_arrays(::Nothing) = ()
witness_arrays(w::WitnessFrame) = species_arrays(w.species)

evolved_aux(::EMState{1}) = ()
evolved_aux(st::EMState{2}) = (convolution_fields(st.aux.layer)..., window_arrays(st.aux.window)..., witness_arrays(st.aux.witness)...)

"""
    evolved(state::EMState) -> Tuple

What a run evolves, as a flat tuple of the state's own arrays (no copy), in this order:

1. per species: the cells and the offsets per axis, the momenta `u[1:3]`, the weights, the status
   and the alive count;
2. the fields of [`field_layout`](@ref), in its order: E and B, and the current J of the last step
   (a step with `Esirkepov()` writes it before reading it; with `NoCurrent()` it is kept as it is);
3. at N = 2, where the state has them: the four convolution fields of the open axis' PML
   (`ψBx`, `ψBz`, `ψEx`, `ψEz`); the moving window's counters, per species `lost` and then per
   species `fast`; the four convolution fields of the window's back-edge PML (`ψBy`, `ψBz`,
   `ψEy`, `ψEz`); the witness bunch's particles, as a species.

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
    return typeof(s)(cell, offset, u, a[2N + 4], a[2N + 5], a[2N + 6], s.q, s.m, s.extras)
end

# Each `*_on(x, a)` returns `x` over the first arrays of `a` and the arrays left.
species_on(::Tuple{}, a::Tuple) = ((), a)
function species_on(sp::Tuple, a::Tuple)
    s = first(sp)
    n = Val(species_length(s))
    rest, a′ = species_on(Base.tail(sp), drop(a, n))
    return (species_on(s, take(a, n)), rest...), a′
end

absorber_on(L, a::Tuple) = (L, a)
function absorber_on(L::PMLFields, a::Tuple)
    ψ = take(a, Val(4))
    return PMLFields(L.node, L.center, ψ..., L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max), drop(a, Val(4))
end
function absorber_on(L::BackPMLFields, a::Tuple)
    ψ = take(a, Val(4))
    return BackPMLFields(L.node, L.center, ψ..., L.scratch, L.cells, L.g, L.n, L.dt, L.sigma_max), drop(a, Val(4))
end

window_on(::Nothing, a::Tuple) = (nothing, a)
function window_on(w::WindowFrame, a::Tuple)
    ns = Val(length(w.lost))
    lost, a = take(a, ns), drop(a, ns)
    fast, a = take(a, ns), drop(a, ns)
    back, a = absorber_on(w.back, a)
    return WindowFrame(w.window, w.dt, w.dx1, w.n1, w.inj, w.g, lost, fast, w.flags, back), a
end

witness_on(::Nothing, a::Tuple) = (nothing, a)
function witness_on(w::WitnessFrame, a::Tuple)
    n = Val(species_length(w.species))
    return WitnessFrame(species_on(w.species, take(a, n)), w.kinj, w.c0, w.nc, w.ppc), drop(a, n)
end

aux_on(st::EMState{1}, a::Tuple) = (st.aux, a)
function aux_on(st::EMState{2}, a::Tuple)
    layer, a = absorber_on(st.aux.layer, a)
    window, a = window_on(st.aux.window, a)
    witness, a = witness_on(st.aux.witness, a)
    return merge(st.aux, (; layer, window, witness)), a
end

"""
    on_evolved(state, arrays::Tuple) -> EMState

An `EMState` with the configuration, the scratch, the cache `K` and the recorder of `state`, whose
evolved arrays are `arrays` (the layout of [`evolved`](@ref)`(state)`, with the array types of the
state's own). No copy. `K` is the cache of whatever was stepped or refreshed last through either
state ([`refresh_caches!`](@ref)).
"""
function on_evolved(st::EMState, a::Tuple)
    length(a) == length(evolved(st)) ||
        throw(ArgumentError("$(length(a)) arrays, the state evolves $(length(evolved(st))) (`evolved`)"))
    species, a = species_on(st.species, a)
    nf = Val(length(st.fields))
    fields = NamedTuple{keys(st.fields)}(take(a, nf))
    aux, _ = aux_on(st, drop(a, nf))
    return EMState(st.numerics, st.grid, st.dt, species, st.ws, fields, st.ρ, st.K, st.recorder, st.setup, aux)
end

# --- the cache ------------------------------------------------------------------------------------

"""
    refresh_caches!(state) -> state

Rebuild what the state caches of its evolved arrays: the kinetic energy `K` at the current
momenta, which the next step's observer reads ([`observe!`](@ref)). `initialize!` and every step
leave it valid; whoever writes the evolved arrays otherwise calls this ([`restore!`](@ref) does,
and the harness at the first step of a solve, whose `u0` its cache has not seen). At N = 2 `K` is
the fixed chunks' sums over the slots as they are. At N = 1 it is a deposition, which in a mode
that sorts reads the sort's segment bounds, so the species are sorted first: the identity
permutation on the particles of a step's end or of `initialize!`, which are sorted, so the bits of
`K` are those the step left.
"""
refresh_caches!(st::EMState{2}) = kinetic!(st)
function refresh_caches!(st::EMState{1})
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, st.grid, ws; workgroupsize = st.numerics.workgroupsize)
    end
    return kinetic!(st)
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
end

"""
    snapshot(state, k) -> EMSnapshot

Save the run of `state` after step `k` (`k = 0`: after `initialize!`): a copy of every evolved
array ([`evolved`](@ref)), on the state's backend, with `k`. A device copy, outside a step. With
the parameters and a state built for the same problem and numerics it is a complete restart
([`restore!`](@ref)); the recorder's histories are the observer's and are not saved
([`histories`](@ref) reads them).
"""
function snapshot(st::EMState, k::Integer)
    k >= 0 || throw(ArgumentError("a snapshot is taken after step k ≥ 0 (0: after initialize!), got k = $k"))
    return EMSnapshot(Int(k), state_units(st), map(copy, evolved(st)))
end

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

# copyto! over two tuples of arrays, unrolled (a foreach over a heterogeneous tuple dispatches
# dynamically and allocates)
copy_arrays!(::Tuple{}, ::Tuple{}) = nothing
@inline function copy_arrays!(dst::Tuple, src::Tuple)
    copyto!(first(dst), first(src))
    return copy_arrays!(Base.tail(dst), Base.tail(src))
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
grid, capacity, element type, or set of absorbers, window and witness bunch), and a snapshot taken
with another time step or cell size. Anything else of the configuration is the caller's to keep
(the charges, the window's start, the witness bunch's injection time), and a restart with other
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

write_evolved!(st::EMState, arrays::Tuple) = (copy_arrays!(evolved(st), arrays); refresh_caches!(st))

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
