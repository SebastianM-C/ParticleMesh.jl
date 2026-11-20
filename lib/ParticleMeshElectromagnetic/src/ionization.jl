# Field ionization of a dopant at N = 2 and 3, deterministic (WP-059, WP-075; A0, A3, A4, A8, §0.4): macro-atoms
# on a fixed lattice whose bound populations follow the rate equations of a field-ionization rate
# in the field at each atom (each level's own law, chosen by its type: `IonizationLevel`), and
# which release their electrons by weight into free slots of an electron species of their own. No random number is drawn: a standard PIC code decides each
# ionization by a Monte Carlo draw against 1 − exp(−W·Δt), which a derivative does not pass
# through, while here the released weight is a smooth function of the field.
#
# - The atoms are an engine `Species` that is never pushed (immobile ions, A2): the lattice of
#   `ppc` points per cell of the engine's quiet start, on every column the plasma is loaded on and
#   across the transverse cells of the description's `rows`. An atom's weight is the dopant density at its place
#   over the points per cell, so the density and the doped section enter through the weights only
#   (A3), and through them the weights of the electrons. With a moving window the atoms shift back
#   with it, the back column's leave, and the injection column's are loaded fresh at their global
#   column, as the plasma's are.
# - Per atom and level j = 1 … L (the transitions the description lists, in order) the bound
#   population f_j, the part of the atom that has released j − 1 electrons and holds its j-th:
#   f_1 = 1 and the others 0 at the start. In a step with the rates W_j of the field |Eⁿ| at the
#   atom, d_j = f_j·(1 − exp(−W_j·Δt)) leaves level j for level j + 1 (the last level's leaves the
#   list), all levels from the populations at the start of the step, so an atom passes one level
#   per step. For one level this is the exact solution in a constant field; the populations are
#   taken at the half steps, as the momenta are, so the rate at tⁿ is the midpoint rule.
# - The released fraction Σ_j d_j is kept per atom in `pending`. At a birth step (every `every`-th
#   step) an atom whose weight times `pending` is positive writes one electron of that weight into
#   a free slot, at the atom's place, with the momentum of an electron at rest at tⁿ taken half a
#   step back (the pusher's −Δt/2 in Eⁿ, Bⁿ, as `initialize!` centres a load), adds `pending` to
#   its `emitted` and zeroes it. Electrons of several levels released in one birth are one
#   particle: they start at one place, at rest. An atom that finds no free slot keeps its
#   `pending` for the next birth step and is counted ([`ionization_lost`](@ref)), so no charge is
#   lost.
# - Charge: an electron is born on top of the ion it leaves, which is immobile, so the birth changes
#   no charge density and no field, and from then on the electron's current is the engine's
#   charge-conserving `deposit_flux!`. Gauss's law therefore holds to roundoff with the ions'
#   charge counted, the atoms' weight times `emitted` at their places ([`ionization_charge!`](@ref)).
#   The energy the ionization takes from the field (the ionization current) is not modelled.
# - Static shapes (§0.4 rule 1): the atoms' buffer is the lattice, full and in the engine's sorted
#   order at every step (columns fastest, then the transverse cells, the in-cell index fastest within a cell; a
#   window's shift writes the fresh column into the slots of the one that left and sorts), so an
#   atom's slot is its place on the lattice. The electrons' buffer has the capacity the
#   description gives; a birth step offers every atom slot i the electron slot count + i behind
#   the electrons' alive count, and the next sort compacts the ones not taken.

"""
    IonizationLevel{T}

One transition of an atom or ion that a static electric field drives, with its numbers in element
type `T`. The rate law is the level's type: [`ADKLevel`](@ref) is the tunnelling rate and
[`TongLinLevel`](@ref) the same with a barrier-suppression factor, and a [`FieldIonization`](@ref)
may list levels of different types. A level type `MyLevel{T} <: IonizationLevel{T}` of isbits
fields plugs in with four methods:

- `ionization_rate_sq(level::MyLevel{T}, E2::T) -> T`: the rate (1/ω₀) from the squared field
  strength `E2 = |E|²` ((mₑcω₀/e)²), a device function within §0.4 (no allocation, no float
  literal). It must be ≥ 0 and C¹ in the field and in the level's numbers (A2), and exactly zero
  with zero partials at `E2 = 0`: an atom in no field releases nothing and takes no electron slot.
- `level_as(S, level) -> MyLevel{S}`: the level in the element type `S`, also a device function.
- `level_numbers(level) -> NamedTuple`: the numbers that are parameters (A3: they enter through
  the rate and may be `Dual`s). Their names are the level's tunables,
  `ionization_level<j>_<name>` ([`tunable_names`](@ref)).
- `with_level_numbers(level, S, numbers::Tuple) -> MyLevel{S}`: the level with these numbers, in
  the order of `level_numbers`, built by its checking constructor, the rest of the level (what is
  configuration, such as a cutoff) kept and converted to `S`.
"""
abstract type IonizationLevel{T} end

"The element type of a level's numbers."
level_eltype(::IonizationLevel{T}) where {T} = T

function ionization_rate_sq end
function level_as end
function level_numbers end
function with_level_numbers end

"""
    ionization_rate(level::IonizationLevel, E) -> rate

The ionization rate of `level` in a field of strength `E ≥ 0` (mₑcω₀/e), in 1/ω₀, by the level's
law ([`IonizationLevel`](@ref)); a device function, generic in the element type.
"""
@inline function ionization_rate(l::IonizationLevel{T}, E::S) where {T,S<:Real}
    R = promote_type(T, S)                     # a field of another type (a host call): both in the common type
    x = R(E)
    return ionization_rate_sq(level_as(R, l), x * x)
end

# The checks the level types' constructors share: the numbers of the tunnelling form.
function check_tunnel_numbers(name, prefactor, field, power, cutoff)
    prefactor >= 0 && isfinite(prefactor) || throw(ArgumentError("$name level's prefactor must be finite and ≥ 0, got $prefactor"))
    field > 0 && isfinite(field) || throw(ArgumentError("$name level's field must be finite and positive, got $field"))
    isfinite(power) || throw(ArgumentError("$name level's power must be finite, got $power"))
    cutoff > 0 && isfinite(cutoff) || throw(ArgumentError("$name level's cutoff must be finite and positive, got $cutoff"))
    return nothing
end

"""
    ADKLevel(prefactor, field, power; cutoff = 64)

One tunnel-ionization transition of an atom or ion ([`IonizationLevel`](@ref)), with its rate in a
static electric field of strength `E` in the form of Ammosov, Delone and Krainov, in A1 units:

    W(E) = prefactor · (field/E)^power · exp(−field/(3E))        (1/ω₀, E in mₑcω₀/e)

In atomic units, for the ionization energy I and the charge Z of the ion left behind,
`field = 2(2I)^{3/2}`, `power = 2n* − |m| − 1` with `n* = Z/√(2I)` and
`prefactor = C²·B·I`, `C² = 2^{2n*}/(n*·Γ(n* + l* + 1)·Γ(n* − l*))`,
`B = (2l + 1)(l + |m|)!/(2^{|m|}·|m|!·(l − |m|)!)`. The conversion to A1 units depends on the
laser frequency and lives in the user layer (A1; `scripts/ionization/atomic.jl` builds a level
from an ionization energy in eV and a wavelength). All three numbers may be `Dual`s.

The rate is taken to zero where it is negligible, smoothly (the AD-smoothness rule of A2): for
`x = field/(3E)` it is multiplied by the C¹ step `s²(3 − 2s)`, `s = clamp(8(cutoff − x)/cutoff, 0, 1)`,
so it is exactly zero for `x ≥ cutoff` (and at `E = 0`), unchanged for `x ≤ 7·cutoff/8`, and its
derivative is continuous. With the default the factor acts below `exp(−56)` of the prefactor's
scale. An atom in a field below the cutoff releases nothing and occupies no electron slot.
No barrier-suppression correction is applied: the form is used as it stands at any field, and
above the field `field/(32n*)` at which the barrier is gone it overestimates the rate
([`TongLinLevel`](@ref) has the correction).
"""
struct ADKLevel{T} <: IonizationLevel{T}
    prefactor::T
    field::T
    power::T
    cutoff::T
end

function ADKLevel(prefactor, field, power; cutoff = 64)
    T = float(promote_type(typeof(prefactor), typeof(field), typeof(power), typeof(cutoff)))   # integers as Float64
    check_tunnel_numbers("an ADK", prefactor, field, power, cutoff)
    return ADKLevel{T}(T(prefactor), T(field), T(power), T(cutoff))
end

level_as(::Type{T}, l::ADKLevel) where {T} = ADKLevel{T}(T(l.prefactor), T(l.field), T(l.power), T(l.cutoff))
level_numbers(l::ADKLevel) = (; prefactor = l.prefactor, field = l.field, power = l.power)
with_level_numbers(l::ADKLevel, ::Type{S}, x::NTuple{3,Any}) where {S} = ADKLevel(x...; cutoff = S(l.cutoff))

# The rate from |E|² (a device function): the test for the cutoff is on E², so no square root and
# no division is taken where the rate is zero, and a `Dual` field of value zero gives a zero with
# zero partials, not a NaN.
@inline function ionization_rate_sq(l::ADKLevel{T}, E2::T) where {T}
    c3 = 3 * l.cutoff
    c3 * c3 * E2 <= l.field * l.field && return zero(T)      # x = field/(3E) ≥ cutoff; a NaN field goes on and gives a NaN
    r = l.field / sqrt(E2)
    x = r / 3
    s = min(8 * (l.cutoff - x) / l.cutoff, one(T))
    return l.prefactor * r^l.power * exp(-x) * (s * s * (3 - 2 * s))
end

"""
    adk_rate(level::ADKLevel, E) -> rate

The ionization rate of an [`ADKLevel`](@ref) in a field of strength `E ≥ 0` (mₑcω₀/e), in 1/ω₀:
its [`ionization_rate`](@ref).
"""
@inline adk_rate(l::ADKLevel, E::Real) = ionization_rate(l, E)

"""
    TongLinLevel(prefactor, field, power, suppression; cutoff = 64)

One field-ionization transition of an atom or ion ([`IonizationLevel`](@ref)) whose rate holds
past the field at which the barrier is gone: the tunnelling rate of an [`ADKLevel`](@ref) of the
same `prefactor`, `field`, `power` and `cutoff`, times the empirical factor of Tong and Lin
(J. Phys. B 38 (2005) 2593), in A1 units

    W(E) = prefactor · (field/E)^power · exp(−field/(3E)) · exp(−suppression·E/field)

In atomic units the factor is `exp(−α·(Z²/I)·E/(2I)^{3/2})` with the fitted number α of the
species (6 for hydrogen), I the ionization energy and Z the charge of the ion left behind, so
`suppression = 4α·n*²`, a pure number (`n* = Z/√(2I)`; the user layer computes it,
`scripts/ionization/atomic.jl`). The barrier is gone at `E_bs = I²/(4Z) = field/(32n*)`, where
the factor is `exp(−α·n*/8)`.

The factor is smooth in the field and in the level's four numbers, which may all be `Dual`s; at
weak fields it goes to 1 and the rate to the tunnelling rate, with the same cutoff. It is a fit
of rates computed up to about twice `E_bs`: far above that the rate falls to zero with rising
field, which no atom does, so a level in fields several times its `E_bs` needs another law.
`suppression = 0` is the [`ADKLevel`](@ref)'s rate, bit for bit.
"""
struct TongLinLevel{T} <: IonizationLevel{T}
    prefactor::T
    field::T
    power::T
    suppression::T
    cutoff::T
end

function TongLinLevel(prefactor, field, power, suppression; cutoff = 64)
    T = float(promote_type(typeof(prefactor), typeof(field), typeof(power), typeof(suppression), typeof(cutoff)))
    check_tunnel_numbers("a Tong–Lin", prefactor, field, power, cutoff)
    suppression >= 0 && isfinite(suppression) || throw(ArgumentError("a Tong–Lin level's suppression must be finite and ≥ 0, got $suppression"))
    return TongLinLevel{T}(T(prefactor), T(field), T(power), T(suppression), T(cutoff))
end

level_as(::Type{T}, l::TongLinLevel) where {T} = TongLinLevel{T}(T(l.prefactor), T(l.field), T(l.power), T(l.suppression), T(l.cutoff))
level_numbers(l::TongLinLevel) = (; prefactor = l.prefactor, field = l.field, power = l.power, suppression = l.suppression)
with_level_numbers(l::TongLinLevel, ::Type{S}, x::NTuple{4,Any}) where {S} = TongLinLevel(x...; cutoff = S(l.cutoff))

# The ADK level's rate with the factor's exponent −suppression·E/field = −suppression/r added to
# the tunnelling exponent: one exponential, and with a zero suppression the same bits.
@inline function ionization_rate_sq(l::TongLinLevel{T}, E2::T) where {T}
    c3 = 3 * l.cutoff
    c3 * c3 * E2 <= l.field * l.field && return zero(T)
    r = l.field / sqrt(E2)
    x = r / 3
    s = min(8 * (l.cutoff - x) / l.cutoff, one(T))
    return l.prefactor * r^l.power * exp(-x - l.suppression / r) * (s * s * (3 - 2 * s))
end

"""
    DopantSection(; start, stop, ramp)

The longitudinal profile of a dopant along axis 1, in the global (lab) coordinate x of the box (of
a moving window's first position), c/ω₀: the factor `r(x − start)·(1 − r(x − stop))` with the C¹
rise `r(s) = sin²(π·s/(2·ramp))` clamped to 0 for `s ≤ 0` and to 1 for `s ≥ ramp`, so the dopant
rises from `start` to `start + ramp`, holds, and falls from `stop` to `stop + ramp`. It multiplies
the dopant's density and enters through the atoms' weights only (A3); all three may be `Dual`s.
`ramp` must be positive and `stop ≥ start`.
"""
struct DopantSection{T}
    start::T
    stop::T
    ramp::T
end
function DopantSection(; start, stop, ramp)
    ramp > 0 || throw(ArgumentError("a dopant section needs a positive ramp length, got $ramp"))
    stop >= start || throw(ArgumentError("a dopant section needs stop ≥ start, got start = $start and stop = $stop"))
    T = float(promote_type(typeof(start), typeof(stop), typeof(ramp)))
    return DopantSection{T}(T(start), T(stop), T(ramp))
end

section_as(::Type{T}, ::Nothing) where {T} = nothing
section_as(::Type{T}, s::DopantSection) where {T} = DopantSection{T}(T(s.start), T(s.stop), T(s.ramp))

"""
    FieldIonization(; density, levels, capacity, section = nothing, ppc = nothing, every = 1, rows = nothing)

A dopant that the fields ionize, of an [`EMProblem`](@ref) at N = 2 (WP-059) or N = 3 (WP-075),
in A1 units, with no random numbers: macro-atoms at rest on the lattice of `ppc = (p₁, …, p_N)`
points per cell (one per axis; one point per cell by default), `((k₁ − ½)/p₁, …, (k_N − ½)/p_N)`
in each cell, on every column the plasma is loaded on and across the transverse cells whose centre
lies inside `rows`: at N = 2 the rows `(y_lo, y_hi)` (c/ω₀), at N = 3 one range per transverse
axis, `((y_lo, y_hi), (z_lo, z_hi))`, either of them `nothing` for the whole axis; `nothing` (the
default) is every transverse cell. An atom's weight is `density` (n_c, the density of dopant
atoms) times the `section`'s factor at its place ([`DopantSection`](@ref), along axis 1; 1 without
one) over `prod(ppc)`.

`levels` lists the transitions the fields drive, in order, each an [`IonizationLevel`](@ref) with
the rate law of its type (an [`ADKLevel`](@ref), a [`TongLinLevel`](@ref)): the atom
starts with all of them bound and releases one electron per transition, level j + 1 only from the
part that has passed level j. Electrons the dopant has lost before the run (an L shell taken as
ionized by the pulse's foot, say) are not listed; they belong to the plasma's density. In every
step each level's bound population loses the fraction `1 − exp(−W·Δt)` of the rate `W` in the
field at the atom, and every `every`-th step (the birth steps) an atom releases what it has lost
since its last birth as one electron of that weight, at rest at the atom's place, into the
electron species of the ionization ([`ionization_electrons`](@ref): charge −1, mass 1, pushed and
deposited as the plasma is, `capacity` slots). What an atom loses in the `every` steps between two
births is born together, at rest at the later birth step, so the momentum the laser leaves an
electron with is that of the birth step's phase: keep `every·Δt` well below the laser period
(`every = 1` costs one particle per releasing atom and step). At a birth step atom slot i writes
into electron slot `alive + i`, so the step needs as many slots as the electrons alive plus the
highest atom slot that releases (the number of atom slots at most), and every atom of non-zero
weight in a field above its level's cutoff releases and takes a slot, however small the weight.
An atom that finds no slot keeps what it has lost for the next birth step and is counted
([`ionization_lost`](@ref)).

`density`, the levels' numbers and the section's are parameters and may be `Dual`s (A3: they
enter through the weights and the rates, never through a position, [`IonizationParams`](@ref));
`ppc`, `every`, `rows` and `capacity` are configuration. The description does not know its N: the
problem checks that `ppc` has N entries and `rows` the form of its N, and resolves `ppc = nothing`.
"""
struct FieldIonization{T,L,S,R,LV<:NTuple{L,IonizationLevel{T}},P}
    density::T
    levels::LV
    section::S
    ppc::P
    every::Int
    rows::R
    capacity::Int
end

# One range of transverse cells' centres: a pair (lo, hi) with lo < hi, in Float64.
function rows_range(r, rows)
    rr = Tuple(r)
    length(rr) == 2 && all(x -> x isa Real, rr) && rr[1] < rr[2] ||
        throw(ArgumentError("rows = $rows: each range is (lo, hi) with lo < hi, got $r"))
    return Float64.(rr)
end

# The rows of a description: `nothing`, one range (N = 2), or one range or `nothing` per transverse axis (N = 3).
rows_of(::Nothing) = nothing
function rows_of(rows)
    r = Tuple(rows)
    length(r) == 2 || throw(ArgumentError("rows = $rows needs (y_lo, y_hi) at N = 2 or ((y_lo, y_hi), (z_lo, z_hi)) at N = 3"))
    all(x -> x isa Real, r) && return rows_range(r, rows)
    all(x -> x === nothing || !(x isa Real), r) && !all(isnothing, r) ||
        throw(ArgumentError("rows = $rows needs (y_lo, y_hi) at N = 2 or ((y_lo, y_hi), (z_lo, z_hi)) at N = 3, either range nothing for the whole axis"))
    return map(x -> x === nothing ? nothing : rows_range(x, rows), r)
end

function FieldIonization(; density, levels, capacity::Integer, section = nothing, ppc = nothing, every::Integer = 1, rows = nothing)
    lv = levels isa IonizationLevel ? (levels,) : Tuple(levels)
    !isempty(lv) && all(l -> l isa IonizationLevel, lv) ||
        throw(ArgumentError("levels must be one IonizationLevel (an ADKLevel, a TongLinLevel) or a non-empty tuple of them, got $levels"))
    section === nothing || section isa DopantSection || throw(ArgumentError("section must be a DopantSection or nothing, got $section"))
    T = promote_type(typeof(density), map(level_eltype, lv)...)
    section === nothing || (T = promote_type(T, typeof(section.start)))
    T = float(T)
    density >= 0 && isfinite(density) || throw(ArgumentError("the dopant density must be finite and ≥ 0, got $density"))
    pp = ppc === nothing ? nothing : Int.(Tuple(ppc))
    pp === nothing || length(pp) in (2, 3) && all(>(0), pp) ||
        throw(ArgumentError("the dopant lattice needs ppc > 0 on each of the N = 2 or 3 axes, got $ppc"))
    every >= 1 || throw(ArgumentError("every must be at least 1 (a birth every `every` steps), got $every"))
    capacity >= 1 || throw(ArgumentError("the ionization electrons need a capacity of at least one slot, got $capacity"))
    rw = rows_of(rows)
    sec = section_as(T, section)
    lvT = map(l -> level_as(T, l), lv)
    return FieldIonization{T,length(lv),typeof(sec),typeof(rw),typeof(lvT),typeof(pp)}(T(density), lvT, sec, pp, Int(every), rw, Int(capacity))
end

"The description in element type `T` (the problem's)."
function ionization_as(::Type{T}, d::FieldIonization{<:Any,L}) where {T,L}
    sec = section_as(T, d.section)
    lv = map(l -> level_as(T, l), d.levels)
    return FieldIonization{T,L,typeof(sec),typeof(d.rows),typeof(lv),typeof(d.ppc)}(T(d.density), lv, sec, d.ppc, d.every, d.rows, d.capacity)
end

# The description of a problem of N axes in its element type `T`: `ppc = nothing` as one point per
# cell on every axis, and an `ArgumentError` unless `ppc` has N entries and `rows` the form of N.
function ionization_as(::Type{T}, d::FieldIonization, ::Val{N}) where {T,N}
    N in (2, 3) || throw(ArgumentError("field ionization is implemented at N = 2 and 3 only"))
    pp = d.ppc === nothing ? ntuple(_ -> 1, Val(N)) : d.ppc
    length(pp) == N || throw(ArgumentError("the dopant lattice of a problem at N = $N needs ppc of $N entries, got $(d.ppc)"))
    rows_fit(d.rows, Val(N)) ||
        throw(ArgumentError("rows = $(d.rows) at N = $N: " * (N == 2 ? "(y_lo, y_hi)" : "((y_lo, y_hi), (z_lo, z_hi)), either range nothing for the whole axis")))
    e = ionization_as(T, d)
    return FieldIonization{T,length(e.levels),typeof(e.section),typeof(e.rows),typeof(e.levels),typeof(pp)}(e.density, e.levels, e.section, pp, e.every,
                                                                                                         e.rows, e.capacity)
end
rows_fit(::Nothing, ::Val) = true
rows_fit(r::NTuple{2,Float64}, ::Val{N}) where {N} = N == 2
rows_fit(r::Tuple, ::Val{N}) where {N} = N == 3

"""
    IonizationParams{T}

The parameters of a [`FieldIonization`](@ref) (A3): the dopant `density` (n_c) and its `section`
([`DopantSection`](@ref) or `nothing`), read where the atoms are loaded (the initial load and a
moving window's refill), and the `levels` (a tuple of [`IonizationLevel`](@ref)s), read by every
step's rates; all may be `Dual`s. The lattice, the birth cadence and the electrons' capacity are
configuration of the state.
"""
struct IonizationParams{T,L,S,LV<:NTuple{L,IonizationLevel{T}}}
    density::T
    levels::LV
    section::S
end

function IonizationParams{T}(density, levels::Tuple, section = nothing) where {T}
    sec = section_as(T, section)
    lv = map(l -> level_as(T, l), levels)
    return IonizationParams{T,length(lv),typeof(sec),typeof(lv)}(T(density), lv, sec)
end

ionization_params(::Type{T}, ::Nothing) where {T} = nothing
ionization_params(::Type{T}, d::FieldIonization) where {T} = IonizationParams{T}(d.density, d.levels, d.section)
ionization_params(::Type{T}, d::IonizationParams) where {T} = IonizationParams{T}(d.density, d.levels, d.section)

# The loader's numbers of the dopant profile: the density, and the section's start, stop and ramp
# (no section: a rise that is over everywhere and a fall that never starts).
function dopant_density_params(p::IonizationParams{T}) where {T}
    s = p.section
    return s === nothing ? (p.density, T(-Inf), T(Inf), one(T)) : (p.density, s.start, s.stop, s.ramp)
end

"The dopant density at the global coordinate `x` along axis 1 (a device function): `p = (density, start, stop, ramp)`."
@inline dopant_density(x, p) = p[1] * ramp_factor(x, p[2], p[4]) * (one(x) - ramp_factor(x, p[3], p[4]))

"""
    IonizationFrame

A [`FieldIonization`](@ref) placed on a grid of N = 2 or 3 axes by the state: the `atoms` (a
`Species{N}` that is never pushed; `w` the atom's weight, and as extras per atom `pending`, the
fraction released since its last birth, `emitted`, the fraction born as electrons so far, and
`bound_1 … bound_L`, the bound population of each level) with the workspace their sort uses, the
`electrons` (the species the births are written into) with theirs, the birth cadence `every`, the
lattice (per axis its first cell `c0` and its cells `nc`: the columns from column 1 along axis 1,
the rows of `rows` along the transverse axes, and `ppc` points per cell), the device counter
`lost` of the births that found no free slot, and the scratch `flags` (one per atom slot). The
dopant's density, section and levels are not here ([`IonizationParams`](@ref)).
"""
struct IonizationFrame{A,AW,E,EW,P<:Tuple,C,F}
    atoms::A
    atoms_ws::AW
    electrons::E
    electrons_ws::EW
    every::Int
    c0::P
    nc::P
    ppc::P
    lost::C
    flags::F
end

"The names and element types of the atoms' extras for `L` levels."
function atom_extras(::Type{T}, L::Integer) where {T}
    names = (:pending, :emitted, ntuple(j -> Symbol(:bound_, j), L)...)
    return NamedTuple{names}(ntuple(_ -> T, L + 2))
end

"The bound populations of the atoms' extras, as a tuple of `L` arrays."
@inline bound_populations(x::NamedTuple, ::Val{L}) where {L} = ntuple(j -> x[j + 2], Val(L))

# The transverse ranges of a description's rows on N axes, one per transverse axis (`nothing`: the whole axis).
transverse_rows(::Nothing, ::Val{N}) where {N} = ntuple(_ -> nothing, Val(N - 1))
transverse_rows(r::NTuple{2,Float64}, ::Val{2}) = (r,)
transverse_rows(r::Tuple, ::Val{3}) = r

# The lattice of a description on a grid, per axis the first cell and the cells: along axis 1 the
# columns from column 1 (the plasma's: all of a static box, those up to the injection column of a
# moving window), along each transverse axis the cells whose centre lies in its range of `rows`.
function ionization_placement(d::FieldIonization, grid::PeriodicGrid{N}, window) where {N}
    n, h = ncells(grid), cellsize(grid)
    tr = transverse_rows(d.rows, Val(N))
    across = ntuple(t -> tr[t] === nothing ? (1, n[t + 1]) : witness_cells(tr[t]..., h[t + 1], n[t + 1], t + 1), Val(N - 1))
    ncols = window === nothing ? n[1] : injection_column(window)
    return ((Int32(1), map(r -> Int32(r[1]), across)...), (Int32(ncols), map(r -> Int32(r[2]), across)...), Int32.(d.ppc))
end

function IonizationFrame(problem::EMProblem{N,T}, grid, backend, numerics, window) where {N,T}
    d = problem.ionization
    L = length(d.levels)
    c0, nc, ppc = ionization_placement(d, grid, window)
    natoms = prod(Int.(nc)) * prod(d.ppc)
    atoms = Species(backend, T, natoms; q = one(T), m = one(T), N, extras = atom_extras(T, L))
    electrons = Species(backend, T, d.capacity; q = -one(T), m = one(T), N, extras = flux_extras(T, N))
    flux = numerics.current isa Esirkepov ? (numerics.shape, rho_stagger(Val(N))) : nothing
    # a birth step adds one electron slot per atom slot behind the electrons' count; the atoms keep
    # cells, since the shift refills a column's atoms in the slots of the back column's
    ews = Workspace(electrons, grid; flux, append = sort_append(numerics, natoms), bins = sort_bins(numerics, Val(N); window = window !== nothing))
    return IonizationFrame(atoms, Workspace(atoms, grid), electrons, ews, d.every, c0, nc, ppc,
                           KernelAbstractions.zeros(backend, Int32, 1), KernelAbstractions.zeros(backend, Int32, natoms))
end

"The ionization's frame of a state, or `nothing` (at N = 1, and at N = 2 without a [`FieldIonization`](@ref))."
ionization_frame(st::EMState) = haskey(st.aux, :ionization) ? st.aux.ionization : nothing

"The ionization's electrons of a state, as a tuple: `()` without ionization."
born_species(::Nothing) = ()
born_species(fr::IonizationFrame) = (fr.electrons,)
born_species(st::EMState) = born_species(ionization_frame(st))
born_workspaces(::Nothing) = ()
born_workspaces(fr::IonizationFrame) = (fr.electrons_ws,)

"The species that deposit a current, with their workspaces in the same order: the problem's, then the ionization's electrons."
deposited_species(st::EMState) = (st.species..., born_species(st)...)
deposited_workspaces(st::EMState) = (st.ws..., born_workspaces(ionization_frame(st))...)

# --- the atoms' load ----------------------------------------------------------------------------

# The fresh atoms of the window columns col0 … col0 + ncol − 1 on the lattice's transverse cells,
# at the window position `origin` (global column = window column + origin): work item i is the
# in-cell index k of the cell (column, transverse cell), columns fastest, then the transverse
# cells (axis 2 fastest), k fastest within a cell (the engine's sorted order), written into slot
# `cell·stride·K + k`, K the points per cell. With `stride = 1` that is slot i, the whole lattice
# in its order (the initial load, `ncol` the lattice's columns); with `ncol = 1` and `stride` the
# lattice's columns it is the slots of the first column of each transverse cell, which a window's
# shift has just retired. `row0` and `nrow` are the lattice's first transverse cell and its
# transverse cells per axis. An atom starts with its first level bound and nothing released; it is
# at rest and stays there.
@kernel function load_atoms_kernel!(a, dp, dx, ppc::NTuple{N,Int32}, col0::Int32, ncol::Int32, row0::NTuple{M,Int32}, nrow::NTuple{M,Int32},
                                    origin::Int32, stride::Int32, ::Val{L}) where {N,M,L}
    i = Int32(@index(Global, Linear))
    T = eltype(a.w)
    K = lattice_points(ppc)
    cell = (i - Int32(1)) ÷ K
    k = (i - Int32(1)) % K
    c1 = col0 + cell % ncol
    ct = lattice_cell(cell ÷ ncol, row0, nrow)
    δ = lattice_offset(T, k, ppc)
    x1 = (T(c1 + origin - Int32(1)) + δ[1]) * dx[1]
    slot = cell * stride * K + k + Int32(1)
    set_position!(a, slot, (c1, ct...), δ)
    @inbounds begin
        a.u[1][slot] = zero(T)
        a.u[2][slot] = zero(T)
        a.u[3][slot] = zero(T)
        a.w[slot] = dopant_density(x1, dp) / T(K)
        a.extras.pending[slot] = zero(T)
        a.extras.emitted[slot] = zero(T)
        a.status[slot] = ALIVE
    end
    b = bound_populations(a.extras, Val(L))
    ntuple(j -> (@inbounds b[j][slot] = ifelse(j == 1, one(T), zero(T)); nothing), Val(L))
end

# The points per cell, as an Int32 (`prod` would widen it to an Int).
@inline lattice_points(ppc::NTuple{2,Int32}) = ppc[1] * ppc[2]
@inline lattice_points(ppc::NTuple{3,Int32}) = ppc[1] * ppc[2] * ppc[3]

# The transverse cell of the 0-based index t over the lattice's transverse cells (axis 2 fastest).
@inline lattice_cell(t::Int32, row0::NTuple{1,Int32}, nrow::NTuple{1,Int32}) = (row0[1] + t,)
@inline lattice_cell(t::Int32, row0::NTuple{2,Int32}, nrow::NTuple{2,Int32}) = (row0[1] + t % nrow[1], row0[2] + t ÷ nrow[1])

# The in-cell offsets of the 0-based in-cell index k (axis 1 fastest), ((k_d + ½)/p_d) per axis.
@inline lattice_offset(::Type{T}, k::Int32, ppc::NTuple{2,Int32}) where {T} =
    ((T(k % ppc[1]) + one(T) / 2) / T(ppc[1]), (T(k ÷ ppc[1]) + one(T) / 2) / T(ppc[2]))
@inline lattice_offset(::Type{T}, k::Int32, ppc::NTuple{3,Int32}) where {T} =
    ((T(k % ppc[1]) + one(T) / 2) / T(ppc[1]), (T(k ÷ ppc[1] % ppc[2]) + one(T) / 2) / T(ppc[2]),
     (T(k ÷ (ppc[1] * ppc[2])) + one(T) / 2) / T(ppc[3]))

# After `nnew` slots were offered behind the count: the count, capped at the capacity.
@kernel function bump_count_kernel!(count, nnew::Int32, cap::Int32)
    @inbounds count[1] = min(count[1] + nnew, cap)
end

# Load the atoms of `ncol` columns from window column `col0` at the window position `origin`, into
# the slots `load_atoms_kernel!` names for `stride`.
function load_atoms!(fr::IonizationFrame, grid, p::IonizationParams, col0::Integer, ncol::Integer, origin::Integer, stride::Integer, wgs)
    a = fr.atoms
    nnew = Int(ncol) * prod(Int.(Base.tail(fr.nc))) * Int(prod(fr.ppc))
    launcher(load_atoms_kernel!, get_backend(a), wgs)(a, dopant_density_params(p), cellsize(grid), fr.ppc, Int32(col0), Int32(ncol),
                                                      Base.tail(fr.c0), Base.tail(fr.nc), Int32(origin), Int32(stride), Val(length(p.levels));
                                                      ndrange = nnew)
    return fr
end

"The start of a run (`initialize!`): no electron, the counter zero, and the atoms of the lattice loaded in their ground state from the parameters."
initialize_ionization!(::Nothing, grid, p, wgs) = nothing
function initialize_ionization!(fr::IonizationFrame, grid, p::IonizationParams, wgs)
    for s in (fr.atoms, fr.electrons)
        fill!(s.status, ParticleMeshCore.FREE)
        fill!(s.count, Int32(0))
    end
    fill!(fr.lost, Int32(0))
    load_atoms!(fr, grid, p, 1, fr.nc[1], 0, 1, wgs)
    fill!(fr.atoms.count, Int32(capacity(fr.atoms)))
    return nothing
end

# --- the step ----------------------------------------------------------------------------------

# One work item per atom slot. An alive atom: the fields at its place, each level's rate, the
# populations after the step from those at its start (level j loses d_j and gains d_{j−1}), and the
# released fraction added to `pending`. At a birth step, an atom whose weight times `pending` is
# positive writes its electron into slot count[1] + i when that slot exists (at the atom's place,
# with the momentum of rest at tⁿ taken half a step back by the pusher), moves `pending` into
# `emitted`, and otherwise keeps it and raises its flag.
@kernel function ionize_kernel!(a, e, flags, @Const(count), f, levels::NTuple{L,Any}, pusher, shape, g::Int32, dt, qm_h, birth::Bool,
                                cap::Int32) where {L}
    i = Int32(@index(Global, Linear))
    deferred = Int32(0)
    @inbounds if a.status[i] == ALIVE
        T = eltype(a.w)
        vN = Val(length(a.cell))
        c = particle_cell(a, i, vN)
        δ = particle_offset(a, i, vN)
        E, B = fields_at(f, c, δ, shape, g)
        E2 = E[1] * E[1] + E[2] * E[2] + E[3] * E[3]
        b = bound_populations(a.extras, Val(L))
        old = ntuple(j -> (@inbounds b[j][i]), Val(L))
        wdt = map(l -> ionization_rate_sq(l, E2) * dt, levels)
        out = ntuple(j -> old[j] * -expm1(-wdt[j]), Val(L))
        ntuple(j -> (@inbounds b[j][i] = old[j] * exp(-wdt[j]) + (j == 1 ? zero(T) : out[max(j - 1, 1)]); nothing), Val(L))
        pending = a.extras.pending[i] + sum(out)
        w = a.w[i] * pending
        if birth && w > zero(T)
            slot = count[1] + i
            if slot <= cap
                u = push_momentum(pusher, (zero(T), zero(T), zero(T)), E, B, qm_h)
                set_position!(e, slot, c, δ)
                e.u[1][slot] = u[1]
                e.u[2][slot] = u[2]
                e.u[3][slot] = u[3]
                e.w[slot] = w
                e.status[slot] = ALIVE
                a.extras.emitted[i] += pending
                pending = zero(T)
            else
                deferred = Int32(1)
            end
        end
        a.extras.pending[i] = pending
    end
    @inbounds flags[i] = deferred
end

"Whether `step!` sorts the species `s` in every step (the charge-conserving current in a mode that sorts), so that its count is the alive count at the start of a step."
sorted_every_step(st::EMState, s) = st.numerics.current isa Esirkepov && sorts(st, s)

# The sort after a window shift moved every particle by one cell: from the order of the step's
# sort where the step sorts the species (`resort_by_cell!`, whose precondition that is), anew where
# it does not, since the particles may then have moved any distance since their last sort.
function sort_after_shift!(st::EMState, s, ws)
    wgs = st.numerics.workgroupsize
    sorted_every_step(st, s) ? resort_by_cell!(s, st.grid, ws; workgroupsize = wgs) : sort_by_cell!(s, st.grid, ws; workgroupsize = wgs)
    return nothing
end

"""
    ionize!(frame, state, params, k)

The ionization of step `k`, before the push: every atom's populations by the rates in Eⁿ at its
place, and, when `k` is a birth step (a multiple of the frame's `every`), the births into the
electrons' slots behind their alive count (the electrons are sorted first where the step does not
sort them, so that the count is the alive count), the count advanced by one slot per atom slot,
and the births that found no slot added to `lost`. `nothing` without a frame.
"""
ionize!(::Nothing, st, p, k) = nothing
function ionize!(fr::IonizationFrame, st::EMState, p::EMParams, k::Integer)
    (; numerics, grid, dt) = st
    fields = particle_fields(st)
    wgs = numerics.workgroupsize
    a, e = fr.atoms, fr.electrons
    be = get_backend(a)
    birth = k % fr.every == 0
    birth && !sorted_every_step(st, e) && sort_by_cell!(e, grid, fr.electrons_ws; workgroupsize = wgs)
    launcher(ionize_kernel!, be, wgs)(a, e, fr.flags, e.count, fields, p.ionization.levels, numerics.pusher, read_shape(numerics),
                                      Int32(guards(grid)), dt, -e.q / e.m * dt / 4, birth, Int32(capacity(e)); ndrange = capacity(a))
    if birth
        launcher(bump_count_kernel!, be, wgs)(e.count, Int32(capacity(a)), Int32(capacity(e)); ndrange = 1)
        sum!(fr.lost, fr.flags; init = false)
    end
    return nothing
end

# A moving window's shift by one cell, to its position after `s_new` shifts: the electrons one cell
# back with the back column's retired and the slots compacted; the atoms one cell back, the
# injection column's atoms written fresh, at their global column, into the slots of the back
# column's (each row's first, the buffer being in sorted order), and the sort that puts the buffer
# back in sorted order, the populations with their atoms.
shift_ionization!(::Nothing, st, p, s_new) = nothing
function shift_ionization!(fr::IonizationFrame, st::EMState, p::EMParams, s_new::Integer)
    wgs = st.numerics.workgroupsize
    (; electrons, atoms) = fr
    launcher(shift_particles_kernel!, get_backend(electrons), wgs)(electrons; ndrange = capacity(electrons))
    sort_after_shift!(st, electrons, fr.electrons_ws)
    launcher(shift_particles_kernel!, get_backend(atoms), wgs)(atoms; ndrange = capacity(atoms))
    load_atoms!(fr, st.grid, p.ionization, fr.nc[1], 1, s_new, fr.nc[1], wgs)
    sort_by_cell!(atoms, st.grid, fr.atoms_ws; workgroupsize = wgs)
    return nothing
end

# --- the checks of the configuration and the parameters -----------------------------------------

check_ionization_params(::Nothing, ::Nothing) = nothing
check_ionization_params(fr, p) = throw(ArgumentError("the parameters $(p === nothing ? "have no" : "have an") ionization, the state $(fr === nothing ? "has none" : "has one")"))
function check_ionization_params(fr::IonizationFrame, p::IonizationParams)
    L = length(fr.atoms.extras) - 2
    length(p.levels) == L || throw(ArgumentError("$(length(p.levels)) ionization levels in the parameters, $L in the state"))
    return nothing
end

# --- observables (host reads, outside a step) ---------------------------------------------------

function need_ionization(st::EMState)
    fr = ionization_frame(st)
    fr === nothing && throw(ArgumentError("the state has no ionization (EMProblem(...; ionization = FieldIonization(...)))"))
    return fr
end

"""
    ionization_electrons(state::EMState) -> Species

The electrons the state's [`FieldIonization`](@ref) has released (the engine's `Species`, charge
−1 and mass 1: cells, offsets, momenta `u = γv` at the last half step, weights in n_c per
particle, status). They are pushed, deposited, retired and shifted as the problem's species are,
and are part of the state's kinetic energy. A sum over them is a smooth function of the
parameters while none of them has left: an electron that an open edge or a moving window's back
edge retires takes its weight out at a step that depends on the parameters, as the plasma's
particles do ([`window_back_band`](@ref), #370). An `ArgumentError` without ionization.
"""
ionization_electrons(st::EMState) = need_ionization(st).electrons

"""
    ionization_state(state::EMState) -> NamedTuple

The dopant atoms of `state`, per atom slot (host arrays): the position `x` and `y`, and at N = 3 `z`
(c/ω₀; window coordinates with a moving window), the weight `w` (n_c per atom), the bound population of each
level `bound` (a tuple of arrays, one per level), the fraction `pending` (released, not yet born),
the fraction `emitted` (born as electrons) and `alive`. An atom of `L` levels has released
`pending + emitted = L − Σ_j (L − j + 1)·bound[j]` electrons per atom, to roundoff (level j holds
the part that has released j − 1); for one level, `bound[1] + pending + emitted = 1`. A device
read: never inside a step.
"""
function ionization_state(st::EMState{N}) where {N}
    fr = need_ionization(st)
    a = fr.atoms
    h = cellsize(st.grid)
    L = length(a.extras) - 2
    pos = NamedTuple{(:x, :y, :z)[1:N]}(ntuple(d -> (Array(a.cell[d]) .- 1 .+ Array(a.offset[d])) .* h[d], Val(N)))
    return (; pos..., w = Array(a.w), bound = map(Array, bound_populations(a.extras, Val(L))), pending = Array(a.extras.pending),
            emitted = Array(a.extras.emitted), alive = Array(a.status) .== ALIVE)
end

"""
    ionization_lost(state::EMState) -> Int

The births that found no free electron slot since `initialize!` (the electrons' `capacity` was too
small at that birth step): each such atom kept what it had released and tried again at the next
birth step, so no charge is lost, but its electrons are born later than they would have been. A
device read: never inside a step.
"""
ionization_lost(st::EMState) = Int(Array(need_ionization(st).lost)[1])

"The charge a dopant atom's ion has gained, `q·w·emitted` (a `deposit!` quantity): the opposite of its born electrons' charge."
struct EmittedCharge end
@inline (::EmittedCharge)(s, i) = @inbounds s.q * s.w[i] * s.extras.emitted[i]

"""
    ionization_charge!(ρ, state::EMState) -> ρ

The charge density of the dopant's ions on the primal nodes: every atom's weight times its
`emitted` fraction, at the atom's place, deposited in the state's mode as the particles' charge is
(with a moving window the x-guards dropped, the transverse guards folded, as [`window_charge!`](@ref)
does). It is the opposite of the charge the born electrons had at their birth, so with it Gauss's
law holds to roundoff in a run with ionization. A diagnostic, outside `step!`.
"""
function ionization_charge!(ρ, st::EMState{N}) where {N}
    fr = need_ionization(st)
    (; numerics, grid) = st
    a = fr.atoms
    fill!(ρ, zero(eltype(ρ)))
    mode = deposit_mode(st, a)
    mode isa Reference && sort_by_cell!(a, grid, fr.atoms_ws)
    deposit!(mode, ρ, a, numerics.shape, rho_stagger(Val(N)), grid, fr.atoms_ws; quantity = EmittedCharge())
    zero_x_edges!(window_frame(st), ρ)
    fold_guards!(ρ, grid)
    return ρ
end
