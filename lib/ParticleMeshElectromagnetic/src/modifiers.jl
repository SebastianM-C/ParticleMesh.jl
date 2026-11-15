# The field modifiers of the mesh solve (WP-057), and the one solve that walks them. Everything
# that changes the fields of a Yee step besides the curl is a modifier: an absorber of an open
# edge, a source, the edges of a box that is not periodic. A state holds its modifiers as a tuple
# (`EMState.modifiers`), and the solve knows none of them by name. This file is the one place the
# interface is written down.

"""
    FieldModifier

Something that changes the fields within the mesh solve, around the solver's sub-steps. A state
holds its modifiers as a tuple (`state.modifiers`) and [`solve_fields!`](@ref) walks it, so a new
boundary or source is a new subtype with the methods below and a place in a state's tuple (given
by the state's constructor, [`field_modifiers`](@ref), or to an existing state by
[`with_modifiers`](@ref)); the solve is not touched.

A modifier has two halves. The state half is the modifier itself: its arrays and coefficients,
made for a grid and a time step (an absorber's tables and convolution fields, an antenna's plane).
The parameter half is what it reads of the run's parameters (A3; the antenna's laser), which a
state never holds: the solve asks for it once per step,

    modifier_params(m, params, k)        # the parameter half at step k; `nothing` by default

and hands it back to the two hooks, which the solve calls around each sub-step `s` (a
[`BHalf`](@ref) or an [`EStep`](@ref): dispatch on it), every modifier in the tuple's order:

    before_substep!(m, mp, fields, grid, s, workgroupsize)    # nothing by default
    after_substep!(m, mp, fields, grid, s, workgroupsize)     # nothing by default

A hook leaves the guards of every component it changes as the next sub-step needs them (filled,
or zero on an edge that is not periodic), stays within the kernel rules of
`docs/plan/architecture-draft.md` §0.4 (no host branch on device data, no allocation) and reads
no parameter from the modifier. The order of the tuple is the order of the corrections after a
sub-step, so it is part of the scheme: [`field_modifiers`](@ref) says which order the module's
own modifiers take and why.

A modifier with arrays that a run evolves (a PML's convolution fields) says so, and they are then
part of the state's evolved arrays ([`evolved`](@ref)) with everything that follows from it: a
snapshot holds them, `restore!` writes them, a state over other arrays ([`state_view`](@ref), the
SciML harness's partition) has the modifier over those arrays, and `initialize!` zeroes them:

    evolved_arrays(m)              # its evolved arrays, a tuple of fixed length; `()` by default
    on_arrays(m, arrays::Tuple)    # the same modifier over other arrays of that layout
    reset_absorber!(m)             # the start of a run; by default every evolved array zero
    shift_absorber!(m, workgroupsize)   # a moving window's shift by one cell; nothing by default

A modifier that keeps state in an array without listing it is not restarted with the run.
"""
abstract type FieldModifier end

"""
    evolved_arrays(m::FieldModifier) -> Tuple

The arrays of the modifier that a run evolves, in a fixed order and number per modifier type; `()`
for a modifier without state of its own (the default). They join [`evolved`](@ref)`(state)`.
"""
evolved_arrays(::FieldModifier) = ()
evolved_arrays(::Nothing) = ()

"""
    on_arrays(m::FieldModifier, arrays::Tuple) -> FieldModifier

The modifier `m` with `arrays` in the place of its [`evolved_arrays`](@ref) (as many, of the same
types) and everything else it holds kept; no copy. A modifier with evolved arrays defines it; one
without is its own (the default).
"""
on_arrays(m::FieldModifier, ::Tuple{}) = m

"The start of a run for a modifier's evolved arrays: zero, unless the modifier says otherwise. `initialize!` calls it ([`reset_convolution!`](@ref))."
reset_absorber!(m::FieldModifier) = (foreach(a -> fill!(a, zero(eltype(a))), evolved_arrays(m)); nothing)

"A moving window's shift by one cell for a modifier's own arrays (a PML's convolution fields move with the fields); nothing by default."
shift_absorber!(::FieldModifier, wgs) = nothing

"""
    BHalf(τ, t, second)

A B half step of the Yee solve, as a [`FieldModifier`](@ref)'s hooks see it: its length `τ`
(Δt/2), the time `t` of the E it differentiates (the start of the step for the first half step,
its end for the second) and whether it is the `second` half step of the step.
"""
struct BHalf{H,S}
    τ::H
    t::S
    second::Bool
end

"""
    EStep(τ, t)

The E step of the Yee solve, as a [`FieldModifier`](@ref)'s hooks see it: its length `τ` (Δt) and
the time `t` of the B it differentiates (the middle of the step).
"""
struct EStep{H,S}
    τ::H
    t::S
end

"The parameter half of a [`FieldModifier`](@ref) at step `k`, from the run's parameters; `nothing` for a modifier that reads none."
modifier_params(::FieldModifier, p, k) = nothing

"What a [`FieldModifier`](@ref) does to the fields before the sub-step `s`; nothing by default."
before_substep!(::FieldModifier, mp, f, grid, s, wgs) = nothing

"What a [`FieldModifier`](@ref) does to the fields after the sub-step `s`; nothing by default."
after_substep!(::FieldModifier, mp, f, grid, s, wgs) = nothing

# One call per modifier, written out for the tuple's length: a loop or a recursion over a
# heterogeneous tuple dispatches dynamically (and allocates) once inference stops unrolling it,
# past 32 elements at the latest; the generated body is static at every length (T-752).
@generated function resolve_params(ms::NTuple{M,FieldModifier}, p, k) where {M}
    return :(Base.Cartesian.@ntuple $M i -> modifier_params(ms[i], p, k))
end

@generated function walk_modifiers!(hook::H, ms::NTuple{M,FieldModifier}, mps::NTuple{M,Any}, f, grid, s, wgs) where {H,M}
    return quote
        Base.Cartesian.@nexprs $M i -> hook(ms[i], mps[i], f, grid, s, wgs)
        return nothing
    end
end

# The first modifier of type `M` in a tuple, or `nothing`: resolved from the tuple's type, so the
# result is inferred at every length.
@generated function pick_modifier(::Type{M}, ms::Tuple) where {M}
    i = findfirst(E -> E <: M, fieldtypes(ms))
    return i === nothing ? :(nothing) : :(ms[$i])
end

# One sub-step of the solve: the modifiers' `before_substep!`, the solver's update, the modifiers'
# `after_substep!`.
function substep!(update!::U, f, grid, s, modifiers, mps, wgs) where {U}
    walk_modifiers!(before_substep!, modifiers, mps, f, grid, s, wgs)
    update!(f, grid, s.τ; workgroupsize = wgs)
    walk_modifiers!(after_substep!, modifiers, mps, f, grid, s, wgs)
    return nothing
end

"""
    solve_fields!(::Yee, f, grid, dt; workgroupsize = nothing)
    solve_fields!(::Yee, f, grid, dt, modifiers::Tuple, params, k; workgroupsize = nothing)

The mesh-solve slot, the one field solve of the module: from Eⁿ, Bⁿ and the current Jⁿ⁺¹ᐟ² to
Eⁿ⁺¹, Bⁿ⁺¹ by `Bⁿ⁺¹ᐟ² = Bⁿ − (Δt/2)∇×Eⁿ`, `Eⁿ⁺¹ = Eⁿ + Δt(∇×Bⁿ⁺¹ᐟ² − Jⁿ⁺¹ᐟ²)`,
`Bⁿ⁺¹ = Bⁿ⁺¹ᐟ² − (Δt/2)∇×Eⁿ⁺¹` ([`yee_b_half!`](@ref), [`yee_e!`](@ref)), on the fields `f` of
[`field_layout`](@ref)`(Val(N))` on a `PeriodicGrid{N}` (N = 1 and N = 2; guards of E filled on
entry, the currents folded). Every component it changes has its guards filled on return.

The second form is step `k` of a run, from the time `(k − 1)·dt`, with a tuple of
[`FieldModifier`](@ref)s: their parameter halves are taken once from `params`
(`modifier_params(m, params, k)`), and each of the three sub-steps is

    before_substep!   of every modifier, in the tuple's order
    the Yee update    (B half step, E step, B half step)
    after_substep!    of every modifier, in the tuple's order

with the sub-step as a [`BHalf`](@ref) (τ = Δt/2; the time of its E is `(k − 1)·dt`, then `k·dt`)
or an [`EStep`](@ref) (τ = Δt, the time of its B `(k − 1/2)·dt`). The solve names no modifier and
calls each hook by static dispatch on the tuple's element types, with no allocation of its own
at any length of the tuple (T-752). With the empty tuple it is the first form, the periodic
solve, bit for bit (T-751); `params` may be anything the modifiers' `modifier_params` take
(`nothing` when none reads a parameter).

In vacuum a mode of wave vector k advances by the ω of the exact discrete relation
sin(ωΔt/2)/Δt = √(Σ_d sin²(k_dΔ_d/2)/Δ_d²) (T-210, T-320), and Σ(Eⁿ)² + Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² is
conserved (T-323); it is stable for Δt below [`courant_limit`](@ref) and grows beyond it (T-325).
It does not check Δt: the state constructor does ([`check_courant`](@ref)).
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid, dt, modifiers::Tuple, p, k::Integer; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    t = (k - 1) * dt
    mps = resolve_params(modifiers, p, k)
    substep!(yee_b_half!, f, grid, BHalf(h, t, false), modifiers, mps, workgroupsize)
    substep!(yee_e!, f, grid, EStep(T(dt), t + h), modifiers, mps, workgroupsize)
    substep!(yee_b_half!, f, grid, BHalf(h, t + T(dt), true), modifiers, mps, workgroupsize)
    return f
end

solve_fields!(solver::Yee, f::NamedTuple, grid::PeriodicGrid, dt; workgroupsize = nothing) =
    solve_fields!(solver, f, grid, dt, (), nothing, 1; workgroupsize)
