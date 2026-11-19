"""
    ParticleMeshCoreEnzymeCoreExt

Enzyme's reverse rules for the engine's transfers and sort (WP-077), through EnzymeRules: when
Enzyme differentiates `deposit!`, `gather!`, `deposit_flux!` (N = 1 and 2), `sort_by_cell!` or
`resort_by_cell!` in reverse mode, it runs the primal and, in the reverse sweep, the engine's
hand-written pullbacks (`deposit_pullback!`, `gather_pullback!`, `flux_pullback!`,
`sort_pullback!`) instead of differentiating the kernels. So Enzyme keeps no tape of the transfer
kernels, and the sort, whose permutation of the species' arrays Enzyme does not differentiate
(#509 and the WP-050 notes), is the inverse permutation of the shadow.

What the rules tape: a copy of the species' arrays (the pullbacks read the positions, weights and
quantities of the forward call, which a later push or sort overwrites, and Enzyme's `overwritten`
does not see writes into arrays reached through the species' reference), the field a gather read,
and the sort's permutation (the next sort overwrites it). Width 1 only (no batched shadows).
Keyword arguments (`dt`, the quantities, `workgroupsize`) are constant, declared so through
`EnzymeRules.inactive_kwarg` (marked experimental in EnzymeCore 0.8): no derivative with respect to
`dt` flows through the current, even when `dt` comes from a differentiated state. The workspace,
the grid, the shape, the stagger and the mode take no adjoint.
"""
module ParticleMeshCoreEnzymeCoreExt

using ParticleMeshCore
using ParticleMeshCore: Species, SpeciesArrays, PeriodicGrid, Workspace, species_data, flat_arrays
import EnzymeCore
using EnzymeCore: Annotation, Const, Duplicated, MixedDuplicated, Active
using EnzymeCore.EnzymeRules: EnzymeRules, AugmentedReturn, RevConfigWidth, needs_primal, needs_shadow

const SpeciesArg{N} = Union{Const{<:Species{N}},Duplicated{<:Species{N}},MixedDuplicated{<:Species{N}}}
const FluxSpeciesArg = Union{SpeciesArg{1},SpeciesArg{2}}
const ArrayArg = Union{Const{<:AbstractArray},Duplicated{<:AbstractArray}}

"The shadow of an annotated argument, `nothing` for a constant one."
shadow(x::Duplicated) = x.dval
shadow(x::MixedDuplicated) = x.dval[]
shadow(::Const) = nothing

"The adjoint the reverse rule returns for an argument: a zero for an `Active` one (the rules give none), else `nothing`."
no_adjoint(x::Active) = EnzymeCore.make_zero(x.val)
no_adjoint(_) = nothing

"A species over copies of the arrays of `s`, for the tape (borrowed: a sort of it would copy back, never swap)."
function snapshot(s::Species)
    a = species_data(s)
    c = SpeciesArrays(map(copy, a.cell), map(copy, a.offset), map(copy, a.u), copy(a.w), copy(a.status), map(copy, a.extras))
    return Species(c.cell, c.offset, c.u, c.w, c.status, copy(s.count), s.q, s.m, c.extras)
end

# The species' arrays as the forward call read them. Always copied: Enzyme's `overwritten` does not
# see a later write into the arrays a species reaches through its reference (T-1064 overwrites the
# weights and offsets after a deposit, and a step's push does).
species_taped(s) = s isa Const ? nothing : snapshot(s.val)

shadow_return(config, x) = needs_shadow(config) ? x : nothing
primal_return(config, x) = needs_primal(config) ? x : nothing

"A zero field like `a` for a constant field whose adjoint a pullback reads (the adjoint of a constant is zero)."
field_shadow(x::Duplicated) = x.dval
field_shadow(x::Const) = fill!(similar(x.val), zero(eltype(x.val)))

# The keyword arguments are constant (`dt`, the quantities, `workgroupsize`): a step reads `dt` from
# its state, which Enzyme would otherwise take as a differentiable keyword and refuse.
for f in (:deposit!, :gather!, :deposit_flux!, :sort_by_cell!, :resort_by_cell!)
    @eval EnzymeRules.inactive_kwarg(::typeof($f), args...; kwargs...) = nothing
end

# --- deposit! ---------------------------------------------------------------------------------

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(deposit!)}, ::Type{RT}, mode::Const, ρ::ArrayArg,
                                      s::SpeciesArg{N}, shape::Const, stagger::Const, grid::Annotation{<:PeriodicGrid{N}},
                                      ws::Annotation{<:Workspace}; kw...) where {RT,N}
    ret = func.val(mode.val, ρ.val, s.val, shape.val, stagger.val, grid.val, ws.val; kw...)
    tape = ρ isa Const ? nothing : species_taped(s)
    return AugmentedReturn(primal_return(config, ret), shadow_return(config, shadow(ρ)), tape)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(deposit!)}, ::Type{RT}, tape, mode::Const, ρ::ArrayArg,
                             s::SpeciesArg{N}, shape::Const, stagger::Const, grid::Annotation{<:PeriodicGrid{N}},
                             ws::Annotation{<:Workspace}; quantity = ChargeWeight(), workgroupsize = nothing) where {RT,N}
    if !(ρ isa Const || s isa Const)
        deposit_pullback!(shadow(s), ρ.dval, something(tape, s.val), shape.val, stagger.val, grid.val; quantity, workgroupsize)
    end
    return map(no_adjoint, (mode, ρ, s, shape, stagger, grid, ws))
end

# --- gather! ----------------------------------------------------------------------------------

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(gather!)}, ::Type{RT}, out::ArrayArg, a::ArrayArg,
                                      s::SpeciesArg{N}, shape::Const, stagger::Const, grid::Annotation{<:PeriodicGrid{N}};
                                      kw...) where {RT,N}
    ret = func.val(out.val, a.val, s.val, shape.val, stagger.val, grid.val; kw...)
    active = !(out isa Const)
    stape = active && !(s isa Const) ? species_taped(s) : nothing
    atape = active && !(s isa Const) ? copy(a.val) : nothing
    return AugmentedReturn(primal_return(config, ret), shadow_return(config, shadow(out)), (atape, stape))
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(gather!)}, ::Type{RT}, tape, out::ArrayArg, a::ArrayArg,
                             s::SpeciesArg{N}, shape::Const, stagger::Const, grid::Annotation{<:PeriodicGrid{N}};
                             workgroupsize = nothing) where {RT,N}
    if !(out isa Const)
        atape, stape = tape
        gather_pullback!(shadow(a), shadow(s), out.dval, something(atape, a.val), something(stape, s.val), shape.val, stagger.val,
                         grid.val; workgroupsize)
    end
    return map(no_adjoint, (out, a, s, shape, stagger, grid))
end

# --- deposit_flux! (N = 1 and 2) ----------------------------------------------------------------

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(deposit_flux!)}, ::Type{RT}, mode::Const,
                                      Jx::ArrayArg, Jy::ArrayArg, Jz::ArrayArg, s::FluxSpeciesArg, shape::Const, stagger::Const,
                                      grid::Annotation{<:PeriodicGrid}, ws::Annotation{<:Workspace}; kw...) where {RT}
    ret = func.val(mode.val, Jx.val, Jy.val, Jz.val, s.val, shape.val, stagger.val, grid.val, ws.val; kw...)
    active = !all(J -> J isa Const, (Jx, Jy, Jz))
    tape = active ? species_taped(s) : nothing
    return AugmentedReturn(primal_return(config, ret), shadow_return(config, map(shadow, (Jx, Jy, Jz))), tape)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(deposit_flux!)}, ::Type{RT}, tape, mode::Const,
                             Jx::ArrayArg, Jy::ArrayArg, Jz::ArrayArg, s::FluxSpeciesArg, shape::Const, stagger::Const,
                             grid::Annotation{<:PeriodicGrid}, ws::Annotation{<:Workspace}; dt, workgroupsize = nothing,
                             q...) where {RT}
    if !(s isa Const || all(J -> J isa Const, (Jx, Jy, Jz)))
        flux_pullback!(shadow(s), field_shadow(Jx), field_shadow(Jy), field_shadow(Jz), something(tape, s.val), shape.val, stagger.val,
                       grid.val; dt, workgroupsize, q...)
    end
    return map(no_adjoint, (mode, Jx, Jy, Jz, s, shape, stagger, grid, ws))
end

# --- sort_by_cell!, resort_by_cell! ------------------------------------------------------------

for f in (:sort_by_cell!, :resort_by_cell!)
    @eval begin
        function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof($f)}, ::Type{RT}, s::SpeciesArg{N},
                                              grid::Annotation{<:PeriodicGrid{N}}, ws::Annotation{<:Workspace}; kw...) where {RT,N}
            ret = func.val(s.val, grid.val, ws.val; kw...)
            tape = s isa Const ? nothing : copy(ws.val.perm)
            return AugmentedReturn(primal_return(config, ret), shadow_return(config, shadow(s)), tape)
        end

        function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof($f)}, ::Type{RT}, tape, s::SpeciesArg{N},
                                     grid::Annotation{<:PeriodicGrid{N}}, ws::Annotation{<:Workspace};
                                     workgroupsize = nothing) where {RT,N}
            s isa Const || sort_pullback!(shadow(s), tape; workgroupsize)
            return map(no_adjoint, (s, grid, ws))
        end
    end
end

end
