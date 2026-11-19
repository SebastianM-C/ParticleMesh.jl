# Spike-only patch: treat Active (scalar) kernel arguments as Const, dropping their adjoints,
# and log which kernels receive them.
using KernelAbstractions, Enzyme
const KAExt = Base.get_extension(KernelAbstractions, :EnzymeExt)
const ACTIVE_SEEN = Set{String}()
@eval KAExt begin
    function _drop_active(func, args)
        map(args) do a
            if a isa EnzymeCore.Active
                push!(Main.ACTIVE_SEEN, string(nameof(func.val.f)) * " :: " * string(typeof(a.val)))
                EnzymeCore.Const(a.val)
            elseif a isa EnzymeCore.MixedDuplicated
                push!(Main.ACTIVE_SEEN, string(nameof(func.val.f)) * " :: mixed " * string(nameof(typeof(a.val))))
                EnzymeCore.Duplicated(a.val, a.dval[])
            else
                a
            end
        end
    end
    function EnzymeRules.augmented_primal(config::RevConfig, func::Const{<:Kernel}, ::Type{Const{Nothing}}, args::Vararg{Any, N};
                                          ndrange = nothing, workgroupsize = nothing) where {N}
        args2 = _drop_active(func, args)
        kernel = func.val
        f = kernel.f
        ndrange, workgroupsize, iterspace, dynamic = launch_config(kernel, ndrange, workgroupsize)
        ctx = _enzyme_mkcontext(kernel, ndrange, workgroupsize, iterspace)
        ctxTy = Core.Typeof(ctx)
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))
        FT = Const{Core.Typeof(f)}
        Mode = split_mode(config, ModifiedBetween)
        TapeType, subtape, aug_kernel = _create_tape_kernel(kernel, Mode, FT, ctxTy, ndrange, iterspace, args2...)
        aug_kernel(f, Mode, subtape, Val(TapeType), args2...; ndrange, workgroupsize)
        return _augmented_return(kernel, subtape, ntuple(_ -> nothing, Val(N)), TapeType)
    end
    function EnzymeRules.reverse(config::RevConfig, func::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args::Vararg{Any, N};
                                 ndrange = nothing, workgroupsize = nothing) where {N}
        subtape, arg_refs, tape_type = tape
        res = ntuple(i -> args[i] isa EnzymeCore.Active ? EnzymeCore.make_zero(args[i].val) : nothing, Val(N))
        args2 = _drop_active(func, args)
        kernel = func.val
        f = kernel.f
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))
        Mode = split_mode(config, ModifiedBetween)
        rev_kernel = similar(kernel, rev)
        rev_kernel(f, Mode, subtape, Val(tape_type), args2...; ndrange, workgroupsize)
        synchronize(backend(kernel))
        return res
    end
end
