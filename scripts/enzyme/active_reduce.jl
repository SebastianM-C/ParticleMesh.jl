# WP-078: Active (scalar or isbits struct) kernel arguments with their adjoints, for Enzyme's
# reverse mode through KernelAbstractions kernels on a GPU. KernelAbstractions' rule refuses an
# Active argument; the WP-050 spike patch (`spike/active_args.jl`) passed it as Const and dropped
# its adjoint, which loses every derivative with respect to a parameter that reaches a kernel as
# a scalar (the antenna's laser, the window's refill densities). Here each thread of the reverse
# kernel writes its adjoint of every Active argument into a per-thread buffer, and the rule
# returns the buffer's sum: on the device for a float, on the host, field by field, for an
# isbits struct (the antenna's laser, the refill's density parameters; small launches).
# `MixedDuplicated` arguments (structs holding arrays and scalars, such as a species) pass as
# `Duplicated` with the shadow struct, as in the spike patch: their arrays' adjoints are kept,
# the adjoints of their scalar fields (charge, mass) are dropped.
#
#     PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl
using KernelAbstractions, Enzyme
const KAExt = Base.get_extension(KernelAbstractions, :EnzymeExt)
const ACTIVE_SEEN = Set{String}()
@eval KAExt begin
    function _mixed_as_duplicated(func, a::EnzymeCore.MixedDuplicated)
        push!(Main.ACTIVE_SEEN, string(nameof(func.val.f)) * " :: mixed " * string(nameof(typeof(a.val))) * " (scalar adjoints dropped)")
        return EnzymeCore.Duplicated(a.val, a.dval[])
    end
    _mixed_as_duplicated(func, a) = a

    # the per-thread adjoint buffer of an Active argument, zeroed
    function _adjoint_buffer(be, a::EnzymeCore.Active, n)
        b = allocate(be, Core.Typeof(a.val), n)
        fill!(b, EnzymeCore.make_zero(a.val))
        return b
    end
    _adjoint_buffer(be, a, n) = nothing

    # the sum of the adjoints: float fields added, any other field the first thread's
    _radd(a::T, b::T) where {T<:AbstractFloat} = a + b
    _radd(a::T, b::T) where {T<:Tuple} = map(_radd, a, b)
    @generated function _radd(a::T, b::T) where {T}
        (T <: Number || fieldcount(T) == 0) && return :(a)
        return Expr(:new, T, [:(_radd(getfield(a, $i), getfield(b, $i))) for i in 1:fieldcount(T)]...)
    end
    _reduce_adjoint(::Nothing, a) = nothing
    _reduce_adjoint(b, a::EnzymeCore.Active{<:AbstractFloat}) = sum(b)
    _reduce_adjoint(b, a::EnzymeCore.Active) = foldl(_radd, Array(b))

    @inline _store_adjoints!(::Tuple{}, ::Tuple{}, I) = nothing
    @inline function _store_adjoints!(bufs::Tuple, d::Tuple, I)
        b = first(bufs)
        b === nothing || (@inbounds b[I] = first(d))
        return _store_adjoints!(Base.tail(bufs), Base.tail(d), I)
    end

    function rev_active(ctx, f::FT, mode::Mode, subtape, ::Val{TapeType}, bufs, args...) where {Mode, FT, TapeType}
        ctx isa ConstArgs && return nothing
        checked = checked_context(ctx)
        _, reverse = EnzymeCore.autodiff_deferred_thunk(mode, TapeType, Const{Core.Typeof(f)}, Const{Nothing},
                                                        Const{Core.Typeof(checked)}, map(Core.Typeof, args)...)
        if __validindex(ctx)
            I = __index_Global_Linear(ctx)
            tp = subtape[I]
            d = reverse(Const(f), Const(checked), args..., tp)[1]
            # one entry per argument of the kernel body: the context, then `args`
            _store_adjoints!(bufs, Base.tail(d), I)
        end
        return nothing
    end

    function EnzymeRules.augmented_primal(config::RevConfig, func::Const{<:Kernel}, ::Type{Const{Nothing}}, args::Vararg{Any, N};
                                          ndrange = nothing, workgroupsize = nothing) where {N}
        args2 = map(a -> _mixed_as_duplicated(func, a), args)
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
        return _augmented_return(kernel, subtape, (_ka_nthreads(ndrange, iterspace),), TapeType)
    end

    function EnzymeRules.reverse(config::RevConfig, func::Const{<:Kernel}, ::Type{<:EnzymeCore.Annotation}, tape, args::Vararg{Any, N};
                                 ndrange = nothing, workgroupsize = nothing) where {N}
        subtape, (nthreads,), tape_type = tape
        args2 = map(a -> _mixed_as_duplicated(func, a), args)
        kernel = func.val
        f = kernel.f
        be = backend(kernel)
        bufs = map(a -> _adjoint_buffer(be, a, nthreads), args2)
        ModifiedBetween = Val((overwritten(config)[1], false, overwritten(config)[2:end]...))
        Mode = split_mode(config, ModifiedBetween)
        rev_kernel = similar(kernel, rev_active)
        rev_kernel(f, Mode, subtape, Val(tape_type), bufs, args2...; ndrange, workgroupsize)
        synchronize(be)
        return map(_reduce_adjoint, bufs, args2)
    end
end
