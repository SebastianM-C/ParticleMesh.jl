"""
    launcher(kernel, backend, workgroupsize)

The kernel instance of one engine launch: `kernel(backend)` for `workgroupsize = nothing` (the
backend's KernelAbstractions default), `kernel(backend, workgroupsize)` otherwise. The
workgroup size selects the launch shape only; no engine kernel's result in `Reference()` mode
depends on it (T-177).

A fixed `workgroupsize` has to fit the kernel it launches. Besides the backend's limits
(`KernelInterface.max_work_group_size(backend)` work-items, and its per-dimension maximum),
each compiled kernel has its own maximum, which depends on the backend, the kernel and the
element type: PoCL lowers it for kernels with a large private stack, such as the 2D flux patch
with `Dual` elements (983 work-items for `Dual{Float32,3}`, the quadratic shape and the
stagger `(Node(), Node())`, 1164 at `(Node(), Center())`, as PoCL reported them, #297). It is
known only once KernelAbstractions has compiled the kernel for its arguments, inside the
launch, and KernelAbstractions has no public query for it, so the engine cannot clamp a
requested size to it. A launch with a size that does not fit is refused before the kernel
runs, with an `ArgumentError` that names the `workgroupsize`, the kernel and the limit it
exceeds (T-500). `workgroupsize = nothing` lets the backend choose a size within the kernel's
maximum, so it fits every kernel.
"""
@inline launcher(kernel, backend, ::Nothing) = kernel(backend)
@inline launcher(kernel, backend, workgroupsize) =
    SizedLaunch(kernel(backend, workgroupsize), kernel, backend, workgroupsize)

"""
    SizedLaunch(instance, kernel, backend, workgroupsize)

The `instance` `kernel(backend, workgroupsize)` of a KernelAbstractions kernel with a fixed
`workgroupsize`, launched like the instance itself (`l(args...; ndrange)`). A launch that
KernelAbstractions refuses with an `ArgumentError` about the work-group (its message names the
work-group or the `workgroupsize`: the size does not fit the kernel or the backend) is refused
again with an `ArgumentError` that keeps KernelAbstractions' message, names the requested
`workgroupsize` and the kernel, and says that `nothing` fits. The refusal comes from
KernelAbstractions' host-side check of the launch geometry, before the kernel runs; nothing
here branches on device data. Any other exception, including an `ArgumentError` that does not
mention the work-group (an argument that cannot be converted for the device, say), passes
unchanged.
"""
struct SizedLaunch{I,K,B,W}
    instance::I
    kernel::K
    backend::B
    workgroupsize::W
end

function (l::SizedLaunch)(args...; ndrange)
    try
        l.instance(args...; ndrange)
    catch err
        err isa ArgumentError && is_workgroup_message(err.msg) || rethrow()
        throw_workgroupsize_error(l, err)
    end
    return nothing
end

is_workgroup_message(msg::AbstractString) = occursin(r"work-?group"i, msg)

@noinline function throw_workgroupsize_error(l::SizedLaunch, err::ArgumentError)
    throw(ArgumentError("workgroupsize = $(l.workgroupsize) does not fit the launch of $(nameof(l.kernel)) on " *
                        "$(nameof(typeof(l.backend))): $(err.msg). " *
                        "The largest work-group a kernel accepts depends on the backend, the kernel and the element " *
                        "type; pass a smaller workgroupsize, or workgroupsize = nothing (the backend's choice, which " *
                        "fits every kernel)."))
end
