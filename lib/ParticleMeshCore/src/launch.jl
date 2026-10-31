"""
    launcher(kernel, backend, workgroupsize)

The kernel instance of one engine launch: `kernel(backend)` for `workgroupsize = nothing` (the
backend's KernelAbstractions default), `kernel(backend, workgroupsize)` otherwise. The
workgroup size selects the launch shape only; no engine kernel's result in `Reference()` mode
depends on it (T-177).
"""
@inline launcher(kernel, backend, ::Nothing) = kernel(backend)
@inline launcher(kernel, backend, workgroupsize) = kernel(backend, workgroupsize)
