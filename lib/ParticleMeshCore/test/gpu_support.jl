# WP-013: one test body for the CPU and for CUDA. A shared body builds its inputs on the host
# (plain `Array`s, as helpers.jl does), runs every engine call through `on(be, ...)`, and checks
# on the host. On the CPU `on` is the plain call, so the CPU tests keep their bits by
# construction; on another backend it moves the arguments to the device, runs the call there and
# copies every argument back into its host original. Needs setup.jl.

using CUDACore: CUDACore, CUDABackend
using KernelAbstractions: @kernel, @index
const Adapt = ParticleMeshCore.Adapt

include("gpu_switch.jl")
const GPU_SWITCH = gpu_switch()
"The device backend of the GPU testsets, or `nothing` when they do not run."
const GPU_BACKEND = GPU_SWITCH.run ? CUDABackend() : nothing

"""
    on(be, f, args...; kw...)

`f(args...; kw...)` with its data on the backend `be`. On `CPU()` this is the plain call. On
another backend each argument (and keyword value) is adapted to `be`; a `Workspace` becomes a
fresh device `Workspace` of the first `Species` argument with the host one's sort state (keys,
permutation, segment bounds) copied in, so a `sort_by_cell!` and a later `deposit!` in two
`on` calls see the same segments. After the call the backend is synchronized and every array
of every argument is copied back into its host original. A returned argument comes back as its
host original, any other array as a host copy.
"""
on(::CPU, f, args...; kw...) = f(args...; kw...)
function on(be, f, args...; kw...)
    gi = findfirst(a -> a isa PeriodicGrid, args)
    gi === nothing && throw(ArgumentError("on: needs a PeriodicGrid argument"))
    grid = args[gi]
    dargs = Any[a isa Workspace ? a : todev(be, a) for a in args]
    for i in eachindex(dargs)
        if args[i] isa Workspace
            j = findfirst(a -> a isa Species, dargs)
            j === nothing && throw(ArgumentError("on: a Workspace argument needs a Species argument"))
            dargs[i] = todev(be, args[i], dargs[j], grid)
        end
    end
    dkw = map(v -> todev(be, v), values(kw))
    r = f(dargs...; dkw...)
    KernelAbstractions.synchronize(be)
    foreach(copyback!, args, dargs)
    i = findfirst(d -> d === r, dargs)
    return i === nothing ? tohost(r) : args[i]
end

todev(be, x) = Adapt.adapt(be, x)
function todev(be, ws::Workspace, ds::Species, grid)
    dws = Workspace(ds, grid)
    for f in (:key, :perm, :seg_start, :seg_stop)
        copyto!(getfield(dws, f), getfield(ws, f))
    end
    return dws
end
tohost(x) = Adapt.adapt(Array, x)

copyback!(h::AbstractArray, d::AbstractArray) = (h === d || copyto!(h, d); nothing)
copyback!(h::Tuple, d::Tuple) = (foreach(copyback!, h, d); nothing)
copyback!(h::NamedTuple, d::NamedTuple) = (foreach(copyback!, values(h), values(d)); nothing)
copyback!(h::Species, d::Species) =
    (foreach(f -> copyback!(getfield(h, f), getfield(d, f)), (:cell, :offset, :u, :w, :status, :count, :extras)); nothing)
copyback!(h::Workspace, d::Workspace) =
    (foreach(f -> copyback!(getfield(h, f), getfield(d, f)), (:key, :perm, :seg_start, :seg_stop)); nothing)
copyback!(h, d) = nothing

@kernel function devmap_kernel!(out, f, @Const(xs))
    i = @index(Global)
    @inbounds out[i] = f(xs[i])
end

"""
    devmap(be, f, xs::Vector) -> Vector

`map(f, xs)`, with `f` evaluated on the backend `be`: on the CPU the plain host `map`, elsewhere
in a kernel, one work item per element. For the device functions of the engine (shape weights,
Philox) that a test calls outside any engine launch.
"""
devmap(::CPU, f, xs::Vector) = map(f, xs)
function devmap(be, f, xs::Vector)
    R = typeof(f(first(xs)))
    out = KernelAbstractions.allocate(be, R, length(xs))
    devmap_kernel!(be)(out, f, todev(be, xs); ndrange = length(xs))
    KernelAbstractions.synchronize(be)
    return Array(out)
end

"A label for `record!` and testset names: \"CPU\" or the device backend's type."
backend_label(::CPU) = "CPU"
backend_label(be) = string(nameof(typeof(be)))
