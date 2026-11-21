# WP-078 spike patch for Enzyme's reverse pass through the trapped-beam step on CUDA, with the
# CUDA package loaded so that Enzyme's CUDA extension supplies the device-copy rules (#552). What
# is left, every override with the engine's bits:
#
# - the refill's and the births' `sum!` of the flags as a one-thread accumulation kernel: the
#   reverse pass of the `sum!` over a view frees the window's `flags`, which the next step's refill
#   then reads (#551);
# - the `sinpi` rule of the density ramp (#545).
#
# WP-078's CUDA follow-up (runs/wp078-cuda-5090) showed that with `using CUDA` the copy kernels in
# the window and PML shifts, the unrolled species loops, the ionization's births in four kernels
# and the checkpoint restore's skip of freed arrays are not needed; WP-083 removed them, and the
# antennas' hand-written reverse rule with them (the laser calls sin and cos, #547).
using ParticleMeshElectromagnetic
@eval ParticleMeshElectromagnetic begin
    # `sum!(acc, flags; init = false)` as one work item adding in index order (integer counts, the same values)
    @kernel function accumulate_flags_kernel!(acc, @Const(flags), n::Int32)
        t = acc[1]
        for i in Int32(1):n
            @inbounds t += flags[i]
        end
        @inbounds acc[1] = t
    end
    accumulate_flags!(acc, flags, n) = (accumulate_flags_kernel!(get_backend(acc))(acc, flags, Int32(n); ndrange = 1); acc)
    # `load_window_columns!` (window.jl), its count of the fast particles by `accumulate_flags!`
    function load_window_columns!(s, grid, load, lost, col0, ncol, origin; fast = nothing, flags = nothing, workgroupsize = nothing)
        nt = Int32.(Base.tail(ncells(grid)))
        nnew = ncol * prod(nt) * prod(load.ppc)
        be = get_backend(s)
        flags === nothing && (flags = KernelAbstractions.zeros(be, Int32, nnew))
        launcher(refill_kernel!, be, workgroupsize)(s, flags, s.count, RampedCosine(), load.p, cellsize(grid), load.ppc, Int32(col0),
                                                    nt, Int32(origin), load.vdrift, load.vth, load.key,
                                                    Int32(capacity(s)); ndrange = nnew)
        launcher(refill_count_kernel!, be, workgroupsize)(s.count, lost, Int32(nnew), Int32(capacity(s)); ndrange = 1)
        fast === nothing || accumulate_flags!(fast, flags, nnew)
        return s
    end
    # `ionize!` (ionization.jl), its count of the lost births by `accumulate_flags!`
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
            accumulate_flags!(fr.lost, fr.flags, length(fr.flags))
        end
        return nothing
    end
    # `ramp_factor` (window.jl): Enzyme has no reverse pass of the device's `sinpi` (`__nv_sinpif`);
    # the same value through `_sinpi`, whose reverse rule (π·cospi) is below
    @noinline _sinpi(x) = sinpi(x)
    @inline function ramp_factor(x, vacuum, ramp)
        t = clamp((x - vacuum) / ramp, zero(x), one(x))
        s = _sinpi(t / 2)
        return s * s
    end
end
using Enzyme: EnzymeRules, Const, Active, Duplicated, Annotation
# inside a rule the device's method overrides do not apply, so the rule calls libdevice itself
# (what CUDA's own `sinpi` and `cospi` call on the device: the same bits)
_dsinpi(x::Float32) = ccall("extern __nv_sinpif", llvmcall, Float32, (Float32,), x)
_dsinpi(x::Float64) = ccall("extern __nv_sinpi", llvmcall, Float64, (Float64,), x)
_dcospi(x::Float32) = ccall("extern __nv_cospif", llvmcall, Float32, (Float32,), x)
_dcospi(x::Float64) = ccall("extern __nv_cospi", llvmcall, Float64, (Float64,), x)
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(ParticleMeshElectromagnetic._sinpi)}, ::Type{RT}, x::Annotation) where {RT}
    y = _dsinpi(x.val)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? y : nothing, nothing, nothing)
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(ParticleMeshElectromagnetic._sinpi)}, dret::Active, tape, x::Active)
    return (dret.val * oftype(x.val, π) * _dcospi(x.val),)
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(ParticleMeshElectromagnetic._sinpi)}, dret::Type{<:Const}, tape, x::Annotation)
    return (x isa Active ? zero(x.val) : nothing,)
end
function EnzymeRules.forward(config::EnzymeRules.FwdConfig, ::Const{typeof(ParticleMeshElectromagnetic._sinpi)}, ::Type{RT}, x::Annotation) where {RT}
    y = _dsinpi(x.val)
    x isa Const && return EnzymeRules.needs_primal(config) ? y : nothing
    dy = x.dval * oftype(x.val, π) * _dcospi(x.val)
    return EnzymeRules.needs_primal(config) ? Duplicated(y, dy) : dy
end
