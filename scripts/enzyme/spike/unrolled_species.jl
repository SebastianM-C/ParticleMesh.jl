# WP-078 spike patch for Enzyme's reverse pass through the trapped-beam step on CUDA. Each
# workaround can be switched off with PM_WA_OFF, a comma-separated list (WP-078's CUDA follow-up
# finds which of them Enzyme's CUDA extension, loaded with `using CUDA`, makes unnecessary):
#
# - `copy`: device-to-device `copyto!` in the window's field shift and the PML shift as a
#   KernelAbstractions copy kernel (Enzyme segfaults compiling CUDA's copy without its CUDA
#   extension, #552; the filter's copy is `device_copy!` in lib since WP-080);
# - `unroll`: the loops over the deposited species, the fields and the PML's convolution fields as
#   unrolled `map`s (#384's workaround; not shown to be needed);
# - `flagsum`: the refill's and the births' `sum!` of the flags as a one-thread accumulation
#   kernel (#551);
# - `ionize4`: the ionization's births in separate kernels (populations, births, settle, unmark;
#   not shown to change anything);
# - `freedskip` (trapped.jl): the checkpoint restore skips arrays the step freed.
#
# Always on: the `sinpi` rule of the density ramp (#545) and the antennas' source rules (#547).
# Every override keeps the engine's bits. With a workaround off, the override that only served it
# is not installed, so the engine's own code runs.
using ParticleMeshElectromagnetic
const WA_OFF = Set(filter(!isempty, split(get(ENV, "PM_WA_OFF", ""), ',')))
wa(name) = !(name in WA_OFF)
println("workarounds on: ", join(filter(wa, ["copy", "unroll", "flagsum", "ionize4", "freedskip"]), ", "),
        "; off: ", isempty(WA_OFF) ? "none" : join(sort!(collect(WA_OFF)), ", "))
const WA_COPY, WA_UNROLL, WA_FLAGSUM, WA_IONIZE4 = wa("copy"), wa("unroll"), wa("flagsum"), wa("ionize4")
@eval ParticleMeshElectromagnetic begin
    # a device copy through a KernelAbstractions kernel, in place of `copyto!` between device arrays
    @kernel function ka_copy_kernel!(dst, @Const(src))
        i = @index(Global, Linear)
        @inbounds dst[i] = src[i]
    end
    ka_copy!(dst, src) = (ka_copy_kernel!(get_backend(dst))(dst, src; ndrange = length(dst)); dst)
    _wa_copy!(dst, src) = $(WA_COPY) ? ka_copy!(dst, src) : copyto!(dst, src)
    # f over the elements of tuples zipped: an unrolled map, or the engine's for loop
    function _wa_each(f, ts::Tuple...)
        if $(WA_UNROLL)
            map(f, ts...)
        else
            for xs in zip(ts...)
                f(xs...)
            end
        end
        return nothing
    end
    # `sum!(acc, flags; init = false)` as one work item adding in index order (integer counts, the same values)
    @kernel function accumulate_flags_kernel!(acc, @Const(flags), n::Int32)
        t = acc[1]
        for i in Int32(1):n
            @inbounds t += flags[i]
        end
        @inbounds acc[1] = t
    end
    accumulate_flags!(acc, flags, n) = (accumulate_flags_kernel!(get_backend(acc))(acc, flags, Int32(n); ndrange = 1); acc)
    _wa_flagsum!(acc, flags, n) = $(WA_FLAGSUM) ? accumulate_flags!(acc, flags, n) : sum!(acc, view(flags, 1:n); init = false)
end
if WA_UNROLL
    @eval ParticleMeshElectromagnetic function current!(::Esirkepov, st::EMState{N}) where {N}
        (; numerics, grid, fields) = st
        T = eltype(st.ρ)
        wgs = numerics.workgroupsize
        _wa_each(deposited_species(st), deposited_workspaces(st)) do s, ws
            sorts(st, s) && resort_by_cell!(s, grid, ws; workgroupsize = wgs)
            nothing
        end
        fill!(fields.Jx, zero(T))
        fill!(fields.Jy, zero(T))
        fill!(fields.Jz, zero(T))
        _wa_each(deposited_species(st), deposited_workspaces(st)) do s, ws
            deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, rho_stagger(Val(N)), grid, ws;
                          dt = st.dt, transverse_currents(Val(N))..., workgroupsize = wgs)
            nothing
        end
        window = window_frame(st)
        _wa_each((fields.Jx, fields.Jy, fields.Jz)) do J
            zero_x_edges!(window, J; workgroupsize = wgs)
            fold_guards!(J, grid; workgroupsize = wgs)
            nothing
        end
        filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
        return st
    end
end
if WA_COPY || WA_UNROLL
    @eval ParticleMeshElectromagnetic begin
        # `shift_window!` (window.jl), its copies and loops by the switches
        function shift_window!(st::EMState{2}, p::EMParams, s_new::Integer)
            (; numerics, grid, fields, aux) = st
            w = aux.window
            wgs = numerics.workgroupsize
            g, n = guards(grid), ncells(grid)
            st.fresh[] = false
            _wa_each((fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)) do a
                _wa_copy!(st.ρ, a)
                launcher(shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
                nothing
            end
            shift_absorbers!(st.modifiers, wgs)
            _wa_each(st.species, st.ws, st.setup, p.species, w.lost, w.fast, w.flags) do s, ws, setup, ps, lost, fast, flags
                launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
                sort_after_shift!(st, s, ws)
                load_window_columns!(s, grid, window_load(setup, ps, p), lost, injection_column(w), 1, s_new; fast, flags, workgroupsize = wgs)
                nothing
            end
            _wa_each(witness_species(st)) do s
                launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
                nothing
            end
            shift_ionization!(ionization_frame(st), st, p, s_new)
            return st
        end
        # `shift_absorber!` (pml.jl), its copy and loop by the switches
        function shift_absorber!(L::PlacedPML, wgs)
            _wa_each(values(L.ψ)) do ψ
                _wa_copy!(L.scratch, ψ)
                launcher(pml_shift_kernel!, get_backend(ψ), wgs)(ψ, L.scratch, Int32(size(ψ, 1)); ndrange = size(ψ))
                nothing
            end
            return nothing
        end
    end
end
if WA_FLAGSUM
    # `load_window_columns!` (window.jl at main 97bdc83: the transverse cell counts as a tuple)
    @eval ParticleMeshElectromagnetic function load_window_columns!(s, grid, load, lost, col0, ncol, origin; fast = nothing, flags = nothing,
                                                                     workgroupsize = nothing)
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
end
if WA_IONIZE4
    @eval ParticleMeshElectromagnetic begin
        # the births in their own kernel, which writes nothing it reads: born atoms are marked 2 in
        # `flags` (deferred 1); `settle_kernel!` moves their `pending` into `emitted`, `unmark_kernel!`
        # resets the marks. The same bits as ionization.jl's single kernel.
        @kernel function birth_kernel!(a, e, flags, @Const(count), f, pusher, shape, g::Int32, qm_h, cap::Int32)
            i = Int32(@index(Global, Linear))
            mark = Int32(0)
            @inbounds if a.status[i] == ALIVE
                T = eltype(a.w)
                pending = a.extras.pending[i]
                w = a.w[i] * pending
                if w > zero(T)
                    slot = count[1] + i
                    if slot <= cap
                        c = particle_cell(a, i, Val(2))
                        δ = particle_offset(a, i, Val(2))
                        E, B = fields_at(f, c, δ, shape, g)
                        u = push_momentum(pusher, (zero(T), zero(T), zero(T)), E, B, qm_h)
                        e.cell[1][slot] = c[1]
                        e.cell[2][slot] = c[2]
                        e.offset[1][slot] = δ[1]
                        e.offset[2][slot] = δ[2]
                        e.u[1][slot] = u[1]
                        e.u[2][slot] = u[2]
                        e.u[3][slot] = u[3]
                        e.w[slot] = w
                        e.status[slot] = ALIVE
                        mark = Int32(2)
                    else
                        mark = Int32(1)
                    end
                end
            end
            @inbounds flags[i] = mark
        end
        @kernel function settle_kernel!(a, @Const(flags))
            i = Int32(@index(Global, Linear))
            @inbounds if flags[i] == Int32(2)
                a.extras.emitted[i] += a.extras.pending[i]
                a.extras.pending[i] = zero(eltype(a.w))
            end
        end
        @kernel function unmark_kernel!(flags)
            i = Int32(@index(Global, Linear))
            @inbounds flags[i] = Int32(flags[i] == Int32(1))
        end
    end
end
if WA_IONIZE4 || WA_FLAGSUM
    # `ionize!` (ionization.jl): the births by `ionize4`, the lost births' count by `flagsum`
    @eval ParticleMeshElectromagnetic function ionize!(fr::IonizationFrame, st::EMState, p::EMParams, k::Integer)
        (; numerics, grid, dt) = st
        fields = particle_fields(st)
        wgs = numerics.workgroupsize
        a, e = fr.atoms, fr.electrons
        be = get_backend(a)
        birth = k % fr.every == 0
        birth && !sorted_every_step(st, e) && sort_by_cell!(e, grid, fr.electrons_ws; workgroupsize = wgs)
        launcher(ionize_kernel!, be, wgs)(a, e, fr.flags, e.count, fields, p.ionization.levels, numerics.pusher, read_shape(numerics),
                                          Int32(guards(grid)), dt, -e.q / e.m * dt / 4, $(WA_IONIZE4) ? false : birth, Int32(capacity(e));
                                          ndrange = capacity(a))
        if birth
            if $(WA_IONIZE4)
                launcher(birth_kernel!, be, wgs)(a, e, fr.flags, e.count, fields, numerics.pusher, read_shape(numerics), Int32(guards(grid)),
                                                 -e.q / e.m * dt / 4, Int32(capacity(e)); ndrange = capacity(a))
                launcher(settle_kernel!, be, wgs)(a, fr.flags; ndrange = capacity(a))
                launcher(unmark_kernel!, be, wgs)(fr.flags; ndrange = capacity(a))
            end
            launcher(bump_count_kernel!, be, wgs)(e.count, Int32(capacity(a)), Int32(capacity(e)); ndrange = 1)
            $(WA_FLAGSUM) ? accumulate_flags!(fr.lost, fr.flags, length(fr.flags)) : sum!(fr.lost, fr.flags; init = false)
        end
        return nothing
    end
end
@eval ParticleMeshElectromagnetic begin
    # `ramp_factor` (window.jl): Enzyme has no reverse pass of the device's `sinpi` (`__nv_sinpif`, #545);
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

# The antennas' sources (antenna.jl, N = 2) with a hand-written reverse rule: Enzyme cannot type
# three values of the B source kernel's tape (the laser's profile), so the device array for it
# fails. A source adds to the fields, so the fields' adjoints pass through; the laser's adjoint is
# Σ_rows ⟨the rows' field adjoints, ∂source/∂laser⟩, from the same source expression with the
# laser's ten float fields as duals (`laser_adjoint_kernel!`). The step size and the time get no
# adjoint (they follow from Δt alone).
import ForwardDiff
struct AntTag end
const ANT_FIELDS = (:a0, :waist, :duration, :delay, :focus, :chirp, :polarization, :phase, :position, :center)
function dual_laser(l::ParticleMeshElectromagnetic.GaussianLaser{T,P,E,C}) where {T,P,E,C}
    (P <: Real && C <: Real) || error("the antenna rule handles a scalar position and center (N = 2)")
    K = length(ANT_FIELDS)
    D = ForwardDiff.Dual{AntTag,T,K}
    v = ntuple(i -> D(getfield(l, ANT_FIELDS[i]), ForwardDiff.Partials(ntuple(j -> T(i == j), K))), K)
    return ParticleMeshElectromagnetic.GaussianLaser{D,D,E,D}(v[1:8]..., v[9], v[10], l.envelope)
end
@eval ParticleMeshElectromagnetic begin
    @kernel function laser_adjoint_kernel!(out, @Const(dA), @Const(dB), a, ld, c, t, ::Val{S}) where {S}
        j = Int32(@index(Global, Linear))
        T = typeof(c)
        l = ld
        if S === :b
            i = a.g + a.ia - Int32(1)
            s = t - l.delay
            ξ = a.xa - l.focus
            sθ, cθ = sincos(l.polarization)
            ec, _, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
            en, _, _ = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
            # Bz[i, j] += c·ec·cθ (dA = the adjoint of Bz), By[i, j] −= c·en·sθ (dB of By)
            @inbounds g = dA[i, j] * (c * (ec * cθ)) - dB[i, j] * (c * (en * sθ))
        else
            i = a.g + a.ia
            s = t + a.ht - l.delay
            ξ = a.xa - a.hx - l.focus
            sθ, cθ = sincos(l.polarization)
            _, bp, _ = laser_field(l, ξ, antenna_dy(a, j, one(T) / 2), s)
            _, _, bs = laser_field(l, ξ, antenna_dy(a, j, zero(T)), s)
            # Ey[i, j] += c·bp·cθ (dA of Ey), Ez[i, j] += c·bs·sθ (dB of Ez)
            @inbounds g = dA[i, j] * (c * (bp * cθ)) + dB[i, j] * (c * (bs * sθ))
        end
        for k in 1:size(out, 2)
            @inbounds out[j, k] = $(ForwardDiff.partials)(g, k)
        end
    end
end
_shadow(x::Enzyme.Duplicated) = x.dval
_shadow(x::Enzyme.MixedDuplicated) = x.dval[]
_shadow(::Enzyme.Const) = nothing
function antenna_adjoint(a::ParticleMeshElectromagnetic.Antenna, fs, c, t, S)
    PEMm = ParticleMeshElectromagnetic
    l = a.laser
    T = typeof(c)
    dA, dB = S === :b ? (fs.Bz, fs.By) : (fs.Ey, fs.Ez)
    nrows = size(dA, 2)
    out = similar(dA, T, nrows, length(ANT_FIELDS))
    PEMm.laser_adjoint_kernel!(KernelAbstractions.get_backend(dA))(out, dA, dB, a, dual_laser(l), c, t, Val(S); ndrange = nrows)
    s = vec(sum(Array(out); dims = 1))
    dl = typeof(l)(s[1:8]..., s[9], s[10], l.envelope)
    z = zero(a.xa)
    return PEMm.Antenna(dl, a.ia, z, map(zero, a.dx), a.g, a.n, map(zero, a.len), z, z)
end
for (f, S) in ((:b_source!, :b), (:e_source!, :e))
    @eval begin
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(ParticleMeshElectromagnetic.$f)}, ::Type{RT},
                                              a::Annotation{<:ParticleMeshElectromagnetic.Antenna{<:Any,<:Any,2}}, f::Annotation, c::Annotation, t::Annotation,
                                              wgs::Annotation) where {RT}
            ParticleMeshElectromagnetic.$f(a.val, f.val, c.val, t.val, wgs.val)
            return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
        end
        function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(ParticleMeshElectromagnetic.$f)}, ::Type{RT}, tape,
                                     a::Annotation{<:ParticleMeshElectromagnetic.Antenna{<:Any,<:Any,2}}, f::Annotation, c::Annotation, t::Annotation,
                                     wgs::Annotation) where {RT}
            fs = _shadow(f)
            da = (a isa Active && fs !== nothing) ? antenna_adjoint(a.val, fs, c.val, t.val, $(QuoteNode(S))) :
                 a isa Active ? Enzyme.make_zero(a.val) : nothing
            return (da, nothing, c isa Active ? zero(c.val) : nothing, t isa Active ? zero(t.val) : nothing, nothing)
        end
    end
end
