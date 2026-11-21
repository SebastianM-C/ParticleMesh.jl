# WP-078 spike patch: the step's loops over the deposited species as unrolled `map`s over the
# species tuples instead of `for` loops (#384: Enzyme's reverse mode fails on a `for` loop over a
# homogeneous tuple of large structs; with two deposited species, the plasma's electrons and the
# ionization's, `current!` makes Enzyme 0.13.221 segfault while it builds the augmented primal).
# `copyto!` between device arrays (in `binomial!`, which the step compiles even with no filter
# passes, and in the shift) makes Enzyme segfault too: a KernelAbstractions copy instead.
# The same bodies as `current!(::Esirkepov, st)` in step.jl and `shift_window!` in window.jl, the bits unchanged.
using ParticleMeshElectromagnetic
@eval ParticleMeshElectromagnetic begin
    # a device copy through a KernelAbstractions kernel, in place of `copyto!` between device arrays
    @kernel function ka_copy_kernel!(dst, @Const(src))
        i = @index(Global, Linear)
        @inbounds dst[i] = src[i]
    end
    ka_copy!(dst, src) = (ka_copy_kernel!(get_backend(dst))(dst, src; ndrange = length(dst)); dst)
    function binomial!(a, scratch, grid::PeriodicGrid{N}, passes::Integer; workgroupsize = nothing, edges = nothing) where {N}
        passes == 0 && return a
        size(scratch) == size(a) || throw(DimensionMismatch("scratch of size $(size(scratch)) for a field of size $(size(a))"))
        src, dst = a, scratch
        launch = launcher(binomial_pass_kernel!, get_backend(a), workgroupsize)
        for _ in 1:passes, d in 1:N
            fill_guards!(src, grid; workgroupsize)
            zero_x_edges!(edges, src; workgroupsize)
            launch(dst, src, CartesianIndex(ntuple(k -> k == d ? 1 : 0, Val(N))), guards(grid); ndrange = grid.n)
            src, dst = dst, src
        end
        src === a || ka_copy!(a, src)
        fill_guards!(a, grid; workgroupsize)
        zero_x_edges!(edges, a; workgroupsize)
        return a
    end
    function current!(::Esirkepov, st::EMState{N}) where {N}
        (; numerics, grid, fields) = st
        T = eltype(st.ρ)
        wgs = numerics.workgroupsize
        map(deposited_species(st), deposited_workspaces(st)) do s, ws
            sorts(st, s) && resort_by_cell!(s, grid, ws; workgroupsize = wgs)
            nothing
        end
        fill!(fields.Jx, zero(T))
        fill!(fields.Jy, zero(T))
        fill!(fields.Jz, zero(T))
        map(deposited_species(st), deposited_workspaces(st)) do s, ws
            deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, rho_stagger(Val(N)), grid, ws;
                          dt = st.dt, transverse_currents(Val(N))..., workgroupsize = wgs)
            nothing
        end
        window = window_frame(st)
        map((fields.Jx, fields.Jy, fields.Jz)) do J
            zero_x_edges!(window, J; workgroupsize = wgs)
            fold_guards!(J, grid; workgroupsize = wgs)
            nothing
        end
        filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
        return st
    end
    # `shift_window!` (window.jl) with its loops over the fields and the species unrolled
    function shift_window!(st::EMState{2}, p::EMParams, s_new::Integer)
        (; numerics, grid, fields, aux) = st
        w = aux.window
        wgs = numerics.workgroupsize
        g, n = guards(grid), ncells(grid)
        st.fresh[] = false
        map((fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)) do a
            ka_copy!(st.ρ, a)
            launcher(shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
            nothing
        end
        shift_absorbers!(st.modifiers, wgs)
        map(st.species, st.ws, st.setup, p.species, w.lost, w.fast, w.flags) do s, ws, setup, ps, lost, fast, flags
            launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
            sort_after_shift!(st, s, ws)
            load_window_columns!(s, grid, window_load(setup, ps, p), lost, injection_column(w), 1, s_new; fast, flags, workgroupsize = wgs)
            nothing
        end
        map(witness_species(st)) do s
            launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
            nothing
        end
        shift_ionization!(ionization_frame(st), st, p, s_new)
        return st
    end
    # `shift_absorber!` (pml.jl): the loop over the convolution fields unrolled, the copy a kernel
    function shift_absorber!(L::PlacedPML, wgs)
        map(values(L.ψ)) do ψ
            ka_copy!(L.scratch, ψ)
            launcher(pml_shift_kernel!, get_backend(ψ), wgs)(ψ, L.scratch, Int32(size(ψ, 1)); ndrange = size(ψ))
            nothing
        end
        return nothing
    end
    # `ionize!` (ionization.jl) in two kernels: the populations (the ionization kernel with no
    # birth, whose reverse pass checks against duals), then the births from the stored `pending`
    # (Enzyme's reverse kernel reads `pending` again after the birth zeroed it, takes the branch
    # with that value and loses the newborns' adjoints back to the atoms), then a kernel that
    # moves the born atoms' `pending` into `emitted`. The same bits as ionization.jl.
    @kernel function birth_kernel!(a, e, flags, @Const(count), f, pusher, shape, g::Int32, qm_h, cap::Int32)
        # writes nothing it reads: born atoms are marked 2 in `flags` (deferred 1), and
        # `settle_kernel!` moves their `pending` into `emitted`
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
    # the branch reads `flags`, which this kernel leaves alone (the reverse kernel takes the
    # branch again from memory); `unmark_kernel!` resets the marks afterwards
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
    function ionize!(fr::IonizationFrame, st::EMState, p::EMParams, k::Integer)
        (; numerics, grid, dt) = st
        fields = particle_fields(st)
        wgs = numerics.workgroupsize
        a, e = fr.atoms, fr.electrons
        be = get_backend(a)
        birth = k % fr.every == 0
        birth && !sorted_every_step(st, e) && sort_by_cell!(e, grid, fr.electrons_ws; workgroupsize = wgs)
        launcher(ionize_kernel!, be, wgs)(a, e, fr.flags, e.count, fields, p.ionization.levels, numerics.pusher, read_shape(numerics),
                                          Int32(guards(grid)), dt, -e.q / e.m * dt / 4, false, Int32(capacity(e)); ndrange = capacity(a))
        if birth
            launcher(birth_kernel!, be, wgs)(a, e, fr.flags, e.count, fields, numerics.pusher, read_shape(numerics), Int32(guards(grid)),
                                             -e.q / e.m * dt / 4, Int32(capacity(e)); ndrange = capacity(a))
            launcher(settle_kernel!, be, wgs)(a, fr.flags; ndrange = capacity(a))
            launcher(unmark_kernel!, be, wgs)(fr.flags; ndrange = capacity(a))
            launcher(bump_count_kernel!, be, wgs)(e.count, Int32(capacity(a)), Int32(capacity(e)); ndrange = 1)
            accumulate_flags!(fr.lost, fr.flags, length(fr.flags))
        end
        return nothing
    end
    # `sum!(acc, flags; init = false)` (window.jl's refill, the ionization's births) as one work item
    # adding in index order: Enzyme's reverse pass of the `sum!` over a view frees the window's
    # `flags` (the next step's refill then reads a freed array). Integer counts, the same values.
    @kernel function accumulate_flags_kernel!(acc, @Const(flags), n::Int32)
        t = acc[1]
        for i in Int32(1):n
            @inbounds t += flags[i]
        end
        @inbounds acc[1] = t
    end
    accumulate_flags!(acc, flags, n) = (accumulate_flags_kernel!(get_backend(acc))(acc, flags, Int32(n); ndrange = 1); acc)
    function load_window_columns!(s, grid, load, lost, col0, ncol, origin; fast = nothing, flags = nothing, workgroupsize = nothing)
        n2 = ncells(grid)[2]
        nnew = ncol * n2 * prod(load.ppc)
        be = get_backend(s)
        flags === nothing && (flags = KernelAbstractions.zeros(be, Int32, nnew))
        launcher(refill_kernel!, be, workgroupsize)(s, flags, s.count, RampedCosine(), load.p, cellsize(grid), load.ppc, Int32(col0),
                                                    Int32(n2), Int32(origin), load.vdrift, load.vth, load.key,
                                                    Int32(capacity(s)); ndrange = nnew)
        launcher(refill_count_kernel!, be, workgroupsize)(s.count, lost, Int32(nnew), Int32(capacity(s)); ndrange = 1)
        fast === nothing || accumulate_flags!(fast, flags, nnew)
        return s
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
