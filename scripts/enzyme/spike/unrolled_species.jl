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
end
