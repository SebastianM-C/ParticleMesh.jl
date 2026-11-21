# WP-084 spike patch: `current!(::Esirkepov, st)` with its loops over the deposited species as
# unrolled `map`s over the species tuples instead of `for` loops over a `zip` (#384). With two
# deposited species of the 3D trapped-beam case (the plasma's electrons and the ionization's),
# Enzyme 0.13.221's reverse pass of the 3D step stops with an IllegalTypeAnalysisException
# ("the use of a Union type") inside `current!` (RTX 5090, 2026-11-21); WP-078 used the same
# unrolling in 2D, which WP-083 found unnecessary there. The same body as step.jl's, the bits
# unchanged.
#
#     PM_ENZYME_PATCH="...;scripts/enzyme/spike/unrolled_current.jl"
using ParticleMeshElectromagnetic
@eval ParticleMeshElectromagnetic begin
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
end
