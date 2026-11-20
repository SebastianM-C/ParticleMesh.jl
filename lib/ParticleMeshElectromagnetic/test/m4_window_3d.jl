# WP-079: the M4 solver in the moving window of the 3D step, T-1100 (its rerun on CUDA is part of
# T-1099, in gpu.jl). After em3d_window.jl, whose `window_vacuum_3d` and `window_gauss_worst_3d`
# it reuses, with the 3D step's helpers before it.

# --- T-1100 ---------------------------------------------------------------------------------

# T-1036 and T-1032 with M4. The vacuum pulse of `window_lwfa_3d` in the window is its static
# twin's before the window moves and, after, on columns 25 … n₁, to T-1036's roundoff, with a bare
# back edge, with a back-edge PML of 6 cells (M4's layer) and with the plane at x = 30 and the
# window starting at the pulse's peak, so that the plane moves back while it launches; the layer
# takes the back edge's reflection to T-1036's fraction of the bare edge's against free space.
# Gauss's law to T-1032's roundoff across the shifts with the laser driving the plasma, all three
# types, both modes, off nodes 1, 2 and the antenna's two columns; and with y, z open (the PML)
# and a back-edge PML off the layers (Float64, both modes). Ampère's law is Yee's under M4, so the
# 2D argument of T-1090 carries over.
function body_T1100(be)
    C = emC("T-1036")
    bare = window_vacuum_3d(nothing; be, solver = M4())
    layer = window_vacuum_3d(PML(; cells = 6); be, solver = M4())
    moving = window_vacuum_3d(nothing; be, position = 30, start = 18, solver = M4())
    @info "T-1100 the M4 3D window against its static twin and free space" bare layer moving
    for r in (bare, layer, moving)
        @test r.shifts > r.n1
        @test r.before <= C * eps(Float64)
        @test r.worst <= C * eps(Float64)
    end
    @test bare.backdiff > 1e-6
    @test layer.near <= emC("T-1036", "C_near") * bare.near
    record!("T-1100 twin", Float64, max(bare.worst, layer.worst, moving.worst) / eps(Float64), be)
    record!("T-1100 back PML over bare", Float64, layer.near / bare.near, be)
    Cg = emC("T-1032")
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    cases = [(T, m, false) for T in TYPES for m in (Reference(), Fast())]
    append!(cases, [(Float64, m, true) for m in (Reference(), Fast())])
    for (T, mode, open) in cases
        prob = window_lwfa_3d(T; bd = open ? (:open, :open) : (:periodic, :periodic))
        st = window_state_3d(prob; mode, be, absorber = open ? pml : nothing, back = open ? PML(; cells = 6, alpha_max = 0.2) : nothing,
                             solver = M4())
        @test st.numerics.solver isa M4
        p = params(prob)
        ia = antenna_planes(st)[1].ia
        cb = open ? 6 : 2
        n = ncells(st.grid)
        rows = open ? ((pml.cells + 2):(n[2] - pml.cells), (pml.cells + 2):(n[3] - pml.cells)) : nothing
        worst = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 4 == 0 || continue
            worst = max(worst, window_gauss_worst_3d(st, p, k, s -> (1:cb..., ia - 1 - s, ia - s); rows))
        end
        @info "T-1100 Gauss" T mode open worst shifts = window_shifts(st, prob.nsteps)
        @test window_shifts(st, prob.nsteps) > prob.cells[1]
        record!("T-1100 Gauss $(open ? "open, back PML" : "periodic") $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= Cg
    end
end

@testset "T-1100 the M4 solver in the 3D moving window: the vacuum pulse is its static twin's, the back-edge PML absorbs, Gauss's law across shifts [A2, A4, A7, A8, WP-079]" begin
    body_T1100(CPU())
end
