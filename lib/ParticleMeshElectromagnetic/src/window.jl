# The moving window at N = 2 (WP-028, A2, A3, §0.4): the box follows the laser at c along axis 1.
# The problem says so (`EMProblem(...; boundaries = (:window, ·), window = MovingWindow(; start))`)
# and gives the plasma a longitudinal profile (`profile = DensityRamp(; vacuum, ramp)`), entering
# through the weights (A3). The grid underneath stays the engine's `PeriodicGrid{2}`; no kernel
# signature changes (B1). What the window changes:
#
# - Axis 1 is not periodic: what lies beyond the box is the undisturbed outside, zero fields ahead
#   of the window and nothing simulated behind it. After every Yee sub-step the x-guard columns of
#   the components it updated are set to zero (they held the periodic images), the deposited
#   current and charge in the x-guards are dropped before the fold (they belong to cells outside),
#   the binomial filter sees zero x-guards in every pass, and a particle that crosses either x edge
#   (its cell jumps by n₁ − 1 across the periodic seam) is retired before the current of that move
#   is deposited. Unlike the open y edge (WP-027), where both sides of the seam are layer rows, the
#   two sides of the x seam are the window's back and front: the engine places a particle's flux
#   stencil at its new cell, so the current of a particle that left through the back would land at
#   the front. Retired first, it takes its charge out of the box with no current, which breaks
#   Gauss's law only on the back nodes its stencil reached (nodes 1 and 2 for the quadratic shape),
#   and those leave the box with the next shifts.
#   Retirements recur at every shift once the plasma reaches the back edge, and which particle
#   goes at which step depends on the parameters, so the back columns' fields are not smooth in
#   them: an observable reads past [`window_back_band`](@ref) (#370, WP-044).
# - The window moves by whole cells at the end of a step: after step k it has moved
#   s(k) = max(0, floor((k·Δt − start)/Δx₁)) cells (c = 1), a host count fixed by k, Δt and Δx₁
#   (static configuration, §0.4 rule 2). A shift copies every E and B component one column back
#   (memory shift through the scratch ρ; the front column and the x-guards become zero), drops the
#   particles of the back column, compacts the slots by the sort, and loads the fresh plasma of the
#   injection column into the free slots behind the alive count (`s.count`, read on the
#   device): the engine's 2D quiet start (`quiet_particle_2d`) at the global column (that column + s), so
#   its draws and weights are those of a static load of the whole domain (A3, A4). The loaded
#   momenta are taken as those at the half step (the plasma ahead of the laser sees no field).
# - The front columns are kept free of fresh plasma: it is loaded on the injection column
#   n₁ − 1 − r − P ([`injection_column`](@ref)), r the reach of the shape's stencil past its cell
#   (`stencil_reach(shape, Node())[2]`, 2 for the quadratic shape) and P the filter passes. The
#   discrete front lags the continuous one (c·(t − start) past the box) by less than a cell, and a
#   particle slower than c only falls further behind the continuous front, so once the window moves
#   no particle loaded there reaches beyond the column after it, and no charge, current or filtered
#   current of the plasma reaches the front face (node n₁ + 1, an x-guard). That node is the one
#   the next shift brings in as the new front node n₁, and its Gauss's law is not kept by the step
#   (its Ey is held at zero). Two things make the new node's law hold: no particle has reached it,
#   and the shift carries Ex on the front face (the edge n₁ + ½, array column n₁) along instead of
#   zeroing it, so D⁻ₓEx there stays what Ey's zero front column needs. A plasma in which nothing
#   reaches the front face needs neither (the cold plasma at rest), a thermal one needs both. The
#   initial load fills columns 1 … n₁ − 1 − r − P likewise. Before the window starts, the front is
#   a fixed edge that the plasma reaches if it moves, so a problem with `start > 0` takes a plasma
#   with no drift and no thermal speed ([`EMProblem`](@ref) refuses the others).
# - The plasma starts neutral: an immobile background of the plasma's own profile (immobile ions,
#   A2) cancels the initial charge, so E⁰ = 0 and no initial solve is needed; Gauss's law then reads
#   D⁻·E = ρ − ρ₀(s), ρ₀(s) the charge of the cold quiet start of the window at its position s
#   ([`window_background!`](@ref)).
# - A drawn speed can reach c: the loader draws the thermal spread in velocity, drift + thermal·z
#   with z standard normal, and the Gaussian tail passes c (#257). The initial load refuses it
#   with an `ArgumentError`, as `check_subluminal` refuses it in a static load. A refill cannot
#   raise inside a step (§0.4), and dropping the particle would leave its charge out of the plasma
#   against the background, so the refill loads it at the drift velocity and counts it on the
#   device ([`window_superluminal`](@ref)).

"""
    MovingWindow(; start = 0)

The moving window of an [`EMProblem`](@ref) at N = 2 (WP-028): the box moves at c along axis 1
(the laser's propagation axis) from the time `start` (1/ω₀, A1), by whole cells: after step k it
has moved `max(0, floor((k·Δt − start)/Δx₁))` cells. The problem's axis 1 boundary is `:window`.
"""
struct MovingWindow{T}
    start::T
end
MovingWindow(; start = 0) = MovingWindow(start)

"""
    DensityRamp(; vacuum, ramp)

A longitudinal density profile along axis 1 (A2: vacuum, ramp, plateau), in the global (lab)
coordinate x of the window's first position, c/ω₀: zero for x ≤ `vacuum`, `sin²(π(x − vacuum)/(2·ramp))`
over the ramp, one beyond `vacuum + ramp`. It is C¹ (the AD-smoothness rule of A2 prefers it to a
linear ramp, whose derivative jumps). It multiplies every species' density and enters through the
particle weights only (A3), in the initial load and in the window's refill; `vacuum` and `ramp`
may be `Dual`s. `ramp` must be positive.
"""
struct DensityRamp{T}
    vacuum::T
    ramp::T
end
function DensityRamp(; vacuum, ramp)
    ramp > 0 || throw(ArgumentError("a density ramp needs a positive length, got $ramp"))
    T = promote_type(typeof(vacuum), typeof(ramp))
    return DensityRamp{T}(T(vacuum), T(ramp))
end

"The profile factor of `r` at the global coordinate `x` (a device function, C¹)."
@inline function ramp_factor(x, vacuum, ramp)
    t = clamp((x - vacuum) / ramp, zero(x), one(x))
    s = sinpi(t / 2)
    return s * s
end

"""
    RampedCosine()

The density profile of a species in a window problem, an isbits callable for the engine's loader:
`n(x, p) = p[1]·(1 + p[2]·cos(p[3]·x[1] + p[4]·x[2]))·ramp(x[1]; vacuum = p[5], ramp = p[6])`, the
[`CosineProfile`](@ref) times the [`DensityRamp`](@ref) (a uniform profile has `p[5] = −∞`). The
parameters enter through the weights (A3).
"""
struct RampedCosine end
@inline (::RampedCosine)(x::NTuple{2}, p) = p[1] * (1 + p[2] * cos(p[3] * x[1] + p[4] * x[2])) * ramp_factor(x[1], p[5], p[6])

# The loader parameters of species d in a window problem.
function window_density_params(problem, d)
    T = eltype(problem)
    k = ntuple(a -> 2 * T(π) * d.mode[a] / problem.length[a], Val(2))
    r = problem.profile
    vac, len = r === nothing ? (T(-Inf), one(T)) : (T(r.vacuum), T(r.ramp))
    return (d.density, d.amplitude, k[1], k[2], vac, len)
end

"""
    WindowFrame

The moving window placed on a grid by the state: the description, the time step and the cell size
along axis 1 it counts its shifts with, the interior columns `n1`, the injection column `inj` (the
plasma is loaded on columns `1 … inj`, [`injection_column`](@ref)), the guard width `g`, the
headroom of the particle buffers, per species the loader parameters, particles per cell, drift,
thermal speeds and Philox key of its refill, `lost`, one device counter per species of the
fresh particles that found no free slot (zero unless the buffers overflow; read with
[`window_lost`](@ref) outside a step), `fast`, one device counter per species of the fresh
particles of the refills whose drawn speed reached c (read with [`window_superluminal`](@ref)),
and `flags`, per species the refill's scratch of one flag per fresh particle that `fast` sums;
`back`, the PML of the back edge placed on the grid ([`BackPMLFields`](@ref)), or `nothing`.
"""
struct WindowFrame{W,T,P,L,F,K}
    window::W
    dt::T
    dx1::T
    n1::Int
    inj::Int
    g::Int
    loads::P
    lost::L
    fast::L
    flags::F
    back::K
end

"""
    injection_column(window::WindowFrame) -> Int

The window column that receives the fresh plasma, `n₁ − 1 − r − P` for a shape whose stencil
reaches `r = stencil_reach(shape, Node())[2]` nodes past its cell and `P` filter passes: the
columns ahead of it stay free of plasma, so nothing the plasma deposits reaches the front face
(the header of `window.jl` says why).
"""
injection_column(w::WindowFrame) = w.inj

"The injection column of a window of `n1` columns for `shape` and `passes` filter passes."
window_injection_column(n1::Integer, shape, passes::Integer) = n1 - 1 - ParticleMeshCore.stencil_reach(shape, Node())[2] - passes

"""
    window_back_band(window::WindowFrame) -> Int
    window_back_band(state::EMState) -> Int

The number of window columns at the back edge that an observable should not read,
`2·(r + P + 1)` for a shape whose stencil reaches `r` nodes past its cell and `P` filter passes
(the same `r + P + 1` columns that [`injection_column`](@ref) keeps free at the front, twice).
The particles the back edge retires (at a shift, or when they cross it) take their charge out
without current, and whether a particle goes at this step or the next depends on the parameters,
so the fields there jump between neighbouring parameter values: the derivative ForwardDiff
carries through a fixed set of retirements is not the slope finite differences see (#370).
The jumps are largest on the back node and fall by a factor of about 4 per column inward,
so `r + P + 1` columns hold the direct break and as many again bring the tail to the level
of the interior (measured, WP-044: T-590). Columns `window_back_band + 1 … n₁` are the ones to read.
"""
window_back_band(w::WindowFrame) = 2 * (w.n1 - w.inj)
window_back_band(st::EMState) = window_back_band(st.aux.window)

"""
    window_shifts(window::WindowFrame, k) -> Int

The number of cells the window has moved after step `k` (`k = 0`: at t = 0),
`max(0, floor((k·Δt − start)/Δx₁))`; a host function of the configuration.
"""
function window_shifts(w::WindowFrame, k::Integer)
    t = k * w.dt - w.window.start
    return t < zero(t) ? 0 : floor(Int, t / w.dx1)
end
window_shifts(st::EMState, k::Integer) = window_shifts(st.aux.window, k)

"The fresh particles of each species that found no free slot in the window's refills (a device read: never inside a step)."
window_lost(st::EMState) = map(l -> Int(Array(l)[1]), st.aux.window.lost)

"""
    window_superluminal(state::EMState) -> NTuple{Int}

Per species, the fresh particles of the window's refills whose drawn speed reached c: the loader
draws each velocity component as `drift + thermal·z` with `z` standard normal, in velocity, so the
Gaussian tail passes c with a probability that grows with the thermal speed (#257). Such a
particle is loaded at the drift velocity instead, so its charge stays in the plasma and Gauss's
law holds, and it is counted here, on the device (raising inside a step would be host branching
on device data, §0.4). The initial load refuses such a draw instead, with an `ArgumentError`
([`initialize!`](@ref)). A device read: never inside a step.
"""
window_superluminal(st::EMState) = map(l -> Int(Array(l)[1]), st.aux.window.fast)

function WindowFrame(problem::EMProblem{2,T}, grid, backend, shape, passes::Integer; back = nothing, workgroupsize = nothing) where {T}
    n1 = ncells(grid)[1]
    inj = window_injection_column(n1, shape, passes)
    inj >= 2 ||
        throw(ArgumentError("a moving window of $n1 columns leaves no plasma columns for this shape and $passes filter passes: the plasma is loaded up to column n₁ − 1 − $(n1 - 1 - inj), which must be at least 2"))
    loads = map(problem.species) do d
        (p = window_density_params(problem, d), ppc = Int32.(d.ppc), vdrift = T.(d.drift), vth = T.(d.thermal),
         key = (UInt32(problem.seed), UInt32(d.stream)))
    end
    lost = map(_ -> KernelAbstractions.zeros(backend, Int32, 1), problem.species)
    fast = map(_ -> KernelAbstractions.zeros(backend, Int32, 1), problem.species)
    flags = map(d -> KernelAbstractions.zeros(backend, Int32, inj * ncells(grid)[2] * prod(d.ppc)), problem.species)
    bk = back === nothing ? nothing : BackPMLFields(back, grid, problem.dt; backend, workgroupsize)
    return WindowFrame(problem.window, problem.dt, T(cellsize(grid)[1]), n1, inj, guards(grid), loads, lost, fast, flags, bk)
end

# --- the x edges ------------------------------------------------------------------------------

# Zero the x-guard columns of a field (every row): ndrange (2g, rows), the first g columns and the last g.
@kernel function zero_x_guards_kernel!(a, g::Int32, shift::Int32)
    i, j = @index(Global, NTuple)
    c = ifelse(Int32(i) <= g, Int32(i), Int32(i) + shift)
    @inbounds a[c, j] = zero(eltype(a))
end

"""
    zero_x_edges!(window, a; workgroupsize = nothing)

Set the x-guard columns of the field `a` to zero (the window's outside): after a field update, in
place of the periodic images, and on a deposited current or charge before the fold, which then
drops the contributions of cells outside the box. With `window = nothing` nothing is done.
"""
zero_x_edges!(::Nothing, a; workgroupsize = nothing) = a
function zero_x_edges!(w::WindowFrame, a; workgroupsize = nothing)
    w.g == 0 && return a
    shift = size(a, 1) - 2w.g
    launcher(zero_x_guards_kernel!, get_backend(a), workgroupsize)(a, Int32(w.g), Int32(shift); ndrange = (2w.g, size(a, 2)))
    return a
end

edges_b!(::Nothing, f, wgs) = nothing
edges_e!(::Nothing, f, wgs) = nothing
function edges_b!(w::WindowFrame, f, wgs)
    zero_x_edges!(w, f.Bx; workgroupsize = wgs)
    zero_x_edges!(w, f.By; workgroupsize = wgs)
    zero_x_edges!(w, f.Bz; workgroupsize = wgs)
    return nothing
end
function edges_e!(w::WindowFrame, f, wgs)
    zero_x_edges!(w, f.Ex; workgroupsize = wgs)
    zero_x_edges!(w, f.Ey; workgroupsize = wgs)
    zero_x_edges!(w, f.Ez; workgroupsize = wgs)
    return nothing
end

"""
    solve_fields!(::Yee, f, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, layer, window; workgroupsize = nothing)

The Yee step with the antennas, the absorbing layer and the moving window's x edges (a
[`WindowFrame`](@ref), or `nothing`): after each sub-step, its antenna correction and its layer
damping, the x-guards of the components it updated are set to zero, the undisturbed outside of the
window. With `window = nothing` it is the step with antennas and layer, bit for bit.
"""
function solve_fields!(::Yee, f::NamedTuple, grid::PeriodicGrid{2}, dt, antennas::Tuple, t, layer, window; workgroupsize = nothing)
    T = eltype(f.Ex)
    h = T(dt) / 2
    absorb_b!(layer, f, workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    back_b!(window, f, grid, h, false, workgroupsize)
    antenna_b!(antennas, f, h, t, workgroupsize)
    absorb_b!(layer, f, workgroupsize)
    edges_b!(window, f, workgroupsize)
    absorb_e!(layer, f, workgroupsize)
    yee_e!(f, grid, dt; workgroupsize)
    back_e!(window, f, grid, dt, workgroupsize)
    antenna_e!(antennas, f, dt, t + h, workgroupsize)
    absorb_e!(layer, f, workgroupsize)
    edges_e!(window, f, workgroupsize)
    absorb_b!(layer, f, workgroupsize)
    yee_b_half!(f, grid, h; workgroupsize)
    back_b!(window, f, grid, h, true, workgroupsize)
    antenna_b!(antennas, f, h, t + T(dt), workgroupsize)
    absorb_b!(layer, f, workgroupsize)
    edges_b!(window, f, workgroupsize)
    return f
end

# Particles that crossed an x edge of the window this step (their cell along axis 1 jumped across
# the periodic seam, by n₁ − 1 ≥ 3 (the window has at least 4 columns), while an ordinary move changes it by at most one) are retired
# before the current of that move is deposited (the header says why).
@kernel function retire_x_kernel!(s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE && abs(s.cell[1][i] - s.extras.old_cell_1[i]) > Int32(1)
        s.status[i] = ParticleMeshCore.FREE
    end
end

retire_x!(::Nothing, st) = nothing
function retire_x!(::WindowFrame, st)
    wgs = st.numerics.workgroupsize
    for s in st.species
        launcher(retire_x_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    return nothing
end

# --- the antennas ------------------------------------------------------------------------------

"""
    window_antennas(antennas, s) -> antennas

The antennas of a static grid placed in the window after `s` shifts: each plane keeps its place
in the lab frame, so its column moves back by `s`; once it is closer than `g + 2` columns to the
back edge it is inactive (the step skips it).
"""
window_antennas(antennas::Tuple, s::Integer) =
    map(a -> Antenna(a.laser, a.ia - Int32(s), a.xa, a.dx, a.g, a.n2, a.ly, a.hx, a.ht), antennas)

# The window passes antenna a in the step after which it has moved ia − g − 1 cells; its pulse must
# be over by then, six durations after its delay (the envelope below Float64 roundoff). With a PML
# on the back edge (WP-034) the plane must also stay out of the layer's columns 1 … cells while it
# emits: it reaches them after ia − cells − 1 shifts, and a plane inside them at the start is refused.
function check_window_antennas(w::WindowFrame, antennas::Tuple)
    cb = back_cells(w.back)
    for a in antennas
        edge = max(Int(a.g), cb)
        s_off = a.ia - edge - 1                    # the shifts after which it no longer acts outside the layer
        cb > 0 && s_off <= 0 &&
            throw(ArgumentError("the antenna's columns $(a.ia - 1) and $(a.ia) lie inside the back-edge PML of $cb cells; place the plane beyond column $(cb + 1)"))
        t_off = w.window.start + s_off * w.dx1
        l = a.laser
        l.delay + 6 * l.duration <= t_off ||
            throw(ArgumentError("laser of delay $(l.delay) and duration $(l.duration): the window passes its antenna (column $(a.ia))$(cb > 0 ? " into the back-edge PML of $cb cells" : "") at t ≈ $t_off, before six durations after the delay; place the plane further forward or start the window later"))
    end
    return nothing
end

# --- the shift and the refill ----------------------------------------------------------------

# One column back: a[i, j] = src[i + 1, j] on the interior columns 1 … n₁ − 1 (array columns
# g + 1 … g + n₁ − 1), zero on the x-guards, and on the front column zero, or (`keep`, Ex) the
# value it had (the front face's Ex carried along, the header says why); every row, the y-guards
# included (they stay the images of their interior rows). One work item per entry.
@kernel function shift_columns_kernel!(a, @Const(src), lo::Int32, hi::Int32, keep::Bool)
    i, j = @index(Global, NTuple)
    i32 = Int32(i)
    @inbounds if lo <= i32 && i32 < hi
        a[i, j] = src[i32 + Int32(1), j]
    elseif keep && i32 == hi
        a[i, j] = src[i, j]
    else
        a[i, j] = zero(eltype(a))
    end
end

# Every alive particle moves one cell back in window coordinates; the particles of the back column
# leave the window and are retired (the slot keeps a valid cell).
@kernel function shift_particles_kernel!(s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = s.cell[1][i] - Int32(1)
        s.cell[1][i] = max(c, Int32(1))
        c < Int32(1) && (s.status[i] = ParticleMeshCore.FREE)
    end
end

"The momentum γv of a drawn velocity, per unit mass, as `initialize!` converts it (a device function)."
@inline function velocity_momentum(v::NTuple{3,T}) where {T}
    γ = one(T) / sqrt(one(T) - (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]))
    return (γ * v[1], γ * v[2], γ * v[3])
end

# The fresh plasma of the window columns col0 … col0 + ncol − 1 at the window position `origin`
# (global column = window column + origin), written behind the alive count: work item i is the
# in-cell index k of the cell (column, row) in the order of the static loader within a column
# (row by row, k fastest), into slot count[1] + i if that slot exists. Positions and draws are the
# engine's quiet start at the global cell, the momenta γv of the drawn velocities; a drawn speed
# that reaches c (or a NaN) is replaced by the drift velocity, and flags[i] says so (1, else 0).
@kernel function refill_kernel!(s, flags, @Const(count), density, p, dx, ppc::NTuple{2,Int32}, col0::Int32, n2::Int32,
                                origin::Int32, vdrift, vth, key, cap::Int32)
    i = Int32(@index(Global, Linear))
    K = ppc[1] * ppc[2]
    cell = (i - Int32(1)) ÷ K                     # 0-based cell index, rows fastest
    k = (i - Int32(1)) % K + Int32(1)
    c2 = cell % n2 + Int32(1)
    c1 = col0 + cell ÷ n2
    @inbounds slot = count[1] + i
    fast = false
    if slot <= cap
        δ, v0, w = quiet_particle_2d(c1 + origin, c2, k, ppc, dx, density, p, vdrift, vth, key)
        fast = !(v0[1] * v0[1] + v0[2] * v0[2] + v0[3] * v0[3] < one(v0[1]))
        u = velocity_momentum(ifelse(fast, vdrift, v0))
        @inbounds begin
            s.cell[1][slot] = c1
            s.cell[2][slot] = c2
            s.offset[1][slot] = δ[1]
            s.offset[2][slot] = δ[2]
            s.u[1][slot] = u[1]
            s.u[2][slot] = u[2]
            s.u[3][slot] = u[3]
            s.w[slot] = w
            s.status[slot] = ALIVE
        end
    end
    @inbounds flags[i] = ifelse(fast, Int32(1), Int32(0))
end

# After a refill of `nnew` particles: the count, capped at the capacity, and the lost particles.
@kernel function refill_count_kernel!(count, lost, nnew::Int32, cap::Int32)
    @inbounds begin
        total = count[1] + nnew
        lost[1] += max(total - cap, Int32(0))
        count[1] = min(total, cap)
    end
end

"""
    load_window_columns!(s, grid, load, lost, col0, ncol, origin; fast = nothing, flags = nothing, workgroupsize = nothing)

Load the fresh plasma of `load` (a species' entry of the [`WindowFrame`](@ref)) on the window
columns `col0 … col0 + ncol − 1` at the window position `origin`, into the slots behind
`s.count` (which must be the alive count, as after a sort), and advance the count; fresh particles
beyond the capacity are counted in `lost` instead. A fresh particle whose drawn speed reaches c
is loaded at the drift velocity and, with a device counter `fast`, counted there (through the
scratch `flags` of at least one entry per fresh particle, allocated when `nothing`). All on the
device.
"""
function load_window_columns!(s, grid, load, lost, col0, ncol, origin; fast = nothing, flags = nothing, workgroupsize = nothing)
    n2 = ncells(grid)[2]
    nnew = ncol * n2 * prod(load.ppc)
    be = get_backend(s)
    flags === nothing && (flags = KernelAbstractions.zeros(be, Int32, nnew))
    launcher(refill_kernel!, be, workgroupsize)(s, flags, s.count, RampedCosine(), load.p, cellsize(grid), load.ppc, Int32(col0),
                                                Int32(n2), Int32(origin), load.vdrift, load.vth, load.key,
                                                Int32(capacity(s)); ndrange = nnew)
    launcher(refill_count_kernel!, be, workgroupsize)(s.count, lost, Int32(nnew), Int32(capacity(s)); ndrange = 1)
    fast === nothing || sum!(fast, view(flags, 1:nnew); init = false)
    return s
end

# The initial load of a window problem's species i, description d (step2d.jl's `initialize!`):
# the window's own loader on its plasma columns 1 … injection_column at its first position, the
# engine's quiet start at the same cells as a static load, the velocities converted to momenta in
# the same kernel. A drawn speed that reaches c is refused here, as `check_subluminal` refuses it
# in a static load (#257): one device count, read on the host outside the step (§0.4).
function load_window_species!(w::WindowFrame, i, s, grid, wgs, d)
    fill!(s.count, Int32(0))
    fill!(w.fast[i], Int32(0))
    load_window_columns!(s, grid, w.loads[i], w.lost[i], 1, injection_column(w), 0; fast = w.fast[i], flags = w.flags[i],
                         workgroupsize = wgs)
    nfast = Int(Array(w.fast[i])[1])
    nfast == 0 ||
        throw(ArgumentError("species $(d.name): $nfast of the speeds drawn for the moving window's initial load reach c (in c); " *
                            "the thermal spread is drawn in velocity, drift + thermal·z with z standard normal, " *
                            "so its Gaussian tail can pass c: lower the thermal speed or the drift"))
    return s
end

"""
    shift_window!(state::EMState{2}) -> state

Move the window of `state` by one cell: E and B one column back (through the scratch ρ, the
x-guards and the front column zero but for Ex on the front face, which is carried along), a PML's
convolution fields with them (front column zero), every particle one cell back with the back column's retired, the
slots compacted by the sort, the fresh plasma of the [`injection_column`](@ref) loaded behind the
alive count, and the kinetic-energy chunks recomputed. Called by `step!` at the end of a step; the window
position it moves to is the caller's (the count of shifts).
"""
function shift_window!(st::EMState{2}, s_new::Integer)
    (; numerics, grid, fields, aux) = st
    w = aux.window
    wgs = numerics.workgroupsize
    g, n = guards(grid), ncells(grid)
    for a in (fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)
        copyto!(st.ρ, a)
        launcher(shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
    end
    shift_absorber!(aux.layer, wgs)
    shift_absorber!(w.back, wgs)
    for (s, ws, load, lost, fast, flags) in zip(st.species, st.ws, w.loads, w.lost, w.fast, w.flags)
        launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
        sort_by_cell!(s, grid, ws; workgroupsize = wgs)
        load_window_columns!(s, grid, load, lost, injection_column(w), 1, s_new; fast, flags, workgroupsize = wgs)
    end
    kinetic!(st)
    return st
end

# The window's moves after step k: one shift per cell it moved in this step.
advance_window!(::Nothing, st, k) = st
function advance_window!(w::WindowFrame, st, k)
    for s_new in (window_shifts(w, k - 1) + 1):window_shifts(w, k)
        shift_window!(st, s_new)
    end
    return st
end

# --- diagnostics -------------------------------------------------------------------------------

"""
    window_charge!(ρ, state::EMState{2}) -> ρ

The charge density of the state's particles on the primal nodes, deposited in the state's mode
(the species sorted first where the mode needs it), with the window's convention: the x-guards
dropped (zero outside), the y-guards folded. A diagnostic, outside `step!`.
"""
function window_charge!(ρ, st::EMState{2})
    (; numerics, grid) = st
    fill!(ρ, zero(eltype(ρ)))
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws)
        deposit!(deposit_mode(st, s), ρ, s, numerics.shape, RHO_STAGGER_2D, grid, ws)
    end
    zero_x_edges!(st.aux.window, ρ)
    fold_guards!(ρ, grid)
    return ρ
end

"""
    window_background!(ρ₀, state::EMState{2}, s) -> ρ₀

The immobile background's charge density in the window at position `s` (after `s` shifts), with
the opposite sign: the charge of the cold quiet start of every species on the window's plasma
columns `1 …` [`injection_column`](@ref) at their global columns, deposited as [`window_charge!`](@ref) deposits. The plasma starts neutral, so
Gauss's law reads `D⁻·E = ρ − ρ₀(s)` at every step. Allocates scratch species: a diagnostic,
outside `step!`.
"""
function window_background!(ρ0, st::EMState{2}, s_win::Integer)
    (; numerics, grid) = st
    w = st.aux.window
    T = eltype(ρ0)
    be = get_backend(ρ0)
    fill!(ρ0, zero(T))
    n = ncells(grid)
    for (s, load) in zip(st.species, w.loads)
        cold = (; load..., vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)))
        sc = Species(be, T, prod(n) * prod(load.ppc); q = s.q, m = s.m, N = 2)
        ws = Workspace(sc, grid)
        load_window_columns!(sc, grid, cold, KernelAbstractions.zeros(be, Int32, 1), 1, injection_column(w), s_win)
        mode = effective_mode(numerics.mode, T)
        mode isa Reference && sort_by_cell!(sc, grid, ws)
        deposit!(mode, ρ0, sc, numerics.shape, RHO_STAGGER_2D, grid, ws)
    end
    zero_x_edges!(w, ρ0)
    fold_guards!(ρ0, grid)
    return ρ0
end
