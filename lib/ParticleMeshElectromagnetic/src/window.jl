# The moving window at N = 2 (WP-028, A2, A3, §0.4): the box follows the laser at c along axis 1.
# The problem says so (`EMProblem(...; boundaries = (:window, ·), window = MovingWindow(; start))`)
# and gives the plasma a longitudinal profile (`profile = DensityRamp(; vacuum, ramp)`), entering
# through the weights (A3), in the initial load and in every refill, from the parameters of the
# step (`EMParams`), not from the state. The grid underneath stays the engine's `PeriodicGrid{2}`; no kernel
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

# The loader parameters of a species in a window problem: those of its cosine and the profile's.
function window_density_params(ps::SpeciesParams{T,2}, r) where {T}
    vac, len = r === nothing ? (T(-Inf), one(T)) : (T(r.vacuum), T(r.ramp))
    return (density_params(ps)..., vac, len)
end

"""
    window_load(setup, ps::SpeciesParams, params::EMParams) -> NamedTuple
    window_load(state, params, i) -> NamedTuple

What [`load_window_columns!`](@ref) loads of a species: the loader parameters `p` of
[`RampedCosine`](@ref) (the species' density, amplitude and wave vector and the parameters'
profile), the particles per cell `ppc`, the drift `vdrift` and thermal speeds `vth`, and the
Philox `key` (the parameters' seed, the species' stream), from the state's `setup` of the species
(configuration) and the parameters. The second form is that of species `i` of a state.
"""
window_load(setup, ps::SpeciesParams{T}, p::EMParams) where {T} =
    (p = window_density_params(ps, p.profile), ppc = Int32.(setup.ppc), vdrift = T.(ps.drift), vth = T.(ps.thermal),
     key = (UInt32(p.seed), UInt32(setup.stream)))
window_load(st::EMState, p::EMParams, i::Integer) = window_load(st.setup[i], p.species[i], p)

"""
    WindowFrame

The moving window placed on a grid by the state: the description, the time step and the cell size
along axis 1 it counts its shifts with, the interior columns `n1`, the injection column `inj` (the
plasma is loaded on columns `1 … inj`, [`injection_column`](@ref)), the guard width `g`,
and what a run evolves of it ([`evolved`](@ref)): `lost`, one device counter per species of the
fresh particles that found no free slot (zero unless the buffers overflow; read with
[`window_lost`](@ref) outside a step), and `fast`, one device counter per species of the fresh
particles of the refills whose drawn speed reached c (read with [`window_superluminal`](@ref));
`flags` is per species the refill's scratch of one flag
per fresh particle that `fast` sums. The window's position is not kept: it is a function of the
step index ([`window_shifts`](@ref)).
What the refill loads is not here: the densities, the profile, the drifts, the thermal speeds and
the seed are parameters ([`window_load`](@ref)). What the window does to the fields within the
mesh solve is not here either: its x edges ([`WindowEdges`](@ref)) and the PML of its back edge
([`BackPMLFields`](@ref)) are field modifiers of the state (`state.modifiers`).
"""
struct WindowFrame{W,T,L,F}
    window::W
    dt::T
    dx1::T
    n1::Int
    inj::Int
    g::Int
    lost::L
    fast::L
    flags::F
end

"""
    WindowClock(start, dt, dx1)

When a moving window moves, apart from the window itself: its start time, the time step and the
cell size along axis 1 ([`window_shifts`](@ref)). Isbits configuration, the part of a
[`WindowFrame`](@ref) that the antennas' modifier keeps ([`AntennaPlanes`](@ref)).
"""
struct WindowClock{T}
    start::T
    dt::T
    dx1::T
end
WindowClock(w::WindowFrame) = WindowClock(w.window.start, w.dt, w.dx1)
WindowClock(::Nothing) = nothing

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
`3·(r + P + 1)` for a shape whose stencil reaches `r` nodes past its cell and `P` filter passes
(the `r + P + 1` columns that [`injection_column`](@ref) keeps free at the front, three times).
The particles the back edge retires (at a shift, or when they cross it) take their charge out
without current, and whether a particle goes at this step or the next depends on the parameters,
so the fields there jump between neighbouring parameter values: the derivative ForwardDiff
carries through a fixed set of retirements is not the slope finite differences see (#370).
The jumps are largest on the back node and fall inward, not at a fixed rate: in WP-043's case
(`r + P + 1 = 3`) at 600 steps, 5250, 2410, 701, 115, 8.5 and 1.1 times the interior's level on
columns 1 to 6; at 750 steps the tail is longer (5 times on column 7, 0.6 on column 9). So
`r + P + 1` columns hold the direct break, twice that brings the 600-step tail to the interior,
and the third is margin for longer runs (measured, WP-044: T-590 with no filter passes and with
two; the WP-044 PR's breakers at 450 to 750 steps). For the linear shape the interior itself is
about 5000 times noisier at the finite-difference scale, so the band is not where its gradients
part. Columns `window_back_band + 1 … n₁` are the ones to read.
"""
window_back_band(w::WindowFrame) = 3 * (w.n1 - w.inj)
window_back_band(st::EMState) = window_back_band(window_frame(st))

"The moving window's frame of a state, or `nothing` (at N = 1, and at N = 2 without a window)."
window_frame(st::EMState) = haskey(st.aux, :window) ? st.aux.window : nothing

"""
    window_shifts(window::WindowFrame, k) -> Int

The number of cells the window has moved after step `k` (`k = 0`: at t = 0),
`max(0, floor((k·Δt − start)/Δx₁))`; a host function of the configuration.
"""
function window_shifts(c::WindowClock, k::Integer)
    t = k * c.dt - c.start
    return t < zero(t) ? 0 : floor(Int, t / c.dx1)
end
window_shifts(w::WindowFrame, k::Integer) = window_shifts(WindowClock(w), k)
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

function WindowFrame(problem::EMProblem{2,T}, grid, backend, shape, passes::Integer) where {T}
    n1 = ncells(grid)[1]
    inj = window_injection_column(n1, shape, passes)
    inj >= 2 ||
        throw(ArgumentError("a moving window of $n1 columns leaves no plasma columns for this shape and $passes filter passes: the plasma is loaded up to column n₁ − 1 − $(n1 - 1 - inj), which must be at least 2"))
    lost = map(_ -> KernelAbstractions.zeros(backend, Int32, 1), problem.species)
    fast = map(_ -> KernelAbstractions.zeros(backend, Int32, 1), problem.species)
    flags = map(d -> KernelAbstractions.zeros(backend, Int32, inj * ncells(grid)[2] * prod(d.ppc)), problem.species)
    return WindowFrame(problem.window, problem.dt, T(cellsize(grid)[1]), n1, inj, guards(grid), lost, fast, flags)
end

# --- the x edges ------------------------------------------------------------------------------

"""
    WindowEdges(g)

The x edges of a moving window as a [`FieldModifier`](@ref): the guard width `g` of the columns
that hold the window's field-free outside in place of the periodic images. It has no arrays and
no parameter half.
"""
struct WindowEdges <: FieldModifier
    g::Int
end
edges_of(w::WindowFrame) = WindowEdges(w.g)

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
function zero_x_edges!(w::Union{WindowFrame,WindowEdges}, a; workgroupsize = nothing)
    w.g == 0 && return a
    shift = size(a, 1) - 2w.g
    launcher(zero_x_guards_kernel!, get_backend(a), workgroupsize)(a, Int32(w.g), Int32(shift); ndrange = (2w.g, size(a, 2)))
    return a
end

# The window's x edges as a field modifier, last in a state's tuple: after each Yee sub-step and
# every other modifier's correction of it, the x-guards of the components it updated are set to
# zero, the undisturbed outside of the window.
function after_substep!(w::WindowEdges, mp, f, grid, ::BHalf, wgs)
    zero_x_edges!(w, f.Bx; workgroupsize = wgs)
    zero_x_edges!(w, f.By; workgroupsize = wgs)
    zero_x_edges!(w, f.Bz; workgroupsize = wgs)
    return nothing
end
function after_substep!(w::WindowEdges, mp, f, grid, ::EStep, wgs)
    zero_x_edges!(w, f.Ex; workgroupsize = wgs)
    zero_x_edges!(w, f.Ey; workgroupsize = wgs)
    zero_x_edges!(w, f.Ez; workgroupsize = wgs)
    return nothing
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
    for s in (st.species..., witness_species(st)...)
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

# The antennas of step k in a moving window: where the window has their planes at the start of the step.
placed_antennas(c::WindowClock, antennas::Tuple, k) = window_antennas(antennas, window_shifts(c, k - 1))

# The window passes antenna a in the step after which it has moved ia − g − 1 cells; its pulse must
# be over by then, six durations after its delay (the envelope below Float64 roundoff). With a PML
# on the back edge (WP-034) the plane must also stay out of the layer's columns 1 … cells while it
# emits: it reaches them after ia − cells − 1 shifts, and a plane inside them at the start is refused.
# `back` is the back edge's placed PML, or `nothing`.
function check_window_antennas(w::WindowFrame, back, antennas::Tuple)
    cb = back_cells(back)
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

Load the fresh plasma of `load` (a species' [`window_load`](@ref)) on the window
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

# The initial load of a window problem's species i, of `load` (its `window_load`) and `name`
# (step2d.jl's `initialize!`): the window's own loader on its plasma columns 1 … injection_column
# at its first position, the engine's quiet start at the same cells as a static load, the
# velocities converted to momenta in the same kernel, from zeroed counters (the alive count and
# the refills' `fast` and `lost`: a run counts from its start). A drawn speed that reaches c is
# refused here, as `check_subluminal` refuses it in a static load (#257): one device count, read
# on the host outside the step (§0.4).
function load_window_species!(w::WindowFrame, i, s, grid, wgs, load, name)
    fill!(s.count, Int32(0))
    fill!(w.fast[i], Int32(0))
    fill!(w.lost[i], Int32(0))
    load_window_columns!(s, grid, load, w.lost[i], 1, injection_column(w), 0; fast = w.fast[i], flags = w.flags[i],
                         workgroupsize = wgs)
    nfast = Int(Array(w.fast[i])[1])
    nfast == 0 ||
        throw(ArgumentError("species $name: $nfast of the speeds drawn for the moving window's initial load reach c (in c); " *
                            "the thermal spread is drawn in velocity, drift + thermal·z with z standard normal, " *
                            "so its Gaussian tail can pass c: lower the thermal speed or the drift"))
    return s
end

"""
    shift_window!(state::EMState{2}, params, s_new) -> state

Move the window of `state` by one cell, to its position after `s_new` shifts: E and B one column back (through the scratch ρ, the
x-guards and the front column zero but for Ex on the front face, which is carried along), a PML's
convolution fields with them (front column zero), every particle one cell back with the back column's retired, the
slots compacted by the sort, the fresh plasma of the [`injection_column`](@ref) loaded behind the
alive count from the parameters ([`window_load`](@ref): the weights of the densities and the
profile at the global column, A3), and the kinetic-energy chunks recomputed. Called by `step!` at
the end of a step; the window position it moves to is the caller's (the count of shifts).
"""
function shift_window!(st::EMState{2}, p::EMParams, s_new::Integer)
    (; numerics, grid, fields, aux) = st
    w = aux.window
    wgs = numerics.workgroupsize
    g, n = guards(grid), ncells(grid)
    for a in (fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)
        copyto!(st.ρ, a)
        launcher(shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
    end
    shift_absorbers!(st.modifiers, wgs)
    for (s, ws, setup, ps, lost, fast, flags) in zip(st.species, st.ws, st.setup, p.species, w.lost, w.fast, w.flags)
        launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
        sort_by_cell!(s, grid, ws; workgroupsize = wgs)
        load_window_columns!(s, grid, window_load(setup, ps, p), lost, injection_column(w), 1, s_new; fast, flags, workgroupsize = wgs)
    end
    for s in witness_species(st)                 # the bunch moves back with the window, unsorted and not refilled
        launcher(shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    kinetic!(st)
    return st
end

# Every modifier's own arrays move with the fields (a PML's convolution fields; nothing otherwise).
@generated function shift_absorbers!(ms::NTuple{M,FieldModifier}, wgs) where {M}
    return quote
        Base.Cartesian.@nexprs $M i -> shift_absorber!(ms[i], wgs)
        return nothing
    end
end

# The window's moves after step k: one shift per cell it moved in this step.
advance_window!(::Nothing, st, p, k) = st
function advance_window!(w::WindowFrame, st, p, k)
    for s_new in (window_shifts(w, k - 1) + 1):window_shifts(w, k)
        shift_window!(st, p, s_new)
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
    zero_x_edges!(window_frame(st), ρ)
    fold_guards!(ρ, grid)
    return ρ
end

"""
    window_background!(ρ₀, state::EMState{2}, params, s) -> ρ₀

The immobile background's charge density in the window at position `s` (after `s` shifts), with
the opposite sign: the charge of the cold quiet start of every species, of the parameters'
densities and profile, on the window's plasma
columns `1 …` [`injection_column`](@ref) at their global columns, deposited as [`window_charge!`](@ref) deposits. The plasma starts neutral, so
Gauss's law reads `D⁻·E = ρ − ρ₀(s)` at every step. Allocates scratch species: a diagnostic,
outside `step!`.
"""
function window_background!(ρ0, st::EMState{2}, p::EMParams, s_win::Integer)
    (; numerics, grid) = st
    w = st.aux.window
    T = eltype(ρ0)
    be = get_backend(ρ0)
    fill!(ρ0, zero(T))
    n = ncells(grid)
    for (s, setup, ps) in zip(st.species, st.setup, p.species)
        cold = (; window_load(setup, ps, p)..., vdrift = (zero(T), zero(T), zero(T)), vth = (zero(T), zero(T), zero(T)))
        sc = Species(be, T, prod(n) * prod(cold.ppc); q = s.q, m = s.m, N = 2)
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
