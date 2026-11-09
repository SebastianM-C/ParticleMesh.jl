# The witness bunch at N = 2 (WP-047, A0, A3, §0.4): test electrons injected at fixed phases of the
# box (of the moving window, when there is one) at a fixed time, pushed with the plasma, and read
# out as their energy gain. The objective of the 2D LWFA optimization is its weighted mean.
#
# - Test particles: the bunch deposits no charge and no current (no beam loading), so the run's
#   fields, and with them every other observable, are those of the same problem without it. It is
#   not a species of the problem: it is not sorted, refilled, deposited or counted in the kinetic
#   energy, and it lives in its own fixed-capacity buffer (`WitnessFrame`).
# - Parameters (A3): the positions are a fixed lattice of `ppc` points per cell on the cells of a
#   fixed region, the same for every parameter value, and the bunch's centre, length and width
#   enter only through the averaging weights, a Gaussian of the lattice positions. A shift of the
#   centre therefore moves the weight across the lattice, never a particle. The injection time,
#   the region and the lattice are configuration (host integers, §0.4 rule 2); the momentum,
#   charge and mass may carry derivatives.
# - Injection: at the start of the first step whose start time (k − 1)·Δt reaches `inject`, every
#   lattice point becomes alive at its place with the momentum `momentum`, taken as the momentum
#   at the half step before (as the window's refill takes its plasma's). Before that step the
#   buffer is empty and the bunch is not pushed.
# - Edges: the bunch moves and leaves as the plasma does. With a moving window its particles shift
#   one column back with every shift, and one that falls behind the back edge, or crosses an x
#   edge, is retired; across an open axis 2 it is retired as the plasma is. A retired witness
#   leaves the mean, which then jumps between neighbouring parameter values, so an objective needs
#   a bunch that stays inside ([`witness_alive`](@ref) counts the ones left).

"""
    WitnessBunch(; inject, region, momentum, center, length, width = Inf, ppc = (1, 1), charge = -1, mass = 1)

A bunch of test particles of an [`EMProblem`](@ref) at N = 2 (WP-047), in A1 units. At the time
`inject` (1/ω₀; the start of the first step at or after it) the particles appear on a lattice of
`ppc = (p₁, p₂)` points per cell, `((k₁ − ½)/p₁, (k₂ − ½)/p₂)` in each cell, on every cell whose
centre lies inside `region = ((x_lo, x_hi), (y_lo, y_hi))` (c/ω₀; with a moving window in window
coordinates, measured from its back edge at the injection, so the lattice is a set of fixed phases
behind the laser), with the momentum `momentum` (u = γv in mₑc, three components, or a number
along axis 1). They carry `charge` (e) and `mass` (mₑ) but deposit nothing: test particles.

Each particle has an averaging weight `exp(−((x − center[1])/length)²/2 − ((y − center[2])/width)²/2)`,
a Gaussian bunch of rms `length` and `width` sampled on the fixed lattice: `center`, `length` and
`width` are the bunch's parameters and enter through the weights only (A3); `width = Inf` weighs
every row alike. The objective is the weighted mean energy gain, [`witness_energy_gain`](@ref).
`inject`, `region` and `ppc` are configuration (plain numbers); `momentum`, `center`, `length`,
`width`, `charge` and `mass` may be `Dual`s.
"""
struct WitnessBunch{T}
    inject::Float64
    region::NTuple{2,NTuple{2,Float64}}
    ppc::NTuple{2,Int}
    momentum::NTuple{3,T}
    center::NTuple{2,T}
    length::T
    width::T
    charge::T
    mass::T
end

function WitnessBunch(; inject::Real, region, momentum, center, length, width = Inf, ppc = (1, 1), charge = -1, mass = 1)
    u = momentum isa Number ? (momentum, zero(momentum), zero(momentum)) : Tuple(momentum)
    T = promote_type(map(typeof, u)..., map(typeof, Tuple(center))..., typeof(length), typeof(width), typeof(charge), typeof(mass))
    T <: AbstractFloat || (T = promote_type(T, Float64))   # integer inputs
    reg = (Float64.(Tuple(region[1])), Float64.(Tuple(region[2])))
    pp = Int.(Tuple(ppc))
    Base.length(u) == 3 || throw(ArgumentError("a witness momentum has three components or is a number along axis 1, got $momentum"))
    Base.length(center) == 2 || throw(ArgumentError("a witness bunch has a centre per axis, got $center"))
    all(r -> r[1] < r[2], reg) || throw(ArgumentError("the witness region $region needs lo < hi on both axes"))
    all(>(0), pp) || throw(ArgumentError("the witness lattice needs ppc > 0 on both axes, got $ppc"))
    inject >= 0 && isfinite(inject) || throw(ArgumentError("the witness injection time must be finite and ≥ 0, got $inject"))
    length > 0 && width > 0 || throw(ArgumentError("the witness bunch needs a positive length and width, got $length and $width"))
    all(isfinite, u) || throw(ArgumentError("the witness momentum must be finite, got $momentum"))
    mass > 0 || throw(ArgumentError("the witness mass must be positive, got $mass"))
    return WitnessBunch{T}(Float64(inject), reg, pp, T.(u), T.(Tuple(center)), T(length), T(width), T(charge), T(mass))
end

"The bunch in element type `T` (the problem's)."
witness_as(::Type{T}, b::WitnessBunch) where {T} =
    WitnessBunch{T}(b.inject, b.region, b.ppc, T.(b.momentum), T.(b.center), T(b.length), T(b.width), T(b.charge), T(b.mass))

"""
    WitnessFrame

The witness bunch placed on a grid by the state: the description, its particles (a `Species{2}`
of one slot per lattice point, with the push's extras; `w` holds the averaging weights), the
injection step `kinj`, the first cell `c0` and the cells `nc` of the lattice per axis, and its
points per cell.
"""
struct WitnessFrame{B,S}
    bunch::B
    species::S
    kinj::Int
    c0::NTuple{2,Int32}
    nc::NTuple{2,Int32}
    ppc::NTuple{2,Int32}
end

"The cells `(first, count)` of one axis whose centre lies inside `[lo, hi]`, for cell size `h` and `n` cells."
function witness_cells(lo, hi, h, n, axis)
    a = ceil(Int, lo / h - one(h) / 2) + 1        # centre (a − ½)·h ≥ lo
    b = floor(Int, hi / h + one(h) / 2)           # centre (b − ½)·h ≤ hi
    1 <= a <= b <= n ||
        throw(ArgumentError("the witness region ($lo, $hi) along axis $axis holds no cell centre of the $n cells of size $h, or leaves the box"))
    return a, b - a + 1
end

"The injection step of a bunch: the first step k whose start time (k − 1)·Δt reaches `inject` (a host loop over the configuration)."
function witness_step(inject, dt)
    k = 1
    while (k - 1) * dt < inject
        k += 1
    end
    return k
end

function WitnessFrame(problem::EMProblem{2,T}, grid, backend) where {T}
    b = problem.witness
    h = cellsize(grid)
    n = ncells(grid)
    (a1, n1), (a2, n2) = ntuple(d -> witness_cells(b.region[d]..., h[d], n[d], d), Val(2))
    kinj = witness_step(b.inject, problem.dt)
    kinj <= problem.nsteps ||
        throw(ArgumentError("the witness bunch is injected at step $kinj (t = $(b.inject)), after the $(problem.nsteps) steps of the problem"))
    cap = n1 * n2 * prod(b.ppc)
    s = Species(backend, T, cap; q = b.charge, m = b.mass, N = 2, extras = flux_extras(T, 2))
    return WitnessFrame(b, s, kinj, Int32.((a1, a2)), Int32.((n1, n2)), Int32.(b.ppc))
end

# One work item per lattice point, in lattice order (cells along axis 1 fastest, then rows; within
# a cell axis 1 fastest): its cell, offset, momentum and averaging weight, and alive.
@kernel function inject_witness_kernel!(s, c0::NTuple{2,Int32}, nc::NTuple{2,Int32}, ppc::NTuple{2,Int32}, dx, u0, center, len, wid)
    i = Int32(@index(Global, Linear))
    T = eltype(s.w)
    K = ppc[1] * ppc[2]
    cell = (i - Int32(1)) ÷ K
    k = (i - Int32(1)) % K
    c1 = c0[1] + cell % nc[1]
    c2 = c0[2] + cell ÷ nc[1]
    δ1 = (T(k % ppc[1]) + one(T) / 2) / T(ppc[1])
    δ2 = (T(k ÷ ppc[1]) + one(T) / 2) / T(ppc[2])
    x1 = (T(c1 - Int32(1)) + δ1) * dx[1]
    x2 = (T(c2 - Int32(1)) + δ2) * dx[2]
    z1 = (x1 - center[1]) / len
    z2 = (x2 - center[2]) / wid
    @inbounds begin
        s.cell[1][i] = c1
        s.cell[2][i] = c2
        s.offset[1][i] = δ1
        s.offset[2][i] = δ2
        s.extras.old_cell_1[i] = c1
        s.extras.old_cell_2[i] = c2
        s.extras.old_offset_1[i] = δ1
        s.extras.old_offset_2[i] = δ2
        s.u[1][i] = u0[1]
        s.u[2][i] = u0[2]
        s.u[3][i] = u0[3]
        s.w[i] = exp(-(z1 * z1 + z2 * z2) / 2)
        s.status[i] = ALIVE
    end
end

"The bunch's particles of a state, as a tuple: `()` without a bunch."
witness_species(::Nothing) = ()
witness_species(w::WitnessFrame) = (w.species,)
witness_species(st::EMState) = witness_species(witness_frame(st))
witness_frame(st::EMState) = haskey(st.aux, :witness) ? st.aux.witness : nothing

"Empty the bunch's buffer (`initialize!`)."
clear_witness!(::Nothing) = nothing
clear_witness!(w::WitnessFrame) = (fill!(w.species.status, ParticleMeshCore.FREE); fill!(w.species.count, Int32(0)); nothing)

"At the start of step `k`: inject the bunch if `k` is its injection step (a host test of the step count, §0.4)."
inject_witness!(::Nothing, grid, k, wgs) = nothing
function inject_witness!(w::WitnessFrame, grid, k, wgs)
    k == w.kinj || return nothing
    b = w.bunch
    s = w.species
    launcher(inject_witness_kernel!, get_backend(s), wgs)(s, w.c0, w.nc, w.ppc, cellsize(grid), b.momentum, b.center, b.length,
                                                          b.width; ndrange = capacity(s))
    fill!(s.count, Int32(capacity(s)))
    return nothing
end

# --- observables (host reads, outside a step) ---------------------------------------------------

"""
    witness_gains(state::EMState{2}) -> NamedTuple

The witness bunch of `state` per particle, in lattice order (host arrays): the position `x` and
`y` (c/ω₀; window coordinates with a moving window), the averaging weight `w`, the energy gain
`gain = m·(γ − γ₀)` in mₑc² (γ of the momentum at the last half step, γ₀ that of the injected
momentum, written `m·(|u|² − |u₀|²)/(γ + γ₀)` against cancellation) and `alive`. An
`ArgumentError` without a bunch. A device read: never inside a step.
"""
function witness_gains(st::EMState{2})
    w = witness_frame(st)
    w === nothing && throw(ArgumentError("the state has no witness bunch (EMProblem(...; witness = WitnessBunch(...)))"))
    s = w.species
    h = cellsize(st.grid)
    u1, u2, u3 = Array(s.u[1]), Array(s.u[2]), Array(s.u[3])
    alive = Array(s.status) .== ALIVE
    u0 = w.bunch.momentum
    m = w.bunch.mass
    a0 = u0[1]^2 + u0[2]^2 + u0[3]^2
    g0 = sqrt(1 + a0)
    gain = map(u1, u2, u3) do a, b, c
        a2 = a * a + b * b + c * c
        m * (a2 - a0) / (sqrt(1 + a2) + g0)
    end
    x = (Array(s.cell[1]) .- 1 .+ Array(s.offset[1])) .* h[1]
    y = (Array(s.cell[2]) .- 1 .+ Array(s.offset[2])) .* h[2]
    return (; x, y, w = Array(s.w), gain, alive)
end

"""
    witness_energy_gain(state::EMState{2}; allow_lost = false) -> T

The objective of WP-047: the weighted mean energy gain of the witness bunch's alive particles,
`Σ wᵢ·gainᵢ / Σ wᵢ` over [`witness_gains`](@ref), in mₑc², summed on the host in lattice order
(deterministic), the gain read at the end of the run so far (the horizon of the run is part of the
objective). Differentiable in every parameter of the run and of the bunch (A3) as long as no
witness is retired: a parameter step across a retirement makes it jump, so by default a retired
witness is an `ArgumentError` (`allow_lost = true` averages over the alive ones, [`witness_alive`](@ref)).
The weights are a Gaussian truncated to the lattice's region, so the region should cover the
centre by several lengths and widths; if the weights of the alive witnesses vanish (the centre far
outside the region), or none is alive (before the injection, or all lost), it is an
`ArgumentError` too, never a NaN. So is an alive witness inside a moving window's back band
([`window_back_band`](@ref)), where observables are not to be read (#370).
"""
function witness_energy_gain(st::EMState{2}; allow_lost::Bool = false)
    g = witness_gains(st)
    any(g.alive) || throw(ArgumentError("no witness particle is alive: the bunch is not injected yet, or every particle left the box"))
    allow_lost || all(g.alive) ||
        throw(ArgumentError("$(count(!, g.alive)) of the $(length(g.alive)) witness particles left the box, so the mean jumps between neighbouring parameter values; widen the window, move the bunch, or pass allow_lost = true"))
    if st.aux.window !== nothing
        edge = window_back_band(st) * cellsize(st.grid)[1]
        any(i -> g.alive[i] && g.x[i] < edge, eachindex(g.x)) &&
            throw(ArgumentError("a witness particle lies in the window's back absorber band (the first $(window_back_band(st)) columns), where the fields jump between neighbouring parameter values (WP-044); move the bunch forward or widen the window"))
    end
    num = sum(g.w[i] * g.gain[i] for i in eachindex(g.w) if g.alive[i])
    den = sum(g.w[i] for i in eachindex(g.w) if g.alive[i])
    den > 0 ||
        throw(ArgumentError("the weights of the alive witnesses vanish (they underflow): the bunch's centre lies far outside the lattice's region"))
    return num / den
end

"""
    witness_alive(state::EMState{2}) -> Int

The number of the bunch's particles alive (all of them from the injection until one leaves the
box). A device read: never inside a step.
"""
function witness_alive(st::EMState{2})
    w = witness_frame(st)
    w === nothing && throw(ArgumentError("the state has no witness bunch"))
    return count(==(ALIVE), Array(w.species.status))
end
