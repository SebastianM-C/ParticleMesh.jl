# The WP-052 case: WP-045's small 2D LWFA run (wake2d_case.jl) at a fixed laser energy, run longer,
# with a WP-047 witness bunch on the axis as the objective. The free parameters are the pulse
# duration τ, the plasma density n₀ and the witness bunch's centre x_c; a₀ follows from τ so that
# the energy the antenna launches stays fixed.
using ParticleMesh, KernelAbstractions
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: witness_frame
using ForwardDiff: ForwardDiff

value_of(x::Real) = x
value_of(d::ForwardDiff.Dual) = ForwardDiff.value(d)

"""
Fixed parameters of the case (A1 units).

- The grid (224 × 64 cells of Δx = λ₀/16, Δy = 4Δx, periodic in y), the laser's waist and delay, the
  vacuum, the ramp and the window's start are WP-045's (`wake2d_case.jl`). The focus is at
  x = 80, 8 c/ω₀ into the plasma.
- `energy` is the laser energy per unit length in z, `laser_energy` = π·a₀²·w·τ/4·(1 + 1/(2w²)),
  that of a₀ = 1.5 at τ = 6 (the run's `low_energy` slice repeats the duration scan at 1/2.25
  of it).
- The run is 830 steps (t ≈ 300). The witness is injected at t = 150, once the laser is about
  40 c/ω₀ into the plasma, and accelerated over about 150 c/ω₀, about half the 1D linear dephasing
  length π/(k_p·(1 − √(1 − n₀))) ≈ 320 at n₀ = 0.072. The injection time and the horizon are part
  of the objective: the duration optimum moves with them (report §4).
- The witness lattice is 2 × 1 points per cell on the two rows at the axis, over window
  x ∈ `region` = (24, 56): the first wake bucket behind the laser (at x ≈ 56) and the decelerating
  phase behind it, so that the Gaussian weights (rms `length` = 2) are not cut off by the lattice's
  end for centres x_c ≥ 30. The best phase is at the bucket's rear (x ≈ 33 to 37).
- The electrons start with u₀ = 30 along x. At u₀ = 10 the decelerated ones behind the bucket fall
  behind the window's back edge (from x ≲ 34.5 at n₀ = 0.1), which `witness_energy_gain` refuses;
  at u₀ = 30 none is lost over the box (n₀ ≤ 0.1, τ = 4 to 6).
"""
const WCASE = (waist = 10.0, delay = 54.0, focus = 80.0, vacuum = 64.0, ramp = 8.0, start = 108.0,
               energy = π * 1.5^2 * 10.0 * 6.0 / 4 * (1 + 1 / (2 * 10.0^2)),
               cells = (224, 64), ppc = (2, 2), dy_cells = 4, nsteps = 830,
               inject = 150.0, region = (24.0, 56.0), u0 = 30.0, length = 2.0)

"""
    a0_at_energy(τ; energy, waist) -> a₀

The peak amplitude of a pulse of duration `τ` and `waist` (in-plane polarization) that carries
`energy` per unit length in z: the inverse of `laser_energy`, so the energy is the same for every τ
and ∂energy/∂τ = 0 through the coupling (the fixed-energy parametrization of WP-052).
"""
a0_at_energy(τ; energy = WCASE.energy, waist = WCASE.waist) = sqrt(energy / (π * waist * τ / 4 * (1 + 1 / (2 * waist^2))))

"""
    witness_problem(duration, density, center; nsteps, t_end, ppl, cells, ppc, dy_cells, energy, focus, inject) -> EMProblem{2}

The case at a pulse duration (1/e field half width), plasma density (n_c) and witness centre
(window x, c/ω₀), with a₀ from the fixed energy; the element type follows the arguments. For
resolution checks: `ppl` cells per λ₀ along x (with `cells` and `dy_cells` to keep the box and Δy),
and `t_end`, which replaces `nsteps` by the steps that reach that time: Δt follows Δx and Δy, so
a fixed step count would shorten the run when they shrink.
"""
function witness_problem(duration, density, center; nsteps = WCASE.nsteps, t_end = nothing, ppl = 16, cells = WCASE.cells, ppc = WCASE.ppc,
                         dy_cells = WCASE.dy_cells, energy = WCASE.energy, focus = WCASE.focus, inject = WCASE.inject)
    T = promote_type(typeof(duration), typeof(density), typeof(center), typeof(focus), typeof(energy))
    z3 = (zero(T), zero(T), zero(T))
    h = 2 * T(π) / ppl
    hy = dy_cells * h
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    t_end === nothing || (nsteps = ceil(Int, t_end / Float64(value_of(dt))))
    a0 = a0_at_energy(T(duration); energy = T(energy))
    pulse = GaussianLaser(; a0, waist = T(WCASE.waist), duration = T(duration), delay = T(WCASE.delay), focus = T(focus),
                          position = 6h, center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(density), zero(T), (1, 0), ppc, z3, z3, 1)
    hy64 = Float64(value_of(hy))
    yc = cells[2] * hy64 / 2
    wb = WitnessBunch(; inject, region = (WCASE.region, (yc - hy64, yc + hy64)), ppc = (2, 1), momentum = T(WCASE.u0),
                      center = (T(center), T(yc)), length = T(WCASE.length))
    return EMProblem("witness2d", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(WCASE.start)),
                     profile = DensityRamp(; vacuum = T(WCASE.vacuum), ramp = T(WCASE.ramp)), witness = wb)
end

function witness_state(duration, density, center; kw...)
    prob = witness_problem(duration, density, center; kw...)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    return st
end

"The objective: the witness bunch's weighted mean energy gain (mₑc²) at the last step (WP-047)."
witness_gain(duration, density, center; kw...) = witness_energy_gain(witness_state(duration, density, center; kw...))

"""
    lattice_x(st) -> Vector{Float64}

The window x of every witness lattice point at its injection, in lattice order (cells along axis 1
fastest, then rows; within a cell axis 1 fastest, then axis 2), from the state's witness frame: the
positions the averaging weights are a Gaussian of (only x: the case's weights do not depend on y).
"""
function lattice_x(st)
    w = witness_frame(st)
    h = Float64(value_of(cellsize(st.grid)[1]))
    p1, p2 = Int.(w.ppc)
    n1, n2 = Int.(w.nc)
    return [(w.c0[1] - 1 + c1 + (k1 - 0.5) / p1) * h for c2 in 0:(n2 - 1) for c1 in 0:(n1 - 1) for k2 in 1:p2 for k1 in 1:p1]
end

"""
    gain_at_centers(st, centers; length) -> Vector

The objective at other witness centres from one run: the witnesses are test particles, so their
gains do not depend on the centre, which enters only through the weights (A3). The weighted mean
of the run's gains with the weights recomputed at each centre, summed in lattice order as
`witness_energy_gain` sums (all witnesses alive, as it requires).
"""
function gain_at_centers(st, centers; length = WCASE.length)
    g = witness_gains(st)
    all(g.alive) || throw(ArgumentError("$(count(!, g.alive)) witnesses were lost"))
    x0 = lattice_x(st)
    gains = Float64.(value_of.(g.gain))
    return map(centers) do c
        w = exp.(-((x0 .- c) ./ length) .^ 2 ./ 2)
        sum(w .* gains) / sum(w)
    end
end

"""
    is_interior_max(xs, Js, x0, J0; rtol = 1e-9) -> Bool

Whether a slice `Js` at `xs` through the point `(x0, J0)` shows a maximum there: `xs` holds `x0`
once, with `J0` there, at least one point strictly on each side of it, and every other `J` below
`J0` by more than `rtol`. A slice cut short by a bound, with points on one side only, fails.
"""
function is_interior_max(xs, Js, x0, J0; rtol = 1e-9)
    at = [isapprox(x, x0; rtol = 1e-12) for x in xs]
    count(at) == 1 || return false
    isapprox(Js[findfirst(at)], J0; rtol) || return false
    any(<(x0), xs[.!at]) && any(>(x0), xs[.!at]) || return false
    return all(J -> J < J0 * (1 - rtol), Js[.!at])
end
