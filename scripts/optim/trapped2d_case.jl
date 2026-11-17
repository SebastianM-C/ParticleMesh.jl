# The WP-065 case: WP-064's ionization case (born2d_case.jl) at a fixed laser energy, with an
# objective over every trapped electron, born by ionization or injected from the plasma itself.
# Its five parameters are the pulse's duration, waist, focus and delay and the plasma's density;
# a₀ follows from the duration and the waist so that the pulse carries the same energy at every
# point (WP-052's coupling, with the waist free as well).
isdefined(@__MODULE__, :BCASE) || include(joinpath(@__DIR__, "born2d_case.jl"))

"""
Fixed parameters of the WP-065 case beyond [`BCASE`](@ref) (A1 units).

- `energy` is the laser energy per unit length in z that the pulse carries through the antenna
  plane, [`laser_energy`](@ref) of the case's pulse at `a0_energy` (π·a₀²·w·τ/4·(1 + 1/(2w²)) at
  the base waist 10 and duration 6). The study's three budgets are those of a₀ = 1.9, 2.175 and
  2.4 at the base pulse (`budgets`), around the onset of the plasma's own injection in WP-064.
- `dopant_fraction`: the dopant's density is this fraction of the plasma's at every density, so
  the density moves both.
- `rise = (lo, hi)` is the trapping weight in the forward momentum u_x: 0 below `lo`, 1 above
  `hi` ([`smooth_rise`](@ref)). The wake's phase velocity at 0.072 n_c has γ_φ ≈ (n_c/n)^½ ≈ 3.7,
  so an electron with u_x above about 3.6 outruns the wake and stays in the bucket; the plasma's
  own electrons quiver below u_x ≈ 1 in the clean runs of WP-064 (their highest γ − 1 is 0.3 to
  1.0), and the born electrons are born near rest.
- `fade = (lo, hi)` is the position weight in window x (c/ω₀), as WP-064's but reaching back to
  just ahead of the window's back band (which ends at x ≈ 3.5 at λ₀/16 and nearer the edge at
  finer grids): the plasma's own electrons, once injected, sit in the buckets behind the first
  one (behind x = 12 in WP-064), and every trapped electron counts. WP-064's first-bucket weight
  (12, 24) is read beside it.
- `x0` is the base point (duration, waist, focus, delay, density); `names` and `scale` give each
  parameter's name and the scale of z = x ./ scale in problem.jl.
"""
const TCASE = (a0_energy = 1.9, budgets = (1.9, 2.175, 2.4), dopant_fraction = 0.01, rise = (3.0, 5.0), fade = (4.0, 10.0),
               x0 = (6.0, 10.0, 80.0, 54.0, 0.072), names = ("duration", "waist", "focus", "delay", "density"),
               scale = (6.0, 10.0, 80.0, 54.0, 0.072))

"""
    pulse_energy(a0, duration, waist) -> W

The energy per unit length in z of the case's pulse ([`laser_energy`](@ref) of a `GaussianLaser`
polarized in the plane, the case's pulse): π·a₀²·w·τ/4·(1 + 1/(2w²)).
"""
pulse_energy(a0, duration, waist) = π * a0^2 * waist * duration / 4 * (1 + 1 / (2 * waist^2))

"The energy budget of the pulse with `a0` at the base duration and waist."
budget(a0 = TCASE.a0_energy) = pulse_energy(a0, TCASE.x0[1], TCASE.x0[2])

"""
    a0_for_energy(duration, waist; energy) -> a₀

The a₀ at which a pulse of `duration` and `waist` carries `energy` per unit length in z, the
inverse of [`pulse_energy`](@ref): the fixed-energy coupling, through which ∂W/∂τ = ∂W/∂w = 0.
"""
a0_for_energy(duration, waist; energy = budget()) = sqrt(energy / (π * waist * duration / 4 * (1 + 1 / (2 * waist^2))))

"""
    trapped_problem(duration, waist, focus, delay, density; energy, ppl, ny, kw...) -> EMProblem{2}

The case at the five parameters, with a₀ from [`a0_for_energy`](@ref) and the dopant at
`TCASE.dopant_fraction` of `density`; the other keywords as [`born_problem`](@ref).
"""
function trapped_problem(duration, waist, focus, delay, density; energy = budget(), kw...)
    a0 = a0_for_energy(duration, waist; energy)
    return born_problem(a0, TCASE.dopant_fraction * density; duration, waist, focus, delay, density, kw...)
end

"The case's state after its run at the five parameters: the run of [`trapped_problem`](@ref)'s problem (and its keywords); a birth without a free slot is an error."
trapped_state(duration, waist, focus, delay, density; ppl::Integer = BCASE.ppl, ny::Integer = 1, kw...) =
    run_to_end(start_of(trapped_problem(duration, waist, focus, delay, density; ppl, ny, kw...); ppl, ny))

# The summand of a slot: weight × kinetic energy × position weight × trapping weight, zero for a free slot.
@inline function slot_trapped(status, c1, o1, u1, u2, u3, w, h1, lo, hi, ulo, uhi)
    x = (c1 - 1 + o1) * h1
    return ifelse(status == ALIVE, w * (sqrt(1 + u1 * u1 + u2 * u2 + u3 * u3) - 1) * smooth_rise(x, lo, hi) * smooth_rise(u1, ulo, uhi), zero(w))
end

# One species' sum, in slot order on the host (as `born_energy`, so that a run with duals gives
# the plain run's bits).
function species_trapped(s, h1, lo, hi, ulo, uhi)
    T = eltype(s.w)
    return foldl(+, Array(slot_trapped.(s.status, s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, h1, T(lo), T(hi), T(ulo), T(uhi))); init = zero(T))
end

function check_rise(rise)
    rise[2] > rise[1] || throw(ArgumentError("the trapping weight needs rise[2] > rise[1], got $rise"))
    return nothing
end

"""
    trapped_parts(st; fade = TCASE.fade, rise = TCASE.rise) -> (; born, plasma)

The two parts of [`trapped_energy`](@ref): the born electrons' (the species of
[`ionization_electrons`](@ref)) and the plasma's (every species of the state; in this case the
electrons, the ions being a fixed background).
"""
function trapped_parts(st; fade = TCASE.fade, rise = TCASE.rise)
    check_fade(st, fade)
    check_rise(rise)
    h = cellsize(st.grid)
    a = h[1] * h[2]
    born = a * species_trapped(ionization_electrons(st), h[1], fade[1], fade[2], rise[1], rise[2])
    plasma = a * foldl(+, map(s -> species_trapped(s, h[1], fade[1], fade[2], rise[1], rise[2]), st.species))
    return (; born, plasma)
end

"""
    trapped_energy(st; fade = TCASE.fade, rise = TCASE.rise) -> J

The objective: the kinetic energy of every trapped electron ahead of the window's back,

    J = Δx·Δy · Σ_i w_i·(γ_i − 1)·S(x_i)·R(u_x,i),

over the alive electrons of the plasma's species and of the ionization's, in
n_c·mₑc²·(c/ω₀)² (per unit length along z). `S` is the position weight
([`smooth_rise`](@ref) of the window x from `fade[1]` to `fade[2]`, zero on the back band), and
`R` the trapping weight, the same C² step in the forward momentum u_x from `rise[1]` to
`rise[2]`: an electron counts once it moves faster than the wake. Unlike WP-064's
[`born_energy`](@ref), which electrons count is decided by their momentum, not their label.
"""
function trapped_energy(st; kw...)
    p = trapped_parts(st; kw...)
    return p.born + p.plasma
end

"The objective at the five parameters (keywords of [`trapped_problem`](@ref), `fade` and `rise`)."
trapped_objective(duration, waist, focus, delay, density; fade = TCASE.fade, rise = TCASE.rise, kw...) =
    trapped_energy(trapped_state(duration, waist, focus, delay, density; kw...); fade, rise)

# The alive particles of a species with their window x, u_x, kinetic energy and weight, as plain numbers.
function species_arrays(s, h1)
    f(a) = Float64.(value_of.(Array(a)))
    alive = Array(s.status) .== ALIVE
    x = (Array(s.cell[1])[alive] .- 1 .+ f(s.offset[1])[alive]) .* h1
    u = map(a -> f(a)[alive], s.u)
    return (; x, ux = u[1], energy = sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2) .- 1, w = f(s.w)[alive])
end

"""
    trapped_summary(st; fade = TCASE.fade, rise = TCASE.rise) -> NamedTuple

Plain numbers for a run's tables: `J` and its parts `J_born` and `J_plasma`; the counted weights
Σ w·S·R of each (`weight_born`, `weight_plasma`, times Δx·Δy) and their mean energies (mₑc²); the
weight of each with u_x above `rise[2]` over the whole window ahead of the back band
(`fast_born_window`, `fast_plasma_window`), which shows trapped charge the position weight leaves
out; the highest u_x of each; and the number of alive particles.
"""
function trapped_summary(st; fade = TCASE.fade, rise = TCASE.rise)
    parts = trapped_parts(st; fade, rise)
    h = Float64.(value_of.(cellsize(st.grid)))
    area = h[1] * h[2]
    band = back_band_end(st)
    counted(q) = area * sum(q.w .* smooth_rise.(q.x, fade[1], fade[2]) .* smooth_rise.(q.ux, rise[1], rise[2]); init = 0.0)
    window(q) = area * sum(q.w[(q.ux .> rise[2]) .& (q.x .>= band)]; init = 0.0)
    b = species_arrays(ionization_electrons(st), h[1])
    ps = [species_arrays(s, h[1]) for s in st.species]
    Jb, Jp = Float64(value_of(parts.born)), Float64(value_of(parts.plasma))
    wb, wp = counted(b), sum(counted, ps)
    return (; J = Jb + Jp, J_born = Jb, J_plasma = Jp, weight_born = wb, weight_plasma = wp,
            mean_energy_born = wb > 0 ? Jb / wb : 0.0, mean_energy_plasma = wp > 0 ? Jp / wp : 0.0,
            fast_born_window = window(b), fast_plasma_window = sum(window, ps),
            ux_max_born = maximum(b.ux; init = 0.0), ux_max_plasma = maximum(q -> maximum(q.ux; init = 0.0), ps),
            particles = length(b.w) + sum(q -> length(q.w), ps))
end
