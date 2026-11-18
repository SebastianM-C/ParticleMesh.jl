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
- `fade = (lo, hi)` is the position weight in window x (c/ω₀), WP-064's: 1 over the first bucket
  behind the pulse, 0 behind x = 12, so J counts every trapped electron of the first bucket,
  born or the plasma's own. The plasma injects itself in this case only into the buckets behind
  the first, wave-breaking charge that grows with the length of wake the window keeps (4.7 times
  the born energy at a₀ = 2.4 at λ₀/16 with the window 4 λ₀ longer, 23 times with it 6 λ₀ longer):
  it is the dark current, reported beside J ([`trapped_summary`](@ref)) and not counted in it.
- `x0` is the base point (duration, waist, focus, delay, density); `names` and `scale` give each
  parameter's name and the scale of z = x ./ scale in problem.jl.
- `free` are the indices of the parameters an optimizer moves: all but the delay, which stays at
  `x0[4]`. ForwardDiff's derivative in the delay does not give J's change over a finite step:
  at λ₀/32 it is 11 to 16 in d ln J/d ln delay where central differences and runs 2⁻¹⁰ apart
  give −0.3, and it changes sign from one grid to the next (WP-065's report).
"""
const TCASE = (a0_energy = 1.9, budgets = (1.9, 2.175, 2.4), dopant_fraction = 0.01, rise = (3.0, 5.0), fade = (12.0, 24.0),
               x0 = (6.0, 10.0, 80.0, 54.0, 0.072), names = ("duration", "waist", "focus", "delay", "density"),
               scale = (6.0, 10.0, 80.0, 54.0, 0.072), free = (1, 2, 3, 5))

"""
    pulse_energy(a0, duration, waist) -> W

The energy per unit length in z of the case's pulse ([`laser_energy`](@ref) of a `GaussianLaser`
polarized in the plane, the case's pulse): π·a₀²·w·τ/4·(1 + 1/(2w²)).
"""
pulse_energy(a0, duration, waist) = π * a0^2 * waist * duration / 4 * (1 + 1 / (2 * waist^2))

"""
The optimizer's box over the free parameters `TCASE.free` (duration, waist, focus, density), in
physical units, with their scales (WP-068): the duration from 4 to 9, the waist from 6 to 14, the
focus from 64 to 110 and the density from 0.05 to 0.10 n_c, so z = x ./ scale stays in [0.6, 1.5].
At the budget of a₀ = 1.9 at the base pulse, a₀ runs across it from about 1.3 (τ = 9, w = 14) to
about 3.0 (τ = 4, w = 6), past the plasma's own injection into the first bucket from a₀ ≈ 2.45 to
2.55 (WP-065).
"""
const TRAPPED_BOX = (lo = [4.0, 6.0, 64.0, 0.05], hi = [9.0, 14.0, 110.0, 0.10], scale = [TCASE.scale[i] for i in TCASE.free])

"""
    trapped_point(free) -> (duration, waist, focus, delay, density)

The case's five parameters from the four free ones (in `TCASE.free`'s order), the delay at the
base point's `TCASE.x0[4]` (every parameter outside `TCASE.free` at its base value), in the free parameters' type.
"""
function trapped_point(free)
    length(free) == length(TCASE.free) || throw(ArgumentError("trapped_point takes the $(length(TCASE.free)) free parameters, got $(length(free))"))
    T = promote_type(map(typeof, Tuple(free))...)
    x = Vector{T}(undef, length(TCASE.x0))
    for i in eachindex(x)
        k = findfirst(==(i), TCASE.free)
        x[i] = k === nothing ? T(TCASE.x0[i]) : free[k]
    end
    return Tuple(x)
end

"The energy budget of the pulse with `a0` at the base duration and waist."
budget(a0 = TCASE.a0_energy) = pulse_energy(a0, TCASE.x0[1], TCASE.x0[2])

"""
    a0_for_energy(duration, waist; energy) -> a₀

The a₀ at which a pulse of `duration` and `waist` carries `energy` per unit length in z, the
inverse of [`pulse_energy`](@ref): the fixed-energy coupling, through which ∂W/∂τ = ∂W/∂w = 0.
"""
a0_for_energy(duration, waist; energy = budget()) = sqrt(energy / (π * waist * duration / 4 * (1 + 1 / (2 * waist^2))))

"""
    trapped_capacity(ppl, ny) -> slots

The born electrons' slots for a run at `ppl` and `ny`: the dopant's macro-particles grow as
ppl²·ny (cells in x and y times the window's travel in cells), and the case at a₀ = 2.4 fills
24,880 slots at λ₀/16 and 204,790 at λ₀/32 with ny = 2 at t = 200, once the pulse has left
the dopant's section (x ≤ 132), so 40,000·ppl²·ny/256 keeps
a margin of about 1.5, and never fewer than `BCASE.capacity` (the runs of WP-064 keep their bits).
"""
trapped_capacity(ppl, ny) = max(BCASE.capacity, cld(40_000 * ppl^2 * ny, 256))

"""
    trapped_problem(duration, waist, focus, delay, density; energy, ppl, ny, capacity, kw...) -> EMProblem{2}

The case at the five parameters, with a₀ from [`a0_for_energy`](@ref) and the dopant at
`TCASE.dopant_fraction` of `density`, and [`trapped_capacity`](@ref) slots by default; the other
keywords as [`born_problem`](@ref).
"""
function trapped_problem(duration, waist, focus, delay, density; energy = budget(), ppl::Integer = BCASE.ppl, ny::Integer = 1,
                         capacity::Integer = trapped_capacity(ppl, ny), kw...)
    a0 = a0_for_energy(duration, waist; energy)
    # the fraction in density's own precision, so that a dual of Float32 stays one
    fraction = convert(typeof(float(value_of(density))), TCASE.dopant_fraction)
    return born_problem(a0, fraction * density; duration, waist, focus, delay, density, ppl, ny, capacity, kw...)
end

"""
The case's state after its run at the five parameters: the run of [`trapped_problem`](@ref)'s
problem (and its keywords) on `backend`, the particles reading the fields through `gather`
([`born_numerics`](@ref): `nothing`, or WP-068's `CompensatedGather()`); a birth without a free
slot is an error.
"""
trapped_state(duration, waist, focus, delay, density; ppl::Integer = BCASE.ppl, ny::Integer = 1, backend = CPU(), gather = nothing, kw...) =
    run_to_end(start_of(trapped_problem(duration, waist, focus, delay, density; ppl, ny, kw...); ppl, ny, backend, gather))

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
out; the highest u_x of each; the dark current, the plasma's electrons with u_x above `rise[2]`
between the back band's end and `fade[1]` (`dark_weight`, and `dark_energy` in J's units); and the
number of alive particles. The dark region starts at the back band's end, which a longer window
(`extend`, as under a power cap) pushes back, so `dark_weight` and `dark_energy` of runs with
different `extend` count different regions and are not comparable.
"""
function trapped_summary(st; fade = TCASE.fade, rise = TCASE.rise)
    parts = trapped_parts(st; fade, rise)
    h = Float64.(value_of.(cellsize(st.grid)))
    area = h[1] * h[2]
    band = back_band_end(st)
    counted(q) = area * sum(q.w .* smooth_rise.(q.x, fade[1], fade[2]) .* smooth_rise.(q.ux, rise[1], rise[2]); init = 0.0)
    window(q) = area * sum(q.w[(q.ux .> rise[2]) .& (q.x .>= band)]; init = 0.0)
    dark(q, v) = area * sum((q.w .* v)[(q.ux .> rise[2]) .& (q.x .>= band) .& (q.x .< fade[1])]; init = 0.0)
    b = species_arrays(ionization_electrons(st), h[1])
    ps = [species_arrays(s, h[1]) for s in st.species]
    Jb, Jp = Float64(value_of(parts.born)), Float64(value_of(parts.plasma))
    wb, wp = counted(b), sum(counted, ps)
    return (; J = Jb + Jp, J_born = Jb, J_plasma = Jp, weight_born = wb, weight_plasma = wp,
            mean_energy_born = wb > 0 ? Jb / wb : 0.0, mean_energy_plasma = wp > 0 ? Jp / wp : 0.0,
            fast_born_window = window(b), fast_plasma_window = sum(window, ps),
            dark_weight = sum(q -> dark(q, 1.0), ps), dark_energy = sum(q -> dark(q, q.energy), ps),
            ux_max_born = maximum(b.ux; init = 0.0), ux_max_plasma = maximum(q -> maximum(q.ux; init = 0.0), ps),
            particles = length(b.w) + sum(q -> length(q.w), ps))
end
