# The trapped-beam case of WP-065 in 3D (WP-084): WP-064's ionization-injection case at a fixed
# laser energy, with the first bucket's trapped electrons as the objective J, on a 3D box. The
# longitudinal geometry is the 2D case's (`born2d_case.jl`: the window of 14 λ₀, the vacuum, the
# ramp, the dopant's section from 108 to 132, the pulse's place, focus and timing, t = 300), and the
# transverse box is its y box along both y and z (128 cells of π/4 with 12 PML cells at each open
# face); the beam is round and polarized along y, and the dopant fills a square tube of the 2D
# case's 36 rows about the axis. The energy budget is the round pulse's (∝ a₀²·w²·τ) at a₀ = 1.9 at
# the base pulse, so the base point has the 2D case's a₀ and peak intensity. J is
#
#     J = Δx·Δy·Δz · Σ_i w_i·(γ_i − 1)·S(x_i)·R(u_x,i)
#
# in n_c·mₑc²·(c/ω₀)³, the 2D sum of `trapped2d_case.jl` with the cell's volume for its area.
# The construction is the one of the grid-and-particle study's 3D driver (`gridppc3d.jl`), with its
# defaults for the GPU: Float32, 2 × 1 × 1 plasma electrons per cell, the compensated gather, Yee.
isdefined(@__MODULE__, :TCASE) || include(joinpath(@__DIR__, "trapped2d_case.jl"))
using ParticleMeshElectromagnetic: CompensatedGather, Yee, ionization_electrons
using ParticleMeshElectromagnetic.ParticleMeshCore: cellsize

"""
    pulse_energy3(a0, duration, waist) -> W

The energy of the 3D case's round pulse ([`laser_energy`](@ref) of a `GaussianLaser` with a
centre of two coordinates): π·√(π/2)·a₀²·w²·τ/4, in the arguments' own type (Float32 and its duals stay so).
"""
function pulse_energy3(a0, duration, waist)
    p = oftype(float(a0 * waist * duration), π)
    return p * sqrt(p / 2) * a0^2 * waist^2 * duration / 4
end

"The 3D energy budget: the pulse's energy at a₀ = `a0E` and the case's base duration and waist."
budget3(a0E = TCASE.a0_energy) = pulse_energy3(a0E, TCASE.x0[1], TCASE.x0[2])

"The a₀ of a round pulse of `duration` and `waist` that carries `energy`, the inverse of [`pulse_energy3`](@ref)."
a0_for_energy3(duration, waist; energy = budget3()) = sqrt(energy / pulse_energy3(one(duration), duration, waist))

"""
    trapped3d_capacity(ppl; nt = 1, ion_ppc = (1, 1, 1), every = 1, factor = 1) -> slots

The born electrons' slots: one per atom slot of the window's dopant lattice (a birth step offers
one slot per atom) and room for the electrons alive, the 2D case's rule (six times the atoms at
λ₀/16, growing with the cells along x) times `factor`. The grid study's λ₀/32 runs held 10 to 15
times the most electrons alive at `factor = 1` and gave the same J at 0.2.
"""
function trapped3d_capacity(ppl::Integer; nt::Integer = 1, ion_ppc = (1, 1, 1), every::Integer = 1, factor = 1)
    atoms = BCASE.length_wavelengths * ppl * (2 * BCASE.half_rows * nt)^2 * prod(ion_ppc)
    return atoms + round(Int, Float64(factor) * max(400_000, 6 * atoms * (ppl / 16) / every))
end

"""
    trapped3d_problem(duration, waist, focus, delay, density; energy, ppl, nt, rows, ppc, ion_ppc, capacity, cfl, every, t_end, float) -> EMProblem{3}

The 3D case at the five parameters: Δx = λ₀/`ppl`, Δy = Δz = π/(4·`nt`) on `rows`·`nt` cells along
each transverse axis, `ppc` plasma electrons and `ion_ppc` dopant atoms per cell, the time step at
`cfl` hundredths of the Courant limit, a₀ from [`a0_for_energy3`](@ref) at `energy`, the dopant at
`TCASE.dopant_fraction` of `density` with the ADK rates of N⁵⁺ and N⁶⁺, and `capacity` born-electron
slots. The element type is `float` for plain arguments, or the duals' when an argument is one.
"""
function trapped3d_problem(duration, waist, focus, delay, density; float::Type{<:AbstractFloat} = Float32, energy = budget3(),
                           ppl::Integer = 16, nt::Integer = 1, rows::Integer = BCASE.rows, ppc = (2, 1, 1), ion_ppc = (1, 1, 1), cfl::Integer = 95,
                           every::Integer = BCASE.every, capacity::Integer = trapped3d_capacity(ppl; nt, ion_ppc, every), t_end = BCASE.t_end)
    ppl % 8 == 0 || throw(ArgumentError("ppl must be a multiple of 8, got $ppl"))
    base(x) = x isa AbstractFloat ? float : typeof(x)
    T = promote_type(base(duration), base(waist), base(focus), base(delay), base(density), float)
    R(x) = x isa AbstractFloat ? T(x) : convert(T, x)
    a0 = a0_for_energy3(R(duration), R(waist); energy = convert(T, energy))
    dopant = convert(T, TCASE.dopant_fraction) * R(density)
    z3 = (zero(T), zero(T), zero(T))
    h = 2 * T(π) / ppl
    ht = T(π) / (4 * nt)
    cells = (BCASE.length_wavelengths * ppl, rows * nt, rows * nt)
    dt = T(cfl) / 100 / sqrt(1 / h^2 + 2 / ht^2)
    nsteps = ceil(Int, t_end / Float64(value_of(dt)))
    yc = cells[2] * Float64(value_of(ht)) / 2
    hw = BCASE.half_rows * π / 4
    pulse = GaussianLaser(; a0, waist = R(waist), duration = R(duration), delay = R(delay), focus = R(focus),
                          position = max(12, 192 ÷ ppl) * (2 * T(π) / 32), center = (cells[2] * ht / 2, cells[3] * ht / 2))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), R(density), zero(T), (1, 0, 0), Tuple(ppc), z3, z3, 1)
    levels = (adk_level(T, :N, 6; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff),
              adk_level(T, :N, 7; wavelength = BCASE.wavelength, cutoff = BCASE.cutoff))
    ion = FieldIonization(; density = dopant, levels, capacity, every, ppc = Tuple(ion_ppc), rows = ((yc - hw, yc + hw), (yc - hw, yc + hw)),
                          section = DopantSection(; start = T(BCASE.section.start), stop = T(BCASE.section.stop), ramp = T(BCASE.section.ramp)))
    return EMProblem("trapped3d", (cells[1] * h, cells[2] * ht, cells[3] * ht), cells, dt, nsteps, (e,), 1, (1, 0, 0);
                     lasers = (pulse,), boundaries = (:window, :open, :open), window = MovingWindow(; start = T(BCASE.start)),
                     profile = DensityRamp(; vacuum = T(BCASE.vacuum), ramp = T(BCASE.ramp)), ionization = ion)
end

"""
The 3D case's numerics: Reference mode, quadratic shapes, PMLs of 12·`nt` cells on the four open
faces and of a quarter wavelength (at least 4 cells) at the window's back, the `solver` and the
particles' `gather` (the compensated gather of WP-067 and WP-076 by default).
"""
trapped3d_numerics(ppl::Integer, nt::Integer = 1; solver = Yee(), gather = CompensatedGather(; k = 1)) =
    EMNumerics(; mode = Reference(), solver, gather, absorber = PML(; cells = BCASE.pml_y * nt),
               back_absorber = PML(; cells = max(4, round(Int, BCASE.pml_x_wavelengths * ppl))))

"""
    trapped3d_start(free; backend, ppl, nt, solver, gather, kw...) -> (st, p, prob)

The 3D case's state at its start at the four free parameters `free` (duration, waist, focus,
density; the delay at its base value, `trapped_point`), their element type or `float`, on `backend`;
the other keywords as [`trapped3d_problem`](@ref).
"""
function trapped3d_start(free; backend = CPU(), ppl::Integer = 16, nt::Integer = 1, solver = Yee(), gather = CompensatedGather(; k = 1), kw...)
    prob = trapped3d_problem(trapped_point(free)...; ppl, nt, kw...)
    st = EMState(prob, trapped3d_numerics(ppl, nt; solver, gather); record = (), backend)
    p = params(prob)
    initialize!(st, p)
    return st, p, prob
end

"The parts of J (the born electrons', the plasma's) of a 3D state: `trapped_parts`'s sums with the cell's volume."
function trapped_parts3(st; fade = TCASE.fade, rise = TCASE.rise)
    check_fade(st, fade)
    check_rise(rise)
    h = cellsize(st.grid)
    v = h[1] * h[2] * h[3]
    born = v * species_trapped(ionization_electrons(st), h[1], fade[1], fade[2], rise[1], rise[2])
    plasma = v * foldl(+, map(s -> species_trapped(s, h[1], fade[1], fade[2], rise[1], rise[2]), st.species))
    return (; born, plasma)
end

"J of a 3D state: the sum of [`trapped_parts3`](@ref)."
trapped_energy3(st; kw...) = (q = trapped_parts3(st; kw...); q.born + q.plasma)
