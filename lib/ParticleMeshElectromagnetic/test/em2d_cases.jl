# WP-024: the 2D problems of em2d.jl and the bitwise digests that its tests and the subprocess
# worker (determinism_worker.jl, T-355 at 1 and 4 threads) share.

zero3(::Type{T}) where {T} = (zero(T), zero(T), zero(T))

"""
    test_particles_2d(T; cells, length, dt, nsteps, u0, E0, B0, ppc = (2, 2), qm = -1, kmode = (1, 0)) -> EMProblem{2}

Test particles at N = 2: one species of zero density (zero weight: no charge, no current) on the
quiet-start lattice, every particle with the momentum `u0` (given as its velocity u0/γ to the
loader), in the uniform background fields `E0`, `B0`.
"""
function test_particles_2d(::Type{T}; cells = (6, 6), length = (T(3), T(3)), dt, nsteps, u0 = zero3(T),
                           E0 = zero3(T), B0 = zero3(T), ppc = (2, 2), qm = -1, kmode = (1, 0)) where {T}
    γ = sqrt(1 + u0[1]^2 + u0[2]^2 + u0[3]^2)
    sp = (SpeciesDescription{T}(:test, T(qm), one(T), zero(T), zero(T), (1, 0), ppc, map(u -> T(u / γ), u0), zero3(T), 1),)
    return EMProblem("test-particles-2d", length, cells, T(dt), nsteps, sp, 1, kmode; E0, B0)
end

"The 1D counterpart of [`test_particles_2d`](@ref): `cells` cells of a box of `length`."
function test_particles_1d(::Type{T}; cells = 6, length = T(3), dt, nsteps, u0 = zero3(T), E0 = zero3(T), B0 = zero3(T),
                           ppc = 2, qm = -1, kmode = 2 * T(π) / length) where {T}
    γ = sqrt(1 + u0[1]^2 + u0[2]^2 + u0[3]^2)
    sp = (SpeciesDescription{T}(:test, T(qm), one(T), zero(T), zero(T), 1, ppc, map(u -> T(u / γ), u0), zero3(T), 1),)
    return EMProblem("test-particles-1d", length, cells, T(dt), nsteps, sp, 1, T(kmode); E0, B0)
end

"""
    plasma_2d(T; density = 1, amplitude = 1/20, mode = (2, 1), cells = (10, 8), ppc = (3, 3), vth = 1/20, nsteps = 8, seed = 1) -> EMProblem{2}

A thermal electron plasma with an oblique cosine perturbation of `mode` on a square box of
side 2π (ω₀/c), Δt at 0.9 of the Courant limit, the diagnostic mode `mode`; `density` and
`amplitude` may be duals (the parameters of T-356).
"""
function plasma_2d(::Type{T}; density = one(T), amplitude = one(T) / 20, mode = (2, 1), cells = (10, 8), ppc = (3, 3),
                   vth = one(T) / 20, nsteps = 8, seed = 1, E0 = zero3(T), B0 = zero3(T)) where {T}
    L = (2 * T(π), 2 * T(π))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), T(density), T(amplitude), mode, ppc, zero3(T), (T(vth), T(vth), T(vth)), 1),)
    return EMProblem("plasma-2d", L, cells, T(dt), nsteps, sp, seed, mode; E0, B0)
end

"The 2D state of `prob` with the `current` (`NoCurrent()` by default, WP-024's prescribed fields), initialized."
function state_2d(prob; mode = Reference(), workgroupsize = nothing, pusher = Boris(), shape = Quadratic, be = CPU(),
                  current = NoCurrent())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, pusher, shape, current); backend = be)
    initialize!(st, prob)
    return st
end

"A hash of the bits of every particle array (both axes, extras included), field, `K` and history of a 2D state."
function state_digest_2d(st::EMState{2})
    h = UInt64(0)
    for s in st.species, a in (s.cell..., s.offset..., s.u..., s.w, s.status, s.count, values(s.extras)...)
        h = hash(bits(a), h)
    end
    for a in (values(st.fields)..., st.K, values(st.recorder.hist)...)
        h = hash(bits(a), h)
    end
    return h
end

"""
    em2d_digests(; workgroupsize = nothing) -> Vector{(T, digest)}

T-355: a Reference run of [`plasma_2d`](@ref) (initialize! and every step) per element type,
one digest each.
"""
function em2d_digests(; workgroupsize = nothing)
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob = plasma_2d(T)
        st = state_2d(prob; workgroupsize)
        run!(st, params(prob), prob.nsteps)
        push!(out, (T, state_digest_2d(st)))
    end
    return out
end

"The momenta of the first alive particle of the first species, on the host."
first_u(st) = (s = first(st.species); (Array(s.u[1])[1], Array(s.u[2])[1], Array(s.u[3])[1]))

# --- WP-023: the 2D step with the charge-conserving current -------------------------------

"""
    plasma_step_2d(T; amplitude = 1/10, cells = (10, 8), ppc = (3, 3), nsteps = 40, seed = 7, vth = 1/10, momentum = :default) -> EMProblem{2}

Two mobile species that cross cells and the periodic seams within a few steps: thermal electrons
(v_th = `vth`, 0.1 by default, in every component) with the drift (0.3, −0.2, 0.1) and an oblique perturbation of
mode (2, 1), and positive particles of mass 4 (v_th = 0.05) drifting along (−0.2, 0.25, −0.15)
with the mode (1, −1), on a box of 2π × 3π/2 (Δx ≠ Δy), Δt at 0.9 of the Courant limit. Both
species take `momentum` (`SpeciesDescription`'s keyword: by default a Maxwell–Jüttner load);
with `momentum = :gaussian_velocity` the loader draws velocities, v_th = 0.1 keeps every |v|
well below c, and at `vth = 3/10` some drawn electron speed passes c and `initialize!` refuses
the load (T-480). The electrons' `amplitude` may be a dual (T-366).
"""
function plasma_step_2d(::Type{T}; amplitude = one(T) / 10, cells = (10, 8), ppc = (3, 3), nsteps = 40, seed = 7,
                        vth = one(T) / 10, momentum = :default) where {T}
    L = (2 * T(π), 3 * T(π) / 2)
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (2, 1), ppc, (3tenth, -2tenth, tenth),
                              (T(vth), T(vth), T(vth)), 1; momentum)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1), ppc, (-2tenth, 5tenth / 2, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2; momentum)
    return EMProblem("plasma-step-2d", L, cells, T(dt), nsteps, (e, p), seed, (2, 1))
end

"""
    em2d_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(T, digest)}

T-363: a Reference run of [`plasma_step_2d`](@ref) with the `Esirkepov()` current, 20 steps,
per element type, one digest each.
"""
function em2d_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob = plasma_step_2d(T; nsteps = 20)
        st = state_2d(prob; workgroupsize, be, current = Esirkepov())
        run!(st, params(prob), prob.nsteps)
        push!(out, (T, state_digest_2d(st)))
    end
    return out
end
