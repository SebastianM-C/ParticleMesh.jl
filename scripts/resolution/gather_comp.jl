# A prototype of a compensated gather, for the study of the particle shape's field loss (included by
# gridppc.jl; no engine code changes). A particle of shape order p reads a field mode of wavenumber
# k at, on average over its offset in the cell, sinc(kΔx/2)^(p+1) of its amplitude: at the laser's
# k = ω₀/c and Δx = λ₀/16 that is 1.9 % less field for the quadratic shape, and J ∝ a₀^24.5. Here
# the fields the particles read (the push and the ionization rates) are a sharpened copy of the
# grid's, E + β(2E_i − E_{i−1} − E_{i+1}) along x, whose transfer 1 + 4β·sin²(kΔx/2) undoes that
# loss at the laser's wavenumber. The mesh solve, the deposition and every read-out keep the grid's
# own fields. Along y nothing is done: the pulse's transverse wavenumbers (≈ 1/waist) lose < 10⁻³.
#
# `compensated_step!` is `step!` of lib/ParticleMeshElectromagnetic/src/step.jl with the gather
# reading the copy; keep the two in step if that function changes.

const PME = ParticleMeshElectromagnetic
const launcher = ParticleMesh.ParticleMeshCore.launcher

"""
    comp_beta(order, ppl) -> β

The β of the sharpening that makes the gather's mean transfer one at the laser's wavenumber for
shape `order` and Δx = λ₀/`ppl`: (1/sinc(π/ppl)^(order+1) − 1)/(4 sin²(π/ppl)). Its small-cell limit
is (order + 1)/24 (1/8 for the quadratic shape).
"""
function comp_beta(order, ppl)
    x = π / ppl
    S = (sin(x) / x)^(order + 1)
    return (1 / S - 1) / (4 * sin(x)^2)
end

@kernel function sharpen_x_kernel!(out, @Const(a), β)
    i, j = @index(Global, NTuple)
    n1 = size(a, 1)
    @inbounds if 1 < i < n1
        out[i, j] = a[i, j] + β * (2 * a[i, j] - a[i - 1, j] - a[i + 1, j])
    else
        out[i, j] = a[i, j]
    end
end

"Write the sharpened copy of every field of `src` into `dst`."
function sharpen!(dst::NamedTuple, src::NamedTuple, β; workgroupsize = nothing)
    for name in keys(src)
        a = src[name]
        launcher(sharpen_x_kernel!, get_backend(a), workgroupsize)(dst[name], a, convert(eltype(a), β); ndrange = size(a))
    end
    return dst
end

"A state that shares every array of `st` but reads its particles' fields from the copy `gf`."
gather_state(st, gf) = EMState(st.numerics, st.grid, st.dt, st.species, st.ws, gf, st.ρ, st.K, st.recorder, st.setup, st.modifiers, st.aux, st.fresh)

"The copy's arrays, one per field of `st`."
gather_fields(st) = map(similar, st.fields)

"`step!` with the particles' gather (the push and the ionization) from the sharpened copy of the fields."
function compensated_step!(st, stg, β, p, k; observer = st.recorder)
    PME.check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PME.window_frame(st)
    observe!(observer, st, k)
    st.fresh[] = false
    PME.inject_witness!(PME.witness_frame(st), p.witness, grid, k, wgs)
    sharpen!(stg.fields, fields, β; workgroupsize = wgs)
    PME.ionize!(PME.ionization_frame(st), stg, p, k)
    for s in (PME.deposited_species(st)..., PME.witness_species(st)...)
        launcher(PME.push_kernel!, get_backend(s), wgs)(s, stg.fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2, dt_dx, n;
                                                        ndrange = capacity(s))
    end
    PME.retire_x!(window, st)
    PME.current!(numerics.current, st)
    PME.retire!(PME.open_absorber(st), st)
    solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    PME.advance_window!(window, st, p, k)
    PME.reads_kinetic(observer, k + 1) && PME.kinetic!(st)
    return st
end

"`run!` with [`compensated_step!`](@ref); β = 0 is `run!` itself."
function run_comp!(st, stg, β, p, nsteps; first = 1)
    β == 0 && return run!(st, p, nsteps; first)
    for k in first:(first + nsteps - 1)
        compensated_step!(st, stg, β, p, k)
    end
    return st
end
