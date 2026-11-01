# T-219: a ForwardDiff derivative through the whole module step against the finite-difference
# floor of A7, measured by the engine's `fd_scan` (Float64, Reference on both sides, common
# random numbers). The parameter is the Landau density perturbation amplitude, which enters
# through the weights only (A3); the functional is the Ex field energy at the last step.

const GRADIENT_KW = (; ppc = 64, t_end = 3)       # 322 steps

"Ex field energy of the Landau case at its last step, as a function of the perturbation amplitude."
function landau_field_energy(α, mode, seed; be = CPU())
    T = typeof(α)
    st, p, prob, _ = case_state("landau", T; mode, seed, amplitude = α, be, GRADIENT_KW...)
    run!(st, p, prob.nsteps)
    return Array(st.hist.WEx)[end]
end

function body_T219(be)
    # the run is long enough for particles to cross cells and the periodic seam
    st, p, prob, _ = case_state("landau", Float64; be, GRADIENT_KW...)
    n = Int32(ncells(st.grid))
    crossed = seam = false
    for k in 1:prob.nsteps
        step!(st, p, k)
        s = st.species[1]
        old, new = s.extras.old_cell, s.cell[1]
        crossed |= any(old .!= new)
        seam |= any(((old .== n) .& (new .== 1)) .| ((old .== 1) .& (new .== n)))
    end
    @test crossed && seam

    seed = 20261031
    α0 = 0.01
    energy(α, mode, seed) = landau_field_energy(α, mode, seed; be)
    ad = ForwardDiff.derivative(α -> energy(α, Reference(), seed), α0)
    hs = α0 .* 2.0 .^ (-2:-2:-20)
    r = fd_scan(energy, α0, hs; seed, ad)
    ratio = r.rel[r.istar] / r.floor_rel
    @info "T-219 gradient" backend = backend_label(be) ad fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
    get!(MEASURED, "T-219 rel/floor", Dict{String,Float64}())[be isa CPU ? "Float64" : "Float64 on $(backend_label(be))"] = ratio
    @test isfinite(ad) && ad != 0
    @test r.rel[r.istar] <= emC("T-219") * r.floor_rel
    return ad
end

@testset "T-219 ForwardDiff through the module step against the fd_scan floor [A3, A7, WP-014]" begin
    body_T219(CPU())
end
