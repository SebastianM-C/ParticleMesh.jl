# WP-043: a gradient through a small laser-wakefield run, T-580 to T-582; WP-044, T-590 and T-591. After window.jl and
# window_cases.jl (`ZERO3`).
#
# The case: a Gaussian pulse launched by the antenna (a₀ = 2 at the focus x = 70, waist 5,
# duration 5) into a cold electron plasma of 0.1 n_c (ω_p ≈ 0.32, λ_p ≈ 20 c/ω₀) behind 64 c/ω₀
# of vacuum and an 8 c/ω₀ sin² ramp, immobile ions, inside a window moving at c from t = 72;
# 16 cells per laser wavelength, Δy = 4Δx, 160 × 16 cells (63 × 25 c/ω₀; the waist is 3.2 cells),
# Δt at 0.95 of the Courant limit, 350 steps (t ≈ 127: the pulse peak ends about 19 c/ω₀, one
# plasma wavelength, into the plateau, with one wake period behind it). It is a small case for
# the derivatives, not an accurate laser: the pulse is about one cycle long (the antenna is exact
# at the carrier only) and its waist is coarsely resolved, and its peak |E_y| at the end is
# 0.82 in vacuum, against about 1.4 that the paraxial 2D beam would have 23 c/ω₀ past the focus
# (measured, cause not checked; not the transverse resolution, since Δy = Δx gives 0.81).
# Every parameter of the gradient enters as A3 says: a₀, the duration and the focus through the
# antenna, the density and the ramp length through the particle weights (in the initial load
# and in the window's refill; the initial window lies in the vacuum, so here all the plasma
# comes from the refill, and the initial load's weight derivatives are T-219's and T-366's). The functional is a wake-amplitude proxy: the soft maximum
# (log-sum-exp at temperature 1/β) of E_x on the axis's node row at the last step, past the window's back band. At this
# operating point its arg-max is the wake's crest about one λ_p behind the pulse; at other points
# it can sit on the pulse instead, so it is not an objective for optimization, only a smooth
# functional of the whole run.
#
# The horizon (#370, WP-044): from about step 376 the window's back edge reaches the plasma, and
# its retirements (charge out with no current, at a step that depends on the parameters) make
# the back columns' fields jump between neighbouring parameter values. Read there, the soft
# maximum is not smooth at the finite-difference scale at 600 steps (a₀'s partial at 3× the
# floor); read past `window_back_band` (T-590), AD and FD agree again (T-591). The functional
# below reads past it.

using Statistics: median, std

"The parameters of the gradient case, in A1 units: a₀, duration, focus, density (n_c), ramp."
const LWFA_GRADIENT_X0 = (2.0, 5.0, 70.0, 0.1, 8.0)
const LWFA_GRADIENT_NAMES = ("a0", "duration", "focus", "density", "ramp")
"The soft maximum's β, in units of eE/(mₑcω₀): 200 against a wake of about 0.09."
const LWFA_GRADIENT_BETA = 200

"""
    lwfa_gradient_problem(x; nsteps = 350) -> EMProblem{2}

The WP-043 case of element type `eltype(x)` with `x = (a0, duration, focus, density, ramp)`.
The pulse's delay is fixed at 36, so the pulse starts below roundoff, and the window starts at
72, after the antenna has finished (delay + 6·duration), for durations up to 6.
"""
function lwfa_gradient_problem(x; nsteps = 350)
    T = promote_type(map(typeof, x)...)
    a0, duration, focus, density, ramp = map(T, x)
    cells = (160, 16)
    h = 2 * T(π) / 16
    hy = 4h
    dt = T(95) / 100 / sqrt(1 / h^2 + 1 / hy^2)
    pulse = GaussianLaser(; a0, waist = T(5), duration, delay = T(36), focus, position = 6h, center = cells[2] * hy / 2)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), density, zero(T), (1, 0), (2, 2), ZERO3(T), ZERO3(T), 1)
    return EMProblem("lwfa-gradient", (cells[1] * h, cells[2] * hy), cells, dt, nsteps, (e,), 1, (1, 0);
                     lasers = (pulse,), boundaries = (:window, :periodic), window = MovingWindow(; start = T(72)),
                     profile = DensityRamp(; vacuum = T(64), ramp))
end

"The state of the WP-043 case after its run, in `mode`."
function lwfa_gradient_run(x, mode; be = CPU(), kw...)
    prob = lwfa_gradient_problem(x; kw...)
    st = EMState(prob, EMNumerics(; mode); backend = be)
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps)
    return st, prob
end

"E_x on the axis's node row (y = L_y/2), every interior column."
function axis_Ex_all(st)
    g, n = guards(st.grid), ncells(st.grid)
    return Array(st.fields.Ex)[(g + 1):(g + n[1]), g + 1 + n[2] ÷ 2]
end

"E_x on the axis's node row, the interior columns past the window's back band (`window_back_band`)."
axis_Ex(st) = axis_Ex_all(st)[(window_back_band(st) + 1):end]

"The soft maximum (1/β)·log Σ exp(β·v), shifted by the primal maximum (constant in the parameters) against overflow."
function soft_maximum(v, β)
    m = maximum(value_of, v)
    return m + log(sum(e -> exp(β * (e - m)), v)) / β
end

"The WP-043 functional: the wake amplitude of the case at `x`."
lwfa_wake_amplitude(x, mode = Reference(); kw...) =
    soft_maximum(axis_Ex(first(lwfa_gradient_run(x, mode; kw...))), LWFA_GRADIENT_BETA)

# --- T-580 ----------------------------------------------------------------------------------

# The case is the one the gradient should be taken through: the window has shifted and refilled,
# the pulse has entered the plateau and driven a wake of a sizeable fraction of the cold
# wave-breaking field E₀ = ω_p (A1 units) behind it, the soft maximum is within 1/β·log(n) of the
# hard one, and no refill ran out of slots.
@testset "T-580 the gradient case drives a wake in the moving window [A0, WP-043]" begin
    st, prob = lwfa_gradient_run(LWFA_GRADIENT_X0, Reference())
    ex = axis_Ex(st)
    E0 = sqrt(LWFA_GRADIENT_X0[4])
    shifts = ParticleMeshElectromagnetic.window_shifts(st, prob.nsteps)
    @info "T-580" shifts maxEx = maximum(ex) E0 ratio = maximum(ex) / E0
    @test shifts > 100
    @test all(iszero, ParticleMeshElectromagnetic.window_lost(st))
    @test 0.1 < maximum(ex) / E0 < 1
    @test minimum(ex) < -maximum(ex) / 2              # an oscillating wake, not a ramp charge
    S = soft_maximum(ex, LWFA_GRADIENT_BETA)
    @test maximum(ex) <= S <= maximum(ex) + log(length(ex)) / LWFA_GRADIENT_BETA
    β = big(LWFA_GRADIENT_BETA)                     # the definition, unshifted, in BigFloat
    @test S ≈ Float64(log(sum(e -> exp(β * big(e)), ex)) / β) rtol = 1e-12
end

# --- T-581 ----------------------------------------------------------------------------------

# The gradient with respect to all five parameters at once (one `ForwardDiff.gradient`, chunk 5,
# Float64, Reference) against the finite-difference floor of A7 on each partial: `fd_scan` over
# six relative steps 2⁻⁴ to 2⁻¹⁴ of the parameter, the cold regular load (the seed changes
# nothing, passed through for common random numbers).
const T581_AD = Ref{Any}(nothing)
@testset "T-581 ForwardDiff through a small LWFA run against the finite-difference floor, five parameters [A3, A7, WP-043]" begin
    x0 = collect(LWFA_GRADIENT_X0)
    ad = ForwardDiff.gradient(x -> lwfa_wake_amplitude(x), x0)
    T581_AD[] = ad
    for i in eachindex(x0)
        f(xi, mode, seed) = lwfa_wake_amplitude(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode)
        hs = x0[i] .* 2.0 .^ (-4:-2:-14)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-581 $(LWFA_GRADIENT_NAMES[i])" ad = ad[i] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-581 rel/floor", Dict{String,Float64}())[LWFA_GRADIENT_NAMES[i]] = ratio
        @test isfinite(ad[i]) && ad[i] != 0
        @test r.rel[r.istar] <= emC("T-581") * r.floor_rel
        @test r.floor_rel <= emC("T-581", "floor_max")    # a primal gone noisy at the FD scale would raise the floor
    end
end

# --- T-582 ----------------------------------------------------------------------------------

# The production element type: `Dual{Float32,3}` seeded in a₀, the duration and the density, the
# three partials of the functional against T-581's Float64 gradient. Float32 over 350 steps of
# particle dynamics is not roundoff-close to Float64, so the scale is the measured relative
# difference against a proposed bound (A7: measure, then propose).
@testset "T-582 Dual{Float32,3} partials of the small LWFA run against the Float64 gradient [A7, A8, WP-043]" begin
    ad = T581_AD[] === nothing ? ForwardDiff.gradient(x -> lwfa_wake_amplitude(x), collect(LWFA_GRADIENT_X0)) : T581_AD[]
    x0 = LWFA_GRADIENT_X0
    xd = (mkdual(x0[1], 1, 0, 0), mkdual(x0[2], 0, 1, 0), mkdual(x0[3], 0, 0, 0), mkdual(x0[4], 0, 0, 1), mkdual(x0[5], 0, 0, 0))
    S = lwfa_wake_amplitude(xd)
    p = Tuple(partials(S))
    ref = (ad[1], ad[2], ad[4])
    rel = maximum(abs.(p .- ref) ./ abs.(ref))
    primal = abs(value(S) - lwfa_wake_amplitude(x0)) / abs(lwfa_wake_amplitude(x0))
    @info "T-582" partials = p ref rel primal
    get!(MEASURED, "T-582 rel", Dict{String,Float64}())["Dual{Float32,3}"] = rel
    @test S isa D3                                  # the production element type, not a Float64 dual
    @test all(isfinite, p) && all(!iszero, p)
    @test rel <= emC("T-582")
    @test primal <= emC("T-582")
end

# --- T-590 ----------------------------------------------------------------------------------

"The long horizon of WP-044: 600 steps, about 224 of them with the back edge in the plasma."
const LWFA_LONG_STEPS = 600

"""
    column_jitter(x0, i; nsteps, h, npoints = 9) -> (jitter, state)

Per interior column of the axis row, the spread (standard deviation) of the second differences
of E_x over a scan of parameter `i` with spacing `h` and `npoints` points centred on `x0`: a
column smooth in the parameter has nearly the same second difference at every point (its
curvature), a column that jumps has a spread. Also the state of the central run.
"""
function column_jitter(x0, i; nsteps, h, npoints = 9)
    m = npoints ÷ 2
    runs = map(-m:m) do j
        st, _ = lwfa_gradient_run(ntuple(k -> k == i ? x0[k] + j * h : x0[k], length(x0)), Reference(); nsteps)
        (axis_Ex_all(st), st)
    end
    ex = reduce(hcat, first.(runs))
    d2 = diff(diff(ex; dims = 2); dims = 2)
    return vec(std(d2; dims = 2)), last(runs[m + 1])
end

# The cause of #370, located: at 600 steps, over a scan of a₀ (spacing 2⁻¹²), the back node's E_x
# jitters far above the plasma's interior, and the first column past the band no more than the
# interior does. The interior's level is the median over the band's end to the middle of the
# window (the plasma behind the pulse; the columns ahead of the wake hold no plasma field and
# would pull a median over the whole window to roundoff).
@testset "T-590 the jitter of the long LWFA run lies in the window's back band [A7, WP-044]" begin
    jit, st = column_jitter(LWFA_GRADIENT_X0, 1; nsteps = LWFA_LONG_STEPS, h = 2.0^-12)
    b = window_back_band(st)
    n1 = length(jit)
    interior = median(jit[(b + 1):(n1 ÷ 2)])
    @info "T-590" band = b interior back = jit[1] / interior past = jit[b + 1] / interior profile = jit[1:(b + 2)] ./ interior
    get!(MEASURED, "T-590 jitter/interior", Dict{String,Float64}())["back node"] = jit[1] / interior
    get!(MEASURED, "T-590 jitter/interior", Dict{String,Float64}())["first past the band"] = jit[b + 1] / interior
    @test b == 2 * (ParticleMeshCore.stencil_reach(EMNumerics().shape, Node())[2] + 1)    # no filter passes here
    @test jit[1] >= emC("T-590", "back_min") * interior
    @test jit[b + 1] <= emC("T-590") * interior
end

# --- T-591 ----------------------------------------------------------------------------------

# T-581 at the long horizon, on the functional past the band: the three partials that exceeded the
# floor at 600 steps when the back band was read (a₀ at 3.0×, focus at 1.09×, density at 1.03×,
# #370) against `fd_scan`'s floor, as in T-581.
@testset "T-591 ForwardDiff through 600 steps of the small LWFA run against the finite-difference floor, past the back band [A3, A7, WP-044]" begin
    x0 = collect(LWFA_GRADIENT_X0)
    sel = (1, 3, 4)
    g(y) = lwfa_wake_amplitude(ntuple(j -> (k = findfirst(==(j), sel); k === nothing ? x0[j] : y[k]), length(x0)); nsteps = LWFA_LONG_STEPS)
    ad = ForwardDiff.gradient(g, x0[collect(sel)])
    for (k, i) in enumerate(sel)
        f(xi, mode, seed) = lwfa_wake_amplitude(ntuple(j -> j == i ? xi : x0[j], length(x0)), mode; nsteps = LWFA_LONG_STEPS)
        hs = x0[i] .* 2.0 .^ (-4:-2:-14)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[k])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-591 $(LWFA_GRADIENT_NAMES[i])" ad = ad[k] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-591 rel/floor", Dict{String,Float64}())[LWFA_GRADIENT_NAMES[i]] = ratio
        @test isfinite(ad[k]) && ad[k] != 0
        @test r.rel[r.istar] <= emC("T-591") * r.floor_rel
        @test r.floor_rel <= emC("T-581", "floor_max")
    end
end
