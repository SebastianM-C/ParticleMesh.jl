# WP-034: the back-edge PML's problems and runs (T-570 to T-575, scripts/em2d/pml_back.jl).
# After window_cases.jl, whose `window_lwfa`, `static_twin` and `window_shifts` it uses.

"A window state with `back` as the back edge's absorber."
function back_state(prob; back = nothing, mode = Reference(), workgroupsize = nothing)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, headroom = 1 / 4, back_absorber = back))
    initialize!(st, prob)
    return st
end

# The back-edge tests' problem: `window_lwfa` with its antenna at x = 10 (column 14) and 72 columns,
# so that the antenna stays out of a layer of up to 8 cells while it emits (the state refuses a
# plane that the layer would reach earlier), and the pulse keeps T-472's distance to the front.
const BACK_POSITION = 10
const BACK_CELLS = (72, 16)
back_lwfa(::Type{T}; kw...) where {T} = window_lwfa(T; position = BACK_POSITION, cells = BACK_CELLS, kw...)

"""
    back_comparison(back; nsteps = 200, extra = 100, position, cells) -> (; shifts, before, far, near, history)

T-472's vacuum pulse in the window with `back` on its back edge, against two references. Its
static twin with the same back edge gives `before` (every column, before the window moves) and
`far` (columns 25 … n₁ after), T-472's comparison of the shift's mechanics. A free-space reference,
a static box with `extra` more columns behind the antenna (moved by as many) and a bare back too far
away for its reflection to come back within the run, gives `near`: the largest difference on the
eight columns behind the layer (cells + 1 … cells + 8), over the whole run. All three are relative
to the reference's largest field at that step; `history` is `near` per step.
"""
function back_comparison(back; nsteps = 200, extra = 100, position = BACK_POSITION, cells = BACK_CELLS)
    prob = window_lwfa(Float64; plasma = false, nsteps, position, cells)
    n = prob.cells
    h = prob.length[1] / n[1]
    win = back_state(prob; back)
    shifts = window_shifts(win, prob.nsteps)
    ref = back_state(static_twin(prob, n[1] + shifts + 8); back)
    free = back_state(window_lwfa(Float64; plasma = false, nsteps, position = position + extra * h,
                                  cells = (n[1] + shifts + 8 + extra, n[2]), start = Inf))
    p = params(prob)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    cb = back === nothing ? 0 : back.cells
    before = near = far = 0.0
    history = zeros(prob.nsteps)
    for k in 1:prob.nsteps
        step!(win, p, k)
        step!(ref, p, k)
        step!(free, p, k)
        s = window_shifts(win, k)
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a, b, f = Array(getfield(win.fields, c)), Array(getfield(ref.fields, c)), Array(getfield(free.fields, c))
            m = maximum(abs, b[(g + 1 + s):(g + n[1] + s), J])
            m > 0 || continue
            d(r) = maximum(abs, a[g .+ r, J] .- b[g .+ s .+ r, J]) / m
            r = (cb + 1):(cb + 8)
            history[k] = max(history[k], maximum(abs, a[g .+ r, J] .- f[g .+ extra .+ s .+ r, J]) / m)
            near = max(near, history[k])
            if s == 0
                before = max(before, d(1:n[1]))
            else
                far = max(far, d(25:n[1]))
            end
        end
    end
    return (; shifts, before, far, near, history)
end

"""
    back_long_run(back; start, nsteps, every = 50, seed = 3) -> (; t, w)

Random fields (standard normal, seed `seed`) in the empty window of `back_lwfa` (no laser, no
plasma) with `back` on its back edge, and the field energy over the window against its start,
every `every` steps: a window that never moves (`start = Inf`) or one that moves from `start`.
"""
function back_long_run(back; start, nsteps, every = 50, seed = 3)
    prob = back_lwfa(Float64; plasma = false, laser = false, nsteps, start)
    st = back_state(prob; back)
    rng = Xoshiro(seed)
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        a = getfield(st.fields, c)
        a .= randn(rng, size(a))
        fill_guards!(a, st.grid)
    end
    g, n = guards(st.grid), ncells(st.grid)
    W() = sum(c -> sum(abs2, view(Array(getfield(st.fields, c)), (g + 1):(g + n[1]), (g + 1):(g + n[2]))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz)) / 2
    W0 = W()
    p = params(prob)
    t, w = [0.0], [1.0]
    for k in 1:nsteps
        step!(st, p, k)
        k % every == 0 && (push!(t, k * Float64(prob.dt)); push!(w, W() / W0))
    end
    return (; t, w)
end
