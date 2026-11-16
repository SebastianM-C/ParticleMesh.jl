# Observers and diagnostics. A step hands the state at its start to an observer (`observe!`); the
# module's own observer is the `HistoryRecorder`, which records the module's physical quantities
# once per step by work items of a fixed order (so deterministic), read back on the host after the
# run. The diagnostic mode and the histories belong to the recorder, not to the step.

const HISTORY_KEYS = (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz, :K, :Ex_re, :Ex_im, :Ey_re, :Ey_im, :Bz_re, :Bz_im)

"""
    RECORD_GROUPS

The groups of quantities a [`HistoryRecorder`](@ref) can record, the values of its `record`
keyword: `:field_energy` (the six energies `WEx … WBz`), `:kinetic` (`K`) and `:modes` (every
Fourier amplitude at the diagnostic mode).
"""
const RECORD_GROUPS = (:field_energy, :kinetic, :modes)
const ENERGY_KEYS = (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz)

# Which groups a recorder records, in its type: the recording kernels are compiled for the
# selection, so a group that is not recorded costs no work item and no history.
struct RecordSelection{E,K,M} end
sel_energy(::RecordSelection{E}) where {E} = Val(E)
sel_kinetic(::RecordSelection{E,K}) where {E,K} = Val(K)
sel_modes(::RecordSelection{E,K,M}) where {E,K,M} = Val(M)
@inline on(::Val{true}) = true
@inline on(::Val{false}) = false

function RecordSelection(record)
    record isa Union{Symbol,Tuple,AbstractVector,AbstractSet} ||
        throw(ArgumentError("record takes a group of $RECORD_GROUPS or a collection of them, got $(repr(record))"))
    groups = record isa Symbol ? (record,) : Tuple(record)
    all(g -> g in RECORD_GROUPS, groups) ||
        throw(ArgumentError("record takes groups of $RECORD_GROUPS, got $record"))
    return RecordSelection{:field_energy in groups,:kinetic in groups,:modes in groups}()
end

"The history keys of `keys` (all of a dimension) that the selection records, in their order."
function recorded_keys(keys, sel::RecordSelection)
    return Tuple(k for k in keys if (k in ENERGY_KEYS ? on(sel_energy(sel)) : k === :K ? on(sel_kinetic(sel)) : on(sel_modes(sel))))
end

@inline function energy_sum(a, g::Int32, n::Int32, ::Type{T}) where {T}
    w = zero(T)
    @inbounds for j in Int32(1):n
        w += a[g + j] * a[g + j]
    end
    return w
end

# Fourier amplitude 2/n·Σ a[j]·e^{−ik xⱼ} of a field whose index j sits at xⱼ = (j − 1 + x0)Δx.
@inline function mode_sum(a, g::Int32, n::Int32, kdx, x0, ::Type{T}) where {T}
    re = zero(T)
    im = zero(T)
    @inbounds for j in Int32(1):n
        φ = kdx * (T(j - Int32(1)) + x0)
        re += a[g + j] * cos(φ)
        im -= a[g + j] * sin(φ)
    end
    return re * 2 / T(n), im * 2 / T(n)
end

@inline record_energies_1d!(::Val{false}, h, f, g, n, dx, k, ::Type{T}) where {T} = nothing
@inline function record_energies_1d!(::Val{true}, h, f, g::Int32, n::Int32, dx, k::Int32, ::Type{T}) where {T}
    @inbounds begin
        h.WEx[k] = energy_sum(f.Ex, g, n, T) * dx / 2
        h.WEy[k] = energy_sum(f.Ey, g, n, T) * dx / 2
        h.WEz[k] = energy_sum(f.Ez, g, n, T) * dx / 2
        h.WBx[k] = energy_sum(f.Bx, g, n, T) * dx / 2
        h.WBy[k] = energy_sum(f.By, g, n, T) * dx / 2
        h.WBz[k] = energy_sum(f.Bz, g, n, T) * dx / 2
    end
    return nothing
end

@inline record_kinetic!(::Val{false}, h, K, len, dA, k, ::Type{T}) where {T} = nothing
# Sums every entry of `K` in index order: at N = 1 the kinetic energy density, deposited with
# guards and not folded; at N = 2 the chunks.
@inline function record_kinetic!(::Val{true}, h, K, len::Int32, dA, k::Int32, ::Type{T}) where {T}
    kin = zero(T)
    @inbounds for j in Int32(1):len
        kin += K[j]
    end
    @inbounds h.K[k] = kin * dA
    return nothing
end

@inline record_modes_1d!(::Val{false}, h, f, g, n, dx, kmode, k, ::Type{T}) where {T} = nothing
@inline function record_modes_1d!(::Val{true}, h, f, g::Int32, n::Int32, dx, kmode, k::Int32, ::Type{T}) where {T}
    half = one(T) / 2
    kdx = kmode * dx
    @inbounds begin
        h.Ex_re[k], h.Ex_im[k] = mode_sum(f.Ex, g, n, kdx, half, T)
        h.Ey_re[k], h.Ey_im[k] = mode_sum(f.Ey, g, n, kdx, zero(T), T)
        h.Bz_re[k], h.Bz_im[k] = mode_sum(f.Bz, g, n, kdx, half, T)
    end
    return nothing
end

@kernel function diagnostics_kernel!(h, f, @Const(K), dx, kmode, n::Int32, g::Int32, len::Int32, k::Int32, sel)
    T = eltype(K)
    record_energies_1d!(sel_energy(sel), h, f, g, n, dx, k, T)
    record_kinetic!(sel_kinetic(sel), h, K, len, dx, k, T)
    record_modes_1d!(sel_modes(sel), h, f, g, n, dx, kmode, k, T)
end

# --- N = 2: parallel reductions in a fixed association order -------------------------------

"""
History keys of a 2D state: the 1D keys, the Fourier amplitude of the longitudinal projection
k̂·E, those of Bx and By, and that of the in-plane projection ê·B, ê = ẑ × k̂ = (−k̂_y, k̂_x).
"""
const HISTORY_KEYS_2D = (HISTORY_KEYS..., :El_re, :El_im, :Bx_re, :Bx_im, :By_re, :By_im, :Bt_re, :Bt_im)
"Rows of the 2D diagnostics' partial sums: six energies and the real and imaginary sums of Ex, Ey, Bz, Bx, By."
const DIAG_ROWS = 16

# e^{−i(θ₁ + θ₂)} from the per-axis tables: (cos θ₁ cos θ₂ − sin θ₁ sin θ₂, sin θ₁ cos θ₂ + cos θ₁ sin θ₂).
@inline phase(c1, s1, c2, s2) = (c1 * c2 - s1 * s2, s1 * c2 + c1 * s2)

# Stage 1: one work item per index j₁ along axis 1 sums its line along axis 2 serially, from
# zero, in increasing j₂: the squares of the six components and the Fourier sums of Ex, Ey, Bz,
# Bx and By, each with the phases of its own stagger (Ex and By at (Center, Node), Ey and Bx at
# (Node, Center), Bz at (Center, Center)). Adjacent work items read adjacent memory.
@kernel function diagnostics_rows_2d_kernel!(P, f, @Const(t1), @Const(t2), n2::Int32, g::Int32, sel)
    j1 = Int32(@index(Global, Linear))
    T = eltype(P)
    energy, modes = on(sel_energy(sel)), on(sel_modes(sel))
    i1 = g + j1
    @inbounds begin
        c1N, s1N, c1C, s1C = t1[j1, 1], t1[j1, 2], t1[j1, 3], t1[j1, 4]
        wEx = wEy = wEz = wBx = wBy = wBz = zero(T)
        exr = exi = eyr = eyi = bzr = bzi = bxr = bxi = byr = byi = zero(T)
        for j2 in Int32(1):n2
            i2 = g + j2
            c2N, s2N, c2C, s2C = t2[j2, 1], t2[j2, 2], t2[j2, 3], t2[j2, 4]
            ex, ey, ez = f.Ex[i1, i2], f.Ey[i1, i2], f.Ez[i1, i2]
            bx, by, bz = f.Bx[i1, i2], f.By[i1, i2], f.Bz[i1, i2]
            if energy
                wEx += ex * ex
                wEy += ey * ey
                wEz += ez * ez
                wBx += bx * bx
                wBy += by * by
                wBz += bz * bz
            end
            if modes
                cs, sn = phase(c1C, s1C, c2N, s2N)
                exr += ex * cs
                exi -= ex * sn
                byr += by * cs
                byi -= by * sn
                cs, sn = phase(c1N, s1N, c2C, s2C)
                eyr += ey * cs
                eyi -= ey * sn
                bxr += bx * cs
                bxi -= bx * sn
                cs, sn = phase(c1C, s1C, c2C, s2C)
                bzr += bz * cs
                bzi -= bz * sn
            end
        end
        if energy
            P[1, j1], P[2, j1], P[3, j1], P[4, j1], P[5, j1], P[6, j1] = wEx, wEy, wEz, wBx, wBy, wBz
        end
        if modes
            P[7, j1], P[8, j1], P[9, j1], P[10, j1], P[11, j1], P[12, j1] = exr, exi, eyr, eyi, bzr, bzi
            P[13, j1], P[14, j1], P[15, j1], P[16, j1] = bxr, bxi, byr, byi
        end
    end
end

@inline function row_sum(P, q::Int, n1::Int32, ::Type{T}) where {T}
    acc = zero(T)
    @inbounds for j in Int32(1):n1
        acc += P[q, j]
    end
    return acc
end

@inline record_energies_2d!(::Val{false}, h, P, n1, dA, k, ::Type{T}) where {T} = nothing
@inline function record_energies_2d!(::Val{true}, h, P, n1::Int32, dA, k::Int32, ::Type{T}) where {T}
    @inbounds begin
        h.WEx[k] = row_sum(P, 1, n1, T) * dA / 2
        h.WEy[k] = row_sum(P, 2, n1, T) * dA / 2
        h.WEz[k] = row_sum(P, 3, n1, T) * dA / 2
        h.WBx[k] = row_sum(P, 4, n1, T) * dA / 2
        h.WBy[k] = row_sum(P, 5, n1, T) * dA / 2
        h.WBz[k] = row_sum(P, 6, n1, T) * dA / 2
    end
    return nothing
end

@inline record_modes_2d!(::Val{false}, h, P, n1, amp, khat, k, ::Type{T}) where {T} = nothing
@inline function record_modes_2d!(::Val{true}, h, P, n1::Int32, amp, khat, k::Int32, ::Type{T}) where {T}
    @inbounds begin
        exr, exi = row_sum(P, 7, n1, T) * amp, row_sum(P, 8, n1, T) * amp
        eyr, eyi = row_sum(P, 9, n1, T) * amp, row_sum(P, 10, n1, T) * amp
        h.Ex_re[k], h.Ex_im[k] = exr, exi
        h.Ey_re[k], h.Ey_im[k] = eyr, eyi
        h.Bz_re[k], h.Bz_im[k] = row_sum(P, 11, n1, T) * amp, row_sum(P, 12, n1, T) * amp
        h.El_re[k] = khat[1] * exr + khat[2] * eyr
        h.El_im[k] = khat[1] * exi + khat[2] * eyi
        bxr, bxi = row_sum(P, 13, n1, T) * amp, row_sum(P, 14, n1, T) * amp
        byr, byi = row_sum(P, 15, n1, T) * amp, row_sum(P, 16, n1, T) * amp
        h.Bx_re[k], h.Bx_im[k] = bxr, bxi
        h.By_re[k], h.By_im[k] = byr, byi
        h.Bt_re[k] = khat[1] * byr - khat[2] * bxr
        h.Bt_im[k] = khat[1] * byi - khat[2] * bxi
    end
    return nothing
end

# Stage 2: one work item adds the rows' partial sums in increasing j₁ and the kinetic-energy
# chunks in order, and writes entry k of the histories the selection records.
@kernel function diagnostics_total_2d_kernel!(h, @Const(P), @Const(K), n1::Int32, nK::Int32, dA, amp, khat, k::Int32, sel)
    T = eltype(P)
    record_energies_2d!(sel_energy(sel), h, P, n1, dA, k, T)
    record_kinetic!(sel_kinetic(sel), h, K, nK, dA, k, T)
    record_modes_2d!(sel_modes(sel), h, P, n1, amp, khat, k, T)
end

# Per-axis phase tables of the diagnostic mode m on n nodes: columns cos and sin of 2π·m·x/n at
# the Node() positions x = j − 1 and at the Center() positions x = j − 1/2. The angle over π,
# m·(2j − 2)/n and m·(2j − 1)/n, is reduced exactly in integers modulo 2 before the division.
@kernel function phase_table_kernel!(t, m::Int, n::Int)
    j = @index(Global, Linear)
    T = eltype(t)
    @inbounds begin
        sn, cs = sincospi(T(mod(m * (2j - 2), 2n)) / T(n))
        t[j, 1] = cs
        t[j, 2] = sn
        sn, cs = sincospi(T(mod(m * (2j - 1), 2n)) / T(n))
        t[j, 3] = cs
        t[j, 4] = sn
    end
end

"""
    HistoryRecorder{N}

The module's observer: the diagnostic histories `hist` of a run at N = 1 or 2 (the keys of
`HISTORY_KEYS`, at N = 2 of `HISTORY_KEYS_2D`, on the state's backend) for the diagnostic mode
`kmode` (the wavenumber at N = 1; the integer mode per axis at N = 2), and in `aux` what only
N = 2 needs: the unit vector `khat` along the mode's wave vector, the phase tables of each axis
(written once) and the per-row partial sums. A state owns one as its default observer
(`state.recorder`, for the problem's `kmode` and `nsteps`); another one, for another mode, length,
cadence or selection, is passed to `step!` or `run!` as `observer`. Read it with
[`histories`](@ref).

    HistoryRecorder(problem::EMProblem, state::EMState; kmode = problem.kmode, nsteps = problem.nsteps,
                    every = 1, record = RECORD_GROUPS)

A recorder for `state`, built for `problem`, with its own histories, covering the steps
1 … `nsteps`. It records at the steps 1, 1 + `every`, 1 + 2`every`, … ([`recorded_steps`](@ref)),
one history entry per recorded step, and it records the groups of `record`
([`RECORD_GROUPS`](@ref): `:field_energy`, `:kinetic`, `:modes`; a symbol or a collection of
them, `()` for none): the histories of the other groups do not exist, and their sums are not
computed. What is recorded has the bits of the same entry of a recorder that records every step
and every group.
"""
struct HistoryRecorder{N,H<:NamedTuple,K,X<:NamedTuple,S<:RecordSelection}
    hist::H
    kmode::K
    aux::X
    nsteps::Int
    every::Int
    record::S
end

function check_cadence(nsteps::Integer, every::Integer)
    nsteps >= 0 || throw(ArgumentError("a recorder covers nsteps ≥ 0 steps, got $nsteps"))
    (every isa Bool || every < 1) && throw(ArgumentError("a recorder records every `every` ≥ 1 steps (an integer), got every = $every"))
    return cld(nsteps, every)
end

function history_arrays(backend, ::Type{T}, keys, sel::RecordSelection, entries::Integer) where {T}
    ks = recorded_keys(keys, sel)
    return NamedTuple{ks}(ntuple(_ -> KernelAbstractions.zeros(backend, T, entries), length(ks)))
end

function HistoryRecorder(backend, ::Type{T}, grid::PeriodicGrid{1}, box, kmode, nsteps::Integer; workgroupsize = nothing, every::Integer = 1,
                         record = RECORD_GROUPS) where {T}
    sel = RecordSelection(record)
    hist = history_arrays(backend, T, HISTORY_KEYS, sel, check_cadence(nsteps, every))
    return HistoryRecorder{1,typeof(hist),T,@NamedTuple{},typeof(sel)}(hist, T(kmode), NamedTuple(), Int(nsteps), Int(every), sel)
end

function HistoryRecorder(backend, ::Type{T}, grid::PeriodicGrid{2}, box, kmode::NTuple{2,Int}, nsteps::Integer; workgroupsize = nothing,
                         every::Integer = 1, record = RECORD_GROUPS) where {T}
    kmode == (0, 0) && throw(ArgumentError("the diagnostic mode must not be (0, 0)"))
    sel = RecordSelection(record)
    n = ncells(grid)
    hist = history_arrays(backend, T, HISTORY_KEYS_2D, sel, check_cadence(nsteps, every))
    tables = ntuple(Val(2)) do d
        t = KernelAbstractions.allocate(backend, T, n[d], 4)
        launcher(phase_table_kernel!, backend, workgroupsize)(t, kmode[d], n[d]; ndrange = n[d])
        t
    end
    k = ntuple(d -> 2 * T(π) * kmode[d] / box[d], Val(2))
    knorm = sqrt(k[1] * k[1] + k[2] * k[2])
    aux = (khat = (k[1] / knorm, k[2] / knorm), tables = tables, partials = KernelAbstractions.zeros(backend, T, DIAG_ROWS, n[1]))
    return HistoryRecorder{2,typeof(hist),NTuple{2,Int},typeof(aux),typeof(sel)}(hist, kmode, aux, Int(nsteps), Int(every), sel)
end

HistoryRecorder(problem::EMProblem{N,T}, st::EMState{N,T}; kmode = problem.kmode, nsteps::Integer = problem.nsteps, every::Integer = 1,
                record = RECORD_GROUPS) where {N,T} =
    HistoryRecorder(get_backend(st.ρ), T, st.grid, problem.length, kmode, nsteps; workgroupsize = st.numerics.workgroupsize, every, record)

"""
    empty_recorder(recorder::HistoryRecorder) -> HistoryRecorder

A recorder of the same mode, length, cadence and selection with histories of its own, zero, and
at N = 2 partial sums of its own; the phase tables, written once, are shared.
"""
empty_recorder(r::HistoryRecorder) =
    typeof(r)(map(h -> fill!(similar(h), zero(eltype(h))), r.hist), r.kmode, own_scratch(r.aux), r.nsteps, r.every, r.record)
own_scratch(aux::NamedTuple{()}) = aux
own_scratch(aux::NamedTuple) = merge(aux, (; partials = similar(aux.partials)))

"The number of steps a recorder covers (its histories hold one entry per recorded step, [`recorded_steps`](@ref))."
Base.length(r::HistoryRecorder) = r.nsteps

"""
    recorded_steps(recorder) -> StepRange
    recorded_steps(state) -> StepRange

The steps whose start a recorder records (of a state: its own recorder's), entry `i` of its
histories being that of step `recorded_steps(recorder)[i]`, at t = (step − 1)Δt.
"""
recorded_steps(r::HistoryRecorder) = 1:r.every:r.nsteps
recorded_steps(st::EMState) = recorded_steps(st.recorder)

"Whether step `k` (a host integer) is on the recorder's cadence: its recorded steps and their continuation past its length, which `check_step` refuses."
records_at(r::HistoryRecorder, k::Integer) = (k - 1) % r.every == 0

"The recorder's selection of groups, as the tuple of [`RECORD_GROUPS`](@ref) it records."
recorded_groups(r::HistoryRecorder) =
    Tuple(g for (g, v) in zip(RECORD_GROUPS, (sel_energy(r.record), sel_kinetic(r.record), sel_modes(r.record))) if on(v))

"""
    reads_kinetic(observer, k) -> Bool

Whether `observer`, shown the state at the start of step `k`, reads the kinetic-energy cache
`state.K`; where it does not, the step before leaves the cache as it was and marks it stale
(`step!`). A [`HistoryRecorder`](@ref) that records `:kinetic` reads it at the steps of its
cadence, 1, 1 + `every`, … (past its length too, so that the last step of a run of a recorder of
every step leaves the cache current), and one that does not record it never does (`record = ()`); a `Tuple` reads it when one of its observers does; anything
else, `nothing` included, is taken to read it at every step, so that a run without an observer
leaves `state.K` current after every step. An observer type of your own that does not read `K`
can say so with a method.
"""
reads_kinetic(r::HistoryRecorder, k::Integer) = on(sel_kinetic(r.record)) && records_at(r, k)
reads_kinetic(os::Tuple, k::Integer) = any(o -> reads_kinetic(o, k), os)
reads_kinetic(f, k::Integer) = true

# The step is a host integer (not device data): the recording kernels write an entry of the
# histories under @inbounds, so a k outside the recorder's steps must not reach a kernel.
check_step(r::HistoryRecorder, k::Integer) =
    1 <= k <= length(r) || throw(ArgumentError("step $k outside the $(length(r)) steps the recorder was allocated for"))

"""
    observe!(observer, state, k)

What a step does first: show `observer` the state at the start of step `k`, at t = (k − 1)Δt,
with Eⁿ, Bⁿ in the fields, the particles at xⁿ with their momenta at n − 1/2, and the kinetic
energy at n − 1/2 in `state.K`. An observer reads the state and writes only what it owns, with
one exception: where the cache `K` is stale (the step before ran with an observer that does not
read it at step `k`, [`reads_kinetic`](@ref)) and this observer reads it, `observe!` first
recomputes it ([`refresh_caches!`](@ref)), so what an observer reads or records is the kinetic
energy of the particles as they are.

- `nothing` observes nothing (the step then has no limit on `k`);
- a [`HistoryRecorder`](@ref) writes, at the steps it records ([`recorded_steps`](@ref)), the
  entry of step `k` of the histories of the groups it records (an `ArgumentError` for a `k`
  outside its steps, recorded or not);
- a `Tuple` of observers is observed in order;
- anything else is called as `observer(state, k)`.

At N = 1 the recorder's entry is one work item's sums over the grid in index order: the field
energies ½Σ(component)²·Δx, the kinetic energy Σ K·Δx and the Fourier amplitude 2/n·Σ a·e^{−ikx}
of Ex, Ey and Bz at the recorder's wavenumber, each over the positions of its own stagger.

At N = 2 it is the field energies ½Σ(component)²·ΔxΔy (per unit length along z), the kinetic
energy Σ m·w·(γ − 1)·ΔxΔy, and at the recorder's diagnostic mode the Fourier amplitude
2/(n₁n₂)·Σ a·e^{−ik·x} of Ex, Ey, Bz, Bx and By, each over the positions of its own stagger, of the
longitudinal projection k̂ₓÊₓ + k̂_yÊ_y and of the in-plane magnetic projection on ê = ẑ × k̂,
k̂ₓB̂_y − k̂_yB̂ₓ (the 2D Weibel mode): two launches with a fixed association order (stage 1: O(n₁)
work items of O(n₂) serial work; stage 2: one work item of O(n₁ + chunks)), so the bits do not
depend on the thread count or the launch shape; the phases come from the recorder's per-axis
tables (no trigonometric evaluation per step).
"""
observe!(::Nothing, st::EMState, k::Integer) = nothing
observe!(f, st::EMState, k::Integer) = (reads_kinetic(f, k) && current_kinetic!(st); f(st, k); nothing)

"The state with its kinetic-energy cache current: recomputed where a step left it stale (`state.fresh`)."
current_kinetic!(st::EMState) = st.fresh[] ? st : refresh_caches!(st)
observe!(os::Tuple, st::EMState, k::Integer) = (foreach(o -> observe!(o, st, k), os); nothing)

observe!(r::HistoryRecorder{M}, st::EMState{N}, k::Integer) where {M,N} =
    throw(ArgumentError("a recorder of N = $M cannot observe a state of N = $N"))

# Entry of step k in the histories, or 0 where the recorder records nothing at step k.
function history_entry(r::HistoryRecorder, k::Integer)
    check_step(r, k)
    (records_at(r, k) && !isempty(r.hist)) || return 0
    return (k - 1) ÷ r.every + 1
end

function observe!(r::HistoryRecorder{1}, st::EMState{1}, k::Integer)
    e = history_entry(r, k)
    e == 0 && return nothing
    on(sel_kinetic(r.record)) && current_kinetic!(st)
    grid = st.grid
    diagnostics_kernel!(get_backend(st.ρ))(r.hist, st.fields, st.K, cellsize(grid), r.kmode, Int32(ncells(grid)), Int32(guards(grid)),
                                           Int32(length(st.K)), Int32(e), r.record; ndrange = 1)
    return nothing
end

function observe!(r::HistoryRecorder{2}, st::EMState{2}, k::Integer)
    e = history_entry(r, k)
    grid = st.grid
    T = eltype(st.ρ)
    n = ncells(grid)
    (size(r.aux.tables[1], 1), size(r.aux.tables[2], 1)) == n ||
        throw(ArgumentError("the recorder's phase tables are for $((size(r.aux.tables[1], 1), size(r.aux.tables[2], 1))) cells, the state has $n"))
    e == 0 && return nothing
    on(sel_kinetic(r.record)) && current_kinetic!(st)
    dx = cellsize(grid)
    be = get_backend(st.ρ)
    # the rows' partial sums serve the energies and the modes; the kinetic energy needs none
    (on(sel_energy(r.record)) || on(sel_modes(r.record))) &&
        launcher(diagnostics_rows_2d_kernel!, be, st.numerics.workgroupsize)(r.aux.partials, st.fields, r.aux.tables[1], r.aux.tables[2],
                                                                            Int32(n[2]), Int32(guards(grid)), r.record; ndrange = n[1])
    diagnostics_total_2d_kernel!(be)(r.hist, r.aux.partials, st.K, Int32(n[1]), Int32(length(st.K)), dx[1] * dx[2],
                                     2 / T(n[1] * n[2]), r.aux.khat, Int32(e), r.record; ndrange = 1)
    return nothing
end

"""
    histories(recorder::HistoryRecorder) -> NamedTuple
    histories(state) -> NamedTuple

Host copies of the diagnostics of a recorder (of the state's own, `state.recorder`), one entry
per recorded step and only the groups it records ([`recorded_steps`](@ref); every step and every
group by default), the entry of step `k` recorded at the start of step `k`, at
t = (k − 1)Δt: the field energies `WEx … WBz` = ½Σ(component)²Δx of Eⁿ and Bⁿ (in mₑc²·n_c·c/ω₀,
A1), the kinetic energy `K` = Σ m·w·(γ − 1)·Δx at t − Δt/2, and the real and imaginary parts of the
Fourier amplitude at `kmode` of Ex, Ey and Bz (each at its own stagger). At N = 2 it also holds
`El_re`, `El_im`, the amplitude of the longitudinal projection k̂·E, `Bx_*`, `By_*` and
`Bt_re`, `Bt_im`, the amplitude of ê·B with ê = ẑ × k̂, and its energies are sums over ΔxΔy
([`observe!`](@ref)).
"""
histories(r::HistoryRecorder) = map(Array, r.hist)
histories(st::EMState) = histories(st.recorder)

"Field energy per step: the sum of the six component energies."
field_energy(h::NamedTuple) = h.WEx .+ h.WEy .+ h.WEz .+ h.WBx .+ h.WBy .+ h.WBz

"""
    kinetic_energy_history(h) -> Vector

The kinetic energy at integer times n = 0 … N − 2 (steps 1 … N − 1), as the mean of the values at
n − 1/2 and n + 1/2 recorded by steps n + 1 and n + 2: `h` holds the histories of every step
(`every = 1`). Given the recorder or the state instead of its histories, one that does not
record every step is refused with an `ArgumentError` (the mean of entries `every` steps apart is
not the energy at a time).
"""
kinetic_energy_history(h::NamedTuple) = (h.K[1:(end - 1)] .+ h.K[2:end]) ./ 2
kinetic_energy_history(r::HistoryRecorder) = kinetic_energy_history(histories(every_step(r)))
kinetic_energy_history(st::EMState) = kinetic_energy_history(st.recorder)

"Total energy (fields plus kinetic) at integer times, steps 1 … N − 1, of the histories of every step; of a recorder or a state as [`kinetic_energy_history`](@ref)."
total_energy_history(h::NamedTuple) = field_energy(h)[1:(end - 1)] .+ kinetic_energy_history(h)
total_energy_history(r::HistoryRecorder) = total_energy_history(histories(every_step(r)))
total_energy_history(st::EMState) = total_energy_history(st.recorder)

every_step(r::HistoryRecorder) =
    r.every == 1 ? r : throw(ArgumentError("the energy at integer times is the mean of two consecutive steps' entries, and the recorder records every $(r.every) steps"))

"Amplitude history |â| of the mode at `kmode` of `:Ex`, `:Ey` or `:Bz` (at N = 2 also `:El`, `:Bx`, `:By`, `:Bt`)."
mode_amplitude_history(h::NamedTuple, c::Symbol) = hypot.(getfield(h, Symbol(c, :_re)), getfield(h, Symbol(c, :_im)))
