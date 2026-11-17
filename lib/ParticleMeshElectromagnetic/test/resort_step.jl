# WP-041: the step's sort from the order of the last sort (`EMNumerics(; resort)`), T-852 to T-855;
# the deposition by nearest-node bins (`EMNumerics(; node_bins)`), T-865.

"""
    resort_digests(T; resort, mode, nsteps) -> Vector{UInt64}

`ion_window`'s run (the window's refills behind the count after every shift, the dopant's births
behind the electrons' count, retirements at the back edge) with `EMNumerics(; resort)`: the digest
of everything the state evolves after every step, and of the workspaces' permutation and segments.
"""
function resort_digests(::Type{T}; resort, mode = Reference(), nsteps = 150) where {T}
    prob = ion_window(T; nsteps)
    st = EMState(prob, EMNumerics(; mode, resort); backend = CPU())
    initialize!(st, prob)
    p = params(prob)
    fr = st.aux.ionization
    out = UInt64[]
    nborn = 0
    for k in 1:nsteps
        step!(st, p, k)
        h = ion_digest(st)
        for w in (st.ws..., fr.electrons_ws), a in (w.perm, w.seg_start, w.seg_stop)
            h = hash(a, h)
        end
        push!(out, h)
        nborn = max(nborn, born(st).n)
    end
    used = all(w -> resort ? w.resort !== nothing && w.resort.ordered[] : w.resort === nothing, (st.ws..., fr.electrons_ws))
    resorts = resort ? map(w -> w.resort.resorts[], (st.ws..., fr.electrons_ws)) : nothing
    return out, used, window_shifts(st, nsteps), nborn, resorts
end

@testset "T-852 the step sorting from the last order gives the bits of the full sort at every step, with window refills and ionization births [A4, WP-041]" begin
    for (T, mode) in ((Float32, Reference()), (Float64, Reference()), (D3, Reference()), (D3, Fast()))
        inc, used_inc, shifts, nborn, resorts = resort_digests(T; resort = true, mode)
        full, used_full, _, _, _ = resort_digests(T; resort = false, mode)
        @test used_inc && used_full
        @test all(>(150), resorts)                # every step's sort and the shifts' (208 and 207 measured)
        @test shifts > 40 && nborn > 0           # the run refills and gives births
        @test inc == full
    end
end

# Whether the slots up to the count of a workspace's last sort hold alive particles each inside
# its bin's segment (its cell's, for `CellBins()`): the sorted state that the window's refill and the next sort start from.
function sorted_state(s, ws, n)
    m = Int(ws.resort.sorted_count[1])
    return all(1:m) do i
        s.status[i] == ParticleMeshCore.ALIVE || return false
        L = ParticleMeshCore.linear_cell(ParticleMeshCore.particle_bin(ws.bins, s, i, Int32.(n)), Int32.(n))
        return ws.seg_start[L] <= i <= ws.seg_stop[L]
    end
end

@testset "T-853 a species the step does not sort is sorted anew at the window's shift, not from an order many steps old [A4, WP-041]" begin
    # Float32 in Fast(): deposited by atomics, so the step sorts neither the species nor the
    # electrons, and the shift's sort cannot start from the order of the last sort
    prob = ion_window(Float32; nsteps = 150)
    st = EMState(prob, EMNumerics(; mode = Fast(), resort = true); backend = CPU())
    initialize!(st, prob)
    p = params(prob)
    fr = st.aux.ionization
    ok, checked, nborn = true, 0, 0
    for k in 1:150
        step!(st, p, k)
        nborn = max(nborn, born(st).n)
        window_shifts(st, k) > window_shifts(st, k - 1) || continue
        checked += 1
        for (s, w) in zip((st.species..., fr.electrons), (st.ws..., fr.electrons_ws))
            ok &= sorted_state(s, w, st.grid.n)
        end
    end
    @test checked > 40 && nborn > 0
    @test ok
end

"A hash of the bits of the state's evolved arrays."
evolved_digest(st) = foldl((h, a) -> hash(bits(a), h), evolved(st); init = UInt64(0))

@testset "T-854 a restart into a used state, or a state over saved arrays, sorts anew and continues the run's bits [A4, WP-041]" begin
    # the snapshot after step 100: the laser is in the plasma, the window shifts and the dopant
    # gives births, so the particles change cells between that step and the state's step 150
    nsteps, k0 = 150, 100
    prob = ion_window(Float32; nsteps)
    p = params(prob)
    new_state() = (st = EMState(prob, EMNumerics(; mode = Reference(), resort = true); backend = CPU()); initialize!(st, prob); st)
    ref = new_state()
    local snap
    nborn = 0
    for k in 1:nsteps
        step!(ref, p, k)
        k == k0 && (snap = snapshot(ref, k))
        k > k0 && (nborn = max(nborn, born(ref).n))
    end
    @test window_shifts(ref, nsteps) - window_shifts(ref, k0) > 10 && nborn > 0
    want = evolved_digest(ref)
    # restore! into the state that ran on to step 150, and into a new one
    for st in (ref, new_state())
        restore!(st, snap)
        for k in (k0 + 1):nsteps
            step!(st, p, k)
        end
        @test evolved_digest(st) == want
    end
    # a state over copies of the saved arrays, from a state that ran to the end
    used = new_state()
    for k in 1:nsteps
        step!(used, p, k)
    end
    view = ParticleMeshElectromagnetic.state_view(used, map(copy, snap.arrays))
    for k in (k0 + 1):nsteps
        step!(view, p, k)
    end
    @test evolved_digest(view) == want
end

# The status, the count and the bits of the alive slots of every species (the plasma, the atoms
# and the born electrons), and the fields' bits: what a state initialized again must repeat (its
# free slots keep what the run before left).
function alive_record(st)
    fr = ParticleMeshElectromagnetic.ionization_frame(st)
    sp = (st.species..., fr.atoms, fr.electrons)
    rec(s) = (alive = Array(s.status) .== ParticleMeshCore.ALIVE;
              (Array(s.status), Array(s.count), map(a -> bits(Array(a)[alive]), (s.cell..., s.offset..., s.u..., s.w))...))
    return (map(rec, sp), map(a -> bits(Array(a)), values(st.fields)))
end

@testset "T-855 a used state initialized again, and a state and a state over other arrays stepping in turn, keep the run's bits with the sort from the last order [A4, WP-041]" begin
    # initialize! on a state that ran, with births from the first steps on: the born electrons'
    # workspace holds an order of electrons that initialize! removes
    prob = ion_in_E(Float32; plasma = true, nsteps = 20)
    p = params(prob)
    st = EMState(prob, EMNumerics(; mode = Reference(), resort = true); backend = CPU())
    initialize!(st, prob)
    foreach(k -> step!(st, p, k), 1:20)
    want = alive_record(st)
    @test Int(ParticleMeshElectromagnetic.ionization_frame(st).electrons.count[1]) > 0   # electrons alive at the end
    initialize!(st, prob)
    foreach(k -> step!(st, p, k), 1:20)
    @test alive_record(st) == want
    # a state and a view over a copy of its earlier arrays share the workspaces: the view's steps
    # leave them with the view's order, and the state's next sort is the full one
    prob = ion_window(Float32; nsteps = 150)
    p = params(prob)
    mk() = (s = EMState(prob, EMNumerics(; mode = Reference(), resort = true); backend = CPU()); initialize!(s, prob); s)
    ref, st = mk(), mk()
    foreach(k -> step!(ref, p, k), 1:150)
    local snap
    for k in 1:100
        step!(st, p, k)
        k == 50 && (snap = snapshot(st, k))
    end
    view = ParticleMeshElectromagnetic.state_view(st, map(copy, snap.arrays))
    foreach(k -> step!(view, p, k), 51:60)
    foreach(k -> step!(st, p, k), 101:150)
    @test evolved_digest(st) == evolved_digest(ref)
end

"""
    node_bins_gap(T; nsteps) -> (worst, gaps, shifts, nborn)

`ion_window`'s Reference run with `EMNumerics(; node_bins)` on and off: per field array, the largest
|on − off| over `eps(T)·max|off|` after `nsteps` steps, and the largest of them.
"""
function node_bins_gap(::Type{T}; nsteps = 150) where {T}
    prob = ion_window(T; nsteps)
    p = params(prob)
    sts = map((true, false)) do nb
        st = EMState(prob, EMNumerics(; mode = Reference(), node_bins = nb); backend = CPU())
        initialize!(st, prob)
        st
    end
    nborn = 0
    for k in 1:nsteps
        foreach(st -> step!(st, p, k), sts)
        nborn = max(nborn, born(sts[1]).n)
    end
    on, off = sts
    gaps = Dict{Symbol,Float64}()
    for f in fieldnames(typeof(off.fields))
        y = getfield(off.fields, f)
        y isa AbstractArray || continue
        m = Float64(maximum(absmax, y))
        m > 0 && (gaps[f] = Float64(maximum(absdiff.(getfield(on.fields, f), y))) / (base_eps(T) * m))
    end
    return maximum(values(gaps)), gaps, window_shifts(on, nsteps), nborn
end

@testset "T-865 the step's deposition by nearest-node bins follows the cells' run to roundoff through the window's shifts and the births [A2, A4, WP-041]" begin
    C = emC("T-865")
    for T in (Float32, Float64, D3)
        worst, gaps, shifts, nborn = node_bins_gap(T)
        @test shifts > 40 && nborn > 0                   # the run refills and gives births
        @test haskey(gaps, :Jx) && haskey(gaps, :Ex)     # the plasma carries current and a wake
        @test record!("T-865", T, worst) <= C
        @test worst > 0                                  # the bins change the order of the sums
    end
end
