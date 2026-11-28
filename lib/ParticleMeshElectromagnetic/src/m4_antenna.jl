# The antenna's B corrections for the M4 solver (WP-079). The plane splits the grid as in
# antenna.jl: E and B points of columns ≥ ia hold the total field, those behind it the scattered
# field. Yee's B difference crosses the plane only between the columns ia − 1 and ia; M4's
# ([`forward_difference`](@ref)) reaches two columns ahead and one behind along x, and averages
# over the neighbouring rows, so a B point of the scattered region reads total-field E (and the
# other way round) at more places: By and Bz on the columns ia − 2, ia − 1 (scattered) and ia
# (total) through the third difference and the β terms of D̃ₓ, and Bx on the columns ia − 1 and ia
# through the x-average of D̃_y and D̃_z. The correction is the general total-field/scattered-field
# rule written once for every stencil: a B point p of region χ_p ∈ {0, 1} must see E_q − (χ_q − χ_p)·E_inc,q
# for every E point q of its stencil, so after the B update
#
#     B_c[p] ← B_c[p] + h·(∇̃×G)_c[p],   G_q = (χ_q − χ_p)·E_inc,q,
#
# the solver's own expression ([`faraday_update`](@ref)) evaluated on G, which is zero except where
# the stencil crosses the plane. At the Yee stencil this is antenna.jl's two B corrections.
# E_inc is needed on the columns ia − 1, ia and ia + 1 (x_a + mΔx, m = −1, 0, 1): the incident
# wave there is that of the plane a time m·kΔx later (k the M4 wavenumber of ω₀, as for the half
# cell of the E correction), with the beam at its own distance from the focus. The E correction is
# antenna.jl's (Ampère's law is Yee's), with the incident B scaled by M4's B/E ratio ρ at ω₀
# ([`AntennaPlane`](@ref)). Only the transverse incident E enters (Ex_inc = 0, as in antenna.jl).

# The incident E of component C (2 or 3) at the array index I as a TF/SF difference seen from a B
# point of region χp: (χ_q − χp)·E_inc at q = I, zero where q is in the B point's own region. `src`
# is what the laser's field at the plane is evaluated from: the time t for a GaussianLaser, the
# time factors of the three columns for a ShapedLaser (whose profile on the three columns is the
# pupil sum itself at N = 2 and the plane's table at N = 3). Indexed like an array by the stencil.
struct IncidentDifference{C,A,S}
    a::A
    src::S
    i0::Int32
    χp::Int32
end
IncidentDifference{C}(a::A, src::S, i0, χp) where {C,A,S} = IncidentDifference{C,A,S}(a, src, Int32(i0), Int32(χp))

"An E component the antenna does not drive (Ex): zero at every index."
struct NoIncident{T} end
@inline Base.getindex(::NoIncident{T}, I::CartesianIndex) where {T} = zero(T)

@inline function Base.getindex(G::IncidentDifference{C}, I::CartesianIndex) where {C}
    a = G.a
    T = typeof(a.xa)
    i = Int32(I[1])
    dq = Int32(i >= G.i0) - G.χp
    dq == Int32(0) && return zero(T)
    e = incident_e(a.laser, a, G.src, i - G.i0, Val(C), Base.tail(Tuple(I)))
    return ifelse(dq > Int32(0), e, -e)
end

# E_inc of component C at the column ia + m and the transverse array index J (rows, and layers
# at N = 3), from the laser's field at the plane: the beam at x_a + mΔx, a time m·kΔx later.
@inline function incident_e(l::GaussianLaser, a::Antenna, t, m::Int32, vc::Val{C}, J::NTuple{1}) where {C}
    T = typeof(a.xa)
    ξ = a.xa + T(m) * a.dx[1] - l.focus
    s = t - l.delay - T(m) * 2 * a.ht
    e, _, _ = laser_field(l, ξ, antenna_dy(a, Int32(J[1]), C == 2 ? one(T) / 2 : zero(T)), s)
    return polarized(l, e, vc)
end
@inline function incident_e(l::GaussianLaser, a::Antenna, t, m::Int32, vc::Val{C}, J::NTuple{2}) where {C}
    T = typeof(a.xa)
    ξ = a.xa + T(m) * a.dx[1] - l.focus
    s = t - l.delay - T(m) * 2 * a.ht
    h = one(T) / 2
    dy = antenna_d(a, 2, l.center[1], Int32(J[1]), C == 2 ? h : zero(T))
    dz = antenna_d(a, 3, l.center[2], Int32(J[2]), C == 3 ? h : zero(T))
    e, _, _, _ = laser_field(l, ξ, dy, dz, s)
    return polarized(l, e, vc)
end
@inline function incident_e(l::ShapedLaser, a::Antenna, g::NTuple{3}, m::Int32, vc::Val{C}, J::NTuple{1}) where {C}
    T = typeof(a.xa)
    ξ = a.xa + T(m) * a.dx[1] - l.focus
    e, _, _ = shaped_field(l, ξ, antenna_dy(a, Int32(J[1]), C == 2 ? one(T) / 2 : zero(T)), g[m + Int32(2)])
    return polarized(l, e, vc)
end
# A tabulated 3D ShapedLaser (zernike_antenna.jl): the table's U on the column ia + m at the place
# p (1: Ey's, 2: Ez's) of the row j, layer k, the first of the twelve numbers at the plane and an
# M4 table's own eight beside it, with the time factor of that column's time.
@inline function column_value(a::Antenna, p, m::Int32, j::Int32, k::Int32)
    V = a.laser.profile.values
    j0, k0 = plane_index(a, 2, j), plane_index(a, 3, k)
    o = ifelse(m == Int32(0), 6 * (p - 1) + 1, 4 * (p - 1) + ifelse(m < Int32(0), 13, 15))
    @inbounds return Complex(V[j0, k0, o], V[j0, k0, o + 1])
end
@inline incident_e(l::ShapedLaser, a::Antenna, g::NTuple{3}, m::Int32, vc::Val{C}, J::NTuple{2}) where {C} =
    polarized(l, real(column_value(a, C - 1, m, Int32(J[1]), Int32(J[2])) * g[m + Int32(2)]), vc)
@inline polarized(l, e, ::Val{2}) = e * cos(l.polarization)
@inline polarized(l, e, ::Val{3}) = e * sin(l.polarization)

# The incident differences seen from a B point of region χp, as the three E components.
@inline incident_set(a, src, i0, χp) =
    (NoIncident{typeof(a.xa)}(), IncidentDifference{2}(a, src, i0, χp), IncidentDifference{3}(a, src, i0, χp))

# B_c[I] += h·(∇̃×G)_c[I] for the B point I of region χp: faraday_update on G from zero is −h·(∇̃×G)_c.
@inline function tfsf_b!(B, a, src, i0, χp, I, hd, vc, vN)
    T = typeof(a.xa)
    G = incident_set(a, src, i0, χp)
    c = faraday_update(zero(T), G, I, hd, a.stencil, vc, vN)
    Bc = component(B, vc)
    @inbounds Bc[I] -= c
    return nothing
end

# One work item per row (N = 2) or (row, layer) (N = 3) of the plane, guards included: By and Bz on
# the columns ia − 2, ia − 1 (scattered) and ia (total), Bx on ia − 1 and ia.
@kernel function m4_antenna_b_kernel!(B, a, hd, src)
    J = @index(Global, Cartesian)
    vN = Val(length(J) + 1)
    i0 = a.g + a.ia
    for (col, χp) in ((i0 - Int32(2), Int32(0)), (i0 - Int32(1), Int32(0)), (i0, Int32(1)))
        I = CartesianIndex(Int(col), Tuple(J)...)
        tfsf_b!(B, a, src, i0, χp, I, hd, Val(2), vN)
        tfsf_b!(B, a, src, i0, χp, I, hd, Val(3), vN)
    end
    for (col, χp) in ((i0 - Int32(1), Int32(0)), (i0, Int32(1)))
        I = CartesianIndex(Int(col), Tuple(J)...)
        tfsf_b!(B, a, src, i0, χp, I, hd, Val(1), vN)
    end
end

# The B source of an M4 antenna after a B half step of length h to the time t: `c` is h/Δx (as the
# Yee kernels take it), the kernel needs h/Δ_d per axis.
function m4_b_source!(a::Antenna{<:Any,<:Any,N}, f, c, src, wgs) where {N}
    T = typeof(c)
    h = c * a.dx[1]
    hd = map(d -> h / d, a.dx)
    rows = ntuple(d -> size(f.Bz, d + 1), Val(N - 1))
    launcher(m4_antenna_b_kernel!, get_backend(f.Bz), wgs)((f.Bx, f.By, f.Bz), a, hd, src; ndrange = rows)
    return nothing
end

b_source!(a::Antenna{<:GaussianLaser,<:Any,2,<:M4Stencil}, f, c, t, wgs) = m4_b_source!(a, f, c, t, wgs)
b_source!(a::Antenna{<:GaussianLaser,<:Any,3,<:M4Stencil}, f, c, t, wgs) = m4_b_source!(a, f, c, t, wgs)
# A ShapedLaser's time factors on the columns ia − 1, ia and ia + 1 at the time t: the incident
# wave on the column ia + m is that of the plane a time m·kΔx later.
function column_time_factors(a::Antenna{<:ShapedLaser}, t::T) where {T}
    l = a.laser
    return ntuple(i -> time_factor(l, t - l.delay - T(i - 2) * 2 * a.ht), 3)
end
b_source!(a::Antenna{<:ShapedLaser,<:Any,2,<:M4Stencil}, f, c, t, wgs) = m4_b_source!(a, f, c, column_time_factors(a, t), wgs)
b_source!(a::Antenna{<:ShapedLaser,<:Any,3,<:M4Stencil}, f, c, t, wgs) = m4_b_source!(a, f, c, column_time_factors(a, t), wgs)
