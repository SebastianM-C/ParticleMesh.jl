# The round beam of a Zernike pupil on the 3D antenna (WP-085, A2, A3): the pupil sum of a
# ZernikeBeam costs a few thousand plane waves per point, too many for every row of the plane in
# every sub-step, so its profile on the plane is a table, built once per parameter point by two
# kernels (the pupil's samples, then their sum at every staggered place of the plane) and read by
# the antenna's kernels with the sub-step's time factor, as the 2D ShapedLaser's rows are. The
# table is a parameter of the run (the laser's profile, A3): a Dual or an Enzyme shadow of the
# coefficients reaches the step through it, and Enzyme's adjoint of the table is the sensitivity of
# an objective to the field at every point of the plane.

"""
    PlaneTable{T}

The profile of a [`ZernikeBeam`](@ref) on the antenna plane it was built for, the profile of a
tabulated 3D [`ShapedLaser`](@ref) ([`tabulate`](@ref)): `beam` the description, `focus` the
laser's focus, `plane` the [`AntennaPlane`](@ref), `center` the laser's axis, and `values` an
array of the state's backend of size (n_y, n_z, 12), per interior row and layer of the plane the
real and imaginary parts of six complex numbers: at the place of Ey (y Center, z Node) the profile
`U` at ξ = x_a − focus, `U − U_yy/2 + U_zz/2` and `U_yz` at ξ = x_a − Δx/2 − focus; at the place of
Ez (y Node, z Center) `U` at the first ξ, `U + U_yy/2 − U_zz/2` and `U_yz` at the second, each of
the nearest image of the axis, and on a row or layer within Δ/1000 of the antipode the mean of the
two images (the antenna's antipode rule, which for an even beam zeroes the odd cross term). Its numbers ([`PulseShapes.parameter_names`](@ref)) are
the beam's; replacing them gives the untabulated laser, which the step refuses until it is
tabulated again.
"""
struct PlaneTable{T,B<:ZernikeBeam,P<:AntennaPlane,C,A<:AbstractArray{T,3}} <: TransverseProfile
    beam::B
    focus::T
    plane::P
    center::C
    values::A
end

Adapt.adapt_structure(to, t::PlaneTable) = PlaneTable(t.beam, t.focus, t.plane, t.center, Adapt.adapt(to, t.values))
Adapt.adapt_structure(to, l::ShapedLaser) =
    ShapedLaser(l.a0, l.delay, l.focus, l.polarization, l.phase, Adapt.adapt(to, l.profile), l.envelope, l.position, l.center)
Adapt.adapt_structure(to, a::Antenna) = Antenna(Adapt.adapt(to, a.laser), a.ia, a.xa, a.dx, a.g, a.n, a.len, a.hx, a.ht, a.stencil, a.ρ)

# The description of a profile: the beam a table was built from, or the profile itself.
table_beam(b::TransverseProfile) = b
table_beam(t::PlaneTable) = t.beam

PulseShapes.transverse_dimensions(::PlaneTable) = 2
PulseShapes.parameter_names(t::PlaneTable) = parameter_names(t.beam)
PulseShapes.parameter_values(t::PlaneTable) = parameter_values(t.beam)
PulseShapes.with_parameter_values(t::PlaneTable, v::Tuple) = with_parameter_values(t.beam, v)
PulseShapes.convert_eltype(::Type{T}, t::PlaneTable{T}) where {T} = t
PulseShapes.convert_eltype(::Type{S}, t::PlaneTable) where {S} = convert_eltype(S, t.beam)
PulseShapes.profile_moments(t::PlaneTable) = profile_moments(t.beam)

# The pupil's samples, one work item each: (pr, pi, κy, κz, wa) of `pupil_sample` in the columns of
# P, the samples of one radial node consecutive (the order of `profile`'s sum).
@kernel function zernike_pupil_kernel!(P, b)
    q = Int32(@index(Global, Linear))
    T = eltype(P)
    i = (q - Int32(1)) ÷ b.angular + Int32(1)
    k = (q - Int32(1)) % b.angular + Int32(1)
    ρ, wρ = radial_node(b.radial, i, T)
    pr, pim, κy, κz, wa = pupil_sample(b, ρ, wρ, k)
    @inbounds begin
        P[q, 1] = pr
        P[q, 2] = pim
        P[q, 3] = κy
        P[q, 4] = κz
        P[q, 5] = wa
    end
end

# The pupil sum at the transverse offsets (dy, dz) for the two distances ξe (the incident E, at the
# plane) and ξb (the incident B, half a cell behind): (U(ξe), the B combination at ξb with the sign
# `sgn` of its U_yy − U_zz term, U_yz(ξb)), over the sum of the weights, as `profile` sums them.
@inline function table_sums(P, nq::Int32, ξe, ξb, dy, dz, sgn)
    T = eltype(P)
    ue = ub = uyy = uzz = uyz = zero(Complex{T})
    s = zero(T)
    for q in Int32(1):nq
        @inbounds pr, pim, κy, κz, wa = P[q, 1], P[q, 2], P[q, 3], P[q, 4], P[q, 5]
        ue += pupil_terms(pr, pim, κy, κz, ξe, dy, dz)[1]
        tb = pupil_terms(pr, pim, κy, κz, ξb, dy, dz)
        ub += tb[1]
        uyy += tb[2]
        uzz += tb[3]
        uyz += tb[4]
        s += wa
    end
    ue, ub, uyy, uzz, uyz = ue / s, ub / s, uyy / s, uzz / s, uyz / s
    um = ub - sgn * (uyy - uzz) / 2
    return real(ue), imag(ue), real(um), imag(um), real(uyz), imag(uyz)
end

# The other image of the axis across the seam, and whether the offset d lies within Δ/1000 of the
# antipode (antipode_weight's band), where the two images are equally near.
@inline seam_image(plane, axis, d) = ifelse(d > 0, d - plane.len[axis], d + plane.len[axis]), iszero(antipode_weight(plane, axis, d))

# table_sums at the nearest image of the axis, and the mean over both images along an axis whose
# offset lies on the antipode (four at a corner): the branch is taken only on those rows and
# layers, and the band is wide against roundoff, so every backend takes the same images.
@inline function image_sums(P, nq::Int32, ξe, ξb, plane, dy, dz, sgn)
    T = eltype(P)
    ya, iy = seam_image(plane, 2, dy)
    za, iz = seam_image(plane, 3, dz)
    a = table_sums(P, nq, ξe, ξb, dy, dz, sgn)
    if iy
        a = a .+ table_sums(P, nq, ξe, ξb, ya, dz, sgn)
    end
    if iz
        a = a .+ table_sums(P, nq, ξe, ξb, dy, za, sgn)
        if iy
            a = a .+ table_sums(P, nq, ξe, ξb, ya, za, sgn)
        end
    end
    n = T((1 + iy) * (1 + iz))
    return a ./ n
end

# One work item per interior (row, layer) of the plane: the twelve numbers of PlaneTable.
@kernel function zernike_table_kernel!(V, @Const(P), plane, center, ξe, ξb, nq)
    J = @index(Global, Cartesian)
    j, k = Int32(J[1]), Int32(J[2])
    T = eltype(V)
    h = one(T) / 2
    jr, kr = plane.g + j, plane.g + k
    y1, z1 = antenna_d(plane, 2, center[1], jr, h), antenna_d(plane, 3, center[2], kr, zero(T))
    y2, z2 = antenna_d(plane, 2, center[1], jr, zero(T)), antenna_d(plane, 3, center[2], kr, h)
    a = image_sums(P, nq, ξe, ξb, plane, y1, z1, one(T))
    b = image_sums(P, nq, ξe, ξb, plane, y2, z2, -one(T))
    @inbounds ntuple(m -> (V[j, k, m] = a[m]; V[j, k, m + 6] = b[m]; nothing), Val(6))
end

"""
    tabulate(state, params) -> params
    tabulate(laser, plane::AntennaPlane, backend; workgroupsize = nothing) -> laser

The parameters with the profile of every 3D [`ShapedLaser`](@ref) of a [`ZernikeBeam`](@ref)
replaced by its [`PlaneTable`](@ref) on the state's antenna plane and backend, the other lasers
as they are; a laser that is tabulated already is tabulated again from its beam (so that a
changed focus is taken up). Two kernels: one work item per pupil sample (radial × angular), then
one per interior (row, layer) of the plane, each a sum over every sample. Call it once per
parameter point, before [`initialize!`](@ref) and the steps (`initialize!(state, problem)` calls
it); inside a function that is differentiated, it carries the derivatives of the beam's numbers
and of the focus into the table. It refuses a beam whose [`PulseShapes.quadrature_radius`](@ref)
is less than the half diagonal of the box's cross-section.
"""
function tabulate(st::EMState, p::EMParams)
    eltype(p) === eltype(st.ρ) ||
        throw(ArgumentError("the parameters have element type $(eltype(p)), the state $(eltype(st.ρ)): build the state from a problem of the parameters' element type"))
    planes = antenna_planes(st)
    length(p.lasers) == length(planes) ||
        throw(ArgumentError("$(length(p.lasers)) lasers in the parameters, $(length(planes)) antenna planes in the state"))
    isempty(p.lasers) && return p
    be = get_backend(st.ρ)
    lasers = map((l, a) -> tabulate(l, a, be; workgroupsize = st.numerics.workgroupsize), p.lasers, planes)
    return EMParams(p; lasers)
end

tabulate(l, ::AntennaPlane, backend; workgroupsize = nothing) = l
tabulate(l::ShapedLaser{<:Any,<:Any,<:PlaneTable}, a::AntennaPlane, backend; workgroupsize = nothing) =
    tabulate(shaped_laser(eltype(l.a0), typeof(l.position), (l.a0, l.delay, l.focus, l.polarization, l.phase), l.profile.beam, l.envelope,
                          l.position, l.center), a, backend; workgroupsize)

function tabulate(l::ShapedLaser{T,<:Any,<:ZernikeBeam}, a::AntennaPlane{S,3}, backend; workgroupsize = nothing) where {T,S}
    T === S || throw(ArgumentError("the laser's element type $T is not the antenna plane's $S"))
    b = l.profile
    check_quadrature(a, b)
    nq = Int(b.radial) * Int(b.angular)
    P = KernelAbstractions.allocate(backend, T, nq, 5)
    launcher(zernike_pupil_kernel!, backend, workgroupsize)(P, b; ndrange = nq)
    V = KernelAbstractions.allocate(backend, T, Int(a.n[2]), Int(a.n[3]), 12)
    c = center_as(T, l.center)
    launcher(zernike_table_kernel!, backend, workgroupsize)(V, P, a, c, a.xa - l.focus, a.xa - a.hx - l.focus, Int32(nq);
                                                            ndrange = (Int(a.n[2]), Int(a.n[3])))
    t = PlaneTable(b, l.focus, a, c, V)
    return ShapedLaser{T,typeof(l.position),typeof(t),typeof(l.envelope),typeof(l.center)}(l.a0, l.delay, l.focus, l.polarization,
                                                                                         l.phase, t, l.envelope, l.position, l.center)
end

function check_quadrature(a::AntennaPlane, b::ZernikeBeam)
    r = sqrt(a.len[2]^2 + a.len[3]^2) / 2
    q = quadrature_radius(b)
    q >= r ||
        throw(ArgumentError("the ZernikeBeam's pupil sum resolves the beam and its wavefront to $q from its axis, less than the half diagonal $r of the box's cross-section: raise its radial nodes or angular points (PulseShapes.quadrature_radius)"))
    return nothing
end

# A 3D ShapedLaser reaches the step through its table: refused untabulated, or with a table made
# for another plane, center or focus (a host comparison of host scalars and types, §0.4 rule 2).
function check_profile(a::AntennaPlane{<:Any,3}, l::ShapedLaser)
    a.stencil isa YeeStencil ||
        throw(ArgumentError("a ShapedLaser's antenna in 3D is implemented for the Yee solver only"))
    t = l.profile
    t isa PlaneTable ||
        throw(ArgumentError("the 3D ShapedLaser's $(nameof(typeof(t))) reaches the antenna through the table of its plane: call `params = tabulate(state, params)` once per parameter point"))
    t.plane === a && t.center === l.center && t.focus === l.focus ||
        throw(ArgumentError("the 3D ShapedLaser's table was built for another plane, center or focus than the laser's: tabulate it again (`tabulate(state, params)`)"))
    return nothing
end

# The interior index (1 to n) of the row j (any index, guards included) along `axis`.
@inline plane_index(a::Antenna, axis, j::Int32) = mod(j - a.g - Int32(1), a.n[axis]) + Int32(1)

# The table's complex number m (1 to 6) at the place p (1: Ey's, 2: Ez's) of the row j, layer k.
@inline function table_value(a::Antenna, p, m, j::Int32, k::Int32)
    V = a.laser.profile.values
    j0, k0 = plane_index(a, 2, j), plane_index(a, 3, k)
    o = 6 * (p - 1) + 2 * m - 1
    @inbounds return Complex(V[j0, k0, o], V[j0, k0, o + 1])
end

# antenna.jl's 3D rows with the table and the time factor `g` of the sub-step's time in place of the
# laser and the time: after a B half step Ey_inc = Re[U·g]·cos θ and Ez_inc = Re[U·g]·sin θ at x_a,
# after the E step Bz_inc = bp·cos θ − byz·sin θ and By_inc = byz·cos θ − bs·sin θ half a cell behind.
@inline function antenna_b_rows(a::Antenna{<:ShapedLaser,<:Any,3}, j::Int32, k::Int32, g, ::Type{T}) where {T}
    l = a.laser
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    return real(table_value(a, 1, 1, j, k) * g) * cθ, real(table_value(a, 2, 1, j, k) * g) * sθ
end

@inline function antenna_e_rows(a::Antenna{<:ShapedLaser,<:Any,3}, j::Int32, k::Int32, g, ::Type{T}) where {T}
    l = a.laser
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    bp, xy = real(table_value(a, 1, 2, j, k) * g), real(table_value(a, 1, 3, j, k) * g)
    bs, xz = real(table_value(a, 2, 2, j, k) * g), real(table_value(a, 2, 3, j, k) * g)
    return bp * cθ - xy * sθ, bs * sθ - xz * cθ
end

b_source!(a::Antenna{<:ShapedLaser,<:Any,3,YeeStencil}, f, c, t, wgs) =
    launcher(antenna_b_kernel_3d!, get_backend(f.Bz), wgs)(f.By, f.Bz, a, c, time_factor(a.laser, t - a.laser.delay);
                                                           ndrange = (size(f.Bz, 2), size(f.Bz, 3)))
e_source!(a::Antenna{<:ShapedLaser,<:Any,3,YeeStencil}, f, c, t, wgs) =
    launcher(antenna_e_kernel_3d!, get_backend(f.Ey), wgs)(f.Ey, f.Ez, a, c, time_factor(a.laser, t + a.ht - a.laser.delay);
                                                           ndrange = (size(f.Ey, 2), size(f.Ey, 3)))

"""
    launched_energy(laser::ShapedLaser) -> W

The energy that the antenna drives through the box's cross-section for a tabulated 3D laser
([`tabulate`](@ref)): the incident field's paraxial flux over the plane's places,
(a0²/2)·∫|A|² ds·ΔyΔz·Σ (cos²θ·|U|² at Ey's places + sin²θ·|U|² at Ez's), from the table. With the
whole beam inside the cross-section it is [`laser_energy`](@ref) to the plane's sampling of the
profile; a wavefront whose wings leave the cross-section launches less, by the fraction of the
beam outside, and the antenna drives the field of the nearest image of the axis there.
"""
function launched_energy(l::ShapedLaser{<:Any,<:Any,<:PlaneTable})
    t = l.profile
    V = t.values
    sθ, cθ = sin(l.polarization), cos(l.polarization)
    ey = sum(abs2, view(V, :, :, 1:2))
    ez = sum(abs2, view(V, :, :, 7:8))
    return l.a0^2 / 2 * envelope_energy(l.envelope) * t.plane.dx[2] * t.plane.dx[3] * (cθ^2 * ey + sθ^2 * ez)
end
