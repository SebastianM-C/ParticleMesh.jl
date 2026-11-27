# A 3D animation of the frames of `scripts/lwfa/animate3d.jl` with a quarter of the volume cut out
# (y ≤ 0 and z ≥ 0 about the laser axis, the rows within HALF_WIDTH of it), rendered with GLMakie.
# On the two cut faces and on the outer faces that the camera sees: the electrons' density n_e/n₀
# in grey with the laser's E_y in red and blue over it. In the cut-out quarter: the laser as the
# two isosurfaces E_y = ±ISO_EY·max|E_y| (of the frame), the wake's electron sheath as a
# translucent isosurface n_e = ISO_NE·n₀, and the born electrons that the wake holds as points
# coloured by γ.
#
#   julia +1.13 --project=scripts/lwfa/render3d scripts/lwfa/render3d/render3d.jl <run dir> [out dir] [first frame]
#
# On a machine without a display, run it under `xvfb-run -s '-screen 0 1920x1080x24'` (Mesa's
# software OpenGL is enough). It reads `<run dir>/frames/frame*.h5` and writes into `out dir`
# (default `<run dir>/render`): `lwfa3d.mp4` (every frame), `lwfa3d.gif` (every 2nd frame, smaller,
# for the README) and the last frame as `lwfa3d_last.png` with its JSON sidecar (F3), and
# `render.toml` with the SHA-256 of the frames it read.
#
# GLMakie, not CairoMakie (F3's plotting package): CairoMakie does not draw 3D scenes with depth.
# This environment holds only the renderer, so that the scripts/ environment does not load
# OpenGL. The picture is not a validation; the run's case and its comparison with two other codes
# are in reports/wp-082-ionization-three-codes-3d.md.

using GLMakie, HDF5, JSON, SHA, Dates
using Meshing: isosurface, MarchingCubes
using GLMakie.GeometryBasics: Mesh, Point3f, TriangleFace

const NE_GREY = 3.0         # n_e/n₀ at full white on the cut faces
const EY_SHOW = 0.1         # |E_y|/a₀ below this leaves the faces' grey alone
const ISO_EY = 0.4          # the laser's isosurfaces, as a fraction of the frame's largest |E_y| (the pulse diffracts past its focus)
const ISO_NE = 2.0          # the sheath's isosurface, n_e/n₀
const GAMMA_RANGE = (3, 15) # the born electrons' colours
const HALF_WIDTH = 16.0     # the rows drawn, within this of the axis (c/ω₀); the frames hold 25

dequant(q, lo, hi) = lo .+ (hi - lo) .* (Float32.(q) ./ 255)

"Read frame file `path`: n_e/n₀ and E_y/a₀ within HALF_WIDTH of the axis, the born electrons and the attributes."
function read_frame(path)
    h5open(path, "r") do f
        a = Dict(k => read_attribute(f, k) for k in keys(HDF5.attributes(f)))
        ny, nz = size(f["ne"])[2:3]
        jw = min(round(Int, HALF_WIDTH / a["dy"]), ny ÷ 2)          # the rows within HALF_WIDTH of the axis, row ny ÷ 2 + 1
        kw = min(round(Int, HALF_WIDTH / a["dz"]), nz ÷ 2)
        J, K = (ny ÷ 2 + 1 - jw):(ny ÷ 2 + 1 + jw), (nz ÷ 2 + 1 - kw):(nz ÷ 2 + 1 + kw)
        a["y0"], a["z0"] = a["y0"] + (first(J) - 1) * a["dy"], a["z0"] + (first(K) - 1) * a["dz"]
        ne = dequant(read(f["ne"])[:, J, K], 0.0f0, Float32(a["ne_max"]))
        ey = dequant(read(f["Ey"])[:, J, K], -Float32(a["ey_max"]), Float32(a["ey_max"]))
        b(name) = (r = read_attribute(f[name], "range"); r[1] .+ (r[2] - r[1]) .* (Float32.(read(f[name])) ./ 65535))
        born = (; x = b("born_x"), y = b("born_y"), z = b("born_z"), gamma = b("born_gamma"))
        return (; ne, ey, born, a)
    end
end

laser_rgb(e) = e > 0 ? RGBf(1.0, 0.3, 0.12) : RGBf(0.2, 0.55, 1.0)

"The colour of a cut face: n_e/n₀ in grey, E_y in red and blue over it."
function face_color(ne, ey)
    g = clamp(ne / NE_GREY, 0, 1)^0.9
    base = RGBf(0.06 + 0.86g, 0.07 + 0.87g, 0.1 + 0.9g)
    α = clamp((abs(ey) - EY_SHOW) / 0.5, 0, 0.9)
    l = laser_rgb(ey)
    return RGBf((1 - α) * base.r + α * l.r, (1 - α) * base.g + α * l.g, (1 - α) * base.b + α * l.b)
end

"The isosurface `a = level` over the nodes `x, y, z` as a mesh (an empty one if there is none)."
function iso_mesh(a, level, x, y, z)
    vs, fs = isosurface(Float32.(a), MarchingCubes(; iso = Float32(level)), x, y, z)
    if isempty(fs)
        vs, fs = fill((0.0, 0.0, 0.0), 3), [(1, 2, 3)]
    end
    return Mesh(Point3f.(vs), [TriangleFace(f...) for f in fs])
end

"Coordinates of the frame's nodes: window x, and y, z from the axis."
function axes_of(fr)
    a = fr.a
    nx, ny, nz = size(fr.ne)
    x = a["x0"] .+ (0:nx - 1) .* a["dx"]
    y = a["y0"] .+ (0:ny - 1) .* a["dy"]
    z = a["z0"] .+ (0:nz - 1) .* a["dz"]
    return x, y, z
end

"The figure, and `show!(i)` that draws frame file `files[i]` into it."
function scene(files; size = (1280, 520))
    fr = read_frame(files[1])
    x, y, z = axes_of(fr)
    jc, kc = findfirst(>=(0), y), findfirst(>=(0), z)        # the axis's node row: the cut faces
    Jq, Kq = 1:jc, kc:length(z)                               # the cut-out quarter: y ≤ 0, z ≥ 0, facing a camera on the −y side
    cut_q(j, k) = (j < jc) & (k > kc)
    xr, yq, zq = range(first(x), last(x); length = length(x)), range(first(y), y[jc]; length = jc), range(z[kc], last(z); length = length(Kq))
    u = size[1] / 1280                                        # the labels scale with the figure
    fig = Figure(; size, backgroundcolor = :black, figure_padding = 0, fontsize = 14u)
    ax = Axis3(fig[1, 1]; aspect = :data, viewmode = :fitzoom, azimuth = 1.36π, elevation = 0.21π, perspectiveness = 0.4,
               backgroundcolor = :black, protrusions = 0)
    hidedecorations!(ax)
    hidespines!(ax)
    # the cut faces: the plane z = 0 over y ≤ 0 and the plane y = 0 over z ≥ 0
    face_z = Observable(zeros(RGBf, length(x), length(yq)))
    face_y = Observable(zeros(RGBf, length(x), length(zq)))
    surface!(ax, x, yq, zeros(length(x), length(yq)); color = face_z, shading = NoShading)
    surface!(ax, [xi for xi in x, _ in zq], zeros(length(x), length(zq)), [zi for _ in x, zi in zq]; color = face_y, shading = NoShading)
    # the outer faces of the three kept quarters that the camera sees: the top (z = z_max, y ≥ 0), the side
    # (y = y_min, z ≤ 0) and the window's back end (x = x_min), so that the cut reads as a notch in a block
    Jk, Kk = jc:length(y), 1:kc
    top = Observable(zeros(RGBf, length(x), length(Jk)))
    side = Observable(zeros(RGBf, length(x), length(Kk)))
    back = Observable(zeros(RGBf, length(y), length(z)))
    surface!(ax, x, y[Jk], fill(last(z), length(x), length(Jk)); color = top, shading = NoShading)
    surface!(ax, [xi for xi in x, _ in Kk], fill(first(y), length(x), length(Kk)), [z[k] for _ in x, k in Kk]; color = side, shading = NoShading)
    surface!(ax, fill(first(x), length(y), length(z)), [yj for yj in y, _ in z], [zk for _ in y, zk in z]; color = back, shading = NoShading)
    laser_p, laser_n, sheath = (Observable(iso_mesh(zeros(Float32, 2, 2, 2), 1, 0:1, 0:1, 0:1)) for _ in 1:3)
    mesh!(ax, laser_p; color = laser_rgb(1), transparency = true, alpha = 0.85)
    mesh!(ax, laser_n; color = laser_rgb(-1), transparency = true, alpha = 0.85)
    mesh!(ax, sheath; color = RGBAf(0.65, 0.8, 1.0, 0.18), transparency = true)
    pts = Observable(Point3f[])
    gam = Observable(Float32[])
    scatter!(ax, pts; color = gam, colormap = :plasma, colorrange = GAMMA_RANGE, markersize = 3.5u)
    title = Observable("")
    Label(fig[1, 1], title; tellwidth = false, tellheight = false, halign = :left, valign = :top, color = :white, fontsize = 20u,
          padding = (24u, 0, 0, 18u))
    Label(fig[2, 1], "ParticleMesh.jl · 3D ionization injection · electron density (grey), laser E_y (red, blue), born electrons by γ (purple $(GAMMA_RANGE[1]) to yellow $(GAMMA_RANGE[2]))";
          tellwidth = false, halign = :left, color = (:white, 0.7), fontsize = 14u, padding = (24u, 0, 14u, 0))
    limits!(ax, first(x), last(x), first(y), last(y), first(z), last(z))
    function show!(i)
        f = read_frame(files[i])
        ne, ey = f.ne, f.ey
        face_z[] = face_color.(ne[:, Jq, kc], ey[:, Jq, kc])
        face_y[] = face_color.(ne[:, jc, Kq], ey[:, jc, Kq])
        top[] = face_color.(ne[:, Jk, end], ey[:, Jk, end])
        side[] = face_color.(ne[:, 1, Kk], ey[:, 1, Kk])
        back[] = [cut_q(j, k) ? RGBf(0, 0, 0) : face_color(ne[1, j, k], ey[1, j, k]) for j in eachindex(y), k in eachindex(z)]
        q = (:, Jq, Kq)
        e = ISO_EY * maximum(abs, ey)
        laser_p[] = iso_mesh(ey[q...], e, xr, yq, zq)
        laser_n[] = iso_mesh(.-ey[q...], e, xr, yq, zq)
        sheath[] = iso_mesh(ne[q...], ISO_NE, xr, yq, zq)
        b = f.born
        inq = (b.y .<= f.a["dy"] / 2) .& (b.z .>= -f.a["dz"] / 2) .&      # in the cut-out quarter: the rest is behind the faces
              (b.y .>= first(y)) .& (b.z .<= last(z))
        pts[] = Point3f.(b.x[inq], b.y[inq], b.z[inq])
        gam[] = b.gamma[inq]
        title[] = "t = $(round(Int, f.a["t"]))/ω₀"
        return f
    end
    return fig, show!
end

function render(run, out = joinpath(run, "render"); first_frame = 1, fps = 20)
    mkpath(out)
    files = sort(filter(endswith(".h5"), readdir(joinpath(run, "frames"); join = true)))[first_frame:end]
    fig, show! = scene(files)
    record(show!, fig, joinpath(out, "lwfa3d.mp4"), eachindex(files); framerate = fps)
    last = show!(length(files))
    save(joinpath(out, "lwfa3d_last.png"), fig)
    open(joinpath(out, "lwfa3d_last.json"), "w") do io
        JSON.print(io, Dict("source" => basename(files[end]), "t" => last.a["t"], "step" => last.a["step"],
                            "born_count" => last.a["born_count"], "born_in_file" => length(last.born.x),
                            "ne_grey_max" => NE_GREY, "ey_faces_above" => EY_SHOW, "iso_Ey_over_frame_max" => ISO_EY,
                            "iso_ne_over_n0" => ISO_NE, "gamma_colorrange" => collect(GAMMA_RANGE), "cut" => "y <= 0 and z >= 0 (from the axis)"), 2)
    end
    small, show_small! = scene(files; size = (800, 324))
    record(show_small!, small, joinpath(out, "lwfa3d.gif"), 1:2:length(files); framerate = fps ÷ 2)
    open(joinpath(out, "render.toml"), "w") do io
        repo = joinpath(@__DIR__, "..", "..", "..")
        println(io, "generator = \"scripts/lwfa/render3d/render3d.jl\"")
        println(io, "commit = \"$(readchomp(`git -C $repo rev-parse HEAD`))\"")
        println(io, "dirty = $(!isempty(readchomp(`git -C $repo status --porcelain --untracked-files=no`)))")
        println(io, "julia = \"$VERSION\"")
        println(io, "updated = \"$(now(UTC))Z\"")
        println(io, "first_frame = $first_frame")
        println(io, "\n[inputs]")
        for f in files
            println(io, "\"frames/$(basename(f))\" = \"sha256:$(bytes2hex(open(sha256, f)))\"")
        end
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    run = ARGS[1]
    render(run, get(ARGS, 2, joinpath(run, "render")); first_frame = parse(Int, get(ARGS, 3, "1")))
end
