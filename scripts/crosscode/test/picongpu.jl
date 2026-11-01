# Tests of the PIConGPU deck generator and reader (WP-017). Included by runtests.jl, which
# defines CC and loads SmileiDeck, SmileiRead and Observables.
include(joinpath(CC, "picongpu.jl"))
using .PIConGPUDeck
using .PIConGPUDeck: picongpu_density, background_field, deposit_form_factor, tsc
using .PIConGPURead
using SHA: sha256

# `constexpr float_64 NAME = <value>;` in a generated .param file.
param_number(text, name) =
    parse(Float64, match(Regex("\\b$name\\s*=\\s*([-+0-9.eE]+)"), text).captures[1])

@testset "T-250 unit conversion to PIConGPU [A1, WP-017]" begin
    # Hand-computed SI values for cases/landau.toml at λ₀ = 0.8 µm with PIConGPU 0.8.0's
    # constants (include/picongpu/param/physicalConstants.param): c = 2.99792458e8 m/s,
    # μ₀ = 1.25663706127e-6, ε₀ = 1/(μ₀c²), m_e = 9.1093837139e-31 kg, e = 1.602176634e-19 C.
    # ω₀ = 2πc/λ₀; c/ω₀ = λ₀/2π; n_c = ε₀ m_e ω₀²/e².
    c0, μ0, me, qe = 2.99792458e8, 1.25663706127e-6, 9.1093837139e-31, 1.602176634e-19
    ε0 = 1 / (μ0 * c0^2)
    λ0 = 0.8e-6
    ω0 = 2π * c0 / λ0
    nc = ε0 * me * ω0^2 / qe^2
    @test nc ≈ 1.7419e27 rtol = 1e-4                  # the textbook 1.1e21 cm⁻³ µm² / λ²
    case = load_case("landau")
    p = picongpu_parameters(case)
    @test p.lambda0 == λ0
    @test p.units.omega0 ≈ ω0 rtol = 1e-15
    @test p.units.density ≈ nc rtol = 1e-14
    # Δx = (2π/10)/64 c/ω₀ = (π/320)(λ₀/2π) = λ₀/640 = 1.25e-9 m, exactly
    @test p.cell_size_SI[1] ≈ 1.25e-9 rtol = 1e-14
    # quasi-1D: the transverse cell is 8 Δx (the 2D Courant limit with Δt = 0.95 Δx), 8 cells
    @test p.quasi1d && p.cells == (64, 8)
    @test p.cell_size_SI[2] ≈ 8 * 1.25e-9 rtol = 1e-14
    @test p.cell_size_SI[3] == p.cell_size_SI[1]      # the depth of a 2D cell
    # Δt = 0.0093266/ω₀ = 0.0093266 λ₀/(2πc)
    @test p.dt_SI ≈ 0.0093266 * λ0 / (2π * c0) rtol = 1e-14
    @test p.steps == 2681
    # density 1 n_c; T = m v_th² = 0.0025 m_e c² = 0.0025 × 510.99895 keV
    e, ion = p.species
    @test e.density_SI ≈ nc rtol = 1e-14
    @test e.temperature_keV ≈ 0.0025 * me * c0^2 / qe / 1e3 rtol = 1e-14
    @test e.temperature_keV ≈ 1.2774974 rtol = 1e-7
    @test e.charge_ratio == 1.0 && e.mass_ratio == 1.0 && e.mobile
    @test ion.charge_ratio == -1.0 && ion.mass_ratio == 1836.0 && !ion.mobile
    # the perturbation: 1 % at mode 1 of 64 cells, k = 10 ω₀/c = 2π/(λ₀/10)
    @test e.profile.amplitude == 0.01 && e.profile.mode == (1, 0) && e.profile.cells == (64, 8)
    k_SI = 2π * e.profile.mode[1] / (64 * p.cell_size_SI[1])
    @test k_SI ≈ 10 * ω0 / c0 rtol = 1e-14
    @test ion.profile === nothing
    # the reference wavelength is recorded and can be set by the case
    @test picongpu_parameters(case; lambda0 = 1.0e-6).units.omega0 ≈ 2π * c0 / 1.0e-6 rtol = 1e-15

    # The generated files carry these values.
    files = param_files(case)
    sim = files["simulation.param"]
    @test param_number(sim, "DELTA_T_SI") ≈ p.dt_SI rtol = 1e-15
    @test param_number(sim, "CELL_WIDTH_SI") ≈ 1.25e-9 rtol = 1e-14
    @test param_number(sim, "CELL_HEIGHT_SI") ≈ 1.0e-8 rtol = 1e-14
    @test param_number(sim, "BASE_DENSITY_SI") ≈ nc rtol = 1e-14
    @test occursin("precision64Bit;", files["precision.param"])
    @test occursin("precision32Bit;", param_files(case; precision = "Float32")["precision.param"])
    @test occursin("#define SIMDIM DIM2", files["dimension.param"])
    part = files["particle.param"]
    @test param_number(part, "temperature") ≈ e.temperature_keV rtol = 1e-15
    dens = files["density.param"]
    @test occursin("0.01", dens)
    @test occursin(r"constexpr float_64 MX_electron = 1;", dens)

    # The matching choices of A5 are written out explicitly.
    sp = files["species.param"]
    @test occursin("particles::shapes::TSC", sp)
    @test occursin("currentSolver::Esirkepov<UsedParticleShape>", sp)
    @test occursin("particles::pusher::Boris", sp)
    @test occursin("maxwellSolver::Yee", files["fieldSolver.param"])
    @test occursin("--currentInterpolation none", run_arguments(p))
    @test occursin("seed::FromEnvironment", files["random.param"])
    # immobile species are neither pushed nor deposit current
    defs = files["speciesDefinition.param"]
    ionflags = match(r"ParticleFlags_ion = MakeSeq_t<(.*?)>;"s, defs).captures[1]
    @test !occursin("particlePusher", ionflags) && !occursin("current<", ionflags)
    eflags = match(r"ParticleFlags_electron = MakeSeq_t<(.*?)>;"s, defs).captures[1]
    @test occursin("particlePusher<UsedParticlePusher>", eflags) && occursin("current<UsedParticleCurrentSolver>", eflags)

    # Two-stream: the cold beams get the drift γ = 1/√(1 − v²) along ∓x and no temperature.
    ts = picongpu_parameters(load_case("two-stream"))
    bm, bp = ts.species
    @test bm.drift.gamma ≈ 1 / sqrt(1 - 0.01) rtol = 1e-15
    @test bm.drift.direction == (-1.0, 0.0, 0.0) && bp.drift.direction == (1.0, 0.0, 0.0)
    @test bm.temperature_keV == 0.0
end

@testset "T-251 the oblique profile in both generators [A5, WP-017]" begin
    c = load_case("landau-oblique-2d")
    p = picongpu_parameters(c)
    @test !p.quasi1d && p.cells == (128, 128)
    sm = SmileiDeck.smilei_parameters(c)
    (Nx, Ny), (Lx, Ly) = sm.number_of_cells, sm.grid_length
    for (s, ss) in zip(p.species, sm.species)
        for i in 0:Nx-1, j in 0:Ny-1
            # PIConGPU: the generated functor at the position FreeFormulaImpl passes (the cell
            # centre, in SI), divided by n_c; Smilei: its profile at the cell centre.
            npic = picongpu_density(p, s, i, j) / p.units.density
            x, y = (i + 0.5) * Lx / Nx, (j + 0.5) * Ly / Ny
            nsm = ss.profile === nothing ? ss.density :
                ss.profile.base + ss.profile.amplitude * cos(2π * (ss.profile.mode[1] * x / ss.profile.lengths[1] +
                                                                    ss.profile.mode[2] * y / ss.profile.lengths[2]))
            abs(npic - nsm) <= 8eps() || (@test abs(npic - nsm) <= 8eps(); break)
        end
        @test true
    end
    # The functor reads the mode, the cell counts and the amplitude of the case.
    dens = param_files(c)["density.param"]
    @test occursin("MX_electron = 2;", dens) && occursin("MY_electron = 1;", dens)
    @test occursin("NX_electron = 128;", dens) && occursin("NY_electron = 128;", dens)
    @test param_number(dens, "AMPLITUDE_electron") == 0.01
end

@testset "T-254 quasi-1D equivalence [A5, WP-017]" begin
    for name in ("landau", "two-stream", "two-stream-256")
        case = load_case(name)
        v = case_view(case)
        p = picongpu_parameters(case)
        files = param_files(case)
        args = run_arguments(p)
        # read back from the generated files: grid, lattice, density, perturbation
        g = match(r"-g (\d+) (\d+)", args)
        Nx, Ny = parse(Int, g[1]), parse(Int, g[2])
        @test Nx == v.cells[1]
        @test occursin("--periodic 1 1", args)
        @test param_number(files["simulation.param"], "CELL_WIDTH_SI") * p.units.omega0 / PIConGPUDeck.C_SI ≈
              v.lengths[1] / v.cells[1] rtol = 1e-14
        @test param_number(files["simulation.param"], "DELTA_T_SI") * p.units.omega0 ≈ case["time"]["dt"] rtol = 1e-14
        @test occursin("-s $(round(Int, case["time"]["t_end"] / case["time"]["dt"])) ", args)
        nc = param_number(files["simulation.param"], "BASE_DENSITY_SI")
        part = files["particle.param"]
        dens = files["density.param"]
        for (s, sv) in zip(case["species"], v.species)
            n = s["name"]
            lat = match(Regex("QuietParam_$n\\s*\\{\\s*using numParticlesPerDimension = mCT::shrinkTo<mCT::Int<(\\d+), (\\d+), 1>"), part)
            nx, ny = parse(Int, lat[1]), parse(Int, lat[2])
            if s["mobile"]
                # the macro-particle count per x-column equals the 1D count per cell
                @test nx * ny * Ny == sv.particles_per_cell[1]
                @test ny == 1
            end
            @test param_number(dens, "BASE_$n") * nc / p.units.density ≈ sv.density rtol = 1e-14
            if sv.mode === nothing
                @test param_number(dens, "AMPLITUDE_$n") == 0.0
            else
                @test param_number(dens, "AMPLITUDE_$n") == sv.amplitude
                @test param_number(dens, "MX_$n") == sv.mode[1] && param_number(dens, "MY_$n") == 0
                @test param_number(dens, "NX_$n") == Nx
            end
            # the density is uniform in y: equal at every row of a column
            ps = only(x for x in p.species if x.name == n)
            @test all(picongpu_density(p, ps, 5, j) == picongpu_density(p, ps, 5, 0) for j in 0:Ny-1)
        end
        # the transverse cells keep the 1D time step inside the 2D Courant limit
        Δx, Δy = p.cell_size_SI[1], p.cell_size_SI[2]
        @test PIConGPUDeck.C_SI * p.dt_SI < 1 / sqrt(1 / Δx^2 + 1 / Δy^2)
        @test p.transverse.cells == Ny && p.transverse.ratio == round(Int, Δy / Δx)
    end
    # A schema-2 case is not quasi-1D.
    @test picongpu_parameters(load_case("landau-oblique-2d")).transverse === nothing
end

"""
The charge density at the nodes (e n_c) of PIConGPU's load, by brute force: Quiet positions
((j + ½)/n per axis, QuietImpl.hpp), weights from the profile at the cell centre
(FreeFormulaImpl.hpp), TSC assignment to the nodes (TSC.hpp).
"""
function brute_force_charge(p)
    Nx, Ny = p.cells
    ρ = zeros(Nx, Ny)
    for s in p.species
        q = -s.charge_ratio                 # charge in e (PIConGPU's ratio is to the electron's)
        l = s.lattice
        wx = [[tsc((a + 0.5) / l[1] - dx) for dx in -1:2] for a in 0:l[1]-1]
        wy = [[tsc((b + 0.5) / l[2] - dy) for dy in -1:2] for b in 0:l[2]-1]
        Kx = sum(wx); Ky = sum(wy)          # summed over the lattice: the in-cell kernel per node offset
        for i in 0:Nx-1, j in 0:Ny-1
            w = q * picongpu_density(p, s, i, j) / p.units.density / prod(l)
            for (ix, dx) in enumerate(-1:2), (iy, dy) in enumerate(-1:2)
                ρ[mod(i + dx, Nx) + 1, mod(j + dy, Ny) + 1] += w * Kx[ix] * Ky[iy]
            end
        end
    end
    return ρ
end

@testset "T-255 the initial field closes Gauss's law on PIConGPU's loaded charge [A5, WP-017]" begin
    # PIConGPU solves no Poisson equation at t = 0. The generator supplies the electrostatic
    # field of the loaded charge as a static FieldBackgroundE (seen by the pusher, added to the
    # grid field that the dumps and energy plugins see). Check it against a brute-force load:
    # Quiet positions ((j + ½)/n per axis, QuietImpl.hpp), weights from the profile at the cell
    # centre (FreeFormulaImpl.hpp), TSC charge assignment at the nodes (TSC.hpp), E_x at
    # (i + ½, j) and E_y at (i, j + ½) (YeeCell.hpp); then (∇·E)(node) = ρ(node).
    @test sum(tsc(x) for x in (-1.3, -0.3, 0.7)) ≈ 1 rtol = 1e-15
    for name in ("landau", "two-stream", "landau-oblique-2d")
        case = load_case(name)
        p = picongpu_parameters(case)
        Nx, Ny = p.cells
        Δx, Δy = p.cell_size  # normalized
        ρ = brute_force_charge(p)
        bg = background_field(p)
        Ex = [bg.Ex(i, j) for i in 0:Nx-1, j in 0:Ny-1]
        Ey = [bg.Ey(i, j) for i in 0:Nx-1, j in 0:Ny-1]
        div = [(Ex[i, j] - Ex[mod1(i - 1, Nx), j]) / Δx + (Ey[i, j] - Ey[i, mod1(j - 1, Ny)]) / Δy for i in 1:Nx, j in 1:Ny]
        ρ1 = ρ .- sum(ρ) / length(ρ)
        @test maximum(abs, ρ1) > 0
        @test maximum(abs.(div .- ρ1)) <= 1e-10 * maximum(abs, ρ1)
    end
    # One particle at the cell centre gives half its charge to each of the two nearest nodes
    # (TSC at ±½), so the form factor is cos(kΔ/2); it tends to 1 for kΔ → 0.
    @test deposit_form_factor(1, 0.3) ≈ cos(0.15) rtol = 1e-14
    @test deposit_form_factor(1000, 1e-6) ≈ 1 rtol = 1e-10
end

"Write a small file in PIConGPU's openPMD layout (HDF5, group-based) and the run's units.toml."
function write_picongpu_fixture(dir, u; Nx, Ny, iterations, Ex, Ey, rho_e, rho_i)
    mkpath(joinpath(dir, "openPMD"))
    open(io -> TOML.print(io, u), joinpath(dir, "units.toml"), "w")
    h5open(joinpath(dir, "openPMD", "simData.h5"), "w") do f
        attrs(f)["meshesPath"] = "fields/"
        attrs(f)["iterationEncoding"] = "groupBased"
        for it in iterations
            g = create_group(f, "data/$it")
            # time is float_X(step) × dt in PIConGPU's units, with timeUnitSI the SI value of one
            attrs(g)["time"] = it * 2.0
            attrs(g)["dt"] = 2.0
            attrs(g)["timeUnitSI"] = u["dt"] * u["time"] / 2.0
            m = create_group(g, "fields")
            E = create_group(m, "E")
            a = attrs(E)
            a["axisLabels"] = ["y", "x"]
            a["gridSpacing"] = [u["cell_size"][2] / 3.0, u["cell_size"][1] / 3.0]   # internal unit: 3 c/ω₀
            a["gridGlobalOffset"] = [0.0, 0.0]
            a["gridUnitSI"] = 3.0 * u["length"]
            a["dataOrder"] = "C"
            # a Julia Nx × Ny array is the C-order dataset F[y][x]
            for (c, F, pos) in (("x", Ex, [0.5, 0.0]), ("y", Ey, [0.0, 0.5]), ("z", (i, j, t) -> 0.0, [0.0, 0.0]))
                m["E/$c"] = [F(i, j, it) * u["efield"] / 7.0 for i in 0:Nx-1, j in 0:Ny-1]
                attrs(m["E/$c"])["unitSI"] = 7.0
                attrs(m["E/$c"])["position"] = pos
            end
            B = create_group(m, "B")
            for k in ("axisLabels", "gridSpacing", "gridGlobalOffset", "gridUnitSI", "dataOrder")
                attrs(B)[k] = attrs(E)[k]
            end
            for (c, pos) in (("x", [0.0, 0.5]), ("y", [0.5, 0.0]), ("z", [0.5, 0.5]))
                m["B/$c"] = zeros(Nx, Ny)
                attrs(m["B/$c"])["unitSI"] = 1.0
                attrs(m["B/$c"])["position"] = pos
            end
            for (s, F) in (("electron", rho_e), ("ion", rho_i))
                n = "$(s)_all_chargeDensity"
                m[n] = [F(i, j, it) * u["charge_density"] / 5.0 for i in 0:Nx-1, j in 0:Ny-1]
                for k in ("axisLabels", "gridSpacing", "gridGlobalOffset", "gridUnitSI")
                    attrs(m[n])[k] = attrs(E)[k]
                end
                attrs(m[n])["unitSI"] = 5.0
                attrs(m[n])["position"] = [0.0, 0.0]
            end
        end
    end
end

@testset "T-252 PIConGPU openPMD reader [WP-017]" begin
    mktempdir() do top
        # 2D: a run of the oblique case's units, 8 × 4 cells
        p = picongpu_parameters(load_case("landau-oblique-2d"))
        u = PIConGPUDeck.units_record(p)
        u["cells"] = [8, 4]
        u["cell_size"] = [0.1, 0.2]
        u["dt"] = 0.05
        Ex(i, j, t) = 1e-3 * (i + 10j) + t
        Ey(i, j, t) = -2e-3 * (i - j) - t
        re(i, j, t) = -1.0 + 0.01 * i
        ri(i, j, t) = 1.0
        d2 = joinpath(top, "two")
        write_picongpu_fixture(d2, u; Nx = 8, Ny = 4, iterations = [0, 16, 32], Ex, Ey, rho_e = re, rho_i = ri)
        fl = read_picongpu_fields(d2)
        @test fl.iterations == [0, 16, 32]
        @test fl.time ≈ [0, 16, 32] .* 0.05 rtol = 1e-14
        @test size(fl.fields["Ex"]) == (9, 5, 3)
        @test fl.fields["Ex"][3, 2, 2] ≈ Ex(2, 1, 16) rtol = 1e-13          # x first, normalized
        @test fl.fields["Ey"][8, 4, 3] ≈ Ey(7, 3, 32) rtol = 1e-13
        @test fl.fields["Ex"][9, :, :] == fl.fields["Ex"][1, :, :]          # periodic copy appended
        @test fl.fields["Ex"][:, 5, :] == fl.fields["Ex"][:, 1, :]
        @test fl.fields["Rho"][4, 3, 1] ≈ re(3, 2, 0) + ri(3, 2, 0) rtol = 1e-13
        @test fl.fields["Rho_electron"][4, 3, 1] ≈ re(3, 2, 0) rtol = 1e-13
        # positions: (i + position) Δ per axis, x first, with Δ read in axisLabels order
        @test fl.x["Ex"][1] ≈ ((0:8) .+ 0.5) .* 0.1 rtol = 1e-14
        @test fl.x["Ex"][2] ≈ (0:4) .* 0.2 atol = 1e-15
        @test fl.x["Ey"][1] ≈ (0:8) .* 0.1 atol = 1e-15
        @test fl.x["Ey"][2] ≈ ((0:4) .+ 0.5) .* 0.2 rtol = 1e-14
        @test fl.x["Rho"] == fl.x["Rho_electron"]
        @test collect(keys(read_picongpu_fields(d2; names = ["Ey"]).fields)) == ["Ey"]
        @test_throws ErrorException read_picongpu_fields(d2; names = ["Jx"])

        # quasi-1D: the y-average in the 1D schema
        q = read_picongpu_fields(d2; names = ["Ex", "Rho"], yaverage = true)
        @test size(q.fields["Ex"]) == (9, 3)
        @test q.fields["Ex"][3, 2] ≈ sum(Ex(2, j, 16) for j in 0:3) / 4 rtol = 1e-13
        @test q.x["Ex"] ≈ ((0:8) .+ 0.5) .* 0.1 rtol = 1e-14
        @test yaverage(fl.fields["Rho"])[:, 1] ≈ q.fields["Rho"][:, 1] rtol = 1e-14

        # scalars: Joule to the units of A1 through units.toml, Smilei's column names
        open(joinpath(d2, "fields_energy.dat"), "w") do io
            println(io, "#step total[Joule] Bx[Joule] By[Joule] Bz[Joule] Ex[Joule] Ey[Joule] Ez[Joule] ")
            for s in (0, 4, 8)
                J = u["energy"]
                println(io, s, " ", 3J * (s + 1), " 0 0 0 ", J * (s + 1), " ", 2J * (s + 1), " 0")
            end
        end
        open(joinpath(d2, "electron_energy_all.dat"), "w") do io
            println(io, "#step Ekin_Joule E_Joule ")
            for s in (0, 4, 8)
                println(io, s, " ", 10 * u["energy"], " ", 11 * u["energy"])
            end
        end
        sc = read_picongpu_scalars(d2)
        @test sc.time ≈ [0, 4, 8] .* 0.05 rtol = 1e-14
        @test sc.columns["Uelm_Ex"] ≈ [1, 5, 9] rtol = 1e-14
        @test sc.columns["Uelm_Ey"] ≈ [2, 10, 18] rtol = 1e-14
        @test sc.columns["Uelm"] ≈ [3, 15, 27] rtol = 1e-14
        @test sc.columns["Ukin"] ≈ [10, 10, 10] rtol = 1e-14
        @test sc.columns["Utot"] ≈ [13, 25, 37] rtol = 1e-14
        @test PIConGPURead.picongpu_output(d2) && !PIConGPURead.picongpu_output(top)
        # the energy unit: per unit depth in 2D, per unit transverse area in quasi-1D
        uq = PIConGPUDeck.units_record(picongpu_parameters(load_case("landau")))
        U0 = p.units.energy * p.units.density * p.units.length^3
        @test u["energy"] ≈ U0 * p.cell_size[1] rtol = 1e-14
        pq = picongpu_parameters(load_case("landau"))
        @test uq["energy"] ≈ U0 * pq.cell_size[1] * 8 * pq.cell_size[2] rtol = 1e-14
        # a seed directory finds the units of its run
        mkpath(joinpath(d2, "seed-1"))
        @test PIConGPURead.picongpu_units(joinpath(d2, "seed-1"))["dt"] == 0.05
    end
end

@testset "T-256 the oblique seed ensemble in both codes [A5, WP-017]" begin
    c = load_case("landau-oblique-2d")
    seeds = case_seeds(c)
    @test length(seeds) == 4 && allunique(seeds) && first(seeds) == 20261031   # WP-019's run is seed 1
    # Smilei seeds patch p with random_seed + p: the seeds are spaced beyond its patch count
    @test minimum(diff(sort(seeds))) > prod(SmileiDeck.smilei_parameters(c).number_of_patches)
    # WP-019's four pre-registered entries are unchanged: the hash of their canonical TOML as
    # registered by WP-019 (computed from main before WP-017 touched the file).
    o = filter(p -> p.first != "damping_rate_mode_ensemble", c["observables"])
    io = IOBuffer(); TOML.print(io, o; sorted = true)
    @test bytes2hex(sha256(take!(io))) == "ac7d0fd307a4a8d99f24fc8157f8bce520923864eb3bddb7fac2c516a0335326"
    e = c["observables"]["damping_rate_mode_ensemble"]
    @test e["method"] == "ensemble_peak_envelope" && e["signal"] == "mode_projection"
    @test e["theory_from"] == "damping_rate_mode"
    for k in ("fit_start", "floor_factor")
        @test e[k] == c["observables"]["damping_rate_mode"][k]
    end
    # PIConGPU: one compile, then one run per seed directory with PIC_SEED set
    mktempdir() do runs
        dir = PIConGPUDeck.prepare_run("landau-oblique-2d"; tag = "t256", runs)
        sh = read(joinpath(dir, "run.sh"), String)
        @test count("pic-build", sh) == 1
        @test occursin("for entry in " * join(("seed-$s:$s" for s in seeds), " ") * "; do", sh)
        @test occursin(raw"PIC_SEED=$seed mpiexec -n 1", sh)
        @test all(s -> isfile(joinpath(dir, "seed-$s", "case.toml")), seeds)
        m = TOML.parsefile(joinpath(dir, "manifest.toml"))
        @test m["seeds"] == seeds && m["code"] == "picongpu" && m["precision"] == "Float64"
        @test m["reference_wavelength_m"] == 0.8e-6
        @test haskey(m["inputs"], "params/fieldBackground.param")
        @test endswith(dir, "t256")
        # the default tag names the code, the case, the precision and the case hash
        d2 = PIConGPUDeck.prepare_run("landau"; precision = "Float32", runs)
        @test occursin(r"picongpu-landau-fp32-[0-9a-f]{8}$", d2)
    end
end
