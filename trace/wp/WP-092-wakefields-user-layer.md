+++
id = "WP-092"
status = "active"
goals = ["A1", "A3", "A5", "A6", "A9"]
depends_on = ["WP-065", "WP-084"]
adds_dependencies = ["Accessors", "Mendeleev", "SpecialFunctions", "Unitful"]
+++
# WP-092: the LWFA user layer as packages, `Wakefields` and `RunManifests`, so that scripts call it instead of including each other

Move the run manifest into `lib/RunManifests`, and the LWFA cases (one TOML schema), the SI conversion of A1, the problem builders, the fixed-energy coupling and the objectives into `lib/Wakefields`, which uses only public names of `ParticleMeshElectromagnetic` and patches no other package. It works when every case built through the packages gives the same problem and the same recorded digests as the scripts it replaces, and no script outside `scripts/enzyme/` includes a file from another script directory or a package's `test/`.
