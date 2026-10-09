# Macrofacet

CPU/double transport for Gaussian-process statistical surfaces. Classic local and global rendering use DDA null tracking over a NanoVDB density grid. Global conditional rendering uses SE GP birth-point conditioning and delta tracking. Procedural means are baked to NanoVDB before tracing.

## Build

Build the C++ renderer with CMake and the existing vcpkg dependencies.
Python is used by the data-analysis tests. LibTorch is enabled by default for batched
Renewal inference; CMake reuses `Torch_DIR` or discovers the selected Python's PyTorch
installation. Set `-DMACROFACET_ENABLE_TORCH=OFF` for an Eigen-only build.

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

Windows builds retain `.cmd` launchers alongside the executables.

## Render

Mesh fields now use the `primitive_macrofacet` index-node bake convention: the
grid map is anchored at world zero, samples are written at integer indices,
and unwritten `sdf`, `density`, and `alpha` values are zero. Re-bake fields made
before this change. New macrofacet bakes write a version 2 sidecar; a three-grid
file from `primitive_macrofacet` can be read when `field.sigma` is specified.

For the shader ball comparison, run from this directory:

```powershell
build\Release\macrofacet_fieldgen.cmd "scenes/shader ball.ply" --sigma 0.001 --alpha 0.5 --x 96 --y 96 --z 96 --out outputs/fields/shader_ball.nvdb
build\Release\macrofacet_experiments.cmd render --config configs/render_shader_ball_nanovdb.json
```

The shader ball config selects `classic_local` and reads the baked alpha grid.
The additional `sigma` grid and JSON sidecar carry scale and provenance; they
do not change the three fields used by PBRT.

```powershell
build\Release\macrofacet_experiments.cmd render --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.cmd render --config configs\render_shader_ball_nanovdb.json
build\Release\macrofacet_experiments.cmd render --config configs\macrofacet_ci.json --mode all --width 16 --height 16 --spp 8
build\Release\macrofacet_experiments.cmd curves --config configs\macrofacet_ci.json --rays 1024 --bins 64
build\Release\macrofacet_experiments.cmd first-passage --config configs\first_passage_kernels.json
build\Release\macrofacet_experiments.cmd first-passage --config configs\collision_state_kernels_training.json
build\Release\macrofacet_experiments.cmd first-passage --config configs\collision_state_matern52_parameter_study.json
build\Release\macrofacet_experiments.cmd first-passage --config configs\collision_state_matern52_training.json
```

The `first-passage` executable only generates CSV/JSON data. Create the plots
in a separate step with the standard-library Python visualizer:

```powershell
python data_analysis\plot_first_passage.py --config data_analysis\configs\first_passage_kernels.json
python data_analysis\plot_first_passage.py --config data_analysis\configs\collision_state_kernels.json
```

Plot selection and slope-density intervals live in `data_analysis/configs/`,
not in the C++ experiment configs. See the
[data-analysis guide](data_analysis/README.md).

The renderer writes PFM images, BMP previews, `render_summary.csv`, and `resolved_config.json`. The `curves` command writes per-ray collision data and a comparison CSV/SVG. See the [configuration reference](docs/CONFIGURATION_REFERENCE.md), [conditional rendering guide](docs/CONDITIONAL_RENDERING.md), [NanoVDB tracing guide](docs/NVDB_TRACING.md), and [scene rendering guide](docs/SCENE_RENDERING.md). Earlier transport derivations and implementation reports remain as historical research notes.


First-passage experiments can also use the complete NanoVDB SDF along a ray.
The [full-field experiment guide](docs/FIRST_PASSAGE_FULL_FIELD.md) uses the
existing `shader_ball_full.nvdb`, validates ray coverage before sampling, and
plots transmittance/hazard with Python.

[Fixed-endpoint sampling](docs/FIXED_ENDPOINT_SAMPLING.md) adds a zero-value
endpoint condition, rejects earlier crossings, and uses slope-flux weights to
visualize the first-hit slope distribution at a specified physical distance.

The independent `first-passage` command samples conditioned 1D Gaussian processes and measures the no-history-approximation survival and hazard curves for several kernels. Its unified `collision_state` mode emits the same censored FPT, survival, hazard, and crossing-slope schema for every supported kernel. Matérn 3/2 uses its exact two-state backend; the other kernels use conditioned circulant grids. See [the first-passage experiment guide](docs/FIRST_PASSAGE_EXPERIMENT.md).

The [Renewal+ reference foundation](docs/RENEWAL_REFERENCE.md) provides shared
cell-wise cubic ray profiles and Matern-3/2 first-passage sampling for positive
exterior and known surface starts. Matern-3/2 now uses `rho(x)=(1+x)*exp(-x)`;
see the guide for length migration and example commands. Existing rendering
modes remain available; the GP reference remains a separate 1D experiment.

[Renewal+ sequence training](python/README.md) collects geometry-disjoint NanoVDB
ray profiles, trains a causal GRU with monotone cumulative hazard and positive
truncated Gaussian mixtures, and evaluates censored distance/speed likelihoods.
Python training uses PyTorch. Batched C++ rendering uses LibTorch on CPU or CUDA;
the scalar Eigen inference backend remains available for comparison.

[Renewal+ C++ rendering](docs/RENEWAL_CPP.md) loads the trained JSON model,
samples distances, crossing speeds and full gradients, and traces multiple
conductor reflections in the `neural_renewal` mode. The `renewal-query` command
evaluates A/B distance, transmittance and speed distributions. See
`configs/render_neural_renewal_shader_ball.json` for a complete scene.
That scene selects `backend="auto"` and `batch_size=4096`, using CUDA when available.
