# Macrofacet

CPU/double transport for Gaussian-process statistical surfaces. Classic local and global rendering use DDA null tracking over a NanoVDB density grid. Global conditional rendering uses SE GP birth-point conditioning and delta tracking. Procedural means are baked to NanoVDB before tracing.

## Build

The project uses the LibTorch CMake package bundled with PyTorch
`2.12.x`, the last release line supported here with C++17. Install the pinned
CUDA 13.0 wheel into the Python interpreter that CMake will discover:

```powershell
python -m pip install -r python\requirements-cuda.txt
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

To select a specific Python installation, add
`-DPython3_EXECUTABLE=B:/path/to/python.exe` to the configure command. CMake
rejects PyTorch versions outside the `2.12.x` line.

CMake automatically locates LibTorch through the selected Python interpreter.
On Windows, it generates `.cmd` launchers alongside the executables. These
temporarily add the wheel's `torch/lib` directory to the child process's `PATH`,
using the installed DLLs without copying them or changing the system environment.
CTest and Visual Studio debugging receive that search path automatically.
Use the `.cmd` launchers for terminal commands; launching an `.exe` directly
still requires `torch/lib` in that terminal's `PATH`.

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

The Matérn 5/2 cumulative-hazard surrogate uses C++ interval-count data and a monotone PyTorch I-spline model. Its complete generation, fitting, export, and inference workflow is documented in the [Python training guide](python/README.md).

First-passage experiments can also use the complete NanoVDB SDF along a ray.
The [full-field experiment guide](docs/FIRST_PASSAGE_FULL_FIELD.md) uses the
existing `shader_ball_full.nvdb`, validates ray coverage before sampling, and
plots transmittance/hazard with Python.

[Fixed-endpoint sampling](docs/FIXED_ENDPOINT_SAMPLING.md) adds a zero-value
endpoint condition, rejects earlier crossings, and uses slope-flux weights to
visualize the first-hit slope distribution at a specified physical distance.

The independent `first-passage` command samples conditioned 1D Gaussian processes and measures the no-history-approximation survival and hazard curves for several kernels. Its unified `collision_state` mode emits the same censored FPT, survival, hazard, and crossing-slope schema for every supported kernel. Matérn 3/2 uses its exact two-state backend; the other kernels use conditioned circulant grids. See [the first-passage experiment guide](docs/FIRST_PASSAGE_EXPERIMENT.md).
