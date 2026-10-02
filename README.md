# Macrofacet

CPU/double transport for Gaussian-process statistical surfaces. Classic local and global rendering use DDA null tracking over a NanoVDB density grid. Global conditional rendering uses SE GP birth-point conditioning and delta tracking. Procedural means are baked to NanoVDB before tracing.

## Build

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

## Render

Mesh fields now use the `primitive_macrofacet` index-node bake convention: the
grid map is anchored at world zero, samples are written at integer indices,
and unwritten `sdf`, `density`, and `alpha` values are zero. Re-bake fields made
before this change. New macrofacet bakes write a version 2 sidecar; a three-grid
file from `primitive_macrofacet` can be read when `field.sigma` is specified.

For the shader ball comparison, run from this directory:

```powershell
build\Release\macrofacet_fieldgen.exe "scenes/shader ball.ply" --sigma 0.001 --alpha 0.5 --x 96 --y 96 --z 96 --out outputs/fields/shader_ball.nvdb
build\Release\macrofacet_experiments.exe render --config configs/render_shader_ball_nanovdb.json
```

The shader ball config selects `classic_local` and reads the baked alpha grid.
The additional `sigma` grid and JSON sidecar carry scale and provenance; they
do not change the three fields used by PBRT.

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball_nanovdb.json
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json --mode all --width 16 --height 16 --spp 8
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json --rays 1024 --bins 64
build\Release\macrofacet_experiments.exe first-passage --config configs\first_passage_kernels.json
```

The renderer writes PFM images, BMP previews, `render_summary.csv`, and `resolved_config.json`. The `curves` command writes per-ray collision data and a comparison CSV/SVG. See the [configuration reference](docs/CONFIGURATION_REFERENCE.md), [conditional rendering guide](docs/CONDITIONAL_RENDERING.md), [NanoVDB tracing guide](docs/NVDB_TRACING.md), and [scene rendering guide](docs/SCENE_RENDERING.md). Earlier transport derivations and implementation reports remain as historical research notes.

The independent `first-passage` command samples conditioned 1D Gaussian processes and measures the no-history-approximation survival and hazard curves for several kernels. It also records survivor \((X,X')\) states and compares the Monte Carlo hazard with the endpoint-conditioned and pointwise approximations. See [the first-passage experiment guide](docs/FIRST_PASSAGE_EXPERIMENT.md).
