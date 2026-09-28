# Macrofacet

CPU/double transport for Gaussian-process statistical surfaces. Classic local and global rendering use DDA null tracking over a NanoVDB density grid. Global conditional rendering uses SE GP birth-point conditioning and delta tracking. Procedural means are baked to NanoVDB before tracing.

## Build

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

## Render

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball_nanovdb.json
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json --mode all --width 16 --height 16 --spp 8
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json --rays 1024 --bins 64
```

The renderer writes PFM images, BMP previews, `render_summary.csv`, and `resolved_config.json`. The `curves` command writes per-ray collision data and a comparison CSV/SVG. See the [configuration reference](docs/CONFIGURATION_REFERENCE.md), [conditional rendering guide](docs/CONDITIONAL_RENDERING.md), [NanoVDB tracing guide](docs/NVDB_TRACING.md), and [scene rendering guide](docs/SCENE_RENDERING.md). Earlier transport derivations and implementation reports remain as historical research notes.
