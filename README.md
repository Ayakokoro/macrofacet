# Macrofacet

CPU/double Classic transport for Gaussian-process statistical surfaces. Rendering uses DDA null tracking over a NanoVDB density grid. Procedural means are baked to NanoVDB before tracing; imported surface-band and full-domain grids use the same DDA sampler.

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
```

The renderer writes Classic PFM images, BMP previews, `render_summary.csv`, and `resolved_config.json`. See the [configuration reference](docs/CONFIGURATION_REFERENCE.md), [NanoVDB tracing guide](docs/NVDB_TRACING.md), and [scene rendering guide](docs/SCENE_RENDERING.md). Earlier transport derivations and implementation reports remain as historical research notes.
