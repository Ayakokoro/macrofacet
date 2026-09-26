# Macrofacet

CPU/double implementation of Classic transport for Gaussian-process statistical surfaces. The renderer uses a Gaussian height field, material NDF, directional extinction, and conductor multiple scattering. Procedural means are baked to NanoVDB before tracing; imported NanoVDB fields are also supported.

## Build

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
```

## Run

```powershell
build\Release\macrofacet_experiments.exe all --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball_nanovdb.json
python scripts\plot_experiments.py outputs\macrofacet_ci
```

The executable accepts `curves`, `render`, and `all`. Render outputs include `render_classic_white.pfm`, `render_classic_directional.pfm`, BMP previews, `render_summary.csv`, and `resolved_config.json`.

See the [configuration reference](docs/CONFIGURATION_REFERENCE.md), [NanoVDB tracing guide](docs/NVDB_TRACING.md), and [scene rendering guide](docs/SCENE_RENDERING.md). Earlier Conditional29 derivations and implementation reports remain as historical research notes and are not part of the current executable path.
