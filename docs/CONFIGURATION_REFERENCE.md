# Configuration reference

The executable runs Classic transport only. Configs under `configs/*.json` no longer contain `modes` or Conditional29 settings. Old `modes`, `reference`, `fixed_flight.birth_gradient`, `transport.conditional29`, `correlated_sampler`, `external_policy`, `hard_depth_cap`, and `next_event_estimation` settings are rejected.

## Top-level blocks

- `schema_version`: currently `1`.
- `seed`: base random seed.
- `field`: procedural mean or imported NanoVDB field, covariance and domain settings.
- `material`: NDF family, conductor `eta_rgb` and `k_rgb`, and optional roughness.
- `transport`: Classic phase proposal, Beckmann mixture weight, and roulette start depth.
- `fixed_flight`: birth position, direction, curve range, and sample counts.
- `numeric`: quadrature and root solver tolerances and limits.
- `render`: camera, image size, samples per pixel, environment, and thread count.
- `output_directory`: output path.

`render`, `curves`, and `all` share one config loader, so these blocks currently remain required. See `configs/macrofacet_ci.json` for a small complete example.

## Field and material

`field.mean_type` selects a procedural mean such as `plane`, `sphere`, `cutaway_sphere`, or `shader_ball`, or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.correlation_lengths` has three components and may include `null` for an infinite length. `field.kernel_rotation` optionally rotates that covariance. A generalized Gaussian material may instead specify a positive scalar `material.roughness`; it is the standard deviation of each isotropic gradient component, giving correlation length `sigma / roughness`. `roughness` and `correlation_lengths` are mutually exclusive.

`material.ndf_family` accepts `generalized_gaussian`, `beckmann_limit`, or `ggx`. GGX uses `material.ggx_alpha`. The conductor uses `eta_rgb`, `k_rgb`, and optional `force_unit_fresnel_for_energy_test`.

## Transport and numeric settings

`transport.classic_phase_proposal` accepts `uniform`, `paper_mixture`, or `target_vndf`; `transport.beckmann_mixture_weight` is used by the paper mixture. `transport.roulette_start_depth` starts path roulette. Classic has a fixed safety depth cap in the renderer.

`numeric.relative_tolerance`, `absolute_tolerance`, `max_quadrature_subdivisions`, and `max_root_iterations` govern optical-depth inversion. `distance_absolute_tolerance` and `distance_relative_tolerance` govern root distances. `render.flight_table_cells` is the initial integration partition. For narrow-band fields, Classic uses DDA null tracking instead.

## Commands

```powershell
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json
build\Release\macrofacet_experiments.exe render --config configs\render_sphere.json --width 64 --height 64 --spp 4
build\Release\macrofacet_experiments.exe all --config configs\macrofacet_ci.json
```

The CLI also supports `--sigma`, `--roughness`, `--preserve-slope`, `--flight-cells`, `--threads`, and `--output`. For imported baked fields, `--sigma` must match the file. `resolved_config.json` records effective parameters.
