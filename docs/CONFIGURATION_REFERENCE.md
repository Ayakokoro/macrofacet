# Configuration reference

The renderer supports Classic local/global DDA null tracking and global conditional GP delta tracking. Commands are `render` and `curves`. Old `fixed_flight`, `modes`, `reference`, `render.flight_table_cells`, and legacy Conditional29 transport settings are rejected.

A config contains `schema_version`, `seed`, `field`, `material`, `transport`, `numeric`, `render`, and `output_directory`. See `configs/macrofacet_ci.json` for a complete small example.

`field.mean_type` selects a procedural mean (`plane`, `sphere`, `cutaway_sphere`, or `shader_ball`) or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.correlation_lengths` has three components and may include `null` for an infinite length. `field.kernel_rotation` optionally rotates the covariance. A generalized Gaussian material may instead specify positive scalar `material.roughness` in the Beckmann alpha convention: its isotropic gradient covariance is `roughness^2 / 2 * I` and correlation length is `sqrt(2) * sigma / roughness`. It is mutually exclusive with `correlation_lengths`.

`material.gp_model` selects `local_tangent` (the default) or `global_pointwise` for classic transport. Local classic uses a tangent-plane GP at each point. Without an alpha grid it inherits the configured gradient covariance; a NanoVDB `alpha` grid overrides it with `alpha(x)^2 / 2 * I` for a Gaussian NDF. Global classic uses the global GP's one-point prior independently at each point: mean value `m(x)`, mean gradient `∇m(x)`, and gradient covariance `sigma^2 P`, where `P` is the SE kernel precision. It uses DDA with a certified bound on the global mean gradient. Global conditional transport instead conditions each flight on its birth observation. For imported NanoVDB fields, set `field.use_alpha_grid` to `false` for global modes; they require `generalized_gaussian` without a spatial alpha grid.

`material.ndf_family` accepts `generalized_gaussian`, `beckmann_limit`, or `ggx`. GGX uses `material.ggx_alpha`. Conductors use `eta_rgb` and `k_rgb`. `transport.classic_phase_proposal` accepts `uniform`, `paper_mixture`, or `target_vndf`; `beckmann_mixture_weight` applies to the paper mixture. `roulette_start_depth` starts path roulette.

`numeric` retains tolerances and budgets used by Gaussian moment and phase sampling. `render` specifies image size, samples per pixel, camera, environment, and optional thread count. The CLI supports `--sigma`, `--roughness`, `--preserve-slope`, `--width`, `--height`, `--spp`, `--threads`, and `--output`. For imported baked fields, `--sigma` must match the file.

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
```

`transport.mode` accepts `classic` (default, follows `material.gp_model`),
`classic_local`, `classic_global`, `global_conditional`, and `all`. The last
choice renders three comparable modes. The CLI also accepts `--mode`.

`global_conditional` constructs two delta-tracking majorants per flight:
an analytic bound near birth and a conditional-moment interval bound for the
remaining distance. The split is computed from the ray and birth observation;
there is no user-selected epsilon cutoff or transmittance integration.
`render_summary.csv` reports `bound_intervals`, `near_candidates`, and
`far_candidates`. When the far constant would cause excessive null events,
certified interval thinning uses additional tracking intervals with the same
far-bound formula. `adaptive_majorant_flights` and `rounded_candidate_steps`
record this acceleration and compensated distance additions. See the
[derivation and validation](CONDITIONAL_RENDERING.md).

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json --mode all --width 16 --height 16 --spp 8
build\Release\macrofacet_experiments.exe curves --config configs\macrofacet_ci.json --rays 1024 --bins 64
```

`curves` sends a batch of parallel rays through the active domain and writes
`transmittance_rays.csv`, `transmittance_curves.csv`, and an SVG plot. See
[conditional rendering](CONDITIONAL_RENDERING.md) for the model and limits.
