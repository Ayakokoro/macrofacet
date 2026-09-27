# Configuration reference

The renderer uses Classic DDA null tracking only. The only executable command is `render`. Old `fixed_flight`, `modes`, `reference`, `render.flight_table_cells`, and Conditional29 transport settings are rejected.

A config contains `schema_version`, `seed`, `field`, `material`, `transport`, `numeric`, `render`, and `output_directory`. See `configs/macrofacet_ci.json` for a complete small example.

`field.mean_type` selects a procedural mean (`plane`, `sphere`, `cutaway_sphere`, or `shader_ball`) or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.correlation_lengths` has three components and may include `null` for an infinite length. `field.kernel_rotation` optionally rotates the covariance. A generalized Gaussian material may instead specify positive scalar `material.roughness` in the Beckmann alpha convention: its isotropic gradient covariance is `roughness^2 / 2 * I` and correlation length is `sqrt(2) * sigma / roughness`. It is mutually exclusive with `correlation_lengths`.

`material.gp_model` selects `local_tangent` (the default) or `global_pointwise`. In local mode, each material point uses a tangent-plane GP with zero mean value and unit normal mean gradient. Without an alpha grid it inherits the configured gradient covariance; the NanoVDB `alpha` grid overrides it with `alpha(x)^2 / 2 * I` for a Gaussian NDF. In global mode, extinction and scattering both use the same global GP's one-point prior: mean value `m(x)`, mean gradient `∇m(x)`, and gradient covariance `sigma^2 P`, where `P` is the SE kernel precision. Set `field.use_alpha_grid` to `false` for imported NanoVDB fields in global mode; the global model requires `generalized_gaussian` and does not accept a spatial alpha grid. Global mode still evaluates pointwise marginals independently along each path; it does not yet condition on a path's birth point. DDA uses a certified bound on the global mean gradient. See `configs/render_shader_ball_nanovdb_global.json` for a comparison config.

`material.ndf_family` accepts `generalized_gaussian`, `beckmann_limit`, or `ggx`. GGX uses `material.ggx_alpha`. Conductors use `eta_rgb` and `k_rgb`. `transport.classic_phase_proposal` accepts `uniform`, `paper_mixture`, or `target_vndf`; `beckmann_mixture_weight` applies to the paper mixture. `roulette_start_depth` starts path roulette.

`numeric` retains tolerances and budgets used by Gaussian moment and phase sampling. `render` specifies image size, samples per pixel, camera, environment, and optional thread count. The CLI supports `--sigma`, `--roughness`, `--preserve-slope`, `--width`, `--height`, `--spp`, `--threads`, and `--output`. For imported baked fields, `--sigma` must match the file.

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
```
