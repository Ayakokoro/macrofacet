# Configuration reference

The renderer uses Classic DDA null tracking only. The only executable command is `render`. Old `fixed_flight`, `modes`, `reference`, `render.flight_table_cells`, and Conditional29 transport settings are rejected.

A config contains `schema_version`, `seed`, `field`, `material`, `transport`, `numeric`, `render`, and `output_directory`. See `configs/macrofacet_ci.json` for a complete small example.

`field.mean_type` selects a procedural mean (`plane`, `sphere`, `cutaway_sphere`, or `shader_ball`) or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.correlation_lengths` has three components and may include `null` for an infinite length. `field.kernel_rotation` optionally rotates the covariance. A generalized Gaussian material may instead specify positive scalar `material.roughness` as the default local GP's Beckmann alpha: its isotropic gradient covariance is `roughness^2 / 2 * I` and correlation length is `sqrt(2) * sigma / roughness`. It is mutually exclusive with `correlation_lengths`. At each point, the local material GP is centred on the tangent plane. Without an alpha grid it inherits the configured gradient covariance; the NanoVDB `alpha` grid overrides it using `alpha(x)^2 / 2 * I` for a Gaussian material NDF. Extinction and scattering both use this local gradient distribution. With a GGX NDF, the grid value is used directly as GGX alpha.

`material.ndf_family` accepts `generalized_gaussian`, `beckmann_limit`, or `ggx`. GGX uses `material.ggx_alpha`. Conductors use `eta_rgb` and `k_rgb`. `transport.classic_phase_proposal` accepts `uniform`, `paper_mixture`, or `target_vndf`; `beckmann_mixture_weight` applies to the paper mixture. `roulette_start_depth` starts path roulette.

`numeric` retains tolerances and budgets used by Gaussian moment and phase sampling. `render` specifies image size, samples per pixel, camera, environment, and optional thread count. The CLI supports `--sigma`, `--roughness`, `--preserve-slope`, `--width`, `--height`, `--spp`, `--threads`, and `--output`. For imported baked fields, `--sigma` must match the file.

```powershell
build\Release\macrofacet_experiments.exe render --config configs\macrofacet_ci.json
```
