# Configuration reference

The renderer supports Classic local/global DDA null tracking and global conditional GP delta tracking. Commands are `render`, `curves`, and the independent 1D `first-passage` experiment. Old `fixed_flight`, `modes`, `reference`, `render.flight_table_cells`, and legacy Conditional29 transport settings are rejected.

A config contains `schema_version`, `seed`, `field`, `material`, `transport`, `numeric`, `render`, and `output_directory`. See `configs/macrofacet_ci.json` for a complete small example.

`field.mean_type` selects a procedural mean (`plane`, `sphere`, `cutaway_sphere`, or `shader_ball`) or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.correlation_lengths` has three components and may include `null` for an infinite length. `field.kernel_rotation` optionally rotates the covariance. A generalized Gaussian material may instead specify positive scalar `material.roughness` in the Beckmann alpha convention: its isotropic gradient covariance is `roughness^2 / 2 * I` and correlation length is `sqrt(2) * sigma / roughness`. It is mutually exclusive with `correlation_lengths`.

For an imported NanoVDB field, the tracing domain is the box through the outermost active SDF nodes, with no extra voxel margin. This clips any density interpolation tail outside that box. A surface-band bake can still have inactive, zero-background SDF locations *inside* the box, so use a full-domain bake when comparing global conditional and classic transport.

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

An optional root `transmittance` object replaces that legacy camera-grid batch
with explicit rays. `modes` accepts `classic_local`, `classic_global`, and
`global_conditional`. `curve.bins` controls linearly spaced distances measured
from each requested origin. `monte_carlo.trials_per_ray` repeats every ray and
mode, and `confidence_level` controls pointwise Wilson intervals. Every ray has
a unique `id`, optional `group`, three-component `origin` and `direction`, a
positive `max_distance`, and an optional `bins` override. Directions are
normalized while loading.

The segment before an outside ray enters the active domain and the segment
after it exits are vacuum. The CSV reports both requested-origin distance and
distance inside the medium. `conditional_birth.policy` is normally
`sample_positive_exterior`; `fixed_observation` additionally requires a
nonnegative `value` and a three-component `gradient`. The latter conditions all
conditional trials on the same GP birth observation.

`comparison.local_transport_reference` is `classic_global`.
`comparison.constant_exponential.fit="censored_mle"` also fits a constant-rate
exponential to each mode using collision distances and finite-domain escape as
right-censored samples. The fit uses medium exposure only, so leading vacuum
does not dilute its rate. Explicit experiments write one long-form curve CSV,
one summary CSV, a ray-geometry CSV, optional raw trial samples, and one SVG per
ray. `--trials` and `--bins` override the JSON budgets for quick checks;
`--rays` remains limited to legacy configs without explicit rays.

## First-passage experiment

`first-passage` uses a separate minimal JSON schema rooted at `first_passage`; it does not require `field`, `material`, `render`, or NanoVDB. See `configs/first_passage_kernels.json`. Its `kernels` array accepts `squared_exponential`, `matern_3_2`, `matern_5_2`, and `rational_quadratic`, each with `id`, `variance`, and `length_scale`; rational quadratic additionally accepts `alpha`.

`grid.max_time` and `grid.step_sizes` define a finest exact Gaussian grid and coupled coarser crossing checks. Every step must divide `max_time` and be an integer multiple of the finest step. `initial_condition` currently supports a fixed value strictly above `process.threshold`. `monte_carlo.trajectories`, `monte_carlo.thread_count`, `curve.bins`, and `state_analysis` control the risk-set hazard, parallelism, survivor-state snapshots, and \((X,X')\)-conditioned future-event grid. `rice_series` enables the start-conditioned Rice expansion; `max_order` currently accepts 1 or 2, while its tolerances and subdivision budget control the nested Gaussian-CDF and time quadratures. A zero thread count selects hardware concurrency without changing per-trajectory random streams. The CLI accepts `--trials`, `--bins`, `--threads`, and `--output` overrides.

Outputs include `first_passage_curves.csv`, `first_passage_summary.csv`, survivor-state CSV files, survival/hazard SVG plots, `first_passage_rice_density.svg`, and a resolved config. The curve CSV reports Monte Carlo first-passage hazard beside the endpoint-conditioned \(\Sigma_1\), pointwise \(\Sigma_2\), ordinary Rice downcrossing intensity, and the start-conditioned Rice terms \(\bar W_1(t)\), \(\int_0^t\bar W_2(u,t)du\), and their order-two density. It also compares Monte Carlo survival with the analytic start-conditioned endpoint probability, survival integrated from \(\Sigma_1\), and the pointwise exponential survival from \(\Sigma_2\). Full definitions and limitations are in [FIRST_PASSAGE_EXPERIMENT.md](FIRST_PASSAGE_EXPERIMENT.md).
