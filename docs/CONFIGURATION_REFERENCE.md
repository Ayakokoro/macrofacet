# Configuration reference

The renderer supports Classic local/global DDA null tracking and global conditional GP delta tracking. Commands are `render`, `curves`, `bake-field`, the independent 1D `first-passage` experiment, and neural `renewal-query`. Old `fixed_flight`, `modes`, `reference`, `render.flight_table_cells`, and legacy Conditional29 transport settings are rejected.

`renewal-query --config <A/B-first-passage-config> --model <hazard-bundle.json>`
evaluates a trained Matern-3/2 or SE model and samples its cumulative law. The
reference configuration's kernel must match the model. It accepts
`--trials`, `--bins`, and `--output`; optional top-level `renewal_query` fields
select normalized query distances, survival intervals, and optical depths.
See [Renewal C++ queries](RENEWAL_CPP.md) for export commands, units, and output.

`transport.mode="neural_renewal"` enables full Renewal+ neural path tracing.
Set `transport.renewal.model` to the JSON from `export-model` and optionally
`transport.renewal.profile_maximum_step` (default `0.25`). This mode requires
unit-decay `matern_3_2` or unit-length `squared_exponential` matching the model,
a positive-definite stationary kernel metric,
`generalized_gaussian`, and no alpha grid. NanoVDB inputs must contain full-domain
SDF data. `render.safety_depth_cap` defaults to `64`; the neural example uses
`128`. See [the neural scene](../configs/render_neural_renewal_shader_ball.json).
The [SE h16 scene](../configs/render_neural_renewal_shader_ball_ply_point_linear_se_h16.json)
loads the SE checkpoint with `rho(x)=exp(-x*x/2)`, `x=distance/ell`.
Model exports must declare `parameterization=unit_length` and `beta=1` for SE;
Matern-3/2 requires `parameterization=unit_decay` and `beta=1`.
The renderer and `renewal-query` reject field/reference versus model kernel mismatches.

`transport.renewal.profile_mode` selects `cubic` (default, full piecewise-cubic
mean profile) or `point_linear` (lazy current-point value/gradient queries with
local tangent extrapolation). The latter reuses the model but approximates the
mean field; it is not an equivalent replacement. NanoVDB steps stop at the next
cell face or the normalized `profile_maximum_step`, whichever comes first.
Only visited cells are checked and sampled. Scalar, LibTorch CPU/CUDA and model
transmittance share this option. See [point-query rendering](RENEWAL_CPP.md#当前点查询point_linear)
and [the PLY example](../configs/render_neural_renewal_shader_ball_ply_point_linear.json).

`transport.renewal.backend` accepts `scalar` (default: Eigen), `torch_cpu`,
`torch_cuda`, or `auto` (CUDA if available, otherwise LibTorch CPU or scalar
when built without LibTorch). Explicit unavailable backends fail instead of
silently switching devices. `transport.renewal.batch_size` limits distinct rays
per submission across all three groups, from 1 to 65536, default 4096.
Initialization plus the first segment count as one ray. Eligible mixture rows
take space first, then initialization, then continued segments; each group is FIFO.
`ray_pool_size` controls resident ray slots independently; 0 (default) uses
`min(65536, 2*batch_size)`, bounded by the number of pixels. Explicit pool sizes
are 1..65536. `auxiliary_batch_minimum` (1..65536, default 256, clamped to actual
batch capacity) accumulates initialization and mixture requests; `maximum_queue_delay`
(0..65536, default 4) makes an auxiliary group eligible after that many completed submissions of waiting.
Zero disables the delay. FIFO overflow takes additional submissions. Tail queues
flush when both CPU work and GPU batches are drained; other requests accumulate
while workers keep producing rays. All ready groups share one upload/readback.
`max_in_flight_batches` (1..4, default 2) enables independent CUDA batch buffers
and completion events. Set 1 to limit the outstanding request window to one batch; CPU tasks remain
asynchronous. Slots stay locked until their ticket is collected, then belong
exclusively to a CPU task until its ready chunk is published. LibTorch CPU renders
use one inference batch with independent ray workers; scalar rendering is unaffected.
While CPU tasks or GPU batches remain active, accumulate at least `batch_size`
eligible distinct rays across the three groups. Only when both have drained may
an underfilled tail submit. This avoids submitting each newly published CPU
chunk as a small GPU batch. Auxiliary age affects group eligibility, not this
overall fullness threshold.
The point-linear neural example uses `auto`, batch 8192 and pool 32768, leaving
ready rays available while another group awaits auxiliary processing or inference.
`resolved_config.json` records both requested and resolved backend;
`render_summary.csv` records the actual backend and inference batch counts.
The calling thread groups ready queues and dispatches completed CPU work. A
separate inference thread exclusively owns the LibTorch session, submits batches,
checks CUDA events and publishes results. The bounded request window includes
queued, executing and completed-but-not-yet-dispatched inference batches, limited
by `max_in_flight_batches` (one for `torch_cpu`). `render.thread_count` controls
additional CPU ray workers (0 selects at most 8, also bounded by pool capacity);
it excludes both the calling scheduler and the inference thread.
The inference thread collects/publishes the current GPU batch before entering
the next LibTorch submission, so CPU results are not delayed by that long host
call. It therefore has one active GPU ticket, while other groups remain queued
or execute on CPU. The configured inference window still bounds outstanding
requests; it is not a promise of that many simultaneously active GPU tickets.
Workers initialize camera rays/new flights, query fields, decide collisions,
sample normals, reflect and accumulate pixels. They publish in 128-ray chunks;
there is no batch-wide CPU barrier. Per-slot RNG and per-pixel accumulation order
are preserved, but task completion changes batch membership; GEMM rounding can
change individual Monte Carlo paths, even on repeated runs with the same config.

A config contains `schema_version`, `seed`, `field`, `material`, `transport`, `numeric`, `render`, and `output_directory`. See `configs/macrofacet_ci.json` for a complete small example.

`field.mean_type` selects a procedural mean (`plane`, `sphere`, `cutaway_sphere`, or `shader_ball`) or `nanovdb` with `grid_file`. Procedural fields specify `sigma`, `domain_min`, and `domain_max`. Imported NanoVDB fields derive domain and sigma from the file. `field.kernel_type` accepts `squared_exponential` (the default), `matern_3_2`, or `matern_5_2`. `field.correlation_lengths` has three components and may include `null` for an infinite length; these are the native metric lengths of the selected family. `field.kernel_rotation` optionally rotates that metric. A generalized Gaussian material may instead specify positive scalar `material.roughness` in the Beckmann alpha convention. It fixes the isotropic gradient covariance to `roughness^2 / 2 * I`, and the renderer derives the family-specific correlation length. It is mutually exclusive with `correlation_lengths`.

For an imported NanoVDB field, the tracing domain is the box through the outermost active SDF nodes, with no extra voxel margin. This clips any density interpolation tail outside that box. A surface-band bake can still have inactive, zero-background SDF locations *inside* the box, so use a full-domain bake when comparing global conditional and classic transport.

`material.gp_model` selects `local_tangent` (the default) or `global_pointwise` for classic transport. Local classic uses a tangent-plane GP at each point. Without an alpha grid it inherits the configured zero-distance gradient covariance; a NanoVDB `alpha` grid overrides it with `alpha(x)^2 / 2 * I` for a Gaussian NDF. Global classic uses the global GP's one-point prior independently at each point: mean value `m(x)`, mean gradient `∇m(x)`, and the same zero-distance gradient covariance. Consequently, roughness-matched SE and Matérn kernels are exactly equivalent in both classic modes. Global conditional transport instead conditions each flight on its birth observation and therefore depends on the full two-point kernel. Its current analytic sampler supports only `squared_exponential`; selecting Matérn fails explicitly until family-specific conditional statistics and certified majorants are implemented. For imported NanoVDB fields, set `field.use_alpha_grid` to `false` for global modes; they require `generalized_gaussian` without a spatial alpha grid.

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

For `initial_condition.type="fixed_value"`, `grid.max_time` and `grid.step_sizes` define a finest exact Gaussian grid and coupled coarser crossing checks. Every step must divide `max_time` and be an integer multiple of the finest step. `monte_carlo.trajectories`, `monte_carlo.thread_count`, `curve.bins`, and `state_analysis` control the risk-set hazard, parallelism, survivor-state snapshots, and \((X,X')\)-conditioned future-event grid. `rice_series` enables the start-conditioned Rice expansion; `max_order` currently accepts 1 or 2, while its tolerances and subdivision budget control the nested Gaussian-CDF and time quadratures.

`initial_condition.type="collision_state"` reuses the same command and output contract for all supported kernels. Its `parameter_space` accepts `explicit`, `cartesian`, or `latin_hypercube` sampling of dimensionless `beta_0`, `beta_a`, and strictly positive `beta_g`; the sampler type is `collision_state_auto`. Matérn 3/2 automatically uses the exact two-state SDE (including spatial means) plus conditional bridges and cubic-Hermite crossing localization. Squared exponential, Matérn 5/2, and rational quadratic automatically use a jointly value/finite-difference-derivative-conditioned circulant grid and cubic-Hermite crossing localization. `sampler_type` and `crossing_slope_method` in every CSV row make this numerical distinction explicit. `minimum_step`, `crossing_tolerance`, `bridge_sigma_margin`, and `max_refinement_depth` apply to the Matérn-3/2 backend; `max_embedding_expansions` applies to conditioned grids. Rice and legacy survivor-state diagnostics must be disabled. Each `grid.step_sizes` entry is an independent convergence resolution and the smallest is marked `training_resolution=1`. The old `collision_state_matern32` type is intentionally not accepted. See `configs/collision_state_kernels_training.json`.

C++ first-passage configs contain only data-generation settings. Plot selection is
kept separately under `data_analysis/configs/` and is consumed by
`data_analysis/plot_first_passage.py`. Its `comparison="kernels"` mode fixes one
state, while `comparison="states"` fixes one kernel. The optional
`plots.crossing_slope_density` section controls slope bins and distance intervals;
it requires C++ generation with `monte_carlo.write_raw_samples=true`. See
[the visualization guide](../data_analysis/README.md).

A zero thread count selects hardware concurrency without changing per-trajectory random streams. The CLI accepts `--trials`, `--bins`, `--threads`, and `--output` overrides.

Fixed-value outputs include `first_passage_curves.csv`, `first_passage_summary.csv`, survivor-state CSV files, optional raw samples, and a resolved config. The curve CSV reports Monte Carlo first-passage hazard beside the endpoint-conditioned \(\Sigma_1\), pointwise \(\Sigma_2\), ordinary Rice downcrossing intensity, and the start-conditioned Rice terms \(\bar W_1(t)\), \(\int_0^t\bar W_2(u,t)du\), and their order-two density. Collision-state mode writes common-schema long-form samples (when enabled), curves, summaries, and the resolved state design. SVGs and the derived crossing-slope histogram CSV are Python analysis outputs, not C++ generator outputs. Full schemas, definitions, and limitations are in [FIRST_PASSAGE_EXPERIMENT.md](FIRST_PASSAGE_EXPERIMENT.md).

## Renewal reference foundation

See [RENEWAL_REFERENCE.md](RENEWAL_REFERENCE.md) for `positive_exterior`,
`profile.maximum_step`, the segment-feature CSV, and the spatial Matern-3/2
state-space sampler. `matern_3_2` now uses `rho(x)=(1+x)exp(-x)`, with unit
normalized derivative variance. To preserve a GP specified with the old explicit
Matern-3/2 length, divide that length by sqrt(3). SE and Matern-5/2 are unchanged.
The removed `transport.first_passage_model` setting is rejected explicitly.

The old Matern-3/2 entries have been removed from `first_passage_kernels.json`
and `collision_state_kernels_training.json`. New reference and training configs
declare `unit_decay` explicitly. [Sequence training](../python/README.md) lives
under `python/configs/` and does not change the rendering transport selection.
`bake-field --config <render-config>` prepares the full NanoVDB field and resolved
configuration without tracing; the sequence collector uses it before reference sampling.
