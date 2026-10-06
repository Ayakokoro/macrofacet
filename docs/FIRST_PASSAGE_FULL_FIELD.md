# Full-field first-passage experiments

For a slope distribution conditioned on the **first collision at an exact
fixed distance**, see [fixed-endpoint sampling](FIXED_ENDPOINT_SAMPLING.md).
That mode samples endpoint-conditioned paths with survival/flux weights; it
does not use the ordinary distance-bin crossing histogram.

The existing `collision_state` experiment can use a spatial `MeanField` instead
of an affine ray mean. Data generation remains C++; all plots remain Python.
No density-grid extinction or renderer null-collision model is used as the
reference: each realization is sampled along the ray and stopped at its first
downcrossing after birth.

## Shader-ball example

Use the existing `outputs/fields/shader_ball_full.nvdb`; do not substitute the
surface-band file `shader_ball.nvdb`.

```powershell
cmake --build build --config Release --parallel 4
build\Release\macrofacet_experiments.cmd first-passage --config configs\shader_ball_full_transmittance_se.json
if ($LASTEXITCODE -ne 0) { throw "First-passage generation failed" }
python data_analysis\plot_first_passage.py --config data_analysis\configs\shader_ball_full_transmittance_se.json
```

The example fixes the origin `(0.00013, -0.0374, 0.041)` and direction proportional to
`(1, -0.2, 0.05)`, and compares observed gradients `(0, -0.25, 0)` and `(0, -1, 0)`.
These are configurable observations, not gradients sampled from a previous
collision. The origin is not required to lie on the **mean** zero surface:
the **random field** is conditioned to `F(origin)=0` regardless of its mean.
The direction is normalized by the loader; gradients are **not** normalized.
The example origin lies inside a voxel cell rather than on a gradient jump.

The SE kernel uses `sigma=0.001`, `ell=0.002`; `grid.max_time=6` means a physical
distance of `0.012`. Step sizes are in `q=t/ell`, not world coordinates. Each
gradient/resolution uses 65,536 realizations by default. Use `--trials 512` and
an alternate `--output` for a cheap generation smoke test.

## Config and probability model

Set `first_passage.process.mean_field` to an existing mean-factory object:

```json
{
  "mean_type": "nanovdb",
  "grid_file": "outputs/fields/shader_ball_full.nvdb",
  "use_alpha_grid": false
}
```

Replace `initial_condition.parameter_space` with `initial_condition.rays`.
Each ray has `id`, `origin`, `direction`, and the observed full `gradient`.
Keep `type: "collision_state"`, `process.mean=threshold=0`, and disable Rice
and survivor-state diagnostics. Do not mix beta states and physical rays.

For a unit ray direction, the sampled process is

\[
F(t)=m_{\rm nvdb}(\mathbf b+t\mathbf d)+\xi(t),\qquad
\xi(0)=-m(\mathbf b),\quad
\xi'(0)=\mathbf g\cdot\mathbf d-\nabla m(\mathbf b)\cdot\mathbf d.
\]

Here `xi` has the stationary isotropic kernel selected by the config. Along
one straight ray, transverse birth-gradient components do not affect the
conditional process for these isotropic kernels. The outgoing projection
`gradient.dot(direction)` must be positive. Only the collision state is kept;
survival history before this birth is still discarded.

`survival` estimates the probability of **no downcrossing anywhere on the
segment**, conditional on birth, not a pointwise positivity probability.
`hazard_mc` is the interval event fraction divided by the interval width in q.

## Preflight, interpolation and numerical accuracy

Before creating experiment outputs:

- Require the NanoVDB sidecar to declare `coverage: "full_domain"`.
- Check every interpolation cell crossed by each whole ray segment: its eight
  SDF corners must be active and finite. A bounding box or metadata flag alone
  is insufficient. Stored SDF values equal to zero are valid, not missing.
- Require the kernel sigma to agree with the bake's stored sigma at float32
  precision. Changing that scale requires a consistent re-bake.
- Reject rays leaving the stored domain rather than using zero background as
  an SDF or implicitly declaring vacuum.

All spatial-mean kernels currently use the conditioned circulant-grid backend,
including Matérn 3/2. Its birth residual derivative is a central-difference
observation, not an exact continuous derivative. The affine Matérn 3/2 mode
retains its existing state-space/bridge backend.

The deterministic profile is precomputed and shared by all realizations.
Crossing intervals are split at both GP-grid nodes and NanoVDB voxel faces.
Within a cell, trilinear SDF interpolation restricted to a ray is cubic; four
value queries reconstruct it. The residual uses cubic Hermite interpolation
with finite-difference slopes. This preserves the full interpolated mean,
without flattening it or smoothing over gradient jumps at voxel faces.
Analytic non-polynomial means (for example `sphere`) instead have a cubic
approximation within each GP interval.

Monte Carlo error, residual grid/interpolation error, and the baked SDF's own
voxel/sign error remain. Compare all configured resolutions; finer GP sampling
does not improve the stored geometry. The two-point derivative condition also
converges with the GP step size. This is not an exact continuous-GP solver.

## Outputs and visualization

The usual samples/curves/summary CSVs remain available. Spatial-mean runs add
`mean_profile_type=full_field`. Their beta fields are derived per kernel:

\[
\beta_0=m(\mathbf b)/\sigma,\quad
\beta_a=\ell\nabla m(\mathbf b)\cdot\mathbf d/\sigma,\quad
\beta_g=\ell\mathbf g\cdot\mathbf d/\sigma.
\]

They describe **birth only**, not the complete non-affine mean. Do not feed
these curves into the existing three-beta affine MLP as equivalent training
data. `resolved_first_passage_config.json` records the field, physical rays,
and preflight/backend semantics.

`first_passage_mean_profiles.csv` records `kernel_id,state_id,q,distance,mean,
beta_mean` at the finest GP grid and all voxel-face breakpoints. It is the
deterministic profile, not a realization or the birth-conditioned mean.

In the Python config, `x_axis: "distance"` shows world distance and converts
density/hazard by dividing their dimensionless values by ell. Survival and
cumulative hazard are unchanged. Omitting it retains the existing q axis.
`plots.mean_profile: true` additionally writes `first_passage_mean_profile.svg`.
The other figures are `first_passage_survival.svg`, `first_passage_hazard.svg`,
and `first_passage_cumulative_hazard.svg`.
