# Analytic scene rendering

The renderer uses the sign convention `mean(x) > 0` for the exterior. The ready-to-run
configurations are:

- `configs/render_plane.json`: `mean(x) = z`, clipped to a finite AABB.
- `configs/render_sphere.json`: `mean(x) = length(x - center) - radius`.
- `configs/render_cutaway_sphere.json`: exact signed distance to a sphere with a quarter
  cut out of its outer layer, exposing a smaller concentric sphere.
- `configs/render_shader_ball.json`: material preview ball with two tilted circular grooves
  and a rounded circular pedestal; `render_shader_ball_conditional.json` uses `conditional29`.

Build once, then render a sharp surface and a broad, volume-like stochastic shell:

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_plane.json --sigma 0.03 --preserve-slope --width 64 --height 64 --spp 8 --flight-cells 32 --output outputs\plane_surface
build\Release\macrofacet_experiments.exe render --config configs\render_plane.json --sigma 0.25 --preserve-slope --width 64 --height 64 --spp 8 --flight-cells 32 --output outputs\plane_volume_like

build\Release\macrofacet_experiments.exe render --config configs\render_sphere.json --sigma 0.03 --preserve-slope --width 64 --height 64 --spp 8 --flight-cells 32 --output outputs\sphere_surface
build\Release\macrofacet_experiments.exe render --config configs\render_sphere.json --sigma 0.25 --preserve-slope --width 64 --height 64 --spp 8 --flight-cells 32 --output outputs\sphere_volume_like
```

Each run writes a linear-radiance `.pfm`, an sRGB `.bmp` preview, `resolved_config.json`, and
render statistics. Use the `directional` preview to judge appearance. The `white` image is an
energy-conservation diagnostic and intentionally has little shape contrast.

The plane config uses the paper mixture proposal. The sphere uses `target_vndf`, which samples its
exact local Gaussian visible-normal distribution even though the mean normal varies over the
object. This changes variance, not the target transport model, and is particularly important for
small `sigma`, where a uniform hemisphere proposal would produce mostly zero-weight samples.

The commands above override the checked-in final-quality budget for quick iteration. Remove the
`--width`, `--height`, `--spp`, and `--flight-cells` overrides to use the values from JSON. For a
narrow sigma or close-up silhouette, 64 or 96 flight-table cells reduce collision-location
quantization.

## Cutaway sphere

Build after adding the new mean-field implementation, then run a small preview:

```powershell
cmake --build build --config Release --parallel 4
build\Release\macrofacet_experiments.exe render --config configs\render_cutaway_sphere.json --width 64 --height 64 --spp 4 --output outputs\cutaway_preview
```

To use the configured 256 x 256, 16 spp render:

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_cutaway_sphere.json
```

The field parameters are `mean_type: "cutaway_sphere"`, `sphere_center`, `inner_radius`
(default 0.45), and `outer_radius` (default 1.0), with `0 < inner_radius < outer_radius`.
The outer layer in `local_x > 0 && local_y > 0` is removed; the inner sphere stays intact.
The two planar cut faces connect the outer sphere to the inner sphere. Keep the camera
on the positive X/positive Y side to see into the cut. All surfaces share the configured
conductor material, including the exposed inner sphere.

This field uses the exact Euclidean signed distance and analytical gradients. At sharp
edges and medial-axis points, where the gradient is not unique, it selects a deterministic
one-sided unit gradient. No VDB file or mesh conversion is needed.

The default is `sigma = 0.02` with isotropic correlation lengths 0.06. To compare a thinner
surface at the same gradient covariance, use:

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_cutaway_sphere.json --sigma 0.01 --preserve-slope --flight-cells 256 --output outputs\cutaway_sigma_0.01
```

The render path uses tabulated optical depth in all modes. Its within-cell collision
location is approximate; increase `flight_table_cells` and compare results for small
sigma or close-up cut edges. Set `modes` to `["conditional29"]` or `["midpoint"]` in a
copy of the config to render the correlated models, using a small image first.

## Shader ball

The shader ball is a procedural material-preview shape with two toroidal grooves and a
horizontal rounded pedestal. Build the executable after adding the field, then render:

```powershell
cmake --build build --config Release --parallel 4
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball.json
```

For a small correlated preview:

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball_conditional.json --width 64 --height 64 --spp 4 --output outputs\shader_ball_conditional_preview
```

The `field` parameters are:

- `mean_type: "shader_ball"`.
- `sphere_center` and `sphere_radius`: position and radius of the main ball.
- `groove_axis`: the shared groove axis, normalized on loading; the pedestal remains vertical.
- `groove_radius`: groove cutter tube radius in world units, with
  `0 < groove_radius <= 0.15 * sphere_radius`.

The two grooves are centered at axial offsets +/-0.3 times the ball radius. The pedestal
radius, height, and edge bevel are respectively 1.08, 0.2, and 0.04 times the ball radius;
its top is at `sphere_center.z - sphere_radius`. The default axis preserves the contact
between the ball and pedestal. All parts use the same configured conductor material.

The field evaluates exact distance to the exposed circular arcs of the grooved ball's
meridian profile and to the rounded pedestal. The two solids meet at most at the ball's
bottom point, leaving both boundaries exposed, so the minimum of their signed distances
is also exact. Analytical gradients use a
deterministic one-sided choice at edges and equidistant points.

Both configs use explicit `material.roughness` and 256 flight-table cells.
Keep sigma small relative to the groove radius to retain the groove detail.
With explicit roughness, `--sigma <value>` automatically keeps the gradient covariance fixed.
The same within-cell flight-location approximation described above applies here.
In a 16 x 16, 2 spp conditional smoke render, reducing the table to 128 cells produced
one numerical failure per environment; the default 256-cell run had none. Keep the
default when starting with this scene and check `render_summary.csv` after changing
sampling settings. This smoke check is not a convergence guarantee.

## Threads

Rendering distributes rows over worker threads. The default is the machine's hardware concurrency;
`--threads <count>` or `render.thread_count` in the JSON pins it to a fixed value.

Each pixel seeds its RNG from its own index, so the output depends only on `seed` and the pixel
index — never on the worker count or the order rows complete in. Rendering at any thread count
therefore produces byte-identical `.pfm` and `.bmp` files and identical path statistics; only the
`seconds` column in `render_summary.csv` changes. Measured on a 32-core machine, a 32x32 midpoint
render scales as 1 thread 165.2 s, 8 threads 16.5 s, 32 threads 7.7 s.

The resolved config records this layout as `rng_stream: "per_pixel_v1"`. It is deterministic
across thread counts, but it intentionally differs from images generated by the older whole-image
serial RNG stream even when the numeric seed is unchanged.

Correlated modes (`conditional29`, `midpoint`) cost roughly 20x classic mode per path, so their
timing is dominated by the hazard table build; `--flight-cells` is the most direct lever on it.

## What sigma changes

For a unit-gradient signed-distance mean, `sigma` is approximately the standard deviation of the
random surface position in world units. A small value produces a narrow, surface-like shell; a
large value produces a broad stochastic shell with more multiple scattering and a volume-like
appearance.

For configurations specified by `field.correlation_lengths`, changing `sigma` without
`--preserve-slope` also changes the gradient covariance
`sigma^2 * kernelPrecision`, hence the normal distribution and roughness. With
`--preserve-slope`, the command rescales kernel precision so gradient covariance stays fixed. This
is the recommended mode for an isolated thickness comparison.

This remains a Gaussian-process statistical *surface* model. It can look volumetric, but it is not
a general participating medium with independently configurable absorption, scattering albedo, or
phase function. A homogeneous volume test should instead use `mean_type: "constant"` in a convex
box.

## Material roughness

For `ndf_family: "generalized_gaussian"`, set a positive scalar `material.roughness` to
control isotropic gradient spread independently of `field.sigma`. For example, these are
the relevant entries in an otherwise complete scene configuration:

```json
"field": {
  "sigma": 0.03,
  "ndf_family": "generalized_gaussian"
},
"material": {
  "type": "conductor",
  "roughness": 0.3
}
```

Here roughness `r` means the standard deviation of **each GP gradient perturbation
component**. The renderer constructs the full consistent kernel with
`gradient_covariance = r^2 * I` and isotropic correlation length `ell = sigma / r`.
It does not change the NDF separately from the field statistics. For signed-distance
means, larger `r` broadens the local normal distribution. This is a GP parameterization,
not GGX/Disney perceptual roughness, and it is not an exact normal-angle or slope standard
deviation. Values greater than 1 are permitted; zero (a perfect-mirror limit) is unsupported.

At `sigma = 0.03`, example values are:

| Roughness | Derived correlation length |
| --- | --- |
| 0.15 | 0.20 |
| 0.30 | 0.10 |
| 0.75 | 0.04 |

Use either `material.roughness` **or** `field.correlation_lengths` in JSON; specifying both
is an error. Roughness mode does not require `kernel_rotation`, since its covariance is
isotropic. Keep the original correlation-length configuration for anisotropic fields or
the Beckmann/GGX baseline modes.

You can override roughness directly from the command line, including on a legacy
correlation-length config:

```powershell
build\Release\macrofacet_experiments.exe render --config configs\render_shader_ball.json --roughness 0.3 --sigma 0.03 --output outputs\shader_ball_roughness_0.3
```

CLI roughness takes precedence over the configured value. With an explicit roughness,
changing sigma in JSON or via `--sigma` automatically derives a new `ell`, so
`--preserve-slope` is redundant. Legacy configurations without explicit roughness keep
their existing sigma/`--preserve-slope` behavior.

At a fixed point, holding roughness fixed keeps the prior normal distribution fixed.
Changing sigma still changes the spatial correlation length, the stochastic layer's
thickness, and potentially the final image, particularly in correlated transport modes.
Sigma, roughness and correlation length are not three independent parameters in this kernel.
`resolved_config.json` records the explicit roughness, its definition, derived isotropic
correlation length and gradient component standard deviations.
