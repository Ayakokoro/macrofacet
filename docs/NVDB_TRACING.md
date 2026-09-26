# NanoVDB tracing

`macrofacet_experiments` prepares an NVDB field for `render`, `curves`, and `all`. Procedural means are baked before tracing. For an imported `mean_type: "nanovdb"` field, `field.grid_file` names an existing grid and its bounds determine the tracing domain. A baked file also fixes `sigma`; a config or CLI sigma must agree with it.

A narrow-band field uses its `density` grid to define the transport band. Classic transport samples candidate events using a certified density majorant and DDA traversal; zero-density cells are vacuum. The actual event rate uses the local density and Classic projected area. Full-domain fields instead integrate the Classic hazard and invert optical depth with safeguarded Newton.

The `alpha` grid can supply local material data when `field.use_alpha_grid` is enabled. It is separate from the density grid. Field generation and import commands are described by `macrofacet_fieldgen --help` and the scene configs under `configs/`.

`render_summary.csv` reports path counts, numerical failures, hazard evaluations, quadrature intervals, root iterations, and optical-depth errors. `resolved_config.json` records the prepared field, derived covariance, and effective sampling method. Use `--flight-cells` or `render.flight_table_cells` to set the initial integration partition for full-domain optical-depth sampling.
