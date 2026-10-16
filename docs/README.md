# Documentation

- [Fixed GPIS function-space sampler](GPIS_FUNCTION_SPACE.md): renewal+ memory, Tungsten-style linear training reference, optional Hermite comparison, fixed H / batch / step and separate model identities.
- [SE camera first hit](FIRST_HIT_SE.md): unconditional first-zero RNO / GRU comparison, segmentation checks, gradient reconstruction, and scalar C++ inference.
- [First-hit network design, sigma=0.02](FIRST_HIT_DISTANCE_DESIGN_SIGMA002.md): paired distance-only hazard/CDF experiments, conditional gradient probes, prefix consistency, and local-baseline cost.
- [Expanded camera holdout, sigma=0.02](FIRST_HIT_CAMERA_HOLDOUT_SIGMA002.md): additional viewpoints, geometry-selected diverse profiles, and fixed-checkpoint evaluation.
- [Multiview training and gradient distributions, sigma=0.02](FIRST_HIT_MULTIVIEW_SIGMA002.md): paired direct/residual retraining, frozen mixture/CDF heads, joint sampling, and failure cases.
- [MLP residual regularization and transmittance supervision, sigma=0.02](FIRST_HIT_RESIDUAL_SUPERVISION_SIGMA002.md): training-only residual L2 and survival-curve losses, validation-selected weights, paired seeds, and tail-error diagnosis.
- [Lightweight causal residual GRU, sigma=0.02](FIRST_HIT_CAUSAL_RESIDUAL_SIGMA002.md): history and matched-capacity controls, fixed physical steps, prefix consistency, step sensitivity, and inference costs.
- [Conditional gradient fitting on the frozen GRU, sigma=0.02](FIRST_HIT_CAUSAL_GRADIENT_SIGMA002.md): matched history ablation, mixture/CDF heads, conditional calibration, sampling, and prefix consistency.
- [Gradient-head step accuracy, sigma=0.02](FIRST_HIT_CAUSAL_GRADIENT_STEP_SIGMA002.md): five recurrent steps, three frozen seeds, fixed reference hits, and state/phase diagnostics.
- [Causal GRU first-hit rendering, sigma=0.02](FIRST_HIT_CAUSAL_RENDER_SIGMA002.md): fixed-step joint distance/speed sampling, full normals, scalar/CPU/CUDA parity, and hybrid rendering.
- [Deployed GRU versus analytic local, per ray](FIRST_HIT_CAUSAL_RAY_COMPARISON_SIGMA002.md): interactive ray rankings, camera pixels, transmittance and conditional gradient distributions, with separate improvement and regression cases.
- [Renewal reference foundation](RENEWAL_REFERENCE.md): shared ray profiles, Matern-3/2 A/B starts, kernel migration.
- [Renewal sequence training](../python/README.md): scene-disjoint data collection, GRU joint likelihood, training and evaluation.
- [Renewal C++ rendering](RENEWAL_CPP.md): model export, distance/normal sampling, batched LibTorch CPU/CUDA path tracing, transmittance, and parity checks.
- [Configuration reference](CONFIGURATION_REFERENCE.md): current JSON fields and command-line options.
- [NanoVDB tracing](NVDB_TRACING.md): field preparation, Classic flight sampling, and diagnostics.
- [Scene rendering](SCENE_RENDERING.md): checked-in scene configs and preview commands.

[Implementation report](IMPLEMENTATION_REPORT.md) and [archive](archive/README.md) preserve historical research results. Use the three guides above for the current executable.
