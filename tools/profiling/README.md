# Renewal CUDA profiling

This optional diagnostic executes the current C++ wavefront renderer on the
current model and field. It generates instrumented copies from four source
files, asserting that each instrumentation anchor occurs exactly once. The
executable's objects satisfy the wavefront/backend/profile symbols before the
corresponding static-library members can be extracted. Normal rendering sources
and binaries have no profiling scopes. Generated source hashes are saved in
`build/renewal_profile/source_hashes.json`.

Build with the existing LibTorch CUDA and NanoVDB configuration:

```powershell
cmake --build build --config Release --target macrofacet_profile_renewal --parallel 4
build/Release/macrofacet_profile_renewal.exe configs/render_neural_renewal_shader_ball_ply.json 4 3 outputs/profile/wall.json wall
```

Arguments are config, spp override, repeat count, JSON output, and mode:

- `baseline`: disable host timers and NVTX ranges in the diagnostic.
- `wall`: record nested inclusive/exclusive host times, without extra GPU barriers.
- `trace`: also delimit the measured frame with `cudaProfilerStart/Stop` for Nsight.

An optional final argument sets the point-query sampling stride (default 64;
1 times every query). `profile.point_query_sampled` measures only those sampled
queries; `point_queries` counts all queries. Estimate the total query time as
`point_query_sampled_thread_seconds * point_queries / point_query_sampled_calls`.
This estimates summed thread time, not elapsed frame time; do not add it to
enclosing wall times. Older serial reports use `profile.point_query_sampled`
in `stages` for the sampled time and calls.
The default avoids millions of timer/map/NVTX operations for `point_linear`.
Always compare with `baseline` to check instrumentation overhead.

`segment.cpu_preprocess` covers slot validation and packing five raw double
features (including CPU log dx). `segment.input_transform` measures dispatch of
device double asinh and float conversion; `segment.output_transform` measures
dispatch of hazard softplus. Their actual CUDA execution is available through
the NVTX kernel attribution. `segment.cpu_postprocess` covers output validation,
result assembly and cached segment updates. Older reports taken before the
device-activation change include CPU asinh/softplus in those CPU scopes.
In `point_linear`, most field queries occur lazily
inside `cpu.hazard_and_decision`, when advancing a surviving ray; they are not
all in `cpu.mean_profile`.

CPU advance now uses a persistent pool controlled by `render.thread_count`.
`cpu.advance_wait` measures coordinator wall time for dispatch, its own share
of work, and the barrier. `cpu.hazard_and_decision` also includes ordered
result commitment. `cpu.advance_work` measures each processed range.
All counters and timers are thread-local during rendering and are merged only
after workers join. `stages` contains coordinator timings; `worker_stages`
contains the sum of background-worker timings. Never add worker times to
coordinator/frame wall time. `worker_point_queries` confirms actual background
query execution; `cpu_workers`, `parallel_advance_batches` and
`serial_advance_batches` describe the dispatch. Baseline mode disables timers.

All modes force `torch_cuda`, run one warmup frame at 1 spp, and retain the
config's resolution, field, camera, seed, batch size, network, and environment.
Only the configured environment is rendered. Field/model loading and warmup are
excluded; per-frame session creation is included. The first measured frame is
saved as `<report.json>.pfm` for comparison with the normal renderer. A top-level
host breakdown must use exclusive times where parent and child scopes overlap.

Example Nsight command (use the installed absolute executable path if needed):

```powershell
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --kill=false --export=sqlite --output=outputs/profile/cuda_trace build/Release/macrofacet_profile_renewal.exe configs/render_neural_renewal_shader_ball_ply.json 4 1 outputs/profile/trace.json trace
nsys stats --report nvtx_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv --output outputs/profile/stats outputs/profile/cuda_trace.sqlite
```

Host GRU/MLP timers measure asynchronous dispatch and possible backpressure;
they do **not** measure GPU kernel execution. Use `nvtx_kern_sum` to attribute
GPU kernels to the NVTX range enclosing their launch API. Nested NVTX ranges
overlap: never add a parent range's GPU time to its children's times. Transfer
API wall times can contain synchronization and must not be equated with DMA
duration. No per-module CUDA synchronization is inserted.

The 2026-10-09 investigation is saved in
[`outputs/renewal_bottleneck_20261009`](../../outputs/renewal_bottleneck_20261009/).
It includes the config snapshot, source hashes, three alternating baseline/wall
runs, raw Nsight traces/CSVs, and production image parity. The investigation's
specific filenames can be summarized and checked with:

```powershell
python tools/profiling/summarize_renewal_profile.py outputs/renewal_bottleneck_20261009
```

The later `point_linear` investigation uses the current sigma-0.02 field and
128x128 camera. Its separate scripts, config snapshot, hybrid-CPU affinity
controls, profiles and image-parity checks are in
[`outputs/renewal_point_linear_bottleneck_20261009`](../../outputs/renewal_point_linear_bottleneck_20261009/).
Use that directory's `summarize.py` for those measurements; the older summary
script intentionally expects the earlier cubic investigation's filenames.

CPU parallel advance validation, repeated baseline timing, thread-safe stage
accounting, and production h16/h64 parity are recorded in
[`outputs/renewal_cpu_parallel_20261009/REPORT.md`](../../outputs/renewal_cpu_parallel_20261009/REPORT.md).
