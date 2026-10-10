# Renewal CPU/CUDA profiling

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

An optional argument after the mode sets the point-query sampling stride (default 64;
1 times every query). `profile.point_query_sampled` measures only those sampled
queries; `point_queries` counts all queries. Estimate the total query time as
`point_query_sampled_thread_seconds * point_queries / point_query_sampled_calls`.
This estimates summed thread time, not elapsed frame time; do not add it to
enclosing wall times. Older serial reports use `profile.point_query_sampled`
in `stages` for the sampled time and calls.
The default avoids millions of timer/map/NVTX operations for `point_linear`.
Always compare with `baseline` to check instrumentation overhead.

A second optional argument sets LibTorch CPU intra-op threads (0 preserves its
default). This is separate from `render.thread_count`, which controls ray advance
workers. Reports record both Torch intra-op and inter-op thread counts. For CPU
timing, select `transport.renewal.backend = "torch_cpu"` in the input config:

```powershell
build/Release/macrofacet_profile_renewal.exe outputs/cpu_config.json 1 3 outputs/profile/cpu.json baseline 512 4
```

`batch.enqueue` contains CPU request packing and enqueueing the network's CUDA
operations, including a nonblocking upload and readback. `batch.collect` contains
completion-event waiting, output validation and state commitment. CPU overhead is
visible in the exclusive time of `batch.session_enqueue` / `batch.session_collect`;
`batch.backend_enqueue` / `batch.backend_collect` enclose the device adapter.
`batch.event_wait` measures host waiting for one completion event. There are no
intermediate `.item()` checks or per-stage GPU barriers.

The buffer-reuse implementation keeps each session Pending record and its vector
capacities after collection. Wavefront BatchWork objects return to a pool only
when their last shared owner (including CPU tasks) releases them. Pool handoff
uses a short mutex; it introduces no batch-wide CPU barrier. Output arrays retain
their high-water storage, including nested mixture vectors; request counts define
their active prefixes. The public collect API still returns exact-size owning
results. The pool grows to cover simultaneous CPU/GPU users, not just the GPU
ticket limit. Shared-pointer control blocks and per-task ready lists can still
allocate; this is not an allocation-free renderer.

The second round packs directly into each backend frame's pinned host input and
decodes a borrowed view of its pinned output after completion-event waiting.
No output view escapes into worker tasks or public results. This removes the
CPU copies from session input to staging and from readback to a temporary vector;
H2D/D2H, numeric checks, and the public request snapshot contract remain.
`batch.prepare_input` now measures buffer preparation/growth separately, nested
inside `batch.session_enqueue`; previously input growth was inside backend enqueue.
`batch.backend_collect` exclusive time now excludes the removed vector copy.
Compare complete timings as well as individual stages across these versions.

Two-round source snapshots, preserved executable hashes, tests, and alternating
performance comparisons are in
[`outputs/renewal_buffer_reuse_20261010`](../../outputs/renewal_buffer_reuse_20261010/).
Run its `benchmark.py round1` before the second-round edits, then
`benchmark.py round2` after rebuilding/testing. Each comparison uses three
alternating baseline pairs followed by separate wall captures, preserving the
preceding binary. Existing run files are never overwritten.

The recorded first-round baseline pairs reduce elapsed time by 2.8%, 6.1%, and
7.8% (median 17.38 to 16.32 s). The second round, compared with the preserved
first-round binary in a new session, reduces time by 3.2% and 11.1% in two pairs
but increases it by 17.9% in the third. That round does not establish a stable
end-to-end speedup. Its separate wall capture reduces backend-collect exclusive
time from 0.311 to 0.008 s, consistent with removing the readback-vector copy;
completion-event waiting is measured separately and is retained. Timing drift
precludes multiplying ratios across sessions into a combined speedup. All 16
measured images are finite with zero numerical failures. One pre-round1 baseline
path and one post-round2 baseline path reach the existing depth cap; these
counters remain in the reports. Each round passes all three CTest suites and
the separate Debug session policy (128,081 and 134,247 C++ checks, respectively).

`cpu.worker_task` encloses one background CPU chunk, with child scopes
`cpu.initialize_work`, `cpu.advance_work`, or `cpu.scatter_work`, followed by
`cpu.prepare_features`. The latter computes initial/segment/mixture geometric
features for that chunk's ready slots before publication. These are summed
worker times, not additional frame wall time. New-flight
construction can occur inside any of these as slots are recycled. `cpu.schedule`
packs request groups on the calling scheduler; `cpu.ready_drain` imports
worker publications; `cpu.dispatch_completed` dispatches collected results.
`cpu.wait_ready` measures scheduler waiting for CPU/inference notifications.
The separate LibTorch thread owns `batch.enqueue`/`batch.collect`, and
`inference.wait_ready` measures its empty-queue waiting. After each submission,
it waits directly on that CUDA event
(`batch.event_wait`), with no short timed polling. This avoids OS timer granularity
amplifying a microsecond GPU wait; the scheduler and CPU workers remain runnable.
The old `cpu.advance_batch`/`cpu.refill`
scopes describe the preceding barrier-based scheduler, not this task pipeline.
Use Nsight's kernel execution intervals intersected with these CPU NVTX ranges to
measure actual overlap. Enqueue host time is NOT GPU execution time, and kernels
launched inside an enqueue range may execute after that host range ends.

JSON counters include `max_in_flight_batches`, `blocking_collects` (event not ready
at the pre-collect check), and `cpu_batches_with_gpu_pending` (another event not
ready when inference results are collected). The last counter is an opportunity indicator, not a
measurement of simultaneous execution. Set `transport.renewal.max_in_flight_batches`
to 1/2 to compare outstanding request windows; background CPU tasks remain asynchronous.
The renderer collects/publishes each GPU result before entering the next long
LibTorch call. Its actual active GPU ticket count is therefore one, and
`cpu_batches_with_gpu_pending` can be zero despite substantial CPU/GPU overlap.
The backend API still supports multiple GPU tickets. Use the execution timeline
to measure overlap between the next GPU batch and the previous CPU batch.
Each in-flight request
retains independent pinned buffers; the inference stream executes GPU batches in
order. GPU tickets are collected FIFO, but CPU completion order changes batch
membership. Per-pixel RNG and accumulation order are retained; trained-model
images need not be bitwise identical between runs because GEMM shapes can change.

Background CPU tasks use a persistent pool controlled by `render.thread_count`,
excluding both the calling scheduler and the dedicated inference thread. The
inference thread collects and publishes results before submitting more work, allowing
the scheduler to dispatch CPU processing during LibTorch calls. Its queued,
executing and completed-but-not-dispatched batches share a bounded window.
There is no per-batch CPU barrier.
The old `cpu.advance_wait` measures the preceding synchronous pool's barrier.
Older reports retain their previous stage names; compare like-for-like scopes.
All counters and timers are thread-local during rendering and are merged only
after workers join. `stages` contains coordinator timings; `worker_stages`
contains the sum of background-thread timings, including the dedicated inference
thread's `batch.*` and `inference.*` ranges. Never add worker times to
coordinator/frame wall time. `worker_point_queries` confirms actual background
query execution; `cpu_workers`, `parallel_advance_batches` and
`serial_advance_batches` describe dispatch (multiple available workers and more
than one 128-ray task versus a single task/worker). Baseline mode disables timers.

The tool honors the config's backend (`torch_cpu`, `torch_cuda`, or `auto`), runs
one warmup frame at 1 spp, and retains the
config's resolution, field, camera, seed, batch size, network, and environment.
Only the configured environment is rendered. Field/model loading and warmup are
excluded; per-frame session creation is included. The first measured frame is
saved as `<report.json>.pfm` for comparison with the normal renderer. A top-level
host breakdown must use exclusive times where parent and child scopes overlap.
`trace` requires CUDA. On CPU, `batch.backend_enqueue` includes synchronous CPU
network execution; upload/download scopes are CPU buffer copies, not device DMA.
The existing build target still requires the CUDA-enabled diagnostic dependencies.

Wavefront uses a private submission entry point. Debug builds retain the public
session's full slot validation. With `NDEBUG`, that entry point skips slot bounds,
duplicate, ownership/status validation and all `seen`/`touched`/`busy` traversal;
the scheduler already guarantees those invariants. A bounded ticket-level check
still enforces collect-before-submit. Public session submissions remain checked
in every build, and all paths retain request sizes, numerical checks, output
validation and completion-event waiting. `batch.session_enqueue` covers either
submission policy so host packing times remain comparable.

Wavefront workers cache geometric features in fixed, per-slot arrays (88 bytes
per slot, 2.75 MiB at pool32768). The inference thread packs these directly by
slot ID; it does not recompute polynomial values, derivatives or segment logs.
GPU feature transforms and network arithmetic are unchanged. Public submissions
still compute their own features. Debug wavefront submissions also recompute
and compare all cached features, catching stale values across slot reuse.
Features share the existing ready-publication synchronization and remain
immutable until the slot's collected result is dispatched to a worker; no new
tasks, locks or per-batch barriers are introduced. Request/result copies and
numeric checks are retained.

Worker-precompute validation and measurements are saved in
[`outputs/renewal_worker_features_20261010/summary.json`](../../outputs/renewal_worker_features_20261010/summary.json).
The Release suites and the separate Debug session policy pass 127,835 C++ checks.
Three alternating baseline pairs measure old/new times of 13.80/14.33,
18.01/17.51, and 19.66/19.29 seconds at 512x512, 1 spp, batch8192/pool32768,
8 workers and affinity 0xffff. Their mixed signs and large time drift do not
establish a stable speedup. Two separate wall pairs, with reversed execution
order, measure old/new session packing at 3.54/4.11 and 3.10/3.75 seconds;
backend dispatch times also drift, so these do not isolate a causal regression.
Worker computation has moved, but packing and copying remain. Run `benchmark.py`,
`reverse_wall.py`, then `analyze.py` from that directory to collect the comparison;
scripts preserve existing run files and require fresh output paths for reruns.

Fast-submit validation and before/after measurements are saved in
[`outputs/renewal_wavefront_fast_submit_20261010`](../../outputs/renewal_wavefront_fast_submit_20261010/).
Its `build_checked_tests.py` runs the same C++ test objects with only the session
compiled without `NDEBUG`, exercising the Debug validation policy against the
installed Release LibTorch ABI; this is not a full Debug build. `benchmark.py`
alternates preserved-old/current binaries with unchanged batch/pool settings,
then captures separate wall profiles. Keep individual timings: process times
drift, so their median ratio alone does not establish a stable speedup.

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

The async batch implementation is validated in
[`outputs/renewal_async_20261010/summary.json`](../../outputs/renewal_async_20261010/summary.json).
That directory contains `benchmark.py` (alternating uninstrumented depth 1/2
runs and a single 512x512 pair), `trace.py` (separate Nsight captures),
`check_production.py` (production/profile image identity), and `summarize.py`
(actual GPU/CPU interval intersections and statistics). Run them in that order
after CTest; do not overlap timing with builds, tests or another renderer.
The small workload regressed while the one large-image comparison improved;
these measurements do not establish a general speedup or high-spp performance.

Current 512x512 async bottleneck measurements are in
[`outputs/renewal_async_bottleneck_20261010/summary.json`](../../outputs/renewal_async_bottleneck_20261010/summary.json).
Run that directory's `run.py`, then `analyze.py` to reproduce host breakdowns,
kernel attribution, actual CPU/GPU overlap, and image identity checks. Only spp is
overridden to 1; batch/pool remain 8192/16384. The runner includes baseline repeats
before and after profiling because absolute process times drifted substantially.
GPU attribution joins CUPTI kernels to launch API correlation IDs and disjoint
NVTX network ranges; nested host/worker timings are not added to frame time.

CPU inference timing before the independent worker task pipeline, for the same h16 `point_linear` scene, is recorded in
[`outputs/renewal_cpu_timing_20261010/summary.json`](../../outputs/renewal_cpu_timing_20261010/summary.json).
Reproduce with that directory's `run.py sweep`, `run.py main`, then `run.py finish`.
The runner fixes affinity to logical CPUs 0--15 on the i9-13900HX, keeps 8 ray
advance workers and batch/pool 8192/16384, and measures 1 spp. The 128x128 sweep
is exploratory; full 512x512 CPU baselines measured 76.10 s with 4 Torch threads,
86.17 s with 8, and 69.74 s with its default (16 under that affinity). Each CPU
setting has one measured frame. CUDA's three baseline frames took 24.10, 21.70,
and 25.65 s (median 24.10 s). These measurements do not establish optimal CPU
threading or 256-spp performance. CPU and CUDA retain their production pipeline
depths of 1 and 2, respectively.

The separate CPU stage capture explicitly sets 16 Torch threads, takes 102.61 s,
and attributes 64.03 s to the CPU backend, including math, tensor operations and
buffer copies. It uses the same paths as the CPU default baseline but is a separate,
instrumented run; its longer time cannot be attributed solely to profiling.
Unlike CUDA host enqueue timings, these CPU stages include synchronous execution.
All measurements have zero numerical failures/depth-cap terminations. The 128x128
CPU diagnostic and production renderer produce byte-identical PFMs; see
`production_parity.json`. GPU/CPU floating-point differences can change individual
Monte Carlo paths, so their image difference is not a model-accuracy measurement.

The final independent-worker pipeline validation is in
[`outputs/renewal_worker_pipeline_20261010/summary_pipeline.json`](../../outputs/renewal_worker_pipeline_20261010/summary_pipeline.json).
Use that directory's `run.py pipeline`, `run.py trace_pipeline`, then
`analyze.py pipeline` to repeat the current-vs-preserved-old-binary comparison.
Both baseline variants use h16, point_linear, 512x512 1 spp, batch8192/pool32768,
8 configured ray threads, affinity 0xffff, and one excluded warmup frame/process.
Old timings are 21.72/25.80 s; new timings are 15.14/14.58 s. Median elapsed time
falls from 23.76 to 14.86 s in these two-frame samples, without a 256-spp claim.
The separate final timeline measures 6.797 s of CPU work during host enqueue and
0.586 s during GPU kernels (25.6% of their 2.291-s interval union). CPU initialization,
advance and scatter each run on 8 background workers. All 63,698,578 point queries
in that capture are on the workers. The event-wait host time is 0.056 s, and the
inference thread's empty-queue waiting is 0.412 s. These overlap with other threads;
do not add them to frame time. One active GPU ticket is intentional: publish its
results before entering the next long ATen call, while independent CPU work proceeds.
The final example pool is four batches to leave ready rays behind auxiliary queues.
All C++ 127427 checks pass. New baseline frames and the trace have zero numerical
failures/depth truncations; each old baseline has one path reaching the existing
128-depth safety cap, explicitly retained in the report. Intermediate candidates
and their source/binary hashes remain separately named in that artifact directory.
