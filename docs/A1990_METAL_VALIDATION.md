# Radeon Pro 560X Metal workload validation — 2026-10-08

The owner ran the corrected HyperL qualification harness in a macOS terminal.
The saved report was inspected and its case/device/completion metadata checked.
Result: **`passed-listed-cases` — four GPU executions and one expected rejection**.
This validates these generated workloads on this exact GPU, not all AMD devices,
full AI models, general numerical conformance or a speedup benchmark.

## Environment and provenance

- Owner-identified MacBook Pro A1990; Intel/x86-64 host.
- Selected device: **AMD Radeon Pro 560X**, exact-name selection, managed buffers.
- Discovery also lists Intel UHD Graphics 630; no Intel execution is qualified.
- Bridge built with existing AppleClang 17/system Metal; CLI uses OpenJDK 21.0.12.
  No new driver or privileged installation.
- Working checkout contains [JSON contract fix `37a8081`](https://github.com/sabbirimon/HyperL/commit/37a8081).
  The report does not embed binary hashes/revisions; this records the observed
  checkout/build workflow rather than a signed execution attestation.
- The fix's [OS CI run](https://github.com/sabbirimon/HyperL/actions/runs/37714848101)
  passes Windows, Ubuntu and macOS jobs; these are separate from the Radeon run.
- Report time: **2026-10-08 07:54:17 Bangladesh time** (01:54:17 UTC).
- Report format: `hyperl-metal-qualification/1`; local file
  `a1990-metal-terminal-compute-retry-report.json` is retained outside source commits.
- Original report SHA-256:
  `9cb207668efc5ec7d714f85f6f80a317362d3570a151b4e039efd4ab22d15521`.
- Host `sw_vers` reports macOS **15.8.1**. The terminal Python's `platform.mac_ver()`
  wrote `10.16` in the original report; that field is not used as authoritative
  OS identification. The original report is preserved unchanged.

## Observed cases

Times are milliseconds from this single run. `gpuMs` is the native command's GPU
timestamp interval, including synchronization; it is not isolated arithmetic time.
Full CLI wall time includes JVM/process startup, JSON/file IO, compilation,
transfer, native completion and mandatory CPU verification. These observations
do not establish throughput or a speedup.

| Workload | Elements | Compile | Submit and wait | GPU interval | Full Metal CLI wall | Outcome |
|---|---:|---:|---:|---:|---:|---|
| multiply → ReLU | 3 | 4.517 | 0.868 | 0.0426 | 294.078 | CPU comparison passed; sample `[0,6,12]` |
| multiply → ReLU | 257 | 1.879 | 1.301 | 0.0550 | 310.131 | CPU comparison passed |
| multiply → ReLU | 65,536 | 1.915 | 2.160 | 0.0939 | 462.787 | CPU comparison passed |
| multiply → add → ReLU | 5 | 117.325 | 0.943 | 0.0411 | 408.688 | Verified output `[0,0,0,0,14]` |
| overflow before ReLU | 1 | — | — | — | — | Expected nonfinite rejection; no GPU dispatch |

All four GPU cases report `gpuCompletionConfirmed: true`, the exact Radeon name,
and `storageMode: managed`. The harness requires explicit `backend: METAL_GPU`
and `cpuVerified: true`; missing/false verification cannot count as a pass.
Elementwise results use the existing `1e-6 * max(1, abs(expected))` CPU tolerance.
The earlier failed report remains evidence of the pre-fix JSON omission.

## Reproduce and remaining acceptance

Build corrected source and follow [the Metal guide](MACOS_METAL.md). Choose a new
report path every time. Published alpha.4 archives precede the JSON fix and remain
unchanged. The agent's restricted execution context has separate empty GPU probes;
this successful run came from the owner's terminal.

Remaining work includes repeated/warm execution with separate transfer timings,
broader shapes/values, device loss/cancellation, memory pressure, native GUI GPU
interaction and other exact devices. Reductions, tensors, model inference/training,
ROCm/HIP and distributed GPU work are not qualified by these cases. The probe's
`inferenceQualified: false` remains correct: discovery and elementwise success
do not qualify a model engine.
