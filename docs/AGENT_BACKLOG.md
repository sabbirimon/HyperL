# HyperL agent handoff and pending features

Updated 2026-10-24. This is a prioritized implementation backlog for **standalone
HyperL**. Read [AGENTS.md](../AGENTS.md), [progress](PROGRESS_2026_10_24.md) and
[validation](VALIDATION.md) before coding. Meshlit has its own repository/checkpoint.

## Current delivered foundation

Bounded f32 CPU/C/Python execution; optional checked OpenCL/Metal bridges; Android
Vulkan qualification; authenticated vector workers and checked gather; placement
estimates; pair capability candidates; AES-GCM datasets/rekey; Ed25519 signatures;
SHA-256/384/512 hashing; bounded classical quantum simulator and optional provider
submission bridges. Evidence scope is described in the linked validation record.

## Required and suggested next work

| Priority | Feature | Acceptance before claiming delivered |
|---|---|---|
| P0 | Connect placement estimates to actual dispatch | Revalidate identity, workload, memory and lane observations at dispatch; handle stale/revoked evidence without using old approvals |
| P0 | Durable jobs and recovery | Persistent correlation, duplicate suppression, checkpoint integrity, restart/cancel tests; no duplicate effects after uncertain completion |
| P0 | Full model sharding (owner requirement) | Actual model/tensor/layer runtime, model-state/KV ownership, exact supported precision, multi-device end-to-end reference and failure tests |
| P0 | Load balancing and network telemetry | Measured transfer/queue/service costs, bounded fairness/backpressure, heterogeneous links, actual throughput/latency comparisons |
| P1 | Native CPU and tensor operators | Versioned shape/dtype contract, SIMD/NUMA evidence, matmul/attention/complex math, reference correctness and full cost accounting |
| P1 | GPU, TPU and NPU adapters | CUDA/HIP/TPU/vendor NPU execution on actual targets; operator/runtime/driver/precision qualification, no disguised fallback |
| P1 | Precision expansion | FP64, BF16, IEEE binary128 and optional higher precision under [precision gates](PRECISION_AND_HPC_PLAN.md); never silently lower precision |
| P1 | Enterprise fabrics and devices | Real RDMA/NVLink/CXL/FPGA/ASIC/DPU integrations with approved drivers/firmware, measured copies and resource lifetimes |
| P1 | Very large datasets and memory tiers | Resumable authenticated partitions, quota-managed spill, NUMA/VRAM/HBM accounting and recovery; storage is not RAM |
| P1 | Cluster/tenant security | Durable identity/revocation, isolation, scoped quotas, auditable stops, signed-adapter lifecycle, adversarial protocol tests |
| P1 | Post-quantum cryptography | Qualified provider ML-KEM/ML-DSA/SLH-DSA and hybrid migration under [crypto plan](CRYPTOGRAPHY_EVOLUTION_PLAN.md); new formats, no downgrade |
| P2 | Quantum provider qualification | Real installed SDK translation/simulation, authorized cloud tests, durable IDs/cost limits, provider cancellation, SDK TLS/redirect audit |
| P2 | Framework and future-protocol adapters | PyTorch/TensorFlow/ONNX/CANN ecosystem integration with versioned schemas/provenance; unsupported requests fail explicitly |
| P2 | In-memory coordination/cache | Optional Redis/RESP-style metadata cache, authoritative ledger separate, bounded eviction and recovery tests |
| P2 | Developer tools | Semantic IDE support, step debugger, profiler and trace correlation after runtime contracts stabilize |
| P2 | Release engineering | Reproducible artifacts, SBOM/provenance and platform qualification; support promises only with maintainers and a maintenance policy |

Suggested features above are proposals, not permission to deploy cloud resources,
spend money, install drivers, relax security or claim hardware support. Use owner
configuration for those operations. Preserve existing licensing and numerical/wire
contracts. Keep keys out of source, prompts, examples and logs.

## Recording each increment

For every meaningful fix, record the symptom, cause, implementation files, test or
reproducer, observed result and remaining limits. Link the commit/PR after pushing.
Update the relevant guide and current status, while preserving dated measurements
as historical evidence. Follow [BUILD_AND_TEST.md](BUILD_AND_TEST.md) before shipping.

## Research follow-up — 2026-10-26

See [math assessment](MATH_RESEARCH_2026_10_26.md). Independent reduction-oracle regression coverage is added. Matrix and scheduling theorem adoption remains gated on concrete algorithms, applicable assumptions and target benchmarks; no production planner or cipher change.

## Owner-requested performance expansion — 2026-10-26

[Mathematics-informed performance roadmap](MATH_PERFORMANCE_ROADMAP.md) defines M0–M5: measured baselines, bounded matrix contracts, practical qualified kernels, research constructions, restricted scheduling and product adoption. These are future gates; no unmeasured speedup or new cipher is claimed.
