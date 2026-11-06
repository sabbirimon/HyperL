# HyperL agent handoff and pending features

## HyperL-owned model engine — 2026-11-06

The owner requests the engine primarily under HyperL, with Meshlit Studio as a client.
[Ordered plan and acceptance](HYPERL_STUDIO_ENGINE_PLAN_2026-11-06.md) covers expert
caching, RAM/VRAM/disk placement, specialist routing and approved device cells.
The first [MoE accounting increment](MOE_RESIDENCY_ACCOUNTING_2026-11-06.md) is
implemented and tested; it does not execute sparse expert paging. Keep existing
CPU/RPC model proof separate from AMD GPU and physical mixed-host qualification.
Agreement work is deferred; preserve current artefacts and conserve build storage.

## Model/sharding/resilience follow-up — 2026-11-02

Delivered: pinned optional native GGUF inference, actual layer/KV split across two
CPU workers, mTLS routes and private local API keys; original f32/HIGHEST JAX decoder
with tensor-parallel weights/KV and checked token-boundary checkpoints; live pull
service/agent, immutable model-profile digests, durable queues/epoch/lease/result
fences, warm sessions and authenticated quota-bounded audit; original native Cauchy
erasure coding with whole-artifact checked repair. [Usage](MODEL_SHARDING_AND_FLEET.md)
and [assurance mapping](HIGH_ASSURANCE_BASELINE.md).

Evidence: 135M pretrained-model local/two-worker token/text agreement, direct and
mTLS routes, two live audited jobs, killed-worker rejection; six JAX tests on four
logical CPU devices; one million synthetic jobs/1,000 durable claim/start/finish
cycles. These are not physical mixed-host/GPU/TPU or million-live-device evidence.

Remaining P0: independent physical-host/architecture qualification and sustained
load; production PKI/revocation/ingress and database replication; multi-controller
checkpoint commit and recovery orchestration; qualified cross-backend collectives;
actual GPU/TPU model acceptance and broader model/numerical conformance. The owner
reported no TPU available and authorized the adapter/tests. Hardware ECC, formal
NIST/FIPS/defence certification and production SLAs are not delivered by software
repair or this control mapping. Preserve the scoped earlier backlog below.


Updated 2026-10-24. This is a prioritized implementation backlog for **standalone
HyperL**. Read [AGENTS.md](../AGENTS.md), [progress](PROGRESS_2026_10_24.md) and
[validation](VALIDATION.md) before coding. Meshlit has its own repository/checkpoint.

## Current delivered foundation

Bounded f32 CPU/C/Python execution; optional checked OpenCL/Metal bridges; Android
Vulkan qualification; authenticated vector workers and checked gather; placement
estimates; pair capability candidates; AES-GCM datasets/rekey; Ed25519 signatures;
SHA-256/384/512 hashing; bounded classical quantum simulator and optional provider
submission bridges. Evidence scope is described in the linked validation record.

## 2026-10-29 update

Native intermediate-buffer liveness is implemented and tested; ordered semantics
and the original admission limit remain intact. Remaining work: actual RSS/energy
measurements, device comparisons and integration qualification. Meshlit Inference
has a separate optional-adapter plan; no license or automatic port is implied.

## 2026-11-02 logical-device increment

Delivered: bounded resident f32 tensor handles/versions, named worker pools,
locality-based dispatch, batched chunk transport and RAM LRU, explicit leases and
job cancellation/status; opaque streaming model storage; advisory automatic/manual
layer residency with LLM size estimates; persistent paginated fleet inventory.
[Contracts, usage and limits](LOGICAL_DEVICES_AND_MODEL_STORAGE.md).

Remaining P0: a qualified LLM operator/model backend, true GPU/TPU device-resident
buffers, tensor-parallel collectives/KV ownership, durable coordinator recovery,
authenticated hierarchical fleet services and cross-cell dispatch. One million
synthetic inventory records do not qualify live fleet scheduling. Existing H0
cost/evidence planner integration and physical mixed-architecture tests remain open.

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

## Broader model formats — 2026-11-06

[Format and adapter plan](MODEL_FORMATS_2026-11-06.md): extend model handling beyond
GGUF, starting with bounded format inspection and a narrow SafeTensors import
into the existing JAX decoder. Container recognition, architecture compatibility,
execution readiness and sharding qualification remain separate. Upcoming formats
require reviewed parsers/adapters; arbitrary downloaded model code is not loaded.
