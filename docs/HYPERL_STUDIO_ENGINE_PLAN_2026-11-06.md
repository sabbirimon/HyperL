# HyperL model execution engine and Meshlit Studio integration

Owner direction: 2026-11-06. Build the engine primarily in **standalone HyperL**;
Meshlit Studio is its first application client. Plan first, implement bounded
increments, test each, and conserve build storage and development credits.
New HyperL implementation retains its Community/Enterprise terms. Earlier Apache
rights and independent engine/model licences remain unchanged.

## Intended result

**HyperL owns memory placement, model sharding, splitting across multiple locations,
expert caching, device scheduling, execution and recovery.** Sharding is a core
engine responsibility, not merely choosing a complete model on another device.

One observable execution service for selecting models, placing compatible model
layers/experts across memory tiers and approved devices, and executing bounded
jobs. HyperL CLI/SDK clients receive the same capabilities as Studio. Studio
adds chat/bot model selection and management dashboards; it does not own a fork
of the execution engine. Existing mobile model defaults remain unchanged.

The first target is an Intel i9-9880H Mac with 16 GB system RAM and Radeon Pro
560X with 4 GB dedicated VRAM. Reserve memory for macOS, the IDE, activations,
KV, transfer buffers and other workloads. Do not add RAM and VRAM as if they were
one uniform allocation. Disk holds verified weights; disk capacity is not RAM.

### Explicit sharding and placement scope

- Layer/pipeline splitting: assign different transformer layers and their required
  state to compatible local or remote workers.
- Tensor splitting: partition supported tensors/operators across compatible workers
  with qualified collectives and numerical contracts.
- Expert splitting: place a supported MoE model's experts across memory tiers or
  workers while preserving its learned router and shared layers.
- Storage splitting: distribute verified model artifacts/chunks across approved
  disks or storage nodes; fetch and verify required chunks before execution.
  Stored fragments alone are not an executable distributed model.
- Multiple locations: the same device's CPU/RAM/GPU, several local devices, and
  explicitly paired remote execution cells. Backend compatibility, bandwidth,
  latency, available memory and owner scope constrain each placement.

Use explicit partition manifests, immutable model identities, bounded transfers,
placement telemetry and cancellation. Rebalancing/recovery must preserve committed
state and must not replay uncertain tool actions. Existing native CPU layer and
logical-CPU JAX tensor evidence remains limited to its tested scope. General expert
paging, mixed-device tensor collectives and automatic cross-location recovery
remain planned acceptance gates.

## Existing foundation and evidence

- Native llama.cpp adapter, model/profile hashes, bounded loopback transport,
  layer/KV placement and authenticated fleet jobs already exist. The November 2
  evidence covers SmolLM2 135M, two CPU workers on one physical host, matching
  short greedy generation and failure rejection. See [model execution](MODEL_SHARDING_AND_FLEET.md)
  and [validation](VALIDATION.md). This is not AMD GPU or large-model proof.
- The original JAX decoder has separate tensor-parallel and checkpoint tests on
  four logical CPU devices. These do not qualify physical multi-GPU/TPU execution.
- Model storage/residency planning and fleet inventory already exist. Extend them
  rather than adding a parallel scheduler. Planning estimates are not allocations.
- Meshlit's embedded HyperL recipe screen is an older, separate integration;
  do not confuse its capabilities with the newer standalone model adapters.
- Studio's second bundled model is Qwen3-4B Q4_K_M, with Qwen2.5-1.5B retained as
  the startup default. It is a dense reasoning model, not an MoE model. Source and
  artifact verification precede native load/reply acceptance.

## Three separate mechanisms

| Mechanism | What it does | Required correctness boundary |
| --- | --- | --- |
| Specialist model routing | Select a complete coding, reasoning, vision or speech model for a task; optionally chain/review outputs | Explicit capabilities, model identity, budget, data destination and supported fallback |
| Native MoE execution | A supported model's learned router selects token-level experts; cache/load those experts across tiers | Preserve the original router, shared layers, tokenizer and exact expert tensors; CPU-reference output checks |
| Layer/tensor distribution | Execute parts of the same model across compatible approved devices | Backend/operator compatibility, transport authentication, memory admission and numerical qualification |

Independent dense models cannot be split into interchangeable MoE experts without
a separately trained compatible architecture. Cloud APIs remain complete remote
model calls: hidden weights cannot participate in local layer/expert sharding.
Licensed downloadable proprietary weights can use a compatible adapter only when
the owner's licence permits it.

## Memory and scheduling design

Use a versioned model manifest: immutable model digest, architecture, quantization,
tokenizer, context/KV format, shared tensors, per-layer experts/router, required
operators, licence, byte sizes and integrity provenance. Keep device capability
snapshots separate from model metadata and runtime allocations.

Maintain bounded VRAM, host RAM and disk tiers. Cache keys include model digest,
layer, expert, precision and backend revision. Pin in-use tensors; eviction cannot
remove them. Reserve KV/activation/scratch and transfer buffers before admission.
Count all stored experts, not merely active parameters. CPU execution and transfer
costs must be included in end-to-end measurements. Memory pressure triggers an
explicit supported plan change or a clean rejection, not silent overcommit.

Start expert prefetch with measured access history and a bounded cache. A prediction
may fetch likely experts early; it may not substitute an expert or skip the learned
router's actual choice. Misses perform the correct load/compute and are counted.
Learned predictors follow only after held-out evaluation, with deterministic
fallback and rollback. Do not label simple access counters as reinforcement learning.

Expose real load location, resident bytes, transfers, cache hits/misses, queue time,
prefill/decode timing, selected backend, thermal state when available, and errors.
Unknown telemetry stays unknown. No raw prompts, credentials or tensors in routine
health logs. Stop cancels owned work and reports uncertain remote termination.

## Ordered implementation and acceptance

| Stage | Deliverable | Acceptance / current state |
| --- | --- | --- |
| H0 | Reconcile existing adapters and publish this plan | Written; preserve older evidence scopes |
| H1 | Extend existing model-plan with conservative MoE/shared/router/cache accounting | Implemented; 13 focused checks passed, including CLI, malformed input and full-weight capacity rejection; no inference claim |
| H2 | Studio client adapter to existing HyperL profile/job service | Pin API/profile identity, capability negotiation, explicit connection and cancellation; compare direct native and Studio execution |
| H3 | Qualified single-device CPU + AMD partial offload | Separate experimental binary; CPU-reference matrix tests, real GGUF reply, peak memory, warm/cold latency and Stop; ship only correct and useful paths |
| H4 | Tiered layer/expert cache for one supported small MoE | Resident reference equivalence, eviction/miss/corruption/low-memory cases, bounded prefetch, cold/warm end-to-end benchmarks |
| H5 | Approved Mac + Samsung / LAN execution cells | Reuse authenticated layer protocol; actual assignments, per-worker allocations, disconnect/thermal failure and project-task tests; WAN separately qualified |
| H6 | Specialist task routing for chat, bots and IDE | Start explicit deterministic recipes; provider budgets and privacy gates, visible routing reason, no silent paid fallback |
| H7 | Optional training and adaptive scheduling | Soup adapter evaluation/export provenance, held-out routing/prefetch metrics, rollback; no autonomous untested weight replacement |
| H8 | Release and additional platforms | Reproducible packaging, notices, signing, upgrade/rollback and physical OS/device acceptance |

Stages H2–H8 remain separate gates. Passing H1 does not prove expert paging or GPU
execution. Complete current Studio two-model/native-menu delivery before broad new
UI work. Do not replace the working CPU backend with an experiment.

## Upstream reuse decisions

- [Soup](https://github.com/MakazhanAlpamys/Soup): already integrated as an optional
  Meshlit host training companion. Its layer streaming is useful research for
  bounded transfer/lifetime handling; training and inference have different
  execution requirements. Revalidate a pinned version and target hardware before
  reuse. The 4 GB upstream results do not certify Radeon 560X compatibility.
- [DeepEP](https://github.com/deepseek-ai/DeepEP): expert dispatch/combine and
  prefetch reference for eligible remote GPU workers. Current upstream requires
  Linux, NVIDIA Hopper-or-newer, CUDA and appropriate interconnects. It is not an
  AMD Intel-Mac backend. Review the separate Ascend adapter independently.
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): backend-specific kernel
  research, not a portable replacement for all devices. Pin and qualify any use.
- Reuse the existing pinned llama.cpp adapter for supported GGUF execution. Add
  native kernels only where a measured unsupported operation justifies them.

## Benefits to HyperL itself

The same model manifests, residency plans, telemetry, cache contracts and job
service become reusable CLI/SDK capabilities for other applications. The work
adds real heterogeneous-model conformance fixtures, bounded memory tests and
reproducible hardware evidence to HyperL. Separate adapters preserve the original
vector ABI and make unsupported hardware visible. Claims of speed, model size or
sharding scale require the accompanying measurements, not a new product name.

## Work and storage discipline

Reuse verified model files and copy-on-write packaging. Keep one latest accepted
build and a rollback candidate; remove only identified superseded generated files.
Protect source changes, signing material, credentials, user data, unique artifacts,
active builds and useful test evidence. Publish a concise checkpoint after each
stage with exact tests, limitations and next commands; avoid repeated full builds
when focused checks establish the changed contract.

## First implementation checkpoint

[MoE residency accounting](MOE_RESIDENCY_ACCOUNTING_2026-11-06.md): 84 Python
tests passed, 5 optional JAX tests skipped; Gradle distribution build successful;
5 native tests passed. Separate AMD prototype: 11 numeric checks and short matching
CPU/GPU output, but slower, so no production promotion. Engine remains the current
priority; Studio agreements were explicitly deferred by the owner.

The isolated GPU [measurement record](evidence/studio-amd560x-partial-offload-20261106.json)
and [research patch](evidence/studio-amd560x-scalar-20261106.patch) are retained here
for HyperL engine work. They are not installed backends; the original upstream
licence and numerical gates still apply. Meshlit retains a copy of the same evidence.

## Broader model formats — 2026-11-06

[Format and adapter plan](MODEL_FORMATS_2026-11-06.md): extend model handling beyond
GGUF, starting with bounded format inspection and a narrow SafeTensors import
into the existing JAX decoder. Container recognition, architecture compatibility,
execution readiness and sharding qualification remain separate. Upcoming formats
require reviewed parsers/adapters; arbitrary downloaded model code is not loaded.
