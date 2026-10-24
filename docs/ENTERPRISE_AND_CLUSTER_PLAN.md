# HyperL enterprise, datacenter and AI-cluster plan

Owner requirements recorded 2026-10-08 (Asia/Dhaka): keep HyperL useful on a single
phone or workstation, and design a path to large datacenters, enterprise AI clusters
and heterogeneous compute. This is a proposed architecture and delivery plan. It
does not add a deployed service, production scheduler or certified enterprise SKU.

**2026-10-24 scope clarification:** dedicated compute clusters for large entities,
mainstream datacenters and hyperscalers are primary targets. Phones are one edge
profile. High-end fabrics, FPGA/ASIC adapters, ultra-large datasets and explicitly
high-precision numerical modes are required tracks; none is enabled by a label or
an untested scale limit. See [fabric negotiation](ENTERPRISE_FABRICS.md).

## 1. Current foundation and deployment profiles

Today: standalone Java 17+ desktop CLI/GUI, bounded f32 CPU reference, portable C ABI
preview, source emission, optional explicitly installed OpenCL bridge, encrypted
local dataset streaming, memory admission, cluster-profile validation and read-only
node metadata probing. Separate owner-terminal evidence covers four bounded Metal
cases on Radeon Pro 560X and a standalone Samsung Adreno 506 Vulkan host has its
own listed-case evidence. These do not qualify full models or mixed clusters.
The full SDK, phone app packages and a production fleet control plane are not delivered. Alpha.6
adds a separate [bounded vector shard executor](DISTRIBUTED_EXECUTION.md), which
has actual same-host process and TLS tests. A separate Android probe now also passes
bounded Samsung ARM64/Mac x86-64 CPU splits over USB and direct Wi-Fi TLS; see
[physical evidence](ANDROID_CROSS_PLATFORM_PROBE.md). Enterprise scheduling, model
partitions, durable recovery and the scale below remain qualification gates.
Preserve existing alpha.5 installers, generated contracts, compensated CPU reduction
and HLM2 authenticated storage/rekey. New orchestration follows the
[Community/Enterprise licence scope](LICENSE_HISTORY.md); prior Apache grants remain.

Use one versioned program/artifact format, adapter negotiation and numerical contract
across these profiles; add optional services around it rather than requiring every
installation to run a datacenter stack.

| Profile | Useful target | Proposed installation footprint and constraints |
|---|---|---|
| Single Android/iOS phone | Local AI preprocessing, vision/security analysis and a cluster client | Native C ABI/runtime wrapper and lightweight mobile UI later; normal app sandbox first, CPU baseline, qualified GPU/NPU adapters; no JVM desktop GUI on iOS/Android |
| Workstation / edge appliance | Development, local inference and small datasets | Current CLI/GUI foundation; future native backend/model engine; optional explicitly configured remote client |
| Small private cluster | Team inference and batch AI work | First authenticated 3–5 independent-host pilot; a durable coordinator and bounded workers; optional desktop client, no GUI required on servers |
| Enterprise datacenter | Shared GPU/NPU fleets, quotas, batch/inference and governance | Headless services with tenant isolation, external identity, metrics, registry, lifecycle and audit; integrate an existing orchestrator first |
| Dedicated hyperscaler cluster | Large managed fleets and isolated organization-owned compute | Hierarchical scheduling/failure domains, bounded metadata fanout, capacity reservations, SLOs, automated draining/upgrades and tested disaster recovery; no demonstrated hyperscale claim yet |
| Multi-datacenter / region | Data residency, disaster recovery and controlled federation | Independent failure domains and local scheduling, explicit cross-site policy and bounded federation; WAN cost/latency included |

A phone remains useful by itself. Remote access is optional and never required for
its local CPU mode. Mobile adapters must pass operator, precision, memory, lifecycle
and device-loss checks. Android root extensions are separate owner-installed plugins;
non-root is the default. iOS uses supported app entitlements/native APIs, not root
assumptions. See the [mobile and edge plan](MOBILE_AND_EDGE_PLAN.md).

## 2. Service architecture to build later

### Large datasets and precision tracks

Work on datasets larger than aggregate RAM through bounded chunks, partitioned
object-storage manifests, resumable integrity-checked transfer and explicit spill.
Account for staging, indexes, checkpoints, network traffic and temporary copies;
storage capacity is not accelerator memory. Keep compute near approved data,
support backpressure and cancellation, and validate restart/reconciliation before
claiming large-scale capacity. Start with measured GB/TB workloads and larger
reproducible deployment gates; no PB/EB operating capacity is established today.

Precision is an explicit workload contract, independent of dataset size. Current
graphs use ordered f32; the separate compensated sum uses binary64 accumulation
followed by finite f32 output. It is not an arbitrary-precision engine. Next tracks:

- FP64 operations with actual backend/operator coverage, reproducibility and error tests.
- Binary128 and explicit BF16/mixed-precision tracks with distinct storage,
  compute, accumulation, output and wire types; BF16 is reduced precision.
- Optional decimal/arbitrary-precision reference providers with requested digits,
  rounding modes, bounded intermediates and deterministic serialization.
- Numerically stable reductions and linear algebra; report conditioning/error
  bounds where applicable and preserve the requested numerical mode across shards.
- Explicit mixed-precision modes only when the caller accepts their error contract;
  unsupported high precision is rejected rather than silently rounded down.

High precision can multiply storage, transfer and compute costs. Admission and
placement must account for bytes per element and precision-specific benchmarks.
Both precision and large-data claims require complete reference-checked workloads,
failure recovery and measured resource use, not an aggregate memory/FLOPS total.
The [precision/HPC implementation gates](PRECISION_AND_HPC_PLAN.md) prioritize
typed admission and an f64 CPU baseline before qualified native acceleration.
Supercomputer jobs select accuracy, latency, throughput or capacity objectives
without silently lowering the accepted precision.

Quantum work is a separate asynchronous workflow type. The
[bounded Python simulator and SDK bridges](QUANTUM_EXTENSION_PLAN.md), including
[OriginQ](ORIGINQ_ADAPTER.md), do not provide a production quantum job ledger or
physical QPU qualification. Provider budgets, durable reconciliation, region scope
and cancellation acknowledgments remain enterprise gates.

Use owner-approved Rust cluster services and C/C++ kernels/adapters, linked by a
versioned C ABI; see [native performance decision](NATIVE_PERFORMANCE_ARCHITECTURE.md).
Separate the control plane from timed native compute and bulk-data transfer:

- **Control API and console:** authenticated submissions, templates, previews, job
  status, policy, tenant/project administration and human review. Desktop/CLI/mobile
  use the same versioned API. Expose no public listener by default.
- **Identity and policy service:** enterprise OIDC/SSO, workload identity, short-lived
  credentials, mTLS, RBAC/attribute checks, scoped roles and approval records. Reuse
  a selected identity provider; do not create a new password/crypto system.
- **Admission and quota service:** validate artifact hash, operators/precision,
  permitted datasets/targets, CPU/RAM/VRAM/scratch/network limits, tenant budget,
  concurrency and deadlines before allocating a lease. Logical scheduling limits
  require separate OS/container/VM enforcement.
- **Job ledger and scheduler adapter:** durable job transitions, optimistic revisions,
  idempotency keys, tenant fairness, priority/aging and external scheduler job IDs.
  Exactly-once effects are not assumed; reconcile retries and publish verified results
  once. One scheduler owns a resource lease; avoid competing allocators.
- **Worker/node agent:** enrollment, bounded observed capabilities, health, exact
  driver/adapter identity, artifact verification, resource reservation, isolated
  dispatch, cancellation acknowledgment and result provenance. Untrusted extensions
  execute in a reviewed VM/sandbox, not in the coordinator process.
- **Artifact and compilation registry:** content-addressed programs/models/kernels,
  signatures/SBOM, approved provider plugins, reproducible compiler records and cache
  identity covering IR/shape/precision/runtime/compiler/driver. No automatic execution
  of newly discovered plugins or instructions embedded in model/dataset metadata.
- **Data and checkpoint service:** tenant-owned object storage handles, manifests,
  integrity checks, encrypted chunks, resumable transfers and bounded staging. Key
  handles come from enterprise KMS/HSM integration; keys are not in programs/prompts.
- **Telemetry and fleet health:** real traces/metrics/logs, bounded cardinality and
  sampling, exporter outage/backpressure handling, privacy/redaction, alerts and
  per-device qualification. No fabricated utilization, memory or speed charts.
- **Gateway integration:** optional LLM/inference and A2A/MCP gateways for authenticated
  clients and agents. Protocol gateways do not themselves provide GPU scheduling,
  isolation or a model engine. Keep Meshlit's gateway work a separate integration.

Proposed durable metadata: tenants/projects, identity-to-role mappings, quotas,
artifact hashes, dataset handles, node leases, job attempts/checkpoints, measurements,
audit events and stop epochs. Use a proven transactional store and well-defined
backup/restore semantics. Database/queue products and versions need an ADR before
implementation; do not select five systems when one satisfies the pilot.

## 3. Useful features and services, in delivery order

| Priority | Feature/service | Benefit and acceptance requirement |
|---|---|---|
| P0 | Headless node capability/qualification report | Distinguish discovered devices from usable operators; stale measurements expire, unavailable remains unknown |
| P0 | Authenticated job submission + durable ledger | Submit/query/cancel bounded work; reject duplicate/unauthorized submissions, survive coordinator restart without publishing duplicate output |
| P0 | Tenant/project identity, isolation and quotas | Separate credentials, storage and device leases; negative tests prove no cross-tenant result or log access |
| P0 | Human/agent policy and global emergency stop | Agents have separately scoped default-off authority; stop is durable, blocks new admission and shows acknowledgments/unreachable nodes |
| P0 | Reproducible execution evidence | Program/model/inputs/adapter hashes, precision, outputs, measured transfers and timings; CPU reference is named, never disguised as GPU |
| P1 | Kubernetes or Slurm scheduler adapter | Use the operator's existing fleet; qualify cancellation, resource ownership, device claims, upgrades and failure reconciliation on one chosen system |
| P1 | Batch queues, fairness and reservations | Bound concurrency; prevent starvation; make preemption/checkpoint support explicit per workload/backend |
| P1 | Topology and data-local placement | Use measured NUMA, PCIe, HBM/unified domains and fabric costs; keep compatible devices together when it improves measured total cost |
| P1 | Native artifact/cache service | Compile once using exact compatibility keys; integrity/signature checks and cache eviction; no cached binary across incompatible drivers |
| P1 | OpenTelemetry export and dashboard | Actual wait/transfer/compile/run/result stages; current/stale/missing values clearly marked; privacy-safe labels and bounded queues |
| P1 | Dataset registry, encrypted staging and checkpoints | Integrity-verified resumable transfers; quota/retention/lineage and recovery from truncated/corrupt checkpoints |
| P1 | Rolling upgrades, drain and rollback | In-flight jobs have pinned ABI/artifacts; drain/resume rules and compatibility tests; restore prior version after failure |
| P2 | Inference service and model registry | Tensor/model engine must exist first; admission, load tests, bounded batching, model version/canary and complete end-to-end correctness |
| P2 | Distributed training/evaluation pipelines | Real collectives/autograd/checkpoint correctness and rendezvous; no claim from f32 elementwise source emission |
| P2 | Chargeback and energy/thermal policy | Meter observed accelerator/time/storage/network with allocation provenance; pricing supplied by operator, no invented dollar savings |
| P2 | Multi-site federation and disaster recovery | Data residency and per-site leases; documented RPO/RTO, restore drills and split-brain prevention before automatic failover |
| P2 | Enterprise support/LTS service | Maintainer/security process, supported platform matrix, signed releases, patch windows and compatibility policy before any SLA or LTS promise |

Useful optional services: deployment/adapter qualification, private registry and
on-prem operations tooling, migration/conformance assistance, benchmark reports,
training and support. Eligible community uses remain free without a hosted account
or activation server. New rights follow [community/enterprise terms](LICENSING.md):
large-entity production needs a paid written agreement after its six-month trial.
Earlier Apache rights remain. This plan creates no SLA, price or certification.

## 4. Proven systems to evaluate before inventing replacements

These are researched integration candidates, not bundled/installed integrations.
Select and pin exact releases, review licenses/transitive components and record an
architecture decision before importing code. Product SDKs/drivers remain separately
licensed. Compatibility is demonstrated per selected OS/driver/hardware combination.

- [Kubernetes Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/resource-management/dynamic-resource-allocation/dra-api/)
  provides device request/allocation objects. Evaluate DRA/device-plugin integration
  for a chosen cluster version; do not assume every vendor driver implements it.
- [Slurm GRES](https://slurm.schedmd.com/gres.html) and
  [heterogeneous jobs](https://slurm.schedmd.com/heterogeneous_jobs.html) provide HPC
  device/job allocation interfaces. An adapter should respect existing job ownership,
  partition/account policy and isolation, not run a second scheduler beside Slurm.
- [Ray resource scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
  is an AI orchestration candidate. Its logical resource admission does not enforce
  physical RAM/CPU limits; HyperL needs actual worker isolation and observed capacity.
  Reuse it only where it avoids a redundant distributed runtime.
- [OpenTelemetry gateway pattern](https://opentelemetry.io/docs/collector/deploy/gateway/)
  is a candidate for per-cluster telemetry collection. Keep telemetry traffic away
  from latency-sensitive paths and bound exporter buffering during backend outages.

Initially integrate one orchestrator, one durable metadata store and one telemetry
path. Re-evaluate architecture after actual pilot measurements. Other fabric, GPU
management, storage and identity integrations need separate qualification/licensing
records. Research CUDA/ROCm/CANN/Metal/DirectML/FPGA APIs through public interfaces;
never infer redistribution rights or cross-vendor kernel support.

## 5. Memory, network and hardware contracts

Memory requests must name domains: host RAM, NUMA/locality, HBM/GDDR, shared unified
memory, SRAM/CXL where observed and SSD/NVMe staging. Count shared domains once;
reserve compiler/runtime/communication overhead as well as tensors. Track free,
reserved and allocated separately; allocation failure and fragmentation can still
occur. Current JVM admission is an estimate, not VRAM accounting or a reservation.

Future encrypted out-of-core work needs supported operators, I/O/transfer costs,
chunk lifetime/refcounts, backpressure and deadlines. Keep keys in provider key
handles, enforce scratch quotas and clean uncommitted output after failures. Do
not advertise HBM discovery, automatic spill or multi-terabyte throughput today.

Networking starts with authenticated IPv6/IPv4 control and bounded HTTPS transfers.
Bulk transport may later use qualified UCX/RDMA/InfiniBand/RoCE/Ethernet/optical
adapters, with explicit fallback selection and end-to-end measurements. Link medium,
nominal line speed or vendor name does not prove usable RDMA or low tail latency.
TLS/authentication/encryption costs stay in reported total performance.

Mixed devices require a correctness-first compatible-operator partition and explicit
transfer accounting. Homogeneous islands may enable optimized fused/AOT/SIMD paths
once benchmarks justify them; "turbo" means a qualified software execution policy,
not automatic clocks/voltage or unchecked precision changes. Thermal/power ceilings
remain operator/device policy. Unsupported precision/backend combinations fail.

## 6. Human console and agent operating model

Planned console pages: fleet/capability inventory; queue/job details and output;
models/artifacts/datasets; quotas/identity; topology/memory; traces/profiler; audit,
upgrades and emergency stop. Keep developer editing in the workbench; use a thin
console API client for server administration. No dummy fleet graphs in today's GUI.

Every measured value has source, unit and observation time; show missing/stale state.
Job details distinguish requested, admitted, dispatched, running, stop requested,
cleanup acknowledged, completed and failed. Show memory pressure and bottleneck
causes. Export evidence with redaction and a reproducible command/configuration.

Separate human administrative scope from scoped agent automation. Humans can enable
or revoke each function and set ceilings on node count, spend, data, execution and
time. Agents cannot resume a globally stopped cluster or grant themselves broader
rights. A stop epoch blocks new leases and requests cancellation of old work. The
console reports acknowledgment coverage and unreachable nodes; a network partition
cannot be described as confirmed remote termination. Workers need expiring leases
and fail-closed rules appropriate to their backend, with device cleanup evidence.

## 7. Size limits and measured rollout gates

The current **configuration-only** cluster profile permits inventory 1–1,000,000
(default 10,000) and active workers 1–256 (default 8). Agent management is separately
default-off with its own bound. These are validated settings, not a million-node
scheduler or a throughput result; the command never dispatches cluster work.

Proposed sequence: one desktop reference -> 3–5 independent CPU hosts -> one qualified
GPU family -> 16/64/256-host trials as resources permit -> larger sharded inventory.
Do not lift executing concurrency until queue/metadata/storage/network capacity,
fault recovery and observability have been measured. Simulated large inventories
are named simulation, separate from live capacity. Mobile clients do not imply
million-phone distributed training or availability.

Pilot gates: authenticate/revoke; tenant denial; real output agreement; cancellation;
worker death; expired lease; duplicate delivery; corrupt artifact/chunk; capacity
exhaustion; connection/body timeouts; coordinator restart; upgrade/drain/rollback.
Then test network partition, backend/device loss, telemetry failure, key rotation,
restore and independent security review. Record exact hosts/OS/drivers, workload,
sample counts, p50/p95/p99, transfer/compile/run costs, peak memory, correctness,
queue fairness and recovery time. Agree RPO/RTO/SLO targets with the pilot operator;
no invented availability, speedup, certification or universal target support.

## 8. Concrete next engineering packages

The 2026-10-24 [sharding contract](SHARDING_EXECUTION_CONTRACT.md) is the detailed
acceptance specification for these packages:

1. H0: versioned capability/memory/link/shard-plan objects and strict bounded
   admission, deterministic estimates and compatibility corpus. No listener or
   dispatched work is implied by a successful plan.
2. H1: extend the existing split/map/gather executor from qualified same-host
   processes to independent physical CPU hosts, and add durable attempt/epoch
   recovery. Preserve existing token/TLS, replay, credit, cancellation and checked
   result behavior while proving restart, revocation and remote failure handling.
3. H2–H4: mixed qualified processors, typed tensor/model path and mandatory real
   model-layer sharding with residency/KV/activation evidence. Retain a whole-model
   baseline and separately report capacity, latency and throughput.
4. H5–H6: measured variable-speed data lanes, topology-aware balancing, qualified
   collectives, then 3–5-host durable operations and one orchestrator integration.
   Redis-compatible caching is optional; the authoritative ledger and ownership
   fencing remain correct if the cache fails or evicts all entries.
5. Operational release: enforced tenant isolation, real metrics, upgrade/backup/
   restore runbooks, SBOM, security review and explicit supported-platform policy.
   Larger fleet tests follow the measured rollout gates above. Dynamic explanation
   views display actual placement/bottlenecks and mark unavailable measurements.

The fixed Android/desktop CPU probe and listed Radeon/Adreno cases have separate
physical evidence. No production enterprise fleet or physical QPU is qualified.
Extend those bounded results through the independent-host, recovery and scale
gates above; fixtures and configured node limits cannot substitute for deployment tests.
