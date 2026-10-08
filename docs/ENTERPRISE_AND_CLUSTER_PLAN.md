# HyperL enterprise, datacenter and AI-cluster plan

Owner requirements recorded 2026-10-08 (Asia/Dhaka): keep HyperL useful on a single
phone or workstation, and design a path to large datacenters, enterprise AI clusters
and heterogeneous compute. This is a proposed architecture and delivery plan. It
does not add a deployed service, production scheduler or certified enterprise SKU.

## 1. Current foundation and deployment profiles

Today: standalone Java 17+ desktop CLI/GUI, bounded f32 CPU reference, portable C ABI
preview, source emission, optional explicitly installed OpenCL bridge, encrypted
local dataset streaming, memory admission, cluster-profile validation and read-only
node metadata probing. No actual GPU is qualified on the local host. The full SDK,
phone packages, server control plane and distributed execution are not delivered.

Use one versioned program/artifact format, adapter negotiation and numerical contract
across these profiles; add optional services around it rather than requiring every
installation to run a datacenter stack.

| Profile | Useful target | Proposed installation footprint and constraints |
|---|---|---|
| Single Android/iOS phone | Local AI preprocessing, vision/security analysis and a cluster client | Native C ABI/runtime wrapper and lightweight mobile UI later; normal app sandbox first, CPU baseline, qualified GPU/NPU adapters; no JVM desktop GUI on iOS/Android |
| Workstation / edge appliance | Development, local inference and small datasets | Current CLI/GUI foundation; future native backend/model engine; optional explicitly configured remote client |
| Small private cluster | Team inference and batch AI work | First authenticated 3–5 independent-host pilot; a durable coordinator and bounded workers; optional desktop client, no GUI required on servers |
| Enterprise datacenter | Shared GPU/NPU fleets, quotas, batch/inference and governance | Headless services with tenant isolation, external identity, metrics, registry, lifecycle and audit; integrate an existing orchestrator first |
| Multi-datacenter / region | Data residency, disaster recovery and controlled federation | Independent failure domains and local scheduling, explicit cross-site policy and bounded federation; WAN cost/latency included |

A phone remains useful by itself. Remote access is optional and never required for
its local CPU mode. Mobile adapters must pass operator, precision, memory, lifecycle
and device-loss checks. Android root extensions are separate owner-installed plugins;
non-root is the default. iOS uses supported app entitlements/native APIs, not root
assumptions. See the [mobile and edge plan](MOBILE_AND_EDGE_PLAN.md).

## 2. Service architecture to build later

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

1. Versioned capability/job/lease/error schemas and compatibility corpus, alongside
   full-SDK planning; a local test agent over numeric loopback, no public service.
2. Owner-configured mTLS/authenticated 3–5-host CPU pilot, durable job ledger,
   result integrity and scoped cancel/stop protocol with failure injection.
3. One orchestrator adapter, enforced worker isolation, tenant quotas and real metrics.
4. One qualified native GPU/model path, then telemetry console and data/checkpoints.
5. Operational release: upgrade/backup/restore runbooks, SBOM, security review and
   an explicit supported-platform policy; larger fleet tests after the prior gates.

Real-device tests remain paused at the owner's request. No live enterprise, cloud
provider or SSH test host has been provided. Finish schema/local unit fixtures first;
keep physical/mobile/GPU/multi-host acceptance pending until infrastructure exists.
