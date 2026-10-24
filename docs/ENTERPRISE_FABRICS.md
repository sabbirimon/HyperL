# Enterprise devices, interconnects and capability negotiation

Owner expansion, 2026-10-24. HyperL must retain a useful phone/desktop path while
allowing qualified enterprise adapters to add capabilities. Pairing alone does
not install a driver, allocate remote memory, confer access or qualify a kernel.

## What becomes available when a device joins?

Use three distinct stages:

1. **Discover and approve:** identify the actual device, runtime, driver, firmware,
   adapter artifact and owner-approved scope. A model name is display metadata.
2. **Negotiate candidates:** intersect supported protocol versions, operations and
   numerical contracts; require fresh qualification and an approved measured path.
   Explain missing capabilities rather than hiding why a feature is unavailable.
3. **Admit and execute:** revalidate current identity, permissions and resource
   budgets, reserve the job, then invoke an actually implemented adapter. Planning
   and negotiation output does not activate execution.

Equal device models can have different firmware, available memory and permissions.
A faster device can contribute more capacity without making every other device
support its instructions. Older devices remain eligible for compatible independent
work. A GPU/FPGA/ASIC can join a CPU-coordinated job through an explicit adapter;
there is no requirement that all processors have the same architecture.

For example, joining an approved server may add a remote tensor-engine candidate.
It does not turn a phone into a CUDA device. A qualified low-latency GPU fabric may
later enable a collective mode, while a slower Ethernet group continues to use
coarser pipeline or batch work. The measured workload cost determines placement.

## Separate device buses, GPU fabrics and network adapters

| Family | What HyperL must represent | Required qualification before execution |
|---|---|---|
| USB, Wi-Fi, ordinary Ethernet | Actual route, authenticated transport, measured payload/latency, interface and freshness | Exact end-to-end workload and disconnect behavior; USB ADB forwarding is a test route, not a production USB adapter |
| PCIe | Host/device topology, negotiated generation/width, root complex, copy and peer-access paths | Driver/runtime support, per-buffer ownership, supported direct-copy path and integrity; advertised lane rate is not measured throughput |
| CXL | Memory/device type, coherent domains, OS-visible allocation and ownership | OS/firmware support, isolation and allocation tests; never sum CXL capacity into GPU VRAM automatically |
| NVIDIA NVLink/NVSwitch | GPU scale-up topology and supported peers/generations | Compatible runtime, actual peer path and workload/collective conformance; not inferred from an installed NVIDIA GPU |
| AMD Infinity Fabric | Qualified accelerator topology and runtime-visible peer access | Exact device/runtime/topology checks and measured GPU transfers/collectives |
| InfiniBand and RoCE | NIC, link, subnet/path, registered buffers, queues and protection domains | Explicit RDMA adapter, driver access, bounds/isolation, failure handling and actual transfers |
| High-speed Ethernet NICs / SuperNICs | NIC capabilities separately from switches, congestion settings and usable link rate | Qualified transport/offloads, measurements under load and tenant limits; a NIC brand does not enable all offloads |
| DPUs / SmartNICs | A separate compute/control endpoint and firmware trust boundary | Explicit offload implementation, signed provenance, bounded privileges and independently observable completion |
| Future/vendor fabric | Versioned namespaced protocol and exact adapter artifact | Unknown extensions remain unavailable until an implemented adapter is approved and tested |

NVLink is a GPU interconnect fabric; ConnectX devices provide Ethernet/InfiniBand
network connectivity. These are different topology layers and can coexist in a
larger installation. See [NVIDIA NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
and the [ConnectX-8 manual](https://networking-docs.nvidia.com/connectx8hw/introduction).
[AMD's architecture documentation](https://www.amd.com/en/technologies/cdna.html)
describes its accelerator interconnects; [Linux CXL documentation](https://cxl.docs.kernel.org/)
describes OS-facing coherent-memory integration. Product specifications are not
HyperL performance measurements or compatibility certification.

## FPGA and ASIC adapters

The existing target vocabulary contains `FPGA_VENDOR`. This is a target contract,
not evidence of a loaded board or bitstream. ASIC execution needs an applicable
qualified backend or a separately versioned target extension; do not relabel a
CPU result as ASIC execution.

Every programmable or fixed-function accelerator offer must identify:

- Board/chip identity, driver/runtime ABI, firmware and exact artifact hash.
- For an FPGA, the approved bitstream/kernel and reconfiguration ownership.
- Supported operations, data types, shapes, precision/error contract and layouts.
- Local memory, host staging, DMA alignment, maximum transfers and buffer lifetime.
- Completion, timeout, reset/cancellation semantics, power/thermal limits and errors.
- A trusted qualification record tied to that exact configuration and workload.

FPGA reconfiguration and firmware installation are separate privileged operations;
they must not occur as a side effect of pairing or loading an untrusted graph.
Fixed-function ASICs accept only supported work. An unsupported operation is a
rejected candidate or an explicitly admitted alternate route, never a claimed
accelerator execution.

## Transport and scheduling choices

### Current library boundary

`HyperLCapabilityNegotiation.negotiate(left, right, lanes, implementedRequests,
nowMs)` is a pure pair-level admission precursor. Its bounded inputs identify
exact feature/protocol/version/workload tuples, approved adapter descriptors,
driver revisions, operator coverage and fresh evidence in both lane directions.
FPGA/ASIC offers additionally identify firmware and configuration artifacts.
The current FPGA gate requires `FPGA_VENDOR`; an ASIC offer cannot use CPU or
FPGA targets as its claimed backend. An applicable existing backend still needs
independent qualification for that ASIC's exact workload and firmware.

The caller's local implementation registry supplies `implementedRequests`.
Advertisements cannot add executable features to that registry. A decision returns
an eligible candidate or a reason it was blocked. `executionActivated=false` and
`resourceReservationGuaranteed=false` remain explicit. Caller-supplied
`qualified=true` and `identityVerified=true` are trusted inputs, not cryptographic
verification performed by this function.
Candidates retain exact left/right offer snapshots, including artifact, driver,
firmware/configuration identities, plus both directed lane observations. Dispatch
must compare these snapshots with current state; matching adapter/lane IDs alone
does not prove that a changed artifact or configuration is still qualified.

Integration excerpt (the variables come from the owner's verified registry):

```kotlin
val agreement = HyperLCapabilityNegotiation.negotiate(
    approvedCoordinator, approvedWorker, observedLanes,
    locallyImplementedRequests, observationTimeMs
)
for (decision in agreement.decisions) {
    // Show candidate eligibility and the reason; do not dispatch from this alone.
    showCapabilityDecision(decision)
}
```

Recheck selected evidence at dispatch, then run workload/resource admission. The
existing H0 estimator and distributed vector executor are separate components;
automatic wiring and enterprise driver loading are not implemented here. See the
unit fixtures for constructing exact tuples; their simulated fabrics are not
physical hardware test evidence.

Prefer an established transport boundary over a new protocol for every NIC.
[UCX](https://openucx.github.io/ucx/index) offers integration paths for RDMA, TCP,
GPU and shared-memory communication. Its presence and underlying transport must
still be explicitly detected, approved and qualified. UCX, NCCL/RCCL, vendor
collectives, RDMA and direct GPU memory operations are integration tracks, not
bundled working adapters in this increment.

The scheduler should use measured application goodput and complete workload time,
including serialization, copies, synchronization and verification. Keep these
outcomes separate:

- **Capacity:** a larger admitted workload can complete across several devices.
- **Latency:** the complete job finishes sooner than its qualified local baseline.
- **Throughput:** more complete jobs finish under a stated load and error bound.

The [Samsung/Mac measurements](ANDROID_CROSS_PLATFORM_PROBE.md) demonstrate why:
both CPU halves return the correct result, but this tiny vector is faster on the
phone alone. H0's [sharding planner](SHARDING_EXECUTION_CONTRACT.md) can explain
local versus distributed candidates; automatic negotiation-to-dispatch wiring,
durable recovery and full model sharding retain their own acceptance gates.

## Security and future extension rules

Capabilities are trusted-owner observations, not remote self-attestation. Changes
to adapter hash, driver/firmware, device identity, topology or permission invalidate
affected qualification. Stale observations and unknown versions fail closed.
High bandwidth never reduces authentication, bounds, integrity checks or revocation.
RDMA and coherent memory do not by themselves provide encrypted cross-tenant access.

Namespaced extensions must preserve numerical semantics, explicit copy/memory
domains and meaningful cancellation. An extension can add candidates without
silently changing existing programs. Basic CPU operations remain independent of
optional vendor SDK installation. The current test hardware does not establish
FPGA, ASIC, RDMA, NVLink, CXL or production cluster operation.

Quantum providers use a separate circuit/job/result boundary and explicit account
approval; pairing or fabric negotiation never enables a QPU. The current Python
SDK bridges have contract-double tests, not installed SDK or hardware qualification.
See [quantum scope](QUANTUM_EXTENSION_PLAN.md), [precision/HPC](PRECISION_AND_HPC_PLAN.md)
and the [programming guide](PROGRAMMING_GUIDE.md).
