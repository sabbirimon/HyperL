# Mixed CPU, GPU, NPU and TPU execution plan

Owner request, 2026-10-23: one HyperL program should eventually use different
processor architectures at the same time, including local phones, workstations
and approved remote nodes. This document defines the path and acceptance checks.
It does not claim that the standalone alpha already schedules mixed accelerators.
See [current validation](VALIDATION.md) and [platform matrix](PLATFORMS.md).

## Expanded owner requirements — 2026-10-24

Genuine multi-device workload and model sharding are required delivery goals.
The [sharding execution contract](SHARDING_EXECUTION_CONTRACT.md) defines the
dependency order, first bounded implementation, resource/network admission,
failure recovery and evidence gates. It is the implementation plan for this
increment; the platform rows below remain qualification tracks, not delivered
mixed-accelerator support.

Enterprise/supercomputer profiles are primary targets; see the
[precision/HPC plan](PRECISION_AND_HPC_PLAN.md) for future f64/BF16/binary128 and
arbitrary-precision contracts. Current f32 graphs and the compensated-sum helper
are not those engines. [Quantum circuits and SDK routes](QUANTUM_EXTENSION_PLAN.md)
form a separate hybrid-workflow boundary; a classical simulator or provider bridge
does not qualify a QPU as a vector worker or create cross-device quantum connectivity.

- Preserve the fixed physical Samsung ARM64/Mac x86-64 CPU probe and extend
  independent-host execution with general placement and durable recovery;
  then prove a real model's layers/state split across devices. Replicated model
  requests and profile grouping do not satisfy model sharding.
- Add measured load balancing, variable-speed lanes, backpressure and lane telemetry.
  Plan memory by physical RAM/VRAM/shared domains plus bounded storage staging.
- Keep an optional Redis-compatible metadata/cache adapter separate from the
  authoritative job ledger, resource leases and tensor bulk transfer.
- Qualify mainstream framework adapters, Huawei CANN/HCCL/MindSpore and DeepSeek
  model/kernel requirements separately. Keep extension points for future devices,
  formats and protocols, with unknown required features rejected.
- Expose a stable typed SDK and a simple Python-style authoring path. Explain-plan
  views show placement, copies, memory budgets and estimated versus observed timing.
  A familiar API cannot erase unsupported NPU operators or unknown measurements.
- Report capacity, latency and throughput separately. Fast execution is a measured
  objective; slower devices remain useful where workload and transfer cost fit.
- Add complex mathematics under explicit precision/error contracts, and established
  cryptographic providers with isolated keys and authenticated transport. Neither
  a future math operator nor a cryptographic algorithm is implied by source emission.

Current alpha.6 evidence includes real bounded CPU shards across independently
started JVM processes, verified TLS/trust/hostname behavior, separate Radeon Metal
and Samsung Adreno Vulkan cases, generated cross-language contracts, compensated
CPU reduction and HLM2 authenticated dataset/rekey. A fixed physical Samsung ARM64/
Mac x86-64 CPU vector split now passes over wired and direct Wi-Fi TLS.
This narrow [physical probe](ANDROID_CROSS_PLATFORM_PROBE.md) is slower than its
small local reference and does not qualify model sharding, mixed accelerators,
automatic placement or durable recovery. Preserve the existing
alpha.5 installers and the [licence history](LICENSE_HISTORY.md). New H0 cost-aware
planning complements [the existing executor](DISTRIBUTED_EXECUTION.md); its output
is a local explanation, not an execution manifest or globally optimal allocation.

## Capability before placement

One immutable program and input manifest should be parsed once into a versioned
typed DAG. Each installed backend advertises observed capabilities, not a vendor
label as a substitute for an operator test:

| Field | Required evidence |
|---|---|
| Identity | Device and runtime/driver version, ABI, adapter build digest and trust state |
| Compute | Exact operator, shape, stride, layout, dtype, quantization and batch limits |
| Memory | Free and reservable bytes by distinct memory domain; unified memory counted once |
| Transfer | Supported copy path, bandwidth/latency sample, buffer ownership and alignment |
| Control | Cancellation deadline, failure isolation, permission scope and lease expiry |
| Measurement | End-to-end and operator timing, memory, energy/thermal status, measurement age |

Capability records are scoped to the exact kernel/model revision and expire after
a policy interval. A missing NPU or TPU SDK remains unavailable. Source emission,
device discovery, a successful CPU run or an external model API does not qualify
that device for a HyperL graph segment.

## Planner and runtime contract

1. Validate the full typed DAG against the CPU reference semantics. Reject
   unsupported side effects or unknown numerical behavior before allocation.
2. Build candidate partitions only at explicit tensor boundaries. Admit every
   partition on at least one verified backend and budget weights, intermediate
   tensors, KV/state where applicable, scratch, double buffers and transfer copies.
3. Estimate total latency and energy as compute **plus** serialization, transfer,
   synchronization and queue delay. Keep a whole-CPU and one-device baseline.
   Prefer task/data parallelism when it avoids per-token activation transfer.
4. Reserve each selected resource under a short lease and produce a signed plan
   binding program hash, adapter revisions, input/output digests, precision,
   partition map, budget, deadlines and allowed data destinations. Human/agent
   scope checks happen before dispatch and remain revocable during execution.
5. Run independent DAG branches concurrently with bounded in-flight buffers and
   backpressure. Pass only typed, integrity-checked tensors across boundaries.
   A lost worker cancels dependent segments; retries require idempotent pure
   segments or an acknowledged checkpoint. Never silently switch precision or
   return a partial result as success.
6. Compare results with the CPU reference within declared per-operator tolerances,
   including edge inputs, repeated runs and mixed-device order. Record all real
   timings and transfer bytes. Change placement only after measured end-to-end
   benefit, not based on aggregate advertised FLOPS.

Continue from the fixed physical CPU probe to general placement, additional network
paths and durable failure recovery. Then qualify CPU/GPU partitions and another actual
architecture after cancellation, device loss and numerical conformance pass.
Move to shaped tensors, matmul/attention and actual model state only after
the typed tensor IR or a reviewed model-engine boundary exists. Direct TPU/NPU paths require their own legal SDK,
driver and operator coverage; CPU fallback is an explicit alternate plan, never
reported as TPU/NPU execution.

## Platform adapters

| Target | First adapter work | Distinct gate |
|---|---|---|
| x86/ARM/RISC-V CPU | Portable C99 reference, then ISA-specific SIMD | Same outputs, bounded memory/cancellation, actual target hardware |
| NVIDIA GPU | CUDA adapter via installed toolkit and driver | Kernel load/dispatch, error propagation, transfer and sustained benchmarks |
| AMD GPU | HIP/ROCm on supported Linux hardware; Metal remains a separate macOS path | Exact GPU/ROCm pairing, operator coverage and numerical tests |
| Intel GPU | Qualified oneAPI/Level Zero or Vulkan path | Driver/version and exact operator/precision checks |
| Apple GPU | Metal host backend; iOS packaging separate | Native device tests, buffer ownership and app lifecycle |
| Android GPU | Vulkan or owner-installed OEM path | Per phone driver/operator tests, thermal limits and permissions |
| NPU/TPU | QNN/Core ML/ANE/NNAPI successor/other supported SDK; datacenter TPU via its supported host runtime | Per-device model lowering, supported operations, dtype, provenance and exact output tests |
| Huawei Ascend | CANN/AscendCL compute, HCCL collectives, MindSpore graph/runtime integration | Versioned Ascend hardware/SDK/fabric evidence; Kirin/HiAI is separate; cross-vendor collectives need a tested bridge |

The phone is still useful offline with CPU execution when the advanced adapters
are absent. Mixed execution never requires a phone to expose a public listener.
Remote control and data channels need mutual authentication, a pinned identity,
scoped enrollment, expiry, revocation, encryption and bounded resources. Do not
use upstream experimental plaintext compute RPC directly on an open network.

## Model qualification, not model branding

HyperL does not currently load or shard full LLMs. A model made by NVIDIA can
run on another vendor only when a compatible format, engine and all operators
are implemented there. As a concrete research set, NVIDIA's official Nemotron
3 Nano 4B Q4_K_M GGUF is 2.84 GB and suitable for a standalone local-engine
benchmark; its hybrid Mamba/Transformer state and chat/tool format require
specific support. Nano 30B A3B has 30B total weights despite 3B active per
token, making it a workstation/server candidate. Larger Super 120B and Ultra
550B belong to multi-GPU or hosted qualification. These are **model-engine**
trials, not evidence that HyperL's current f32 vector DAG can execute them.

For the first HyperL model-sharding milestone, implement and test a typed dense
transformer operator subset before hybrid/MoE variants. Compare a whole model on
one GPU, replicated workers (throughput), layer/pipeline splits (capacity) and
tensor splits (fast interconnect) separately. A mixed CPU/GPU/NPU plan wins only
when measured total latency, energy, correctness and recovery improve for the
chosen workload. Avoid using the nominal context window as a memory promise.

## Acceptance matrix

- Conformance: exact small f32 graphs, extreme/NaN rejection, dtype conversion,
  deterministic and declared-tolerance paths, actual CPU/GPU mixed output.
- Admission: incompatible operator, insufficient memory, stale profile, changed
  driver, expired lease and absent accelerator all fail before dispatch.
- Lifecycle: Stop/revoke during computation, worker disconnect, repeated
  submission, retry and result publication have unambiguous states.
- Performance: report model/kernel hash, hardware, OS, driver, precision, input
  shape, p50/p95/p99 wall time, transfer bytes/time, RSS/VRAM, power/thermal and
  the single-device baseline. No speedup claim from a microkernel alone.
- Distribution: test the same contract on Linux, macOS, Windows, Android and
  container builds individually; document unsupported backends on each platform.

## Primary technical references

- [NVIDIA official Nemotron 3 Nano 4B GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/tree/main)
- [NVIDIA Nemotron 3 Nano 4B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
- [NVIDIA Nemotron 3 Nano 30B NVFP4 files](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/tree/main)
- [NVIDIA Nemotron 3 Super model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
- [NVIDIA Nemotron 3 Ultra model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
- [AMD ROCm vLLM parallelism guidance](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html)
- [llama.cpp RPC scope and security warning](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
