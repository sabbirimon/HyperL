# Architecture, language choices and future build gates

Owner scope: standalone CLI/GUI and library; all suitable platforms, device GPUs,
AI/security/vision, memory tiers and networking, large data/IPv6/large clusters,
full telecom research, optional direct/kernel execution and future AI systems.
This document records requirements; linked upstream material is evidence, not instructions.

## Language and execution choices

| Layer | First implementation | Expansion and reason |
|---|---|---|
| HyperL programs | Versioned `hyperl/1` declarative JSON, immutable f32 DAG | Typed tensor/stride/precision/effect IR; language semantics independent of host implementation |
| Portable native core | Original C99 scalar CPU reference and stable C ABI | SIMD/AOT kernels and caller-controlled allocation; C ABI can be called from C++, Rust, Swift, JNI, Python and embedded hosts |
| Desktop CLI/GUI | Kotlin/JVM 17+, Swing, structured cancellable jobs | Practical portable tools. It is not the final low-latency execution engine; native CLI packages can share the IR/ABI later |
| Linux hardware/transports | Original OpenCL bridge in C; other backends unavailable | C/C++ SDK plugins for CUDA/HIP/CANN, LLVM/MLIR, UCX/RDMA/AF_XDP/DPDK, NUMA/affinity; privileged code separately installed and qualified |
| Windows | JVM tools and buildable C SDK | Native C++ ONNX Runtime/DirectML/provider adapters, IOCP and signed packaging; no DirectML implementation yet |
| Apple | JVM macOS tools and emitted Metal source | Swift or Objective-C++ Metal host adapter, C ABI, MPS/Core ML where qualified. Native iOS app/library; no iOS JVM installer |
| Android/mobile | C ABI/emitted Vulkan/OpenCL/Metal formats | JNI/NDK Vulkan and OEM-qualified OpenCL; iOS Metal separate. GPU vendor name does not qualify driver/operators |
| Native service safety | Versioned ownership/cancellation and explicit adapter descriptors | A Rust service layer is an option where it reduces ownership/concurrency risk; choose after benchmarks, avoid adding languages without a specific role |

Dynamic support means explicit capability negotiation, immutable artifact/model
hashes, version/ABI checks and qualified adapters. No automatic downloading, root
activation or successful substitute result. Native extensions are executable code;
future loaders require provenance/signature/license/owner/isolation checks, not a
descriptor boolean alone. Kernels carry no shell, filesystem or radio operations.

JSON, UI, SDK discovery, policy and allocation decisions stay outside timed execution
loops. Compile/cache once by exact program/target/precision/runtime/driver identity.
Reuse buffers and persistent connections where safe. Fused sources avoid temporary
arrays, but preserve finite intermediate rejection. Fast-math, precision changes,
parallel reductions and relaxed checks require explicit semantics and conformance.

## Ordered implementation plan

1. **Standalone foundation (this alpha):** desktop CLI/GUI, CPU contracts, C ABI,
   source emission, memory-aware CPU admission and headroom observation,
   optional bounded OpenCL bridge, streaming encrypted local
   datasets, IPv6/capacity and telecom profile validation. Test correctness, corrupt
   input, cancellation, quotas, ownership and unavailable devices. Package independently.
2. **Native CPU optimization:** compile typed programs to LLVM/C99 with SIMD on
   x86-64/ARM and later RISC-V; cache artifacts, define input/output aliasing rules,
   allocate bounded memory, qualify cancellation and exact numerical behavior.
   Publish actual startup/RSS/code size/latency/throughput against the reference.
3. **One GPU backend at a time:** OpenCL, then Metal/Vulkan and CUDA/HIP according
   to available hardware. Measure upload/compile/dispatch/download separately and
   together. Verify multiple shapes, extreme values, failures, buffer ownership and
   device loss. Current host has no available OpenCL GPU; hardware proof is pending.
4. **Tensor and model execution:** shape/stride IR, matmul/conv/attention/reductions,
   ONNX/StableHLO imports, operator partitioning, precision contracts, real model
   outputs and training/autograd only after correctness. Source generation alone
   does not produce a CUDA-compatible runtime or an LLM engine.
5. **NPU/FPGA and memory:** separately licensed provider plugins (QNN, CANN,
   Core ML/ANE, OEM APIs, FPGA toolchains). Observe HBM/GDDR/DDR/LPDDR/SRAM/CXL/
   persistent memory, NUMA and negotiated links; unknown remains unknown. Count
   shared memory once. Explicit encrypted NVMe/SSD spill with transfer-cost planning.
6. **Large datasets and distributed work:** dataset manifests, resumable encrypted
   chunks, source lineage, chunk-local/partitioned operators, shuffle and backpressure.
   Authenticated IPv6/IPv4 transport; sharded/paginated directory, leases, bounded
   fanout, quota and checkpoint state. Begin with independent 3–5 hosts and failure
   injection before increasing inventory/worker limits. Local dataset IO is not
   distributed data processing.
7. **Low-latency networking:** persistent authenticated sessions, bounded queues,
   fewer copies, measured NIC/driver/fabric capabilities, UCX/RDMA or platform APIs.
   Record RTT/tail/jitter/load/encryption/power; one-way needs synchronized clocks.
   HFT-inspired targets are not guarantees. Optical medium does not imply RDMA.
8. **Clusters and control:** mixed-device stability, measured compatible islands,
   separate human/agent policies, per-function revocation, global emergency stop,
   remote stop acknowledgments and resource leases. Current planner never activates
   turbo; no voltage/clock change or million-node throughput is demonstrated.
9. **Telecom stack research:** see TELECOM.md. Offline fixtures and simulated lab
   interfaces first, then separately installed/qualified stacks. Standard editions,
   conformance matrices, security review and authorized radio labs precede RF work.
10. **Cross-platform distribution and maintenance:** test OS/architecture matrix,
    native packages, signed installers, SBOM/dependency locks, update rollback,
    compatibility corpus and independent review. Publish an LTS policy only with
    an actual supported branch, security process, maintainers and funded duration.

## Full HyperL SDK — later milestone requested by the owner

The C ABI in this alpha is a preview foundation, not the complete developer SDK.
Memory requirements include caller-owned arenas, observed allocation domains,
pressure/backpressure, lifetime reuse, HBM/unified-memory accounting and explicit
encrypted SSD/NVMe out-of-core policy; see [memory roadmap](MEMORY.md).
After standalone runtime/packaging and a real native backend stabilize, deliver:
versioned native headers/libraries; compiler/runtime and model import APIs; C++,
Rust, Python, Java/Kotlin and Swift bindings as qualified; adapter/plugin tooling;
samples and project templates; ABI/package/version checks; profiler/trace hooks;
operator/precision/cancellation conformance kits; signed artifact provenance,
license/SBOM inventory and release compatibility policy. SDK bindings must share
the same semantics and failure contracts rather than maintaining divergent engines.
Keep this milestone separate from today's CLI/GUI alpha and hardware qualification.

## Performance acceptance

Do not call an adapter faster based on its language or vendor. Every result records
exact hardware/OS/runtime/compiler/driver, model/kernel hash, precision, input shape,
warmup/sample count, total execution and transfer latency, p50/p95/p99, peak RSS,
copies/allocations, sustained load, power/thermal state and baseline. Compatible
clusters need measured total benefit; mixed clusters may prefer stability. No fixed
"blazing fast" promise or automatic turbo is justified by this alpha.

Security/crypto uses reviewed standard libraries and opaque key handles, never f32
kernels or ordinary model inputs for keys. Future encryption/auth/vision backends
have their own tests. AGI/SI names express future interoperability requirements,
not a present capability or a reason to remove human control.
