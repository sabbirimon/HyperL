# HyperL and the AMD ecosystem

Researched 2026-10-08 from primary AMD documentation. HyperL currently has HIP
source emission and an original macOS Metal host preview; it has **no qualified
ROCm/HIP GPU runtime, Ryzen NPU, Vitis FPGA, ZenDNN or AMD model-serving adapter**.
The A1990 Radeon Pro 560X is the first requested local target, through **Metal**.
The owner's Terminal now passes four bounded GPU cases and one expected preflight
rejection on this exact Radeon. [Hardware record](A1990_METAL_VALIDATION.md).

## Hardware qualification tracks

| Track | HyperL today | Required next evidence |
| --- | --- | --- |
| Intel Mac with Radeon Pro 560X | Four bounded Metal cases pass in the owner's Terminal with CPU verification; managed storage/completion confirmed | Wider conformance, repeated/transfer benchmarks, device-loss/cancellation and GUI checks |
| Instinct / CDNA datacenter GPUs | HIP source only | Exact supported GPU/OS/driver/ROCm tuple; native dispatch, HBM observation and correctness |
| Radeon / RDNA workstations | HIP/OpenCL source and host boundaries only | Per-SKU support matrix, memory ownership, actual kernels and total transfer timings |
| Ryzen integrated GPU | No qualified Ryzen iGPU adapter | Keep iGPU and NPU identities separate; exact runtime and OS/device tests |
| Ryzen AI NPU | Planned separate provider | Supported Ryzen AI runtime/model format, quantization, graph partitions and device qualification |
| EPYC / Zen CPU | Portable C reference tested on other CPU hosts; no EPYC optimization evidence | Actual EPYC conformance, NUMA/SIMD behavior and selected native math/library benchmarks |
| Adaptive SoC / FPGA | No FPGA lowering/runtime | Separate Vitis AI/toolchain/bitstream/runtime/license and physical target qualification |

AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
covers supported Linux/Windows configurations, with device and software constraints.
It does not establish ROCm support for this macOS Radeon. A compiler accepting HIP
source does not prove that every AMD GPU, NPU or FPGA can run it.

## Foundational compute and libraries

| Component | Purpose | HyperL delivery / plan |
| --- | --- | --- |
| ROCm driver/runtime | AMD GPU execution environment | No installer/driver control; later explicit provider detection and native dispatch |
| HIP / HIPIFY | C++ portability and selected CUDA-source migration | Small generated HIP kernels only; future reviewed compile/load path and migration examples |
| rocBLAS / hipBLAS / hipBLASLt | Matrix/math libraries | Planned operator/layout/dtype adapters; no matrix API in current f32 DAG |
| MIOpen | Deep-learning primitives | Planned selected operator adapters and real reference-model tests |
| RCCL | GPU collective communication | Planned multi-GPU/fabric path; no distributed execution or collective today |
| MIGraphX | AMD inference optimization/runtime | Planned model import through a qualified runtime; no engine import today |
| Profiling / AMD SMI / RDC | Tracing and device operations | Planned read-only observation first; explicit operator configuration for privileged actions |

Reference: [HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/).
HIP portability is an API/source/toolchain capability, not universal CUDA binary
compatibility. CUDA-specific libraries, unsupported calls, precision and driver
behavior require separate work. Preserve each component's actual license; the
entire AMD ecosystem is not one uniform permissive redistribution grant.

## Frameworks and serving

- **PyTorch, JAX, TensorFlow and Hugging Face workflows:** qualify specific installed
  versions/operators/models on an AMD target. HyperL's existing CPU tensor-copying
  preview does not qualify ROCm tensors, autograd or complete models.
- **vLLM / SGLang:** reuse selected reviewed inference services in the later
  [AI Services plan](AI_SERVICES_PLAN.md); no such service starts in this alpha.
- **Triton Inference Server:** select a ROCm-enabled distribution and backend;
  do not assume any NVIDIA container runs unchanged on AMD. AMD documents its
  [ROCm server](https://rocm.docs.amd.com/projects/triton-inference-server/en/latest/)
  and [MIGraphX-backed example](https://rocm.docs.amd.com/projects/triton-inference-server/en/latest/examples/triton-inference-server-examples.html).
  Triton Inference Server and the Triton kernel compiler are different projects.
- **ZenDNN:** later optional EPYC inference adapter, distinct from GPU ROCm.
  [AMD ZenDNN guide](https://docs.amd.com/r/en-US/57300-ZenDNN-user-guide).
- **Ryzen AI / Vitis AI:** separate NPU/adaptive-compute providers with their own
  supported platform and model requirements. [Ryzen AI](https://ryzenai.docs.amd.com/en/latest/),
  [Vitis AI](https://vitisai.docs.amd.com/en/latest/).

## Delivery order

1. Expand the exact Radeon's passing Metal cases with broader conformance and measured performance; keep inaccessible execution contexts explicit.
2. Qualify one supported ROCm workstation/datacenter target, without silent fallback.
3. Add native HIP dispatch and selected math/model adapters through proven libraries.
4. Add an actual AMD model-serving pilot, then RCCL/multi-host failure tests.
5. Qualify Ryzen NPU, EPYC optimizations and FPGA paths as independent tracks.

See the [full software stack](SOFTWARE_STACK.md), [NVIDIA matrix](NVIDIA_COMPATIBILITY.md)
and [platform matrix](PLATFORMS.md). No benchmark, certification or complete
vendor-stack compatibility is claimed before measured acceptance.
