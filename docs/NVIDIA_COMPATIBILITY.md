# HyperL and the NVIDIA ecosystem

Researched 2026-10-08 using primary NVIDIA documentation. Current HyperL evidence:
**zero NVIDIA GPU architecture families qualified on hardware**. CUDA source
emission and an optional installed-PyTorch CUDA preview exist; no CUDA device is
available on this AMD Mac. No Tensor Core, FP4/FP8 model, multi-GPU or AI-factory
compatibility is established by those interfaces.

## Architecture qualification backlog

| NVIDIA family | HyperL current status | Qualification requirement |
| --- | --- | --- |
| Pascal | Unqualified; source generation only | Separate legacy toolkit/framework/driver combination and real device checks |
| Volta | Unqualified; source generation only | Separate legacy toolkit/framework/driver combination and real device checks |
| Turing | Unqualified; source generation only | Pin a compatible toolkit/driver/framework build and test supported f32 kernels |
| Ampere | Unqualified; source generation only | Test actual SKU, compiler, precision and transfers; no Tensor Core claim |
| Ada Lovelace | Unqualified; source generation only | Actual SKU/driver/framework qualification and measured end-to-end behavior |
| Hopper | Unqualified; source generation only | Actual GPU/operator qualification; tensor, HBM and multi-GPU paths separately |
| Blackwell | Unqualified; source generation only | Current compatible tooling and hardware; no automatic FP4/FP8/Tensor Core lowering |
| Rubin / Vera Rubin platform | Future qualification backlog | Pin released platform/SDK/driver details and qualify actual hardware; platform naming does not provide a backend |

Eight families are tracked here, including Turing omitted from the owner's pasted
overview. They are a backlog, not eight supported architectures. Maxwell and older
devices need separately scoped legacy work if requested.

NVIDIA's [architecture/driver matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
shows that compatibility depends on the GPU and toolkit/driver versions. CUDA 13
removed offline compilation/library support for pre-Turing families; Pascal and
Volta cannot be treated as targets of every newest toolkit. See
[NVIDIA's CUDA 13 explanation](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/).
Rubin GPUs and Vera CPUs form a broader system platform, as described in
[NVIDIA's Vera Rubin announcement](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-delivers-world-class-supercomputers-for-science).

## Software coverage

| NVIDIA component / capability | HyperL today | Planned integration / acceptance |
| --- | --- | --- |
| CUDA Toolkit/runtime | Emits small CUDA f32 kernel source; optional installed-PyTorch CUDA dispatch unrun | Explicit native compile/load/dispatch, runtime/driver errors, CPU conformance and transfer/timing evidence |
| cuBLAS/cuDNN | No direct adapter | Supported operator/layout/dtype mapping through existing libraries; licensed installation and real tests |
| TensorRT / TensorRT-LLM | No engine import or serving adapter | Versioned model/engine/operator compatibility, real model correctness, ownership and measured inference |
| NCCL | No collectives/data plane | Qualified devices, topology, multi-host failure/cancellation and collective semantics |
| NIM | No service client or model execution | Explicit authenticated endpoint, model/API contract, metering, data policy and actual integration tests |
| NeMo | No framework/agent-suite integration | Reviewed library/API interoperability; model artifacts and workflow semantics separately |
| Omniverse / Blueprints | No application/workflow import | Selected useful workflows with dependency/license review and real end-to-end samples |
| Run:ai / Base Command Manager | No lifecycle/scheduler connector | Adapter to later HyperL control services with bounded scopes and live test environment |
| MIG / vGPU | No partition or isolation management | Observe actual profiles; require privileged operator setup and prove isolation/quota behavior |
| Kubernetes GPU operators | No installed cluster integration | Reviewed orchestration adapter and qualified pilot, not bundled drivers/operator privileges |
| NVLink / Spectrum-X / BlueField | No native fabric/DPU integration | Observe physical topology and use qualified transports; no advertised latency/throughput before measurement |

The organizing reference is NVIDIA's
[AI Enterprise software overview](https://docs.nvidia.com/ai-enterprise/software/latest/overview.html).
This matrix maps integration goals; it does not imply equivalent features, NVIDIA
certification, support contracts or a license to redistribute proprietary software.

## Delivery order

1. Finish local AMD Mac Metal qualification without substituting another GPU.
2. Qualify one available NVIDIA GPU through the installed framework preview.
3. Add a native CUDA adapter with stable ownership/precision/error contracts.
4. Add selected matrix/model primitives through proven libraries and qualify them.
5. Add actual multi-GPU/cluster controls, collectives and scheduler integrations.
6. Qualify additional families and deploy profiles only with measured evidence.

Source/operator compatibility, tensor/data sharing, graph import, binary ABI,
runtime dispatch and complete application compatibility are different promises.
The [full software stack](SOFTWARE_STACK.md) and [interop guide](LIBRARY_INTEROPERABILITY.md)
keep those boundaries visible.
