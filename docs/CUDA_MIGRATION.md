# CUDA migration and accelerator support

Use **Migrate CUDA file** or **Migrate source folder** in HyperL Studio. Select the target and an output parent. Studio writes a fresh subdirectory with translated files and a machine-readable `migration.json`. The original source stays in place.

## Choose a translation path

| Input | Target | Output and prerequisites |
| --- | --- | --- |
| CUDA/C++ files or a source tree | `hip` | Bundled AMD HIPIFY translates supported CUDA headers, kernel code and API calls. Requires local Perl 5, including Perl supplied by Git for Windows. Output keeps relative names for include resolution. Review warnings and compile with the target HIP SDK. |
| CUDA translation units and their headers | `sycl` | Invokes installed SYCLomatic `c2s` or Intel `dpct` with the project include root. Requires the SDK and CUDA headers. Projects needing custom defines, include paths or build databases should run the vendor tool with their build configuration. |
| One standalone pointwise CUDA kernel | `wgsl`, `spirv`, `metal`, `hlsl` | HyperL parses the complete supported kernel, validates typed WGSL with Naga and writes WGSL, SPIR-V binary, Metal 2.1 source or HLSL. The native `translate-shader` command also accepts compute WGSL directly. |
| Same pointwise kernel | `opencl` | OpenCL C with explicit storage buffers, entry point and element count. |
| Same pointwise kernel | `stablehlo`, `onnx-text` | A rank-one tensor graph for a compatible compiler or framework. ONNX output is ONNX's text format. The ONNX qualification script builds a binary in memory and specializes the length. |

From a checkout, after building the Rust engine, run these commands with a fresh output directory for each invocation.

```sh
cargo build --release --locked --manifest-path engine/Cargo.toml
python3 studio/cuda_migrate.py examples/cuda/add.cu out-metal --target metal --engine engine/target/release/hyperl-engine
python3 studio/cuda_migrate.py examples/cuda out-hip --for-stack rocm
python3 studio/cuda_migrate.py examples/cuda/add.cu out-onnx --for-stack qnn
hyperl-engine translate-shader engine/src/projection.wgsl spirv projection.spv
```

For packaged builds, use `HyperL-Studio --migrate-cuda SOURCE NEW_DIR --target TARGET`. `HyperL-Studio --accelerators` lists stack IDs. `--for-stack` selects a concrete translation route when one exists and reports the SDK integration required otherwise. Packaging details are in [Build and test](BUILD_AND_TEST.md).

## Kernel and host interface

The [sample kernel](../examples/cuda/add.cu) shows the supported grammar. It accepts up to eight float pointers followed by an integer count, one one-dimensional global index, a bounds check and an assignment from two read-only operands using add, subtract or multiply. The parser checks the complete input and rejects extra statements, offset indexing, macros, multidimensional launch expressions and unsupported types. Use the project route for general CUDA. Inline PTX, warp intrinsics, architecture-specific libraries, build scripts and device algorithms can require manual porting.

The report supplies argument bindings, graph operand order, output binding, entry name, workgroup size and launch limits. Allocate distinct buffers, bind the active element count and match the target alignment. The shader path uses groups of 64 in one dimension. Metal reserves buffer 30 for Naga's runtime-array sizes in bytes. Graph callers supply equal-length f32 vectors and write the result to the active prefix. Native binaries, drivers, quantization variants and whole LLM graphs each require their own adapter.

## Running a translated operation

- **Metal**. Generate the sample into `out-metal`, then run `swift scripts/qualify_translated_metal.swift out-metal` on macOS. The harness compiles the generated source on each available GPU, dispatches three workgroups, checks 137 sums and checks 55 guard values. Both the Radeon Pro 560X and Intel UHD Graphics 630 passed.
- **OpenCL**. The same fixture ran on both GPUs through `native/opencl_bridge.c`. For this input/output ordering, put the generated `kernel.cl` beside the bridge's little-endian f32 `input_0.bin` and `input_1.bin`, then use `--run DEVICE LENGTH 2 DIRECTORY`. Both devices returned all 137 expected values.
- **ONNX Runtime**. Install a matching ONNX and ORT distribution for your target. Run `python scripts/qualify_translated_onnx.py out-onnx --provider CPUExecutionProvider`, or select the installed vendor EP and its documented `--provider-options` JSON. The script specializes shapes, checks the model, tests numeric results, and verifies node provider assignment in an ORT profile. CPU passed with ORT 1.18.1. QNN, CANN and DirectML need their own provider builds and hardware.
- **Framework arrays**. `hyperl.portable_kernel.execute(report, {0: a, 1: b}, device=(device_type, device_id), to_host=snapshot)` executes the migrated operation via the supplied array framework. The caller configures the SDK and provides a synchronizing f32 host snapshot. The adapter checks DLPack device identity, shapes, precision, finite outputs and CPU agreement. Frameworks must support these protocols on the selected device. A JAX CPU array passed all 137 elements; the StableHLO module also passed MLIR verification.

## Accelerator and framework routes

A target in this table names the artifact HyperL can produce. A vendor SDK may accept only particular operators, shapes, driver versions or data types. **SDK port** means the integration still needs implementation. The table is generated from the same stack definitions used by the CLI.

| Stack | HyperL output | Integration work and primary reference |
| --- | --- | --- |
| AMD ROCm / HIP (`rocm`) | hip | Build migrated sources with the ROCm SDK. [Reference](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) |
| Intel oneAPI / DPC++ (`oneapi`) | sycl | Source migration requires installed c2s or dpct and CUDA headers. [Reference](https://oneapi-src.github.io/SYCLomatic/) |
| Apple Metal (`metal`) | metal | Pointwise shader, group 0 buffers. See the physical qualification below. [Reference](https://developer.apple.com/metal/) |
| Apple MLX (`mlx`) | SDK port | Port graph operators to MLX or integrate a Metal custom kernel. [Reference](https://ml-explore.github.io/mlx/build/html/index.html) |
| Huawei CANN / Ascend (`cann`) | onnx-text | Specialize the graph and qualify an installed CANN provider or vendor importer. [Reference](https://www.hiascend.com/en/software/cann) |
| Qualcomm QNN / Hexagon (`qnn`) | onnx-text | Select a QNN backend. HTP may require quantization and static shapes. [Reference](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) |
| OpenCL (`opencl`) | opencl | Pointwise OpenCL C kernel for a conforming compiler and device. [Reference](https://www.khronos.org/opencl/) |
| SYCL (`sycl`) | sycl | Migrate using SYCLomatic. Select a SYCL implementation for the target. [Reference](https://www.khronos.org/sycl/) |
| Vulkan compute (`vulkan`) | spirv | Compute SPIR-V for a host with matching descriptor layouts. [Reference](https://docs.vulkan.org/guide/latest/compute_shaders.html) |
| Windows DirectCompute (`directcompute`) | hlsl | Compile emitted HLSL for the chosen Direct3D feature level. [Reference](https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader) |
| Windows DirectML (`directml`) | onnx-text | Qualify the graph with DmlExecutionProvider on Windows. [Reference](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) |
| WebGPU (`webgpu`) | wgsl | WGSL compute plus buffer and dispatch metadata. [Reference](https://www.w3.org/TR/WGSL/) |
| Triton kernel compiler (`triton`) | SDK port | Port the operator to Triton and qualify the selected backend. [Reference](https://github.com/triton-lang/triton) |
| PyTorch / Inductor (`pytorch`) | pytorch | Emits a rank-one f32 function with an explicit input binding map. CPU Inductor full-graph compilation and eager MPS execution matched exact expected results on 137 elements for add, subtract and multiply with PyTorch 2.2.2. [Reference](https://docs.pytorch.org/docs/stable/torch.compiler.html) |
| JAX / XLA (`jax`) | stablehlo | StableHLO tensor operation. HyperL also has an explicit PJRT graph path. [Reference](https://openxla.org/stablehlo) |
| TensorFlow / XLA (`tensorflow`) | stablehlo | Supply the module to an accepting XLA toolchain. TensorFlow model import needs an adapter. [Reference](https://openxla.org/xla) |
| Mojo (`mojo`) | SDK port | Implement the operator using the Modular SDK. [Reference](https://docs.modular.com/mojo/) |
| Kokkos (`kokkos`) | SDK port | Map the launch, views and execution space in C++. [Reference](https://kokkos.org/) |
| RAJA (`raja`) | SDK port | Map the loop body and execution policy in C++. [Reference](https://github.com/LLNL/RAJA) |
| alpaka (`alpaka`) | SDK port | Port the kernel and select an accelerator implementation. [Reference](https://github.com/alpaka-group/alpaka) |
| Halide (`halide`) | SDK port | Express the algorithm and schedule for a chosen target. [Reference](https://halide-lang.org/) |
| ArrayFire (`arrayfire`) | SDK port | Map to array operators and select an installed backend. [Reference](https://arrayfire.org/docs/) |
| Numba (`numba`) | SDK port | Select a maintained target extension and port the kernel. [Reference](https://numba.readthedocs.io/) |
| Taichi (`taichi`) | SDK port | Port fields and kernels to the Taichi runtime. [Reference](https://docs.taichi-lang.org/) |
| NVIDIA Warp (`warp`) | SDK port | Port the operator and qualify the selected Warp device. [Reference](https://github.com/NVIDIA/warp) |
| ZLUDA (`zluda`) | SDK port | Test a compatible CUDA binary and driver with the external compatibility layer. [Reference](https://github.com/vosen/ZLUDA) |
| SCALE (`scale`) | SDK port | Compile the original CUDA with the licensed vendor toolchain. [Reference](https://docs.scale-lang.com/stable/) |
| AMD HIPIFY (`hipify`) | hip | Pinned translator included. A local Perl 5 interpreter runs it. [Reference](https://github.com/ROCm/HIPIFY) |
| SYCLomatic (`syclomatic`) | sycl | Optional installed source migration tool. [Reference](https://github.com/oneapi-src/SYCLomatic) |
| Moore Threads MUSA / MUSIFY (`musa`) | SDK port | Use the vendor migration workflow and qualify the generated MUSA sources. [Reference](https://github.com/MooreThreads/CV-CUDA_musa) |
| Hygon DTK / DCU (`dtk`) | hip | HIP source handoff. Check APIs against the exact DTK release and DCU. [Reference](https://developer.sourcefind.cn/) |
| Cambricon NeuWare / BANG C / BANGPy (`neuware`) | SDK port | Port the operation with the MLU SDK or a supported framework plugin. [Reference](https://github.com/Cambricon/torch_mlu) |
| MetaX MACA (`maca`) | SDK port | Requires the matching MACA compiler, migration path and GPU. [Reference](https://developer.metax-tech.com/) |
| Biren BIRENSUPA (`biren`) | SDK port | Requires a SUPA compiler/runtime integration and a BR device. [Reference](https://www.birentech.com/product/software/birensupa/) |
| Iluvatar CoreX / IXUCA (`ixuca`) | SDK port | Check CUDA source or binary compatibility with the specific CoreX release. [Reference](https://www.iluvatar.com/) |
| Enflame TopsRider (`topsrider`) | SDK port | An installed torch-gcu plugin can expose GCU tensors. Verify model coverage on a device. [Reference](https://github.com/EnflameTechnology/torch-gcu) |
| Baidu Kunlunxin XPU (`kunlun`) | SDK port | Use the vendor XPU runtime or Paddle integration. [Reference](https://github.com/PaddlePaddle/Paddle) |
| Alibaba T-Head Hanguang / requested Jun Wu stack (`hanguang`) | SDK port | Public Hanguang chip evidence. Jun Wu SDK identity and access need confirmation. [Reference](https://www.alibabacloud.com/blog/payment-hanguang-800-alibabas-first-ai-inference-chip_595482) |
| TileLang (`tilelang`) | SDK port | Port to the TileLang DSL and select a supported backend. [Reference](https://github.com/tile-ai/tilelang) |
| DeepGEMM-Ascend (`deepgemm-ascend`) | SDK port | GEMM-specific library with its own shape, precision and hardware constraints. [Reference](https://github.com/deepseek-ai/DeepGEMM-Ascend) |
| PaddlePaddle (`paddle`) | SDK port | Select the operator and installed custom-device plugin. [Reference](https://github.com/PaddlePaddle/PaddleCustomDevice) |
| MindSpore (`mindspore`) | SDK port | Port the graph or use a compatible model importer on the target. [Reference](https://www.mindspore.cn/en) |
| OneFlow (`oneflow`) | SDK port | Requires framework operator dispatch and a qualified configuration. [Reference](https://github.com/Oneflow-Inc/oneflow) |
| MegEngine (`megengine`) | SDK port | Requires framework operator dispatch and a qualified configuration. [Reference](https://github.com/MegEngine/MegEngine) |
| FlagGems (`flaggems`) | SDK port | Check the backend-specific Triton operator implementation. [Reference](https://github.com/flagos-ai/FlagGems) |
| FlagScale (`flagscale`) | SDK port | Distributed framework integration requires model and collective qualification. [Reference](https://github.com/flagos-ai/FlagScale) |

## Provenance

AMD HIPIFY is pinned at [118b213](https://github.com/ROCm/HIPIFY/tree/118b213d89bea7bb7bedddae3e5ec7b14d4a26c5). HyperL verifies the bundled script digest before use. Naga 24 is pinned in Cargo.lock. Translation performs no driver installation or network download.

## Translation performance

The HIP project path initializes AMD's translator once per batch and preserves file-level warning counts. The baseline started one Perl process per source file. The benchmark times the Python CLI through output publication into a fresh directory. Each implementation receives one warm-up, then five interleaved measurements; translated file SHA-256 values, source paths, input hashes and warning counts must match. The Metal case uses the same native engine binary on both sides.

| Target and workload | Before, median | Batched, median | Ratio |
| --- | ---: | ---: | ---: |
| HIP, 1 file / 315 bytes | 199 ms | 191 ms | 1.04× |
| HIP, 24 files / 7,735 bytes | 2.64 s | 0.65 s | 4.08× |
| HIP, 128 files / 41,396 bytes | 44.46 s | 6.29 s | 7.07× |
| Metal, 1 file / 226 bytes | 280 ms | 269 ms | 1.04× |

This Intel macOS development host had concurrent activity. The baseline 128-file samples varied from 30.38 to 162.71 seconds. These small, distinct kernels exercise per-file overhead. Single-file differences were within run-to-run variability. Large translation units, packaged app startup, SDK compilation, inference and GPU execution have separate costs. See [all samples](../benchmarks/results/translate-macos-intel.json).

To reproduce, check out baseline commit `3ea5ca5` into a separate worktree, build the engine once, and run:

```sh
python3 benchmarks/translate.py /path/to/baseline --engine engine/target/release/hyperl-engine
```
