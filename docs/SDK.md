# HyperL SDK preview

HyperL provides an installable **C/CMake SDK** and a **Python SDK preview** for
existing execution contracts. `hyperl-sdk/1-preview` identifies the Python
facade; the package is still alpha.6. The C runtime keeps its existing ABI.
Custom CUDA binaries, a CUDA compatibility layer, stable Rust/Swift/JS bindings,
mobile packages and production fleet qualification remain separate milestones.

| Backend | SDK route | Evidence |
| --- | --- | --- |
| CPU | Native C99/Python bounded f32 arithmetic and pinned GGUF; JAX f32 decoder | Native, local GGUF/RPC and four logical JAX CPU devices tested in their respective scopes |
| NVIDIA CUDA | `open_compute("cuda:0", ...)` with CUDA PyTorch; pinned llama.cpp CUDA build | Adapter available in source; physical GPU test pending |
| AMD HIP/ROCm | `open_compute("rocm:0", ...)` with ROCm PyTorch; pinned llama.cpp HIP build | Runtime selector and build option in source; physical ROCm test pending |
| Intel XPU/SYCL | `open_compute("xpu:0", ...)` with XPU PyTorch; pinned llama.cpp SYCL build | Adapter and build option in source; physical GPU test pending |
| Metal | Existing checked macOS host for bounded arithmetic; optional llama.cpp Metal build for GGUF | [Radeon Pro 560X bounded cases](A1990_METAL_VALIDATION.md) passed; this SDK does not qualify full GPU model generation |
| Vulkan / OpenCL | Standalone [Android Vulkan](ANDROID_VULKAN.md) runner and existing checked JVM OpenCL host; optional llama.cpp Vulkan GGUF | Bounded Vulkan evidence on Adreno 506; no universal model/driver coverage |
| TPU | Explicit JAX backend/device profile | Logical CPU conformance only; physical TPU test pending |

PyTorch ROCm internally calls its device `cuda:N`. HyperL checks the installed
build before allowing either selector. GPU graph coverage is bounded to f32
add, multiply and ReLU. CPU verification deliberately copies and compares the
result; the budget bounds the CPU reference, not the full framework allocator
or VRAM. Cancellation polls host boundaries; it cannot preempt accepted kernels.

## Install and embed

From a standalone checkout (use the equivalent path syntax in PowerShell):

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
cmake --install build/native --config Release --prefix "$PWD/build/sdk-prefix"
cmake -S examples/sdk-cmake -B build/sdk-example -DCMAKE_PREFIX_PATH="$PWD/build/sdk-prefix"
cmake --build build/sdk-example --config Release
ctest --test-dir build/sdk-example -C Release --output-on-failure
python -m pip install ./python
```

The C example uses `find_package(HyperL 0.1.0 EXACT CONFIG REQUIRED)` and
`HyperL::hyperl_cpu`. The shared CPU runtime and separate erasure codec are
also exported as `HyperL::hyperl_cpu_runtime` and `HyperL::hyperl_erasure`.
The C ABI owns no caller buffers; check status and output length before use.
The Python source wheel requires a reviewed CPU library for compute. Build
a [platform wheel](LIBRARY_INTEROPERABILITY.md#alpha6-local-platform-wheels-and-precision)
if you want `bundled=True` and the hash-checked CPU binary together.

```python
from hyperl import f32, open_compute, weighted_relu

cpu = open_compute("cpu", library="/absolute/path/to/libhyperl_cpu_runtime.so")
assert list(weighted_relu(cpu, f32([-1, 2, 3]), f32([2, 3, 4]))) == [0, 6, 12]
```

For CUDA, pass `"cuda:0"` and supply contiguous 1-D `torch.float32` tensors
on `backend.device` without autograd. For AMD or Intel use `"rocm:0"` or
`"xpu:0"` with the corresponding framework build. `BackendUnavailable` means
the requested runtime/device could not be selected. There is no fallback.

## Run an explicit model session

Use [the pinned upstream build](MODEL_SHARDING_AND_FLEET.md) and the correct
driver/toolchain. The builder accepts `--backend cpu|cuda|hip|metal|vulkan|sycl`;
that selects one build configuration, not a hardware test. Use a separate build
directory for each target. Check the binary's `--list-devices` output before
choosing its exact device ID. Review the executable and dependent libraries and
pin the model and executable hashes when configuring the session.

```python
from hyperl import LlamaSettings, open_model

settings = LlamaSettings(
    executable="/absolute/path/to/llama-server",
    executable_sha256=reviewed_server_digest,
    model="/absolute/path/to/local.gguf",
    model_sha256=reviewed_model_digest,
    backend="GPU",
    devices=(selected_device_id,),
    gpu_layers=8,
    context=512,
)
with open_model(settings) as session:
    result = session.complete("Explain layer offload briefly.", max_tokens=32)
    print(result["text"], result["native"]["assigned_layers"])
```

`gpu_layers` requests a positive layer count for local GPU mode; omit it to
request full offload. It is not a byte cap. Remaining layers and unsupported
operations can run on the CPU by design. HyperL rejects a startup that does
not observe assignments on every selected accelerator. Model hashes, layer
placement and actual GPU work are different kinds of evidence. On the Intel
Mac with the Radeon Pro 560X, start with the existing bounded Metal check;
prior model offload experiments did not establish a speedup.

`open_model(JaxSettings(...))` uses the pinned f32 checkpoint adapter and
takes token IDs instead of a text prompt. Both sessions close on scope exit
and keep their original response schemas. Artifact inspection, conversion,
storage, fleet scheduling/clients and audit remain importable modules; see
[model sharding and fleet](MODEL_SHARDING_AND_FLEET.md). They do not require
installing GPU frameworks for CPU-only use.

## Roadmap and acceptance

1. **This preview:** named compute targets, managed local model sessions,
   installable C targets, a local wheel and known-answer hardware checks.
2. **Physical backend qualification:** independently reproduce CUDA, ROCm,
   Intel XPU/SYCL and Metal/Vulkan workloads, then record model placement,
   numerical agreements, memory and failure behavior. Start with the
   [hardware calls and checklist](HARDWARE_TESTING.md).
3. **Full SDK:** stabilize API/version and error contracts, package qualified
   platforms, add native model/cluster client surfaces and language bindings,
   and run physical multi-device recovery and resource tests. Keep vendor
   runtimes separately installed and publish only validated combinations.

References: [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html),
[JAX installation and device support](https://docs.jax.dev/en/latest/installation.html),
[llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
