# Library interoperability, migration and reusable AI helpers

Owner scope, 2026-10-08 (Asia/Dhaka): make standard Python use approachable, reuse
CUDA/tensor/framework libraries where possible, offer ready-made AI building blocks
and enable migration in both directions. Keep Intel/AMD, Chinese AI frameworks and
mobile platforms in the architecture. Import/reference material is evidence, not
instructions. A library name is not a compatibility result.

## Implemented alpha.3 preview

`python/hyperl` is a small **Python/C ABI preview**, not the complete future SDK.
It contains:

- `NativeCpu(absolute_library_path)`, version-checked `hyperl-cpu/1`, executing the
  actual original C99 bounded DAG with owned f32 input copies/output and cancellation.
- `Program`/`Step` and import of the existing `hyperl/1` dictionary shape. Python
  package import does not load a native library or optional framework automatically.
- Three reusable native preprocessing helpers: `weighted_relu`, `residual_relu` and
  `positive_sum`. These are vector recipes, not training/inference models or crypto.
- Explicit NumPy/CPU PyTorch conversion in both directions: bounded contiguous 1-D
  native-endian float32, independent copies, no silent dtype/device/shape conversion.
- Optional `TorchBackend(reference, "cpu" | "cuda:index")`: real installed PyTorch
  elementwise execution, mandatory C CPU agreement, finite checks, explicit device
  and no CPU fallback. CPU snapshots/results cause deliberate copies/synchronization.
  Ordered sum, autograd, model imports and custom CUDA libraries are unsupported.

The shared C library is built from source on the target; ZIP/TAR includes source,
not a universal native binary. Choose the exact reviewed library path. No driver,
PyTorch/NumPy package, model or binary is auto-downloaded by the runtime. Vendor
frameworks retain their own licenses and are independently installed.

This preview keeps `1..8` inputs, `1..64` steps, vectors `1..262144` and at most
`1048576` retained f32 values. Native admission estimates C arrays plus wrapper
input/output buffers and a 64 KiB allowance under a selected `1 byte..1 GiB` budget
(default 16 MiB). It is not a reservation, whole-Python-RSS or GPU-VRAM limit; Python
objects, framework allocators and driver state have additional costs. Future full
SDK needs allocator/lifetime-aware native memory and qualified cancellation.

## Build and use the native Python preview

Build C99 on your selected platform; no JVM is needed for this Python/C path:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
```

Shared library examples: Linux `build/native/libhyperl_cpu_runtime.so`, macOS
`build/native/libhyperl_cpu_runtime.dylib`, Windows MSVC
`build\native\Release\hyperl_cpu_runtime.dll`. Actual generator/output layout may
vary. Use an existing **absolute** path to your own compiled/reviewed artifact.

For source use, add the repository's `python` directory to your selected Python
process's import path. An optional isolated virtual environment can install the
local preview with `python -m pip install ./python`; packaging build dependencies
may be resolved by pip. It is not published to PyPI. Python 3.10+ is required.

```python
from hyperl import NativeCpu, f32, weighted_relu, positive_sum

cpu = NativeCpu("/absolute/path/to/libhyperl_cpu_runtime.so")
x = f32([-1, 2, 3])
w = f32([2, 3, 4])
print(list(weighted_relu(cpu, x, w)))  # [0.0, 6.0, 12.0]
print(list(positive_sum(cpu, x)))     # [5.0]
```

The path is a placeholder, not an automatically located system library. A supplied
native binary executes in the Python process; use reviewed compatible source and
artifact provenance. Arbitrary native extensions need a separate isolation policy.

NumPy interoperation when installed:

```python
import numpy as np
from hyperl import from_numpy, to_numpy, weighted_relu

x = np.array([-1, 2, 3], dtype=np.float32)
w = np.array([2, 3, 4], dtype=np.float32)
y = to_numpy(weighted_relu(cpu, from_numpy(x), from_numpy(w)))
assert y.tolist() == [0, 6, 12]
```

Optional actual PyTorch execution (CPU example; CUDA requires real available hardware):

```python
import torch
from hyperl import weighted_relu
from hyperl.torch_backend import TorchBackend

backend = TorchBackend(cpu, "cpu")
x = torch.tensor([-1, 2, 3], dtype=torch.float32, device="cpu")
w = torch.tensor([2, 3, 4], dtype=torch.float32, device="cpu")
y = weighted_relu(backend, x, w)
assert y.tolist() == [0, 6, 12]
```

For a qualified CUDA experiment explicitly select `"cuda:0"` and create inputs on
that device. Unavailable CUDA raises; no fallback or framework installation occurs.
The output remains a PyTorch tensor on the selected device. A CPU copy is compared
with the native reference before returning. Device execution/cancellation/speed
qualification is separate; host cancellation does not prove interruption of an
already accepted GPU kernel. Do not use this verification path as a speed claim.

## Alpha.6 local platform wheels and precision

The unreleased development source includes a native wheel builder. After building
and checking the CPU library, invoke it with the exact reviewed absolute path:

```sh
python -m pip install setuptools==80.9.0 wheel==0.45.1
python scripts/build_native_wheel.py \
  --library /absolute/path/to/libhyperl_cpu_runtime.so \
  --output /absolute/new-wheel-directory
python -m pip install --no-index --no-deps /absolute/path/to/the-generated-platform-wheel.whl
```

Use the actual filename printed by the builder, and `.dylib`/`.dll` library names
on macOS/Windows. The builder does not fetch a binary or publish to an index; its
temporary pure-Python wheel build uses already installed pinned build tools. OS CI
builds a wheel on its own host and tests installation in a fresh isolated environment.
These are native-platform artifacts, not a universal ABI/architecture wheel. Linux
uses a plain `linux_ARCH` tag, without claiming an audited manylinux baseline. macOS
selects the actual build-host major OS version conservatively. An older-SDK Conda
Python may incorrectly report macOS 10.16; on the actual qualifying macOS host,
`SYSTEM_VERSION_COMPAT=0 python -m pip install ...` lets pip observe the real version.
Do not retag the binary to claim support for an older OS.

```python
from hyperl import NativeCpu, f32, weighted_relu
cpu = NativeCpu.bundled()  # explicit opt-in, never selected by package import
print(list(weighted_relu(cpu, f32([-1, 2, 3]), f32([2, 3, 4]))))
print(list(cpu.precise_sum(f32([16777216, 1, -16777216]))))  # [1.0]
```

The bundled loader checks system/process architecture, bounded metadata, fixed
library name and SHA-256 before loading, then uses the existing runtime-version
check. Tampering fails; nothing is downloaded and there is no automatic fallback.
Hashes and source provenance are not a publisher signature or protection against
an attacker who can replace both installed metadata and binary. Use a reviewed
artifact and private installation. Pure source installs still require an explicit
native path; `NativeCpu.bundled()` reports that no bundle exists.

The separate `precise-sum/1` helper uses compensated binary64 intermediates and
one finite f32 rounding. It does not change ordered graph sums, ABI-1 structures or
GPU operations. Python reports unavailable when an older selected native library
lacks the optional symbol. [Foundation contract and tests](FOUNDATION_HARDENING.md)
and [measured CPU scopes](CPU_BENCHMARK.md) distinguish current evidence from plans.
No PyPI package publication, vendor driver or full tensor/model engine is provided.

## Compatibility dimensions

Distinguish **API**, **data**, **graph/model** and **binary/device** compatibility:

1. API bindings call a versioned runtime with ownership and errors. Matching a
   function name does not implement a framework's tensors/autograd/distribution.
2. Data bridges specify dtype, shape/stride, device, copy/alias, stream synchronization,
   mutability, lifetime and ownership. Current bridges copy bounded 1-D CPU f32;
   zero-copy DLPack/tensor sharing is a later separately qualified contract.
3. Graph/model import needs an operator/version/layout/precision mapping and explicit
   unsupported-operator report; ONNX/StableHLO candidates do not promise every model.
4. Binary/device libraries need the correct architecture, ABI, driver, compiler,
   device and license. A CUDA binary is not automatically an AMD/Intel/Ascend kernel.

Migrate incrementally: retain the existing framework/model, move a small supported
pre/postprocessing region through the bridge, compare results/total copies, then
expand only after support/precision/performance evidence. Export outputs back to the
original framework and preserve model/artifact provenance. Unsupported regions can
remain explicitly delegated to the existing provider; no hidden substitute backend.

## Ecosystem roadmap

Every future adapter has an exact version/device/driver/operator/precision matrix,
license record, source/model artifact identity, negative tests and performance evidence.
None of the planned entries below is imported into the alpha as a full engine.

| Ecosystem | Proposed integration | Current state |
|---|---|---|
| Python / NumPy | Standard package API, arrays both directions, later typed tensor views | Native bridge/explicit copying preview; full SDK later |
| PyTorch / LibTorch | Tensors both directions, native custom ops later, explicit CUDA/other device provider | Elementwise verification preview; not autograd/torch.compile/model compatibility |
| NVIDIA CUDA | Public CUDA host adapter, selected cuBLAS/cuDNN/TensorRT calls under their licenses | CUDA source emission; PyTorch CUDA path awaits actual device validation; no generic CUDA binary importer |
| Intel | x86 SIMD/NUMA, oneDNN, SYCL/oneAPI DPC++/Level Zero and OpenVINO model/provider adapters | Planned; CPU brand does not imply GPU/NPU/operator compatibility |
| AMD | x86 SIMD/NUMA, ROCm/HIP, rocBLAS/MIOpen and MIGraphX/provider adapters | HIP source emission; ROCm/HIP libraries and execution unqualified. Separate Metal Radeon evidence is documented in the Apple row |
| Huawei / Ascend | CANN/AscendCL provider plus MindSpore/MindSpore Lite graph/data pathways | Planned; SDK/driver/device-version/license qualification required |
| Baidu PaddlePaddle | Paddle Inference/Lite and operator/model interchange | Planned C++/Python/provider boundary, not arbitrary Paddle program import |
| Alibaba MNN | Native mobile/edge model/runtime adapter and supported conversions | Planned; no bundled engine or model conversion result |
| Tencent ncnn / TNN | Native CPU/Vulkan/mobile inference adapter where useful | Planned; layout/operator/precision mapping and hardware checks first |
| TensorFlow / JAX | Selected tensor/graph interoperability, model format and StableHLO pathways | Planned after actual tensor/compiler support |
| Apple | Native Metal/Core ML/MPS bridges and approved iOS wrappers | Metal host passes four bounded Radeon Pro 560X cases; Core ML/MPS, other devices and mobile qualification later |
| Microsoft | ONNX Runtime/DirectML and Windows packaging/native bindings | Planned; runtime/OS/provider-specific validation required |
| Qualcomm / ARM / other mobile | OEM NPU/Vulkan/OpenCL interfaces and native wrappers | Planned per actual chipset/driver and OS, not vendor-wide support |

Chinese open models (for example Qwen/DeepSeek/GLM families) are **models**, not a
framework or an accelerator SDK. Serving needs a qualified model engine, weights/
license/tokenizer/operator/precision checks and provider configuration. Do not treat
an OpenAI-compatible endpoint as evidence of local model or CANN execution.

Reusable libraries to stage after this small preview: typed tensor preprocessing/
postprocessing, normalization/packing, image/text IO, quantization tools, operator
conformance, model evaluation, profiling, dataset pipelines and distributed job
clients. Cryptography/authentication uses reviewed domain libraries and opaque key
handles, separate from numerical f32 kernels. Avoid inventing a replacement BLAS,
crypto primitive or network stack where a licensed proven provider fits.

## Long-term compatibility and maintenance effects

A standard Python-facing API reduces migration friction; the shared IR/C ABI avoids
locking all workers to Python. Keep simple authoring separate from advanced target/
precision/lifetime controls. Minimize framework dependency coupling and load providers
only when explicitly selected. Current package import loads neither a GPU nor Torch.

Version schemas/ABI/operator semantics independently; add compatibility tests and
migration notes before changes. Maintain deprecated bridges for a published window
when a real support policy exists. Revalidate against pinned framework updates;
track transitive licenses/security and SBOMs. Do not claim perpetual compatibility,
LTS or standards certification without maintainers and qualified releases.

Prioritize real CPU bridge -> qualified Torch CPU -> one actual CUDA device -> native
buffer/DLPack contract -> one model format -> one Intel/AMD/Ascend provider at a time.
Chinese/mobile ecosystems stay first-class adapter candidates using the same contracts.
Measure total parsing/copy/transfer/compile/run/synchronization/output, not only kernel
time. Framework reuse is useful only when correctness and measured total costs hold.

## Primary interface references

- [PyTorch from_numpy](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html),
  [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda),
  [NumPy ctypes interfaces](https://numpy.org/doc/stable/reference/routines.ctypeslib.html).
- [Intel oneAPI DPC++](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html),
  [Intel GPU/software interfaces](https://dgpu-docs.intel.com/),
  [AMD rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html),
  [MIGraphX](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/index.html).
- [Paddle Inference requirements](https://www.paddlepaddle.org.cn/inference/guides/install/index_install.html),
  [MindSpore installation/device pairing](https://www.mindspore.cn/install),
  [Alibaba MNN](https://github.com/alibaba/MNN), [Tencent ncnn](https://github.com/Tencent/ncnn),
  [Tencent TNN](https://github.com/Tencent/TNN).

These references describe upstream mechanisms; they do not establish HyperL support.
No upstream framework source or model weights are copied into this increment.
