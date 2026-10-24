# HyperL use cases and ready-made library

HyperL currently supplies bounded finite **f32 vector preprocessing**, local
developer tools, a native C ABI preview and encrypted local dataset streaming.
It is useful before or beside an existing AI model. Full models, training,
attention, matrix multiplication and distributed execution are later milestones.
This guide distinguishes runnable workflows from the expansion plan.

## 1. Start with a ready-made recipe

Install a [desktop package](DESKTOP_INSTALLERS.md) or extract the portable archive
with Java 17+. Set `hyperl` below to your actual CLI launcher path.

```sh
hyperl library
hyperl recipe weighted_relu > weighted.hyperl.json
# Open weighted.hyperl.json in the workbench with Open.
# Or supply exactly x and w in a local JSON file:
hyperl recipe-run weighted_relu examples/inputs.json
```

The supplied `examples/inputs.json` has `x=[-1,2,3]`, `w=[2,3,4]`; weighted ReLU
returns `[0,6,12]`. `recipe` writes a versioned workspace containing editable
program and sample inputs. It does not execute the program. Use the workbench
Validate, Analyze, Memory plan and Run CPU actions before adapting the recipe.

| Recipe ID | Formula | Sample output | Practical use |
|---|---|---|---|
| `add` | x + y | `[3,-1,4]` | Combine already aligned feature vectors |
| `multiply` | x × w | `[-2,6,12]` | Per-feature weighting or masks |
| `relu` | max(x, 0) | `[0,2,3]` | Remove negative responses |
| `weighted_relu` | max(x × w, 0) | `[0,6,12]` | Weighted positive feature stage |
| `residual_relu` | max(x + residual, 0) | `[0,1,3]` | Elementwise skip connection |
| `affine` | x × w + bias | `[-1,7,13]` | Per-element scale and offset |
| `affine_relu` | max(x × w + bias, 0) | `[0,7,13]` | Scale, offset and activation |
| `residual_affine_relu` | max(x × w + bias + residual, 0) | `[0,6,13]` | Combined preprocessing chain |
| `dot` | ordered sum(x × w) | `[16]` | Weighted feature score, unnormalized similarity |
| `sum` | ordered sum(x) | `[4]` | Small local aggregation |
| `positive_sum` | ordered sum(max(x, 0)) | `[5]` | Aggregate positive responses |
| `squared_norm` | ordered sum(x × x) | `[14]` | Squared vector energy; no square root |

These examples use x=`[-1,2,3]`, w=`[2,3,4]`, y=`[4,-3,1]`, bias=`[1,1,1]`,
residual=`[1,-1,0]`. Each workspace includes only the recipe's required keys.
All binary elementwise operations require equal lengths: scalar broadcasting
and automatic dtype conversions are unsupported. `affine` is an elementwise
scale/offset, **not** a dense neural-network layer.

Each input has 1–262,144 values; graphs have 1–8 inputs and 1–64 instructions;
total retained input/intermediate vectors cannot exceed 1,048,576 f32 values.
Default array budget is 16 MiB. JVM admission also observes heap headroom;
this estimate is not a physical memory reservation or whole-process RSS limit.
Nonfinite inputs and intermediate overflow fail, even if a later ReLU would
hide the value. Reductions use ordered f32 addition and reject intermediate
overflow. Parallel/reassociated reductions are not silently substituted.

## 2. Python beside NumPy or an existing AI pipeline

Build the C runtime, install the Python preview into your chosen environment,
then select its absolute shared-library path. Native installers include the
host library under their application payload's `native/` directory; portable
archives contain source to build. Python and NumPy remain optional, separately
installed dependencies.

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
python3 -m pip install ./python
```

```python
from hyperl import NativeCpu, f32, affine_relu, dot, squared_norm

cpu = NativeCpu('/ABSOLUTE/PATH/libhyperl_cpu_runtime.so')
# Use libhyperl_cpu_runtime.dylib on macOS or hyperl_cpu_runtime.dll on Windows.
x, w, bias = f32([-1, 2, 3]), f32([2, 3, 4]), f32([1, 1, 1])
print(list(affine_relu(cpu, x, w, bias)))  # [0.0, 7.0, 13.0]
print(list(dot(cpu, x, w)))              # [16.0]
print(list(squared_norm(cpu, x)))        # [14.0]
```

Python exports `add`, `multiply`, `relu`, `weighted_relu`, `residual_relu`,
`affine`, `affine_relu`, `residual_affine_relu`, `dot`, `sum_values`,
`positive_sum` and `squared_norm`; `LIBRARY` and `recipe_program(id)` let you
inspect their immutable declarative programs. All helpers accept an optional
`cancelled` callback and the explicitly configured backend. NativeCpu checks
its byte budget before copying inputs and checks the loaded ABI version.

For NumPy, use `from_numpy` and `to_numpy`: bounded contiguous one-dimensional
float32 arrays only. CPU Torch copies use `from_torch`/`to_torch`; tensors
requiring gradients are rejected. The optional TorchBackend preview can dispatch
elementwise regions on an explicitly selected device after CPU comparison.
Unavailable CUDA is an error, and its reductions remain unsupported. These
bridges copy data; they do not import arbitrary CUDA/cuDNN/TensorFlow operators,
provide autograd, or promise zero-copy interoperability. See the
[interop guide](LIBRARY_INTEROPERABILITY.md) for exact tested boundaries.

## 3. Offline work on a single Android phone

The separate Meshlit port adds **Settings → HyperL libraries** to both app flavors.
Choose any of the twelve recipes; inspect/edit JSON; Validate its graph and
memory estimate; Run CPU; review the result, measured total time and backend.
Stop cancels local work. Copy output explicitly places it on the OS clipboard.
The editors accept up to 65,536 characters each and preview at most 256 values.

This workflow needs no root, VM, cloud account or internet. It can support local
sensor-feature preparation or small score/energy calculations. It does not
capture sensors automatically, run a GPU/model or convert the device into a
distributed transformer worker. Meshlit's independent bundled model runtime
remains responsible for chat inference. Android GPU adapters are a later gate.

CPU execution/source generation use Meshlit's per-function HyperL control and
latched emergency stop. The controller admits one active operation, rejects busy
requests without queueing, applies a ten-second cooperative deadline and uses
the selected memory budget. Disabling the function or global stop cancels active
work before publication. Editor contents are transient in this screen; persist
them externally only through a future explicit storage workflow. No inputs are
uploaded or delegated to agents. Physical-phone validation remains paused.

## 4. Author a kernel, then choose an accelerator explicitly

The same JSON/workspace can be validated on CPU and exported as CUDA, HIP,
OpenCL, Metal, Vulkan GLSL or C99 source. In the phone screen, Metal/Vulkan source
generation and Copy output support an external developer workflow. Emission is
not driver installation, compilation, GPU execution or hardware qualification.
Reduction recipes cannot be emitted for the current GPU paths.

On macOS, the Metal bridge can be selected explicitly:

```sh
hyperl metal-probe /ABSOLUTE/PATH/hyperl-metal
hyperl metal-run /ABSOLUTE/PATH/hyperl-metal 'AMD Radeon Pro 560X' \
  examples/elementwise.json examples/inputs.json
```

The A1990 Radeon evidence covers only the four listed elementwise cases and
overflow rejection. [Recorded hardware results](A1990_METAL_VALIDATION.md) are
separate from agent-shell availability, Apple Silicon, Intel GPU and model
qualification. Measure transfer, compile, dispatch and full-call cost separately
before choosing acceleration; a small GPU dispatch time is not total speedup.

## 5. Protect local dataset transfers

Existing `keygen`, `data-import` and `data-export` commands stream bounded local
datasets with authenticated encryption, manifests and integrity checks. Keep the
private key outside the dataset/installation directory. Choose new destinations
and explicit maximum sizes; failed integrity checks do not publish valid output.
See [the programming guide](USER_GUIDE.md) for complete runnable commands.

This supports preparing local research data for later AI processing. The f32
library is not a cryptographic primitive. No remote synchronization, automatic
NVMe spill, public server or regulatory certification follows from local encryption.

## 6. Embed the portable C ABI

C/C++, Rust FFI and other hosts can link the scalar C99 runtime through
`sdk/native/include/hyperl.h`. Specify vectors, ordered operations, output
capacity and cancellation callback. Check every returned status and length;
retain caller buffers until completion. Inputs are copied by the runtime; callers
must not mutate their arrays concurrently while snapshotting is underway.
No filesystem, shell, network or privileged hardware operation exists in the IR.

Use this for a bounded preprocessing component or conformance oracle. The
current native reference is not SIMD-optimized and has no hard real-time guarantee.
The full SDK, package registries and native IDE/debugger integrations remain later.

## 7. Enterprise and datacenter expansion

Today, isolated developer workstations can use the local library and explicit
authenticated read-only node metadata probe. Capacity and telecom profiles are
planning/validation tools, not operating distributed jobs or a telecom stack.

Useful next services are tenant quotas, authenticated node enrollment, durable
jobs/checkpoints, model-runtime adapters, telemetry, operator approvals, paginated
fleet views and acknowledged cluster emergency stop. C/C++ kernels/accelerator
adapters and Rust control services remain the chosen performance/safety direction.
Start with a small hardware qualification cluster before promising large-node
capacity, HBM/NVLink/RDMA/CXL efficiency or heterogeneous turbo performance.

See [the enterprise plan](ENTERPRISE_AND_CLUSTER_PLAN.md),
[AI services plan](AI_SERVICES_PLAN.md) and
[production acceptance gates](PRODUCTION_AND_MESHLIT_PLAN.md). NVIDIA, AMD,
Intel, Apple, Huawei/Chinese frameworks, mobile NPUs and telecom providers require
individual version/license/security/hardware validation. Naming them is not
evidence of compatibility. Signed deployment artifacts, lifecycle testing and
independent security review are still required for a production release.
