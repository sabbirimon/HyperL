# Help test HyperL on real hardware

We welcome a single reproducible run, including failures and unsupported-device
reports. Start from the revision linked in the [GitHub hardware testing call](https://github.com/sabbirimon/HyperL/issues?q=is%3Aissue+%22Hardware+testing+call%22).
Record the exact commit with `git rev-parse HEAD`.

| Hardware | First useful contribution |
| --- | --- |
| NVIDIA with CUDA PyTorch | Run `cuda:0`; report GPU model, driver and CUDA build |
| AMD with ROCm PyTorch | Run `rocm:0`; report GPU model and ROCm build |
| Intel with XPU PyTorch | Run `xpu:0`; report GPU model and driver/runtime |
| Apple Silicon or Radeon/Intel Mac | Run the [Metal qualification](MACOS_METAL.md); note physical device selection |
| Android Adreno | Follow the [Vulkan qualification](ANDROID_VULKAN.md) on an authorized test device |
| ARM64 / x86-64 CPU | Run the CPU SDK check and [installed C example](SDK.md) |
| Two authorized machines | Follow the [TLS split tests](DISTRIBUTED_EXECUTION.md); distinguish CPUs, GPUs and physical hosts |

## Run the Python check

Install HyperL in an isolated environment and build its C CPU reference with
[the SDK commands](SDK.md). Accelerators additionally need an independently
installed framework build and driver supported for that exact platform.
On macOS with the C runtime built in `build/native`:

```sh
python -m hyperl verify-compute --library "$PWD/build/native/libhyperl_cpu_runtime.dylib" --target cpu
```

On Linux substitute `libhyperl_cpu_runtime.so`; on Windows use the absolute
path to the built `hyperl_cpu_runtime.dll`. With a platform wheel use
`--bundled` in place of `--library PATH`. Replace `cpu` with one exact
accelerator selector and try `--elements 65536` after the default test passes.

The check executes multiply/ReLU against an independent known answer. The
accelerator path checks the result device and also compares with the native CPU
reference. It exits 0 with a result hash, 3 when the requested runtime or device
is unavailable, and nonzero on failure. The end-to-end timing includes input
conversion, transfers, execution and verification, but excludes backend setup.
It is a diagnostic, not a kernel speed benchmark. The test does not establish
model offload, cancellation, training or multi-GPU performance.

For model offload, share the local [GGUF SDK](SDK.md) configuration without private
paths, backend/device, requested and observed layers, runtime and model digests,
and a CPU comparison with the same prompt and sampling settings. Record context,
startup, steady generation, VRAM and any unexpected CPU placements separately.

Use the repository's [issue chooser](https://github.com/sabbirimon/HyperL/issues/new/choose)
and [reporting guide](TEST_REPORTING.md). Link any related run or source change.
Code contributions follow [CONTRIBUTING.md](../CONTRIBUTING.md). Driver fixes,
documentation, packaging and reproducible reports all help.
