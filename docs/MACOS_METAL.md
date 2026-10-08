# macOS Metal preview and A1990 Radeon qualification

Metal supports compatible Intel-based Macs as well as Apple silicon. Intel Mac
hosts may expose Intel and AMD GPUs. See Apple's
[Intel multi-GPU guide](https://developer.apple.com/documentation/metal/finding-multiple-gpus-on-an-intel-based-mac).
The owner identified this machine as A1990; macOS reports Radeon Pro 560X (4 GB VRAM)
and Intel UHD 630, with Metal support. Hardware listing is not an execution test.

The original Objective-C++ bridge uses existing system Foundation/Metal frameworks,
explicit device-name selection and bounded generated elementwise MSL. No new driver,
privilege or downloaded executable is installed. The macOS bridge is a source
preview; it must be compiled with a reviewed local Apple toolchain.

## Build and use

From the HyperL source directory:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
./gradlew --no-daemon installDist
```

Use a maintained Java 17+ runtime for the CLI. The native bridge itself uses the
Apple frameworks rather than Java. Probe with an explicit absolute binary path:

```sh
build/install/hyperl/bin/hyperl metal-probe "$PWD/build/native/hyperl-metal"
build/install/hyperl/bin/hyperl metal-run "$PWD/build/native/hyperl-metal" \
  "EXACT NAME FROM PROBE" examples/elementwise.json examples/inputs.json
```

The CLI uses the **exact** name. It rejects missing or ambiguous names; it never
substitutes the Intel GPU or CPU for the requested Radeon. In the workbench, select
`METAL_GPU`, enter the explicit bridge path under GPU setup, click Probe GPU and
choose the returned index. The frontend resolves that index to a name and the
native run pins the exact name, protecting against enumeration-order changes.

For a binary release, native sources/CMake are under `sdk/`. Build with
`cmake -S sdk -B build/native -DCMAKE_BUILD_TYPE=Release`; use the archive's own
`bin/hyperl` and `scripts/qualify_macos_metal.py` instead of a Gradle source launcher.

## A1990 test from macOS Terminal

Run the reviewed local harness in normal macOS Terminal, from the source checkout:

```sh
python3 scripts/qualify_macos_metal.py \
  --cli "$PWD/build/install/hyperl/bin/hyperl" \
  --bridge "$PWD/build/native/hyperl-metal" \
  --output "$PWD/a1990-metal-terminal-report.json"
```

This explicitly executes those two local binaries. Set `JAVA_HOME` to your existing
maintained JDK if the launcher cannot locate it. It selects the **sole Radeon Pro
560X** by name; `--device-name` can explicitly select another exact GPU. No matching
device yields an `unrun` report and exit 3, never a CPU result labeled as GPU work.
The report is a new file and never overwrites an existing report.

The generated cases cover 3, 257 and 65,536-element multiply/ReLU, a small
multiply/add/ReLU dependency chain and rejection of overflow before ReLU. Every
successful GPU request is checked against the CPU reference; device/completion
metadata are included. This only qualifies the listed cases, not full models,
all f32 values, every AMD device or a speedup.

## Memory, timing and cancellation boundaries

- Use managed buffers with `didModifyRange` and explicit output synchronization
  for non-unified Intel/AMD Macs; use shared buffers on unified-memory devices.
  [Apple managed-resource synchronization](https://developer.apple.com/documentation/metal/synchronizing-a-managed-resource-in-macos).
- Disable fast math, use MSL 2.0, pad threadgroups with shader bounds checks, validate
  finite inputs/results and compare GPU outputs against CPU within the documented
  `1e-6 * max(1, abs(expected))` tolerance. Reductions/tensors/models are unsupported.
- Limits: eight inputs, 262,144 elements/vector, one-million input elements in the
  raw bridge (1,048,576), 64 KiB source, bounded frontend graph/retention and output files.
- Compile and submit/wait timings are observed; optional GPU timestamps cover the
  submitted command, including its synchronization. CLI wall time also includes
  startup, IO and mandatory CPU verification. They are not speedup measurements.
- Host process deadline: 20 seconds, then bounded two-second process cleanup.
  GUI index selection performs a separate bounded probe before the bounded run.
  A submission marker is removed only after the native API confirms command end.
  If completion remains unconfirmed, the application blocks new Metal requests.
  Killing the host does not prove GPU kernel cancellation or driver preemption.

## Current evidence

Local AppleClang 17 compiles the adapter on macOS 15.8.1. Its actual probe returns
an empty device list; a separate Metal/OpenCL probe also sees no compute GPU in
this execution session despite macOS listing both installed GPUs. The cause is
not established here; restricted process/device access is a possibility. The
session harness reports **unrun**, with no hardware dispatch or performance claim.
CPU/pre-dispatch/error tests pass; actual GPU tests skip. The owner's request now
authorizes testing this Mac's GPU; phone/two-device validation remains paused.

### Terminal discovery and post-alpha.4 result-format fix

The owner's normal Terminal probe exposes both the AMD Radeon Pro 560X and Intel
UHD Graphics 630. The first qualification attempt failed with `Device or completion
mismatch`, before recording any passed cases. This differs from the empty device
list in the agent's process; broader file/network access does not itself establish
GPU visibility.

The alpha.4 CLI encoded `backend` and `cpuVerified` as default-valued properties.
Its JSON serializer omitted those fields, although the harness requires an explicit
verification flag. The source fix makes both fields required and sets them only
after native completion and CPU comparison succeed. A serialization regression
test reproduces the omission without claiming hardware execution. The harness
continues to reject absent/false verification, wrong devices or incomplete commands,
and now retains bounded metadata with a specific failure reason.

Rebuild `installDist` from the corrected source before retrying, and choose a **new**
report path. The published alpha.4 archives retain their original contents; this
source fix does not replace them. The owner's subsequent Terminal rerun **passed
all listed cases on AMD Radeon Pro 560X**: four GPU requests with CPU comparison,
managed-buffer synchronization and confirmed completion, plus expected overflow
rejection before dispatch. See [the hardware record](A1990_METAL_VALIDATION.md) for
provenance, timings and remaining acceptance. Intel/Apple silicon execution, full
models and general performance qualification remain separate.
