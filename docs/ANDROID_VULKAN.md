# Android GPU build and qualification

On 2026-10-08 the owner resumed single-phone GPU testing. A connected Samsung
Galaxy A20s **SM-A207F**, Android 11/API 30, executes generated HyperL kernels
on its **Adreno (TM) 506** through system Vulkan. **Eleven finite-output cases
match the CPU reference, and two overflow cases correctly reject output.**
This is a native ADB-shell qualification experiment. It is not an Android HyperL
installer, a JNI runtime wired into Meshlit, a full tensor/model engine or a
production qualification. No root, network upload, driver installation or new
APK is needed. Other Android vendors, Mali/Immortalis/Xclipse GPUs and iOS require
separate devices and adapters. The unrelated two-phone cluster tests stay paused.

## What the runner does

`native/vulkan_bridge.cpp` is an original C++17 host using the installed Android
Vulkan loader. `--probe` enumerates real physical devices, compute queues, limits
and driver/API metadata. `--run INDEX N INPUT_COUNT DIRECTORY` runs one reviewed
SPIR-V module with sorted input bindings, a 32-bit length push constant and
64-thread workgroups. No software/CPU/other-GPU fallback is allowed.

The host requires 1–262,144 elements, 1–8 inputs, at most 256 KiB shader bytecode,
little-endian IEEE f32 and sufficient device limits. It admits at most 32 MiB of
Vulkan allocations and currently requires host-visible coherent storage memory.
It uses shader-write → host-read synchronization and a two-second completion
fence. An unconfirmed completion exits the process with failure; it never returns
a success marker. The host rejects nonfinite inputs/outputs and creates a fresh
output with exclusive/no-symlink semantics. It does not guarantee GPU preemption,
whole-device isolation, a hard process-RSS bound or a fleet emergency stop.

`scripts/build_android_vulkan.py` uses **NDK 28.2.13676358**, Android API 24 ARM64,
static C++ runtime linking, the existing HyperL CLI emitter/CPU reference and
that NDK's `glslc`/`spirv-val` with Vulkan 1.0 target. The SDK must already be
installed at the explicit path; the script downloads nothing. Each new bundle
contains source/toolchain hashes, shader/input/reference hashes and retained
HyperL/Android-toolchain license notices. A manifest checks integrity; it is
not a publisher signature or authorization to execute an unknown bundle.

`scripts/qualify_android_vulkan.py` requires an existing owner-started loopback
ADB server and one authorized Samsung (or explicit `--serial`). It checks model,
API/ABI, bundle integrity, exact GPU identity, completion, finite outputs and CPU
agreement before recording each pass. Multiple eligible GPUs require `--gpu-name`.
The f32 comparison tolerance is `1e-6 * max(1, abs(reference))`; these listed
examples do not qualify every rounding/subnormal/fusion behavior.

Only generated synthetic vectors, bytecode and the reviewed executable are pushed
to a unique `/data/local/tmp/hyperl-vulkan-UUID` directory. The runner removes that
directory after success or failure and records cleanup; failure to clean fails the
report. It records no phone serial, credentials, app data, raw logs or identifiers
beyond the stated model/GPU/platform metadata. Individual ADB subprocesses are
bounded to 30 seconds and each native run uses Android `timeout 20`. A report is
created at a new local path, never replaced. No listener on the phone is started.

## Build and run

First build the HyperL CLI with Java 17+ using `./gradlew installDist` (JDK 21
is used in CI). Commands below use placeholder **absolute paths**; replace them
with your own installed SDK/checkout/output paths.

```sh
python3 scripts/build_android_vulkan.py \
  --ndk /absolute/Android/sdk/ndk/28.2.13676358 \
  --cli /absolute/HyperL/build/install/hyperl/bin/hyperl \
  --output /absolute/new-android-vulkan-bundle
```

Unlock the owner-selected phone, enable Developer options → USB debugging and
accept the Mac's RSA prompt. Use a data-capable USB cable. In a normal Terminal:

```sh
/absolute/Android/sdk/platform-tools/adb start-server
/absolute/Android/sdk/platform-tools/adb devices -l
```

Proceed when the intended device shows `device`. `unauthorized` requires the
phone owner's approval; an empty list is not a successful detection. In a
restricted desktop-agent session, a daemon started inside that session may be
unable to use macOS USB interfaces. The owner must start the USB-capable server
in their normal Terminal; the harness does not auto-start it or bypass permissions.

```sh
python3 scripts/qualify_android_vulkan.py \
  --adb /absolute/Android/sdk/platform-tools/adb \
  --bundle /absolute/new-android-vulkan-bundle \
  --gpu-name 'Adreno (TM) 506' \
  --output /absolute/new-samsung-gpu-report.json
```

For multiple connected phones provide `--serial` for the authorized target.
For a manually configured separate local ADB server provide `--adb-port`.
The current harness intentionally checks Samsung manufacturer to preserve this
owner-authorized test scope. Generalizing it needs a selected-device contract and
the corresponding physical vendor checks. It does not use Wi-Fi pairing or root.

An already owner-paired TLS wireless endpoint can also be selected explicitly
with `--serial PHONE_IP:DEBUG_PORT`. The owner enables/pairs Android wireless
debugging separately; this script does not turn it on or authorize a host.

## Observed results

The first actual report is `hyperl-android-vulkan-qualification/1`, generated at
**2026-10-08 04:38:45 UTC (10:38:45 Asia/Dhaka)**. Raw evidence is local, with
SHA-256 `88b90ce2e31da2bcf4ef1afa0e7a8b561981f8c70fb225624e9f899ad686ba43`.
Native source SHA-256:
`27c114f087b03a945a06d3f84e5cf9de43d6fdfe69455a070026cbe48229ab61`.
Bundle-manifest SHA-256:
`05cca2929e1916c370f61d5854f8be74afb8dc46ddf73617730eb798db406c55`.
The later notice-preserving bundle has separate integrity metadata; the original
report/bundle remain unchanged. The device reports Vulkan **1.1.128**, integrated
GPU type, vendor ID 20803 and opaque driver version 2149539840. Do not interpret
that vendor driver value as a standard Vulkan version.

| Cases | Actual result |
|---|---|
| Eight elementwise recipes, three elements each | add, multiply, ReLU, weighted/residual ReLU, affine, affine-ReLU and residual-affine-ReLU pass CPU comparison |
| Weighted ReLU, 257 and 65,536 elements | Both pass, including partial-workgroup tail handling at 257 |
| Fractional weighted ReLU | `[0, 0.030000001192092896, 0]`, CPU-verified |
| Positive and negative f32 overflow before ReLU | Both explicitly dispatched by this negative-test harness, rejected by the native output check, no result published |
| Temporary files | All removed after the run |

The normal CPU preflight rejects those two overflow programs before GPU dispatch;
the direct native negative tests deliberately exercise the generated shader's
finite-intermediate marker and the host's rejection boundary.

For 65,536 elements the observed native pipeline creation takes **16.4321 ms**,
submit/fence wait **2.55885 ms**, and the ADB native-run command **206.835 ms**.
Vulkan allocations total **786,432 bytes**. This is one observation, not a GPU
timestamp, p95/throughput result, full upload/download/verification measurement
or CPU speedup. CPU reference generation runs on the build host. Cold/warm cache,
driver overhead, thermal/battery/sustained load and alternative memory paths need
separate performance experiments. No performance promise follows from detection.

### USB and TLS wireless reruns

On 2026-10-08, the same notice-preserving bundle was tested through explicit USB
and paired TLS wireless targets. Each run passed all thirteen listed cases
(eleven CPU comparisons and two expected rejections), confirmed Adreno 506
completion and removed its temporary files. ADB wireless access does not change
these kernels into remote model inference or an in-app Meshlit GPU backend.

| Transport | Pipeline creation, 65,536 elements | Submit/fence wait | ADB run/wait wall time |
| --- | ---: | ---: | ---: |
| USB | 16.6934 ms | 2.58333 ms | 240.884 ms |
| TLS wireless | 17.0516 ms | 2.72365 ms | 686.041 ms |

Each cell is one observation, not a repeated latency distribution or speedup.
The ADB measurement includes host/transport/command overhead; the native wait
is not a hardware timestamp. Mac builds were running during these checks.
Local report SHA-256 values, with serials/addresses absent from reports:

- USB, 2026-10-08 05:55:06 UTC:
  `0a642e2bbf52bc3b2d25d6a9e12d5f42c4effd117317fe8da9d838d45f12153b`.
- TLS wireless, 2026-10-08 05:55:42 UTC:
  `0202aa8bd69367f4cf999b30d26ad1677834b163e0f72ca4d8a8bf23a348e0ae`.

## Next integration gates

1. Add an Android JNI/library adapter with application-UID device discovery,
   ownership/cleanup, explicit memory admission and acknowledged completion.
2. Wire it behind Meshlit's HYPERL human/per-function/global-stop policy, without
   presenting the current CPU screen as GPU execution. Handle uncertain GPU
   completion separately from cooperative CPU cancellation.
3. Test actual app run/stop, revocation, busy/deadline, process/background/rotation,
   low memory, device loss and thermal/battery conditions on this phone.
4. Add the broader f32 conformance corpus and other physical vendors. Reductions,
   matmul/attention, complete models, NPU backends and clusters remain separate.

References consulted for SDK/API design (original implementation, no copied sample
code): [Android shader compilation](https://developer.android.com/ndk/guides/graphics/shader-compilers),
[Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html),
[ADB protocol/commands](https://android.googlesource.com/platform/packages/modules/adb/+/refs/heads/main/docs/user/adb.1.md).
