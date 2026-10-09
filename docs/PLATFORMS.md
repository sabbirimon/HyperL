# Platform and accelerator matrix

| Platform | Available source/tooling | Qualification still needed |
|---|---|---|
| macOS x86-64 | Tested JVM CLI/GUI panel, C CPU library/C99 execution; Metal passes four bounded Radeon Pro 560X cases in the owner's Terminal | Native window/GPU interaction, wider Metal conformance; OpenCL execution and other exact GPUs |
| Linux x86-64/ARM64 | JVM distribution; C/CMake SDK; OpenCL source bridge for installed runtime | CI and actual devices; CUDA/HIP/CANN/transports not integrated |
| Windows x86-64/ARM64 | JVM `.bat` launcher/installer; C SDK CMake/MSVC path | CI, actual native GUI/installer and accelerator tests |
| Apple Silicon macOS | JVM/C source and macOS Metal host source preview | ARM64/Metal hardware runs, signing and package checks |
| iOS/iPadOS | C API and Metal source suitable for native integration | Swift/Objective-C host, Xcode/device qualification, sandbox lifecycle and installable app/library |
| Android ARM64/x86-64 | Portable C API and Vulkan/OpenCL source; ARM64 Vulkan qualification runner passes 11 outputs/two expected rejections on Galaxy A20s / Adreno 506 | App-UID JNI/runtime/package, broader f32/lifecycle/thermal tests, x86-64 and other OEM GPUs. Desktop Swing is not an Android app |
| RISC-V/other CPUs | C99 source and versioned ABI; JVM tools where a supported JRE exists | Toolchain/ISA/ABI/float conformance, hardware execution and actual packages |
| FPGA/embedded/RTOS/bare-metal | C core integration surface | Allocator/OS/driver/isolation adapters, toolchain and cancellation; no direct/kernel backend exists |

Device GPUs: Apple Metal and Android Vulkan are separate backend paths. OpenCL on
a mobile device is vendor/driver-dependent; no support follows from Android alone.
References: [Apple Metal](https://developer.apple.com/metal/),
[Android Vulkan prerequisites](https://developer.android.com/ndk/guides/graphics/getting-started),
[Khronos OpenCL](https://www.khronos.org/opencl/).

NVIDIA CUDA/TensorRT, AMD HIP/ROCm, Huawei CANN/AscendCL/MindSpore, Qualcomm QNN,
Kirin HiAI, MediaTek NeuroPilot/Neuron and Apple Core ML/ANE are separate SDK/provider
milestones. XRING public NPU SDK availability remains unverified. NPUs are not
generic device GPUs. SDK detection, free research access and source emission are
not license/redistribution grants or inference qualification.

## First standalone CI result

[Run 37697211217](https://github.com/sabbirimon/HyperL/actions/runs/37697211217)
passed all three Ubuntu, Windows and macOS jobs at `0203267`, including JVM
tests/distribution, CMake/CTest and Python installer checks. References to CI
qualification in the matrix above now have that runner evidence; ARM/mobile/OEM
architectures, desktop interaction, accelerators and signing remain separate.

## Metal host preview — alpha.4

The macOS host uses system Metal and supports explicit compatible Intel/AMD or
Apple silicon device selection. Local AppleClang 17 builds it successfully, but
the agent's process sees no GPUs. The owner's Terminal passes four bounded Radeon
Pro 560X GPU cases with CPU comparison and one expected overflow rejection.
See [the hardware record](A1990_METAL_VALIDATION.md) and
[Mac/Metal guide](MACOS_METAL.md); other GPUs, full models and performance need
separate evidence.
See [NVIDIA architecture/software coverage](NVIDIA_COMPATIBILITY.md) and the
[full HyperL platform](SOFTWARE_STACK.md) for current versus future interfaces.

## Android Vulkan experiment — 2026-10-08

Owner-authorized single-phone tests now execute generated kernels on Samsung
SM-A207F, Android 11/API 30, Adreno 506, Vulkan 1.1.128. Eleven outputs match the
CPU reference; two overflow checks reject, with temporary files removed. This is
an ADB-shell native runner, not Meshlit GPU integration or a phone installer.
[Reproduce and inspect the scope](ANDROID_VULKAN.md). Other phone/cluster tests
remain separate acceptance gates.
