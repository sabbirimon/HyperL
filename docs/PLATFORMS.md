# Platform and accelerator matrix

| Platform | Available source/tooling | Qualification still needed |
|---|---|---|
| macOS x86-64 | Locally tested JVM CLI/GUI panel, C CPU library, C99 kernel execution; OpenCL bridge compiled/probed | Native window interaction; GPU probe returned no devices; no GPU execution proof |
| Linux x86-64/ARM64 | JVM distribution; C/CMake SDK; OpenCL source bridge for installed runtime | CI and actual devices; CUDA/HIP/CANN/transports not integrated |
| Windows x86-64/ARM64 | JVM `.bat` launcher/installer; C SDK CMake/MSVC path | CI, actual native GUI/installer and accelerator tests |
| Apple Silicon macOS | JVM/C source and Metal emitter | ARM64/Metal hardware runs, signing and package checks |
| iOS/iPadOS | C API and Metal source suitable for native integration | Swift/Objective-C host, Xcode/device qualification, sandbox lifecycle and installable app/library |
| Android ARM64/x86-64 | Portable C API and Vulkan/OpenCL source | JNI/NDK runtime/package, actual Adreno/Mali/Immortalis/Samsung/other driver checks. Desktop Swing is not an Android app |
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
