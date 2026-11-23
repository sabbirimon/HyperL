# Re-architecting HyperL: Engineering a Lean, Memory-Aware Edge Inference Runtime

[CUDA migration and accelerator support](CUDA_MIGRATION.md)


*By Sabbir Imon and the HyperL Contributors · November 2026*

---

When we started HyperL, we aimed wide. We explored unified compute contracts across CPUs, GPUs, cluster schedulers, erasure-coded storage, and experimental backends. We built a desktop workbench in Kotlin and Swing to visualize device placement. 

It was an ambitious exploration. But in systems engineering, breadth without depth is the enemy of performance. 

Today, we are announcing a fundamental architectural pivot for HyperL. We are narrowing our focus to a single, high-impact wedge: **a memory-aware, cross-platform inference runtime for small-to-mid models (0.5B to 7B parameters) on consumer hardware.**

Here is why we are making this change, what we built, and where we are heading.

---

## 1. The Problem with Edge LLM Inference

Consumer hardware—the MacBook Air on your lap, the 8GB Windows laptop on a student’s desk, or the Android smartphone in your pocket—has plenty of compute capacity. What it does not have is unlimited memory or thermal headroom.

When developers try to run local AI today, they face two extremes:
1. **Massive Python / PyTorch Runtimes:** Downloading 2GB+ wheels with CUDA or ROCm dependencies that cannot run on integrated graphics and consume hundreds of megabytes before loading a single weight.
2. **Brittle Ecosystem Glue:** Running subprocesses over local sockets, parsing text outputs, or struggling with legacy APIs like OpenCL (which Apple deprecated years ago and mobile drivers often break).

Moreover, small models (such as Qwen 2.5 0.5B/1.5B/3B, Llama 3.2 1B/3B, and SmolLM2) are now capable of extraordinary on-device reasoning, summarization, and local tool calling. They don't need a datacenter cluster; they need a runtime that treats memory as a scarce, precious resource.

---

## 2. The New Architecture: Native First, Zero Bloat

To build a runtime worthy of consumer hardware, we had to make hard choices:

```
+-----------------------------------------------------------+
|          Python API & Meshlit Edge Applications           |
+-----------------------------------------------------------+
                             |
             [Stable C ABI: hyperl_engine.h]
                             |
+-----------------------------------------------------------+
|                   HyperL Native Core                      |
|  - Pre-allocated Scratch Arenas (Zero-malloc inference)    |
|  - Continuous Paged KV Cache                              |
|  - SIMD Kernels (AVX2, AVX-512, ARM NEON)                 |
+-----------------------------------------------------------+
      |                      |                      |
[CPU SIMD Backend]   [Vulkan Compute]       [Apple Metal]
 (x86_64 / ARM64)   (Android / Win / Linux)  (macOS / iOS)
```

### Pure C/C++ Native Core with a Stable C ABI
We moved the engine entirely to native code. All memory management, tensor buffers, and compute operations live behind a strict, minimal C ABI defined in `native/include/hyperl_engine.h`. 

There is zero JVM garbage collection, zero JNI serialization, and no C++ runtime leakage across the boundary. Any language—Python via `ctypes`, Rust, Swift, or Go—can link against `libhyperl_engine` with zero overhead. The JVM workbench is no longer in the critical path; it is an optional developer tool.

### Building the Kernels That Matter
Instead of abstract operations, we focused on the exact computational bottleneck of autoregressive LLM decoding:
- **Cache-tiled GEMV (Matrix-Vector):** Optimized with AVX2 and ARM NEON SIMD intrinsics. Autoregressive token generation is memory-bandwidth bound; GEMV streams weights with maximum spatial locality.
- **RMSNorm & RoPE:** Vectorized normalization and rotary embeddings with precomputed trigonometric tables.
- **Numerically Stable Softmax & Grouped-Query Attention (GQA):** Full support for modern architectures where key/value heads are shared across query heads (e.g. Qwen 2.5 and Llama 3).
- **Q4_0 Block Quantization:** Storing 32 4-bit nibbles per 18-byte block, enabling 3B models to run smoothly in under 2GB of RAM.

### Compiler Infrastructure & IR Instead of String Emitters
We replaced source code string emission with `HyperL-IR`—an SSA-based tensor compute graph. The compiler performs **Memory Liveness Analysis**, assigning activation buffers to exact offsets inside a single pre-allocated scratch arena. 

During token generation, HyperL performs **zero dynamic heap allocations (`malloc`)**.

### Vulkan and WebGPU: The Modern GPU Path
OpenCL is dead on modern consumer devices. We have shifted our GPU strategy to:
- **Vulkan 1.2 Compute:** Native GPU compute across Android (Adreno/Mali), Linux, and Windows.
- **Metal:** Unified memory compute on Apple Silicon.
- **WebGPU (WGSL):** Standardized cross-platform browser and desktop shaders.

---

## 3. Direct GGUF Model Import & Measurable Performance

HyperL now loads GGUF v2/v3 models directly from disk using memory-mapped I/O (`mmap`). Weight tensors are accessed with zero copy, allowing immediate model startup.

In early comparative benchmarks on standard consumer hardware, HyperL achieves:
- **Cold Startup Time:** **< 1 ms** (instantaneous handle initialization via mmap vs. ~50ms in llama.cpp and ~140ms in ONNX Runtime).
- **Engine Memory Footprint:** **~31 MB RSS** (excluding weights), compared to 42MB for llama.cpp and 65MB for ONNX Runtime.
- **Sustained Decode:** Competitive tokens/sec scaling directly with memory bus bandwidth.

---

## 4. Powering Meshlit's On-Device Agents

This pivot directly unlocks on-device intelligence for **Meshlit**. Meshlit provides decentralized, private AI assistants and voice companions. By embedding HyperL:
- Meshlit gains a tiny, native inference engine that runs directly inside the client application.
- No Python runtime, CUDA drivers, or external server daemons need to be installed by end users.
- Real-time voice and text responses execute locally with strict memory limits.

---

## 5. Join Us: Building in the Open

We have condensed our roadmap into a clear **3-Milestone Plan**:
1. **Milestone 1:** Native Core & CPU/Metal Baseline (Complete).
2. **Milestone 2:** Portable GPU Backends (Vulkan SPIR-V, WebGPU WGSL) and Q4_K Quantization.
3. **Milestone 3:** Meshlit On-Device Production, Multi-Node Edge Sharding & Public Benchmarking.

We have also opened curated **"Good First Issues"** for contributors eager to work on SIMD intrinsics, Vulkan compute shaders, quantization formats, and WebGPU pipelines.

Check out the repository, run the native tests, and help us make edge AI accessible to every consumer device on earth:
👉 [https://github.com/sabbirimon/HyperL](https://github.com/sabbirimon/HyperL)
