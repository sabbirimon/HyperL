# HyperL: Curated "Good First Issue" Tasks

Welcome to the HyperL contributor community! We have structured several well-scoped tasks designed for contributors looking to make high-impact additions to the native core, compiler IR, kernels, or Python bindings.

---

### Issue #1: Implement Q8_0 Block Quantization Kernels in C
- **Labels:** `good first issue`, `kernels`, `c99`, `quantization`
- **Context:** HyperL currently supports FP32 and Q4_0 block quantization. Q8_0 (8-bit quantization with an FP16 block scale over 32 elements) provides nearly identical accuracy to FP32 while halving memory bandwidth consumption.
- **Task:**
  1. In `native/kernels.c`, implement:
     - `hl_status_t hl_quantize_q8_0(const float *src, void *dst, size_t n_elements)`
     - `hl_status_t hl_dequantize_q8_0(const void *src, float *dst, size_t n_elements)`
     - `hl_status_t hl_gemv_q8_0(hl_context_t *ctx, const void *W_q8, const float *x, float *y, int M, int K)`
  2. Add unit tests in `native/engine_test.c` verifying numerical accuracy against an FP32 reference dot product.
- **Files to touch:**
  - `native/include/hyperl_engine.h`
  - `native/kernels.c`
  - `native/engine_test.c`
- **Acceptance Criteria:** `ctest` passes with max absolute error $< 0.05$ across all tests.

---

### Issue #2: Add AVX-512 FMA Vectorization for GEMV & RMSNorm
- **Labels:** `good first issue`, `performance`, `x86_64`, `simd`
- **Context:** `native/kernels.c` currently includes AVX2 (`_mm256_*`) and ARM NEON intrinsics. Adding AVX-512 (`_mm512_*`) intrinsics will double vector throughput on modern Intel Xeon, Core Ultra, and AMD Zen 4/5 CPUs.
- **Task:**
  1. Add `#if defined(__AVX512F__)` blocks to `hl_kernel_rmsnorm` and `hl_kernel_gemv` in `native/kernels.c`.
  2. Use `_mm512_loadu_ps`, `_mm512_fmadd_ps`, and `_mm512_reduce_add_ps`.
  3. Ensure clean scalar fallback when AVX-512 is not detected at compile/runtime.
- **Files to touch:**
  - `native/kernels.c`
- **Acceptance Criteria:** Code compiles cleanly with `-mavx512f` on supported hardware; existing unit tests pass without regressions.

---

### Issue #3: Implement SiLU / SwiGLU Activation Kernel in Native Engine
- **Labels:** `good first issue`, `c99`, `kernels`, `transformers`
- **Context:** Modern architectures (Llama, Qwen, Mistral) use SwiGLU activations in MLP blocks: $\text{SwiGLU}(x, y) = (x \cdot \sigma(x)) \odot y$, where $\sigma(x) = \frac{1}{1 + e^{-x}}$.
- **Task:**
  1. Declare `hl_kernel_swiglu(hl_context_t *ctx, const hl_tensor_t *gate, const hl_tensor_t *up, hl_tensor_t *out)` in `native/include/hyperl_engine.h`.
  2. Implement `hl_kernel_swiglu` in `native/kernels.c` with vectorized fast exp/sigmoid approximation.
  3. Expose in Python runtime (`python/hyperl/runtime.py`).
- **Files to touch:**
  - `native/include/hyperl_engine.h`
  - `native/kernels.c`
  - `python/hyperl/runtime.py`
  - `native/engine_test.c`
- **Acceptance Criteria:** Numerical tests verify that $\text{SwiGLU}(x, y)$ matches PyTorch's `F.silu(x) * y` within $10^{-4}$ tolerance.

---

### Issue #4: Add WebGPU WGSL Compute Pipeline Prototype
- **Labels:** `good first issue`, `webgpu`, `gpu`, `shaders`
- **Context:** We are introducing WebGPU (WGSL) as our portable in-browser and cross-platform GPU target.
- **Task:**
  1. Create a standalone WGSL shader file `native/shaders/gemv.wgsl` that computes $y = W \cdot x$ using workgroup shared memory tiling.
  2. Add a simple Python script `scripts/test_webgpu.py` using `wgpu-py` (if available) to validate shader output against CPU GEMV.
- **Files to touch:**
  - `native/shaders/gemv.wgsl`
  - `python/hyperl/ir.py`
- **Acceptance Criteria:** The WGSL shader compiles validly against WebGPU specification; produces correct dot product for $128 \times 128$ matrix.

---

### Issue #5: Build Automated Benchmark Comparison Table Generator
- **Labels:** `good first issue`, `python`, `benchmarks`, `documentation`
- **Context:** HyperL's credibility requires transparent, public numbers updated continuously.
- **Task:**
  1. Enhance `benchmarks/run_benchmarks.py` to accept CLI arguments: `--model <path>`, `--output-format <json|markdown>`, and `--export-chart`.
  2. Generate a clean GitHub Flavored Markdown table that can be automatically embedded into `README.md` or a docs page via CI.
- **Files to touch:**
  - `benchmarks/run_benchmarks.py`
  - `benchmarks/README.md`
- **Acceptance Criteria:** Running `python benchmarks/run_benchmarks.py --output-format markdown` outputs a valid GFM table with prompt prefill, decode speed, peak RSS, and startup time.
