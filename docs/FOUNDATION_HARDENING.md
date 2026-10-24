# Alpha.6 foundation improvements

This unreleased development increment addresses reproducible review findings.
Earlier alpha.4/alpha.5 reviews describe an older snapshot: bounded Radeon Metal
and Samsung Vulkan execution already have separate hardware evidence. They still
do not establish full models, training, CUDA/HIP or a production cluster service.

## Implemented changes

| Review finding | Current change | Evidence boundary |
|---|---|---|
| Python and Kotlin accept different identifiers | `contracts/hyperl-1.json` generates JVM/Python/C constants and IDE schema bounds; all require the same ASCII identifiers, counts and arities | Actual same-JSON JVM/C tests, including rejected names, forward references, shape/overflow and declaration bounds |
| Limits are repeated in several runtimes | Generated files are checked in CI with `scripts/generate_contract.py --check`; backend host input bounds reuse the contract | This is the existing bounded vector contract, not a new tensor language |
| Dense native/security code is difficult to audit | C/Metal/Vulkan sources are formatted; the CPU and encrypted-data paths have explicit validation, stages and failure cleanup | Readability and tests do not establish a complete security audit |
| Native malformed inputs need stronger tests | Strict CPU compiler warnings; 4,096 seeded C cases covering 16 exercised admission/error categories, plus ASan/UBSan CI | The corpus is deterministic, not exhaustive or coverage-guided fuzzing |
| Ordered f32 reduction loses small contributions | A separate compensated binary64 `precise-sum/1` helper, rounded once to finite f32, exists in C, Python and JVM | Existing ABI-1 graph `sum` remains ordered f32; no graph opcode or GPU reduction changed |
| Footer detection uses a substring | HLM2 records have explicit types and exact fields; authenticated HLM1 records are recognized by exact structure | Independent authenticated malformed-record fixtures reject unknown/extra fields and string-valued integers |
| Flat chunk files and no key rotation | Grouped chunk directories, authenticated key IDs and direct decrypt/re-encrypt `data-rekey` | Quota is 4 TiB; tests use bounded small/multi-chunk files, not a measured 4 TiB deployment |
| Python installation needs a separately built binary | An explicitly built platform wheel can bundle the reviewed C runtime; `NativeCpu.bundled()` verifies its target/hash before loading | Local Mac wheel tested; OS CI builds its own artifact. No PyPI publication, universal wheel or manylinux qualification |
| No reproducible performance baseline | A checked local benchmark records Python API, native ABI, NumPy and a preallocated naive-C baseline | [CPU results](CPU_BENCHMARK.md) expose overhead; no application/GPU/model speedup is claimed |

The concrete bounds remain 1–8 inputs, 1–64 steps, 1–262,144 elements per vector
and at most 1,048,576 retained f32 values. The generated contract does not make
runtime Python objects, JVM heap admission or vendor memory into reserved VRAM.

## Explicit precision choice

```python
from hyperl import NativeCpu, f32
cpu = NativeCpu.bundled()  # explicit opt-in after a compatible local wheel install
print(list(cpu.precise_sum(f32([16777216, 1, -16777216]))))  # [1.0]
```

The unchanged graph `sum` returns 0 for these values. The new helper uses Neumaier
compensation with binary64 intermediates; input length, selected memory admission,
finite values/results and cancellation still apply. It is not exact real-number
arithmetic. C declares `hl_sum_precise` as an additional optional symbol without
changing `hl_execute`, existing operation numbers or ABI-1 structures. An older
selected native library remains usable for graph execution and reports the new
helper as unavailable. The JVM CLI accepts `sum-precise INPUTS.json [budget]`,
with exactly one `x` input vector. These helpers are CPU-only.

## Next implementation gates

| Milestone | Work needed before claiming delivery |
|---|---|
| Faster reusable native execution | Prepared graph/buffer lifetimes, liveness reuse and vectorization under unchanged admission/cancellation/numerical rules; compare equivalent complete workloads |
| Tensors and model inference | Versioned dtype/shape/layout/operator contracts; reuse a reviewed model runtime, pinned model/tokenizer provenance, real end-to-end tests and unsupported-operator errors |
| NVIDIA CUDA / AMD HIP | Explicit vendor adapter and installed SDK/driver, owner-selected physical hardware, exact-device numerical and total-transfer timing checks; generated source is insufficient |
| VRAM/NUMA and spill | Actual provider allocator observations, reservation/lifetime accounting, failure/pressure tests and authenticated spill recovery before any automatic spill feature |
| Language/IDE tools | A versioned parser and semantic diagnostics, LSP integration and debugger/profiler behavior; current JSON editor/schema remain the delivered authoring tools |
| Rust fleet services | Authenticated durable jobs, scoped quotas, isolation, retry/stop/failure protocols and multi-host acceptance; the current cluster configuration is not a scheduler |

Those milestones are independent projects and qualification gates. No unavailable
backend is represented as a successful stub or silent CPU replacement. Licence
scope, prior Apache rights and third-party notices remain in [LICENSING.md](LICENSING.md).
