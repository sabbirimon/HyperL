# Precision and high performance computing plan

Owner requirement, 2026-10-24. Enterprise, supercomputer and hyperscale workloads
are primary HyperL targets. Phones and desktops provide useful compatibility tests;
they do not define the eventual workload size or numerical limits. This is a
proposed implementation plan, not a claim that new dtypes or HPC backends execute.
Baseline source: `07b8d51`; actual evidence remains in [validation](VALIDATION.md).

## 1. Current numerical baseline

`hyperl/1` graphs use finite f32 vectors and f32 outputs. Graph `sum` preserves
input order and f32 accumulation. The separate CPU `precise-sum/1` helper accepts
f32 inputs, uses compensated binary64 accumulation, then rounds once to finite
f32 output. It improves selected sums but is neither an f64 graph engine nor exact
arithmetic. Existing ABI, wire and reduction behavior must remain compatible.
The physical Android/desktop test qualifies one bounded f32 workload, not HPC scale.

## 2. Planned types and explicit meaning

| Type / policy | Representation and first purpose | Status / qualification needed |
|---|---|---|
| f32 / binary32 | 4-byte elements; current bounded graph semantics | Preserve existing finite-value and ordered-reduction contract |
| f64 / binary64 | 8-byte elements for scientific computing and accuracy-sensitive stages | Future typed graph, CPU reference, wire/C ABI and qualified native adapters |
| BF16 | 2-byte elements; reduced storage/transfer for eligible AI operations | Future conversions and explicitly selected accumulator/output types; never a precision upgrade from f32 |
| f128 / IEEE binary128 | 16-byte elements; explicitly identified binary128 arithmetic | Future provider with exact format/rounding tests; hardware versus software execution reported separately |
| Arbitrary binary precision | Requested significand bits, exponent bounds and bounded provider storage | Future MPFR-backed reference/provider; precision and allocation limits required |
| Decimal precision | Explicit decimal radix, digits, exponent and rounding policy | Separate future provider/wire format; binary floating point is not decimal arithmetic |
| Complex values | Typed real/imaginary components with declared component precision | Future operators and error contracts; account for both components and scratch |

BF16 has one sign bit, eight exponent bits and seven stored fraction bits. Its
exponent width matches binary32, while its significand is much shorter; it does
not provide f32 accuracy. [AMD AOCL BF16 format](https://docs.amd.com/r/en-US/68552-AOCL-api-guide/BFloat16-Format).
Do not equate `long double`, a 16-byte C object or a compiler type name with portable
binary128. Compiler support depends on the target and ABI; it also does not prove
hardware acceleration. [GCC floating types](https://gcc.gnu.org/onlinedocs/gcc-14.3.0/gcc/Floating-Types.html).
MPFR provides arbitrary binary precision and specified rounding; decimal parsing
does not make its arithmetic decimal. Its exponent/subnormal behavior needs explicit
configuration when serving as an IEEE reference. [MPFR overview](https://www.mpfr.org/),
[MPFR numerical semantics](https://mpfr.org/mpfr-current/mpfr.html).

## 3. Contract before optimization

- Version dtype, shape/stride, byte order, wire encoding and alignment independently
  of a host language type. Never transport raw C structures or round f64/f128 data
  through a JSON parser restricted to binary64. Bind dtype/shape/payload digests.
- Specify input/storage, multiply/compute, accumulation, reduction and output types
  separately; BF16 inputs with f32 accumulation remain mixed precision. Conversion
  nodes are visible in the accepted plan and count toward its error and time costs.
- Declare rounding at conversions and operations, FMA contraction, overflow/underflow,
  signed zero and subnormal handling. Preserve current nonfinite rejection; any
  future NaN/infinity support needs its own contract and canonical wire rules.
- Separate ordered reductions, fixed-tree reproducible reductions, compensated
  reductions and tolerance-based fast reductions. A fixed tree alone cannot promise
  cross-backend bit identity without matching arithmetic and compiler behavior.
- Require workload error budgets: absolute/relative/ULP criteria as appropriate,
  accumulation depth, residual/convergence or model-quality checks. Do not claim
  universal accuracy from a wider dtype or from one passing random-input test.
- Record native versus emulated arithmetic, provider/compiler/driver revision and
  exact shape/operator coverage. Unknown or unsupported requirements fail admission.

## 4. Four explicit optimization objectives

| Objective | Optimize while preserving the accepted numerical contract |
|---|---|
| Accuracy | Meet the requested dtype, rounding and error budget; use a qualified higher-precision provider where required |
| Latency | Minimize complete request time including queue, conversion, copies, synchronization and verification |
| Throughput | Maximize sustained valid completed work under stated batch, concurrency, fairness and latency limits |
| Capacity | Complete a workload beyond one admitted device's memory using genuine sharding and bounded staging |

The user or job policy selects allowed precision changes. No objective silently
downgrades f64/f128 to f32/BF16. Offer a changed plan with measured tradeoffs when
the requested contract cannot fit. Capacity gains may increase completion time.

## 5. Admission and enterprise execution

Use checked byte counts for every typed tensor, padding, conversion, accumulator,
workspace, communication buffer and checkpoint. MPFR/decimal allocations include
limbs, metadata and provider scratch, not only requested bits divided by eight.
Count shared NUMA/UMA domains once; distinguish HBM/VRAM, host RAM and storage.
Enforce quotas, cancellation, bounded queues and worker isolation before allocation.

Qualify batched native SIMD/BLAS/tensor-engine paths with data reuse, tiling and
explicit NUMA placement; keep UI/configuration outside measured kernels. Group
compatible accelerators for collectives, then connect groups using authenticated,
measured lanes. Integrate explicit Slurm/Kubernetes scheduling and native-fabric
adapters only after their independent tests. [Enterprise fabrics](ENTERPRISE_FABRICS.md).
Require tenant isolation, job/epoch fencing, rank/worker-loss handling, bounded
retry, integrity-checked checkpoints and recovery before a production scale claim.

## 6. Dependency order and acceptance gates

| Gate | Bounded deliverable | Evidence required |
|---|---|---|
| P0: typed contract | Versioned dtype/rounding/wire/admission schema; reject unsupported execution | Round-trip bit patterns, malformed sizes, overflow, incompatible versions and unchanged f32 fixtures |
| P1: f64 CPU baseline | Typed f64 vectors and ordered reduction in one reference/native path | Independent high-precision oracle; cancellation, conversion boundaries, cancellation-prone sums, subnormals and overflow tests |
| P2: native HPC kernels | One qualified CPU SIMD/BLAS path and one applicable accelerator path | Matrix/operator coverage, exact or declared tolerance checks, memory/NUMA observations and transfer-inclusive baseline comparison |
| P3: mixed precision | BF16 storage/compute plus explicit accumulation/output policy | Conversion ties, range/error budgets and real workload quality; no inferred tensor-core/NPU support |
| P4: extended precision | Binary128 provider, then bounded arbitrary binary/decimal and complex tracks | Pinned/licensed provider, independent rounding/conversion tests, exponent limits, scratch bounds and measured emulation cost |
| P5: distributed HPC | Typed shards, qualified collectives, durable recovery and batch scheduler integration | Physical 3–5-host pilot, then measured 16/64/256-host steps; strong/weak scaling, contention, failure and tenant-isolation tests |

P2/P3 depend on the typed tensor/operator foundation; full-model sharding remains
mandatory under [H3/H4](SHARDING_EXECUTION_CONTRACT.md), independently of adding f64.
Report accuracy, useful work/s, complete latency, peak bytes, transfer goodput and
energy where observed, with warm-up, workload and hardware provenance. Peak vendor
FLOPS, node-count configuration and dtype negotiation are not execution evidence.

**First implementation:** P0 plus a bounded f64 CPU path with independent numerical
fixtures and byte admission. Keep f32 behavior unchanged; qualify native throughput
and mixed precision afterward. Wider precision and hyperscale performance remain
separate measured delivery gates, with no advertised maximum cluster size yet.
