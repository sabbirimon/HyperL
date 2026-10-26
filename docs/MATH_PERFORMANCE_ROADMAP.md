# Mathematics-informed performance expansion

Owner-requested future expansion, 2026-10-26. The aim is a faster, more capable
HyperL with measured evidence. This roadmap is not an implementation claim.
See [research assessment](MATH_RESEARCH_2026_10_26.md) for the pinned OpenAI source,
verification limits and current exact-arithmetic regression work.

## Delivery sequence

| Gate | Work | Required acceptance |
| --- | --- | --- |
| M0 — baseline | Record representative vector, reduction and future matrix shapes, dtype, host/driver, memory budget and input checksums. Separate cold-start, transfer, execution and total time. | Reproducible correctness oracle; warmup/repetitions; peak memory; p50/p95 timing; cancellation and error paths. No claim from one favorable sample. |
| M1 — matrix foundation | Add explicit bounded shape/stride/layout contracts and a reference matrix operation under a new supported operation/version boundary. Preserve existing graph/ABI behavior. | Nonfinite/overflow/shape/alias/bounds tests; deterministic reference results; owned-buffer lifetime and cancellation. Tensor/model execution remains a distinct deliverable. |
| M2 — practical acceleration | Evaluate reviewed platform BLAS/Accelerate or qualified GPU libraries behind optional adapters. Use blocked kernels, buffer reuse and operation fusion only where semantics allow. | Compare to M0 on real small/large shapes, including transfer and allocation; publish observed crossover and memory, accuracy tolerances and unsupported targets. No silent CPU fallback or fast-math. |
| M3 — new matrix constructions | Evaluate OpenAI math family 107's arithmetic-complexity construction after obtaining an executable, auditable algorithm and checking its assumptions. | Independent proof/implementation review; finite-size constants, memory and float behavior; outperform the qualified M2 baseline on intended workloads. Otherwise retain as research. |
| M4 — scheduling research | Prototype family 124 only for unit jobs on three identical workers; use independent feasibility checks. Compare matching/placement candidates on explicit workload graphs. | Reject nonmatching assumptions. For real heterogeneous fleets, model measured speeds, transfer, memory, energy/thermal constraints and failures; compare to current planner before dispatch integration. |
| M5 — product adoption | Version and expose only successful implementations with an explicit opt-in until qualified. | Host/device matrix, regression suites, rollback and documented scope. Meshlit consumes a separately reviewed adapter; new restricted HyperL code is not copied into its Apache tree. |

M0–M5 are future gates. Existing `precise-sum/1` independent oracle tests are
implemented; a new fast matrix kernel or theoretical scheduling implementation is
not. Improving performance does not justify changing numerical or wire semantics.

## Security and power

Keep current authenticated encryption and signatures. A hardness result is not a
cipher specification. Post-quantum experiments require separately versioned suites,
a reviewed standardized provider, test vectors, interoperability and downgrade
protection; they cannot be bundled into a matrix optimization.

Battery and thermal impact must be measured on intended devices. Faster arithmetic
may still lose overall if it needs more memory, copies or radio transfers. Record
energy only when a real supported measurement is available; never infer it from
runtime alone. No numeric speedup, energy target or LTS support promise is made
before measurements.

## Immediate next implementable slice

Complete M0/M1 with small bounded matrices and a reference oracle first. Reuse
existing native admission/cancellation patterns; keep policy/UI outside timed
loops. Then benchmark M2 before attempting M3. This is the lowest-risk path to
practical speed while preserving a research track for the new mathematics.
