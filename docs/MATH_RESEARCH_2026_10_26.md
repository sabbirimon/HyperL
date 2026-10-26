# Mathematical research qualification — 2026-10-26

The reviewed OpenAI math snapshot is
`1581d4cbdd69a0faf34b2089614fe2a814316a6a`. Its
[README](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/README.md)
reports 719 manuscripts in 372 families, with partial formalization. The
[revision history](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/history.md)
includes withdrawals and repairs. This review scanned family titles and selected
abstracts/scope notes; it did not independently verify all proofs or run Lean.

## Candidate decisions

1. [Matrix multiplication family 107](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/docs/107.md)
   is research-relevant to future tensors. Its arithmetic exponent does not
   establish a practical float32 kernel, memory bound, crossover or GPU speedup.
   Require a concrete construction, numerical semantics and end-to-end comparison
   with qualified native kernels before implementing a new backend.
2. [Scheduling family 124](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/docs/124.md)
   assumes unit-duration jobs on exactly three identical machines. It cannot be
   directly substituted for heterogeneous placement with communication costs.
   A future restricted oracle must reject inputs outside those assumptions.
3. Matching/cache-placement candidates need explicit graph/metric assumptions and
   measured execution plus transfer costs. Pure mathematical improvements are not
   automatically product improvements.
4. Lattice hardness does not change HLASv1 or qualify post-quantum cryptography.
   NIST's [ML-KEM standard](https://csrc.nist.gov/pubs/fips/203/final) is a separate
   standardized key-establishment mechanism; any new suite needs reviewed provider
   code, interoperability, downgrade protection and explicit wire versions.

These are engineering assessments. No upstream proof/code was imported and no
novel cipher, new matrix kernel, proof completion or speedup is claimed.

## Applied verification improvement

`PreciseReductionOracleTest` uses exact `BigDecimal(Double)` values of binary32
inputs as an independent oracle. It checks 200 seeded cancellation fixtures plus
subnormal residual and extreme-magnitude cases against the existing compensated
`precise-sum/1` primitive. This is finite regression evidence, not a universal
rounding theorem. Ordered-f32 `hyperl/1`, ABI 1 and all native kernels are unchanged.

Validation: JVM 91 tests, zero failures/errors, two skipped; `installDist` and
`distZip` pass. Native build and all three CTests pass; Python script suite 53 pass.
Native Metal probe coverage is discovery, not a new GPU qualification. No paid
calls, model downloads, driver installation, commits or pushes were made.

New tests/documentation follow the repository's existing Community/Enterprise
terms; previous Apache grants remain intact. No newly restricted HyperL code was
copied into Meshlit.

## Owner-requested performance expansion — 2026-10-26

[Mathematics-informed performance roadmap](MATH_PERFORMANCE_ROADMAP.md) defines M0–M5: measured baselines, bounded matrix contracts, practical qualified kernels, research constructions, restricted scheduling and product adoption. These are future gates; no unmeasured speedup or new cipher is claimed.
