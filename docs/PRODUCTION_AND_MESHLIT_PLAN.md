# Production hardening, libraries and Meshlit delivery

Owner request, 2026-10-08: production-grade HyperL, a Meshlit app port, a rich
ready-made library, detailed use cases and macOS/Windows/Linux installers.
Production readiness is an acceptance result, not a version-name change. Keep
HyperL standalone and Meshlit as separate repositories and review branches.

## Current implementation increment

1. Harden the bounded runtime: validate the complete graph and shape/memory cost
   before copying inputs; explicit memory budget; finite ordered reductions;
   bounded concurrent admission and deadlines; cancellation; no backend fallback.
2. Provide twelve shared recipes: vector add/multiply, ReLU, weighted/residual
   ReLU, affine/affine-ReLU/residual-affine-ReLU, dot product, sum, positive sum
   and squared norm. Provide Kotlin/JSON/Python examples and actual output tests.
   Recipes reuse the versioned execution contract; unsupported GPU reductions
   remain unavailable. These are preprocessing building blocks, not a model engine.
3. Port the portable contracts, memory admission and recipes to Meshlit core-gpu.
   Add a working Settings → HyperL screen: recipe selection, editable program and
   inputs, validation/memory estimate, CPU execution, Stop and bounded output.
   Both app flavors share the screen. The Mac Metal host is not an Android driver.
4. Package the desktop workbench with a bundled maintained Java runtime using
   jpackage on each target OS. Produce macOS DMG, Windows MSI and Linux DEB/RPM
   where the runner toolchain permits; keep portable ZIP/TAR and CLI available.
   Record architecture, hashes and unsigned-preview status. No signing keys in Git.
5. Publish an honest use-case guide with runnable examples, prerequisites,
   backend selection, data/privacy boundaries, deployment and explicit limits.

## Validation and release gates

- Runtime/library tests cover real output, malformed graphs, shape/memory
  rejection, overflow, cancellation, concurrency and unavailable backends.
- Cross-repository conformance fixtures and source provenance prevent silent
  drift. Run standalone JVM/native/Python checks and Meshlit core/app tests,
  both APK flavors and fatal lint. Inspect packaged classes/assets and docs.
- Build installers on their actual OS/architecture and inspect their payloads.
  Build/CI success is separate from interactive installer/upgrade/uninstall tests.
- The owner resumed single-phone GPU qualification; the standalone native runner
  passes listed Adreno 506 Vulkan cases. App-UID/JNI integration and two-phone
  cluster acceptance remain pending. [Exact test scope](ANDROID_VULKAN.md).
  Do not infer Android
  runtime, thermal, accessibility or vendor GPU acceptance from JVM tests/APKs.
- Radeon evidence qualifies only the previously listed elementwise workloads.
  Expand values/shapes, repeat/warm/transfer measurements, device loss and memory
  pressure before general GPU qualification or performance claims.
- A production release additionally needs target-specific installation/rollback,
  independent security review, dependency/SBOM and vulnerability response,
  owner-controlled Windows/macOS signing/notarization, reproducibility and a
  stated supported-platform lifecycle. No debug-key production signing fallback.

## Subsequent platform milestones

Add typed tensors and qualified model runtimes before advertising inference or
training libraries. Reuse reviewed ONNX Runtime, PyTorch/LibTorch, oneDNN,
CUDA/cuBLAS/cuDNN, ROCm libraries, Metal/MPS/Core ML and Chinese-framework adapters
through explicit provider/version/license contracts. No arbitrary binary import.
Native Android/iOS GPU/NPU adapters, Rust cluster services, acknowledged fleet
stop, authenticated remote execution and distributed collectives have separate
operator configuration and hardware/failure acceptance. Single-phone local
preprocessing remains useful while these later milestones develop.
