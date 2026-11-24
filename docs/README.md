# HyperL documentation index

[CUDA migration and accelerator support](CUDA_MIGRATION.md) covers CUDA project migration, portable kernel compilation, vendor SDK requirements and hardware results.


Start with the [native CLI and graph compiler](NATIVE_ENGINE.md) and processor management plan (archived). JVM and desktop installer guides below describe older releases.



[Native model sharding and fleet](MODEL_SHARDING_AND_FLEET.md) · [Error correction and assurance mapping](HIGH_ASSURANCE_BASELINE.md)


[Logical devices, LLM residency and fleet inventory](LOGICAL_DEVICES_AND_MODEL_STORAGE.md) — 2026-11-02 development APIs and limits.

Updated 2026-10-24. Alpha.5 is the published release; alpha.6 additions are development
source. Guides separate working code, estimates, provider contracts and future plans.

## Learn, build and diagnose

- [Programming guide with diagrams](PROGRAMMING_GUIDE.md)
- [Installation and CLI/workbench reference](USER_GUIDE.md)
- [Build, validation and packaging](BUILD_AND_TEST.md)
- [Troubleshooting and debugging](TROUBLESHOOTING.md)
- Progress and bug-fix record (archived)
- Agent backlog and upcoming features (archived)
- [Current and historical validation](VALIDATION.md)
- [Share measured hardware results and feedback](TEST_REPORTING.md)

## Runtime and developer APIs

[Language](LANGUAGE_V1.md) · [Recipes](USE_CASES.md) · [Python/framework interoperability](LIBRARY_INTEROPERABILITY.md) ·
[Memory](MEMORY.md) · [Dataset format](DATASET_FORMAT.md) · [Distributed execution](DISTRIBUTED_EXECUTION.md) ·
[Cryptography](CRYPTOGRAPHY.md) · Post-quantum migration (archived) ·
Quantum APIs (archived) · [OriginQ](ORIGINQ_ADAPTER.md) ·
[Diagnostics](CODE_DIAGNOSTICS.md) · [Editor integration](ide/README.md).

## Hardware and deployments

[Platforms](PLATFORMS.md) · [NVIDIA](NVIDIA_COMPATIBILITY.md) · [AMD](AMD_COMPATIBILITY.md) ·
[Metal](MACOS_METAL.md) · [Android Vulkan](ANDROID_VULKAN.md) ·
[Android/Mac CPU qualification](ANDROID_CROSS_PLATFORM_PROBE.md) ·
[Android probe project](../android/probe/README.md) · [Desktop installers](DESKTOP_INSTALLERS.md) ·
[Enterprise fabrics](ENTERPRISE_FABRICS.md) · Enterprise clusters (archived) ·
Mobile/edge (archived) · Precision/HPC (archived).

## Architecture and plans

Architecture/roadmap (archived) · [Software stack](SOFTWARE_STACK.md) ·
[Sharding contract](SHARDING_EXECUTION_CONTRACT.md) · Heterogeneous execution (archived) ·
[Native performance architecture](NATIVE_PERFORMANCE_ARCHITECTURE.md) ·
[Foundation hardening](FOUNDATION_HARDENING.md) · AI services (archived) ·
Developer experience (archived) · Developer tools (archived) ·
UI design (archived) · Owner requests (archived) ·
Production and Meshlit boundary (archived) · [Telecom research](TELECOM.md).

## Provenance and historical records

[Licensing](LICENSING.md) · [License history](LICENSE_HISTORY.md) ·
[Python license history](../python/LICENSE_HISTORY.md) · [Contribution rules](../CONTRIBUTING.md) ·
[Contributor agreement](CONTRIBUTOR_AGREEMENT.md) · [Upstream reuse](UPSTREAM_REUSE.md) ·
[Font provenance](FONT_PROVENANCE.md) · [Mac hardware evidence](A1990_METAL_VALIDATION.md) ·
[CPU baseline](CPU_BENCHMARK.md) · [2026-10-08 research](RESEARCH_2026_10_08.md) ·
[2026-10-08 validation](VALIDATION_2026_10_08.md).

The October 24 audit reviewed the current guide/plan corpus. Dated research,
measurements, license history, contributor terms and font provenance are preserved
as records rather than rewritten as new evidence. Read the current validation and
platform matrix before interpreting historical capability statements.
