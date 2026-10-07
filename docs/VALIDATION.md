# Standalone alpha validation — 2026-10-08

Actual local host: macOS/x86-64, Homebrew OpenJDK 21.0.12, Apple Clang 17,
Gradle 9.4.1, Kotlin 2.4.10. No Android SDK is required by this standalone build.
This evidence concerns HyperL standalone, not the separate Meshlit Android app.

```sh
./gradlew --offline --no-daemon --max-workers=1 test installDist distZip distTar
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
python3 -m unittest discover -s scripts -p 'test_*.py' -v
```

Final JVM/package command: **BUILD SUCCESSFUL in 57s**. **25 cases: 24 pass,
1 skipped, 0 failures/errors**. Portable C CTest: **1 pass**. Python installer:
**2 pass**. The C library was previously configured with CMake Release and compiled
on this actual host; the final C command reused that unchanged compilation.

JVM coverage includes numerical/version/DAG/shape/cancellation/bounds behavior,
generated C99 compilation and actual `[0,6,12]` execution with overflow rejection,
source-only boundaries, adapter predicates, compatible-island planning, real Swing
panel CPU/source buttons in headless mode, strict JSON, encrypted multi-chunk data
round-trip and wrong-key/quota/chunk/manifest corruption, empty datasets, policy
memory estimates/low heap/budget rejection and actual heap observation.

A real local IPv6 `::1` HTTP fixture verifies authenticated read-only node metadata,
redirect rejection, oversized body rejection and complete-response timeout. This is
loopback/in-process evidence; remote HTTPS endpoints and distributed compute are
unrun. Snapshot fixtures exercise memory policy; they do not qualify HBM, bus widths,
VRAM, RSS limits or large-scale performance.

The optional original OpenCL bridge compiled against the installed macOS framework
and probed actual devices: **no available OpenCL GPU**. The real GPU execution test
is therefore the one skipped case; it never substitutes CPU hardware. Other bridge
tests exercise real bounded host subprocess/failure contracts. Emitted CUDA/HIP/
Metal/Vulkan/OpenCL sources do not establish those backend executions.

One earlier combined check hit the 5-second generated-C host-process deadline while
other large builds were active. The isolated compiler/kernel recheck passed, and
the final combined JVM check also passed without extending the deadline. An earlier
memory policy treated free OS pages as allocatable memory and rejected dataset
checks. That policy was corrected: environment free pages are advisory, while CPU
admission enforces the selected byte budget and observed JVM headroom. Failed
attempts are not counted as successful validation.

The desktop preview is an actual Swing panel rendered headlessly, not proof of a
native visible window/session. [OS matrix CI run 37697211217](https://github.com/sabbirimon/HyperL/actions/runs/37697211217)
completed successfully for Ubuntu, Windows and macOS at source `0203267`. Every
job built/tested the JVM distribution, compiled/ran the portable C CPU contract
and ran the Python installer checks. This establishes those runner build paths;
it does not qualify native GUI sessions, OEM mobile devices or vendor accelerators.
C ABI/IEEE semantics and source compatibility still need ARM/RISC-V/mobile target
builds and hardware checks. Full SDK, tensor/model engine, native mobile packages,
qualified GPU/NPU/FPGA runtimes, distributed scheduler, automatic spill, large-node
capacity, kernel/fabric performance, telecom stack and certification remain later.

See [memory boundaries](MEMORY.md), [platform matrix](PLATFORMS.md),
[programming/install guide](USER_GUIDE.md) and [ordered roadmap](ARCHITECTURE_AND_ROADMAP.md).
