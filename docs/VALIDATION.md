# Standalone alpha validation — 2026-10-08

## Distributed vector execution — 2026-10-09, unreleased alpha.6

Current `test installDist distZip` passes: **61 JVM cases, 59 passed, two
unavailable-GPU skips, zero failures/errors**. Twelve distributed cases use actual
HTTP workers for split/gather, overlapping bounded dispatch, graphs exceeding
single-reference retention, exact ordered terminal reduction, shape rejection,
bad authentication, altered/replayed frames, false output, deadlines and credit
cleanup, stale worker identity, effective concurrency and rejection of approximate
GPU shards for exact ordered reduction and genuine busy-worker rejection without
replay. Strict native CMake/CTest passes three existing cases; fourteen Python
installer/qualification contract tests pass. The updated OpenCL probe compiles
with strict warnings and reports no available GPU here; that is no new GPU pass.

The installed CLI starts **two independent JVM processes on this Mac**. They
compute eight 512-element shards covering 4,096 elements, use both actual workers
and return the exact local reference output. A third independent worker passes
trusted TLS; clients reject both an untrusted certificate and a trusted
certificate with the wrong hostname. Tokens, private key and input fixtures stay
in a temporary private directory; cleanup stops all workers.
`scripts/qualify_distributed.py` reproduces this explicit local qualification.

A freshly built macOS native launcher with a bundled JDK repeats the same
two-worker computation and TLS checks with `JAVA_HOME` removed and `PATH` empty
for every HyperL process. The fixture's certificate-generation tool alone uses
the explicitly installed JDK. Both launcher checks use the final tested application
JAR. The launcher check is not a signed installer or native
compute-engine migration; the workers still use the stated JVM CPU reference.

These are same-host processes and loopback HTTP/TLS, not physical multi-device,
Internet/NAT, Linux NUMA hardware, mixed GPU/NPU, production tenancy, durable
recovery or low-latency fabric evidence. No speedup or VRAM/storage pooling is
claimed. Reported total includes control, transfer, verification and execution.
The Windows/Linux/macOS CI step is added; new-head remote CI is separate.
The initial CI identified a Unix-only missing-key test path and a response/slot
cleanup race causing HTTP 429 on consecutive shards. The test uses the current
platform's absolute path; worker admission now waits at most 100 ms for cleanup
and still rejects genuine concurrent overload. The corrected local full suite
passes; the corrected remote head needs its own result.
The corrected Windows computation/TLS checks subsequently pass but expose a
fixture cleanup failure: stopping the batch wrapper left its JVM holding stderr
open. The helper now stops only each exact live fixture process tree on Windows.
Windows cleanup qualification remains tied to its new CI result.
See [implementation and ownership limits](DISTRIBUTED_EXECUTION.md).

## Cross-language/native/data hardening — unreleased alpha.6

The actual local `test installDist distZip distTar` check passes in **30 seconds**:
**49 JVM cases, 47 passed, two unavailable OpenCL/Metal execution skips, zero
failures/errors**. New tests cover separate precise reduction, independently
encrypted legacy HLM1 export/rotation, HLM2 multi-chunk rotation, authenticated
invalid record types/fields/numbers, retained source, wrong-key and no-overwrite
publication. An initial HLM2 writer omitted default-valued types; its two failing
tests were retained and the corrected writer explicitly serializes record types
before this passing run.

Strict AppleClang 17 Release CMake/CTest: **three passes** (CPU contract, 4,096
seeded stress cases covering all sixteen declared categories, Metal availability
probe). Debug ASan/UBSan CTest also reports three passes; only the C targets are
instrumented. Local macOS leak detection is off because this sanitizer runtime
does not support it. The separate Ubuntu CI sanitizer job enables leak detection;
its new-head result is separate from the local run. The deterministic corpus is
not exhaustive or a claim of coverage-guided fuzzing.

Actual Python/native preview: **ten cases, eight passed, two uninstalled
PyTorch/CUDA skips**. Cross-language conformance: three test methods pass,
covering sixteen same-JSON cases through actual JVM/Python/C, 120 seeded DAGs
against NumPy 2.3.3 ordered-f32 arithmetic, and precise reduction/cancellation/
overflow cases. Fourteen existing installer/GPU-harness script tests pass.
Generated contract/schema consistency passes. [Measured CPU baseline](CPU_BENCHMARK.md)
records real overhead and the distinct cost scopes rather than claiming a speedup.

A local macOS 15/x86-64 native platform wheel builds and installs in a clean
virtual environment: **three installed-package checks pass**, including actual
native execution/cancellation and rejection of corrupt binary/path/target metadata.
An older-SDK Python's initial pip attempt rejected the macOS 15 wheel because it
reported compatibility version 10.16; the retry with `SYSTEM_VERSION_COMPAT=0`
uses the actual host OS and passes without retagging the artifact. Wheel metadata
retains the development licence and prior Apache notices. No index/release upload
or other architecture claim follows from this local installation.

The formatted/current Android Vulkan host compiles under NDK 28.2.13676358 with
strict warnings; all thirteen generated shaders compile and validate. **No phone
is visible in this fresh ADB inventory**, so there is no new physical-device pass.
Historical exact Samsung/Radeon evidence below remains distinct. New-head OS CI,
installer signing and fresh hardware execution require their own results.

[Implemented review fixes and next gates](FOUNDATION_HARDENING.md) ·
[Encrypted format and publication semantics](DATASET_FORMAT.md).

## Community/enterprise licence transition — unreleased alpha.6

The affected JVM/package check passes in **39 seconds**: 44 JVM cases, 42 passed,
two unavailable-GPU skips, zero failures/errors; installDist, ZIP and TAR build.
The actual installed CLI reports `0.1.0-alpha.6` and retains `hyperl/1`. Native
CMake build/CTest passes two checks (CPU contract and Metal probe); fourteen
packaging/qualification script checks pass. The Metal probe is availability
evidence, not a newly qualified GPU.

Actual ZIP/TAR inspection confirms ten licence/history/Python metadata files
match current source. A Python wheel builds as `hyperl-experimental 0.1.0a6`, with
`License-Expression: LicenseRef-HyperL-Community-1.0` and four accompanying files:
LICENSE, NOTICE, Apache-2.0.txt and LICENSE_HISTORY.md. Each matches its package
source; the original Apache text is byte-identical to pre-transition commit
`67b30e09741f4bba1672a891644fc73abb1f9855`. No release/tag, historical alpha.5 asset,
Meshlit grant or dependency licence is replaced. These local packages are not
published installers, signing evidence or proof of legal enforceability.

[Licence scope and legal-review limits](LICENSING.md) explain the free uses,
group thresholds, six-calendar-month production trial and prior Apache rights.
Qualified counsel review remains required before commercial enforcement. No
automated check establishes patent clearance or every contributor's authority.

The owner-authorized main ruleset is Active, targets only `main`, blocks deletion
and force pushes, requires pull requests/resolved review conversations and eight
strict GitHub Actions checks. No bypass actor or second-person approval is set;
the single owner can merge their own checked PR. `dev` remains unprotected.
The configuration does not merge a PR or change release/licence history.

## Workbench themes and editor focus

The actual `test installDist distZip` run succeeds in 1m41s: **44 JVM cases,
42 passed, two unavailable-GPU skips, zero failures/errors**. The new GUI check
switches the real component tree to Paper, verifies saved preferences, retains
edited code/input, toggles editor Focus, and executes the real CPU result `[12]`.
Actual Swing panel renders at 1360×840 were visually inspected in Graphite, Aurora,
Paper and Focus. These are offscreen component renders, not native OS-window or
GPU execution evidence. The current installers/release screenshots predate this UI
increment; source packaging and installer publication are separate.

## Physical Samsung GPU qualification

The owner resumed single-phone GPU testing. The original ARM64 C++17 Vulkan host
compiles with NDK 28.2.13676358/API 24 and `-Wall -Wextra -Werror`; all thirteen
generated shaders compile and pass SPIR-V validation. On Samsung Galaxy A20s
SM-A207F, Android 11/API 30, **Adreno 506 / Vulkan 1.1.128**, eleven finite-output
cases pass CPU comparison and two explicitly dispatched overflow cases reject
without publishing results. Completion and temporary-device cleanup are confirmed.
[Full scope, hashes, commands and observations](ANDROID_VULKAN.md).

Local required JVM/package check succeeds in 15s (`test` reuses the previously
passing unchanged JVM suite; installDist/distZip execute). CMake/CTest: two pass.
Script checks: fourteen pass, including eight new fail-closed metadata/integrity
checks; these unit fixtures do not replace the actual phone evidence above.
Initial ADB enumeration was empty because the session-started macOS daemon could
not use USB interfaces. After the owner started ADB in normal Terminal and
authorized the Samsung, the existing loopback client reached the real device.
No permissions or root authorization were bypassed. No APK/JNI adapter, full
model, sustained-load/thermal, CPU speedup or other mobile GPU is qualified.
The published alpha.5 installers predate this test-runner increment.

## Alpha.5 production-hardening candidate

Local JVM/package checks pass in 48 seconds: **43 tests, 41 pass, two unavailable
GPU tests skip, zero failures/errors**. Native C/Metal-probe CTest: **two pass**;
the probe detects no GPU in the restricted agent shell. Script checks: **six pass**.
Python/native preview: **ten tests, eight pass, two optional Torch checks skip**;
the real shared C library and NumPy execute locally. The twelve-recipe test checks
actual outputs, including dot/sum/energy, and ordered intermediate-overflow rejection.
An initial Python test imported a library inventory under the existing binary-path
variable name; this test-name collision was repaired before the passing rerun.

New JVM checks cover full-graph admission before copies, actual recipe values,
strict malformed-input rejection, capacity/deadline/cancellation and overflow.
Coordination fixtures are explicitly identified; they do not qualify a backend.
The C contract additionally rejects a nonfinite ordered partial sum without
publishing a changed caller output. Language and native ABI remain version 1.

The Intel macOS jpackage app image builds and its bundled-runtime CLI checks pass:
capabilities/version, twelve library entries and weighted ReLU `[0,6,12]` with
JAVA_HOME removed and PATH empty. Native DMG creation fails at `hdiutil create`
in this restricted session. Earlier packaging attempts exposed macOS's positive
version constraint and case-insensitive launcher collisions; both are fixed.
OS-specific installer CI is separate evidence, recorded in release packaging
reports. No native GUI, interactive install/uninstall, signing/notarization or new
GPU qualification is inferred from the app image. Historical alpha.4 Radeon
Terminal results remain exactly scoped in [their hardware report](A1990_METAL_VALIDATION.md).

See [native installers](DESKTOP_INSTALLERS.md), [twelve-library use cases](USE_CASES.md)
and [production/Meshlit gates](PRODUCTION_AND_MESHLIT_PLAN.md). The Meshlit source
port is reviewed and tested separately in its own repository; phones remain paused.

Alpha.5 native installers are now published. The immutable tag source is
`d330d2bc46c427421c9d3dc04aec9ec2e9170b76`. All four
[target package jobs](https://github.com/sabbirimon/HyperL/actions/runs/37721587473)
pass; their initial publication step failed because it assumed flat artifact
directories. The corrected collector reused the exact tagged artifacts and
[publication recovery passes](https://github.com/sabbirimon/HyperL/actions/runs/37722682361).
No tag or older release was replaced. All five published installer sizes/digests
match their target-build reports. Windows/Linux/macOS source checks also pass at
`fda4d92`, `d330d2b` and `415d7de`; workflow-only cleanup is separate from runtime
qualification. The published `desktop-validation.json` and `SHA256SUMS` retain
the complete per-target evidence. Interactive lifecycle/signing gates remain open.

## Historical foundation evidence

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

## Developer workbench increment — alpha.2

Local JVM/package validation after the polished UI/editor tools passes in **2m16s**:
**28 cases, 27 pass, one GPU hardware skip, zero failures/errors**. This includes
actual new workspace round-trip/no-overwrite/example/unsafe-hook and shape
rejection checks, and real panel Validate, Format JSON, Memory plan and Find next
actions in addition to CPU execution and Metal source emission. FlatLaf 3.7.2
no-natives and RSyntaxTextArea 4.0.1 are pinned; component licenses are retained.
The initial API integration compile failure was repaired before this passing run.

The UI design/developer tooling plans and local IDE schema/workspace are included.
Native desktop file dialogs/focus/accessibility and actual VS Code/JetBrains UI
interaction remain unrun. An IDE task/configuration is not a shipped language
server, debugger or complete SDK. The earlier alpha.1 OS CI result above is
historical; the UI increment needs its own OS matrix run before claiming it passed.

## Final alpha.2 diagnostics and polished-layout check

After the final editor-layout and diagnostic changes, the combined JVM/package
command passes in **24s**: **33 cases, 32 pass, one unavailable GPU hardware skip,
zero failures/errors**. New diagnostics checks cover real CPU sample verification,
context completions, operation/reference repairs, exact-text hash/stale-edit
rejection, input/shape/memory/backend errors and actual overflow localization
before ReLU. The real GUI Analyze action is exercised as well.

Visual QA uses the actual Swing panel and CPU button result rendered to a buffered
image. RSyntaxTextArea font metrics are initialized against that actual image in
the headless preview helper; native window interaction is still separate.

OS CI [37700269194](https://github.com/sabbirimon/HyperL/actions/runs/37700269194)
passes all Ubuntu, Windows and macOS jobs for the workbench/diagnostics source
`12a991f`. The subsequent bounded-report refinement also passes locally in
**22s**, retaining **33 cases, 32 pass, one GPU skip, zero failures/errors**. It
caps issues/paths and rejects excessive root fields/input declarations before
expanding diagnostic output. A malformed-field fixture verifies bounded rejection.
The refinement receives its own source CI run; local results do not imply a
GPU, native GUI session or general-language bug detector.

## Reference-driven polish and Python preview — alpha.3

Final UI source/packaging check passes locally in **29s** with **33 JVM cases:
32 pass, one unavailable OpenCL GPU skip, zero failures/errors**. Real GUI checks
now verify actual bundled Inter/JetBrains Mono font families and select Source
emission/Execution tabs before exercising the existing CPU/Metal source actions.
The visual layout uses static vector gradients; no timer, browser engine or runtime
font download is added. Native GUI/window/accessibility remains separate acceptance.

The unchanged C99 core now builds both static and shared artifacts. Actual macOS
shared-library build and CTest pass (**one C contract case**). The bundled Python
3.12 runtime with NumPy 2.3.5 validates the explicit shared-library bridge and recipes:
**nine cases, seven pass, two skip, zero failures/errors**. PyTorch is not installed
locally, so Torch CPU/CUDA tests skip rather than emulate that framework. NumPy
conversion uses actual installed NumPy. Cancellation callback failures/interrupts
complete C cleanup; overflow before ReLU, shapes/budgets, retained graph bounds,
existing program import, copy ownership and real recipes are checked. An initial
recipe fixture incorrectly expected ReLU(-1+2) to be zero; its expected value was
corrected to one before the passing run. Installer checks still pass (**two cases**).

CI [37705354983](https://github.com/sabbirimon/HyperL/actions/runs/37705354983)
passes all Windows, Ubuntu and macOS jobs for source `2d58892`. It uses pinned NumPy
2.3.3 across all three OS jobs and PyTorch 2.9.1 CPU on Linux. Actual Python native
and tensor tests report **eight pass, one unavailable CUDA skip on Linux** and
**seven pass, two uninstalled PyTorch/CUDA skips on Windows/macOS**, with no failures.
The C contract case and JVM/distribution checks pass on each OS. The subsequent
darker, wider and glass-inspired UI pass requires its own source checks and renders.
That final local pass succeeds in **29s**, retaining **33 JVM cases: 32 pass, one
unavailable OpenCL GPU skip, zero failures/errors**. ZIP/TAR and installed distribution
build successfully. Actual CPU-result renders are checked at 1280x820 and 1000x700,
including Source emission and GPU setup tabs at the smaller size; the wider code
pane, reduced spacing, darker surfaces and restrained glass chrome remain readable.
No actual CUDA GPU, vendor-specific
Intel/AMD/Ascend libraries, phone package, enterprise service or full SDK is qualified.
The source archive includes Python preview and native source, not precompiled
universal native libraries or vendor runtimes. Enterprise/mobile/performance/SDK
interoperability documents record plans; they are not successful deployments.

## Thin-strip workbench and Metal host preview — alpha.4

The final local JVM command succeeds in **38s**: **38 cases, 36 pass, two GPU
execution skips, zero failures/errors**. OpenCL and Metal actual GPU tests skip
because neither API sees a device in this session. The Metal bridge compiles
with AppleClang 17; CTest reports **two passes** (C CPU contract and Metal discovery).
An explicit fault-injection host leaves a submission marker; the real frontend
rejects it and blocks further Metal requests when completion is uncertain. This
fixture never invokes a GPU or fabricates successful hardware output. One new test
initially had a non-void JUnit signature; it was corrected before this passing run.

Python installer checks: **two pass**. Actual shared C/NumPy/recipe checks: **nine
cases, seven pass, two uninstalled PyTorch/CUDA skips**, zero failures/errors.
The earlier Linux CI Torch CPU result remains historical until alpha.4 CI completes.
The source release includes an original Objective-C++ Metal host and a bounded
exact-device Terminal harness; compiled vendor runtimes/drivers are not bundled.

The owner identifies A1990; macOS lists Radeon Pro 560X and Intel UHD 630 with
Metal support. Actual Metal and OpenCL process probes expose no GPU; the cause is
not established. The local Radeon harness returns **unrun (exit 3)** with no GPU
cases executed, no fallback and no performance claim. The owner authorized this
Mac's GPU test; phone/two-phone hardware acceptance stays paused. See
[Metal guide](MACOS_METAL.md) for a concrete normal-Terminal qualification command.

The actual Swing CPU panel renders correctly at 1280x820 and 1000x700 with thin
header/status strips, retaining approximately 75 additional pixels of editor
height versus alpha.3. Headless rendering does not qualify native GUI interaction.
The [platform overview](SOFTWARE_STACK.md), [AMD matrix](AMD_COMPATIBILITY.md),
[NVIDIA matrix](NVIDIA_COMPATIBILITY.md) and [AI Services plan](AI_SERVICES_PLAN.md)
record implementation status separately from proposed enterprise/model services.

## Post-alpha.4 Metal JSON contract fix — 2026-10-08

The owner's normal Terminal reports the exact Radeon and Intel devices. Its first
qualification attempt fails with `Device or completion mismatch` and no recorded
passed cases. The report is preserved; it is not counted as successful qualification.
The CLI's default JSON serializer omitted the default-valued `backend` and
`cpuVerified` properties. A new wire-format fixture reproduces the missing backend
assertion before the fix (**one failing test**); that fixture uses no physical GPU.
Making those fields required restores explicit verification metadata after the
existing completion and CPU checks. The harness still rejects absent/false flags,
wrong devices and incomplete commands, and retains bounded metadata for diagnosis.

After the source fix, `./gradlew --offline --no-daemon --max-workers=1 test installDist
distZip` succeeds in **28s**: **39 JVM cases, 37 pass, two unavailable GPU skips,
zero failures/errors**. Native build/CTest: **two passes**. Python script checks:
**six passes**, including four metadata-contract cases; these are fixtures, not GPU
qualification. No native kernel, numerical tolerance or device fallback was changed.

An initial Gradle invocation could not create its local socket under the current
sandbox (`Operation not permitted`); it ran after the specific network permission
was granted. This permission does not demonstrate GPU access. The Radeon
Terminal rerun was pending at this build; its subsequent result is recorded below.
This fix's new OS CI results were also pending at this build. Published alpha.4
archives are unchanged; the corrected working-source distribution must be rebuilt.

The subsequent [OS CI run 37714848101](https://github.com/sabbirimon/HyperL/actions/runs/37714848101)
passes all Windows, Ubuntu and macOS jobs at source `37a8081`, including the JSON
regression and script contract checks. CI runner tests remain separate from the
owner's physical Radeon execution below.

## Owner-Terminal Radeon Metal execution — 2026-10-08

The saved retry report records **`passed-listed-cases` on AMD Radeon Pro 560X** at
07:54:17 Bangladesh time. Inspection confirms four passed GPU cases (3, 257 and
65,536-element multiply/ReLU; 5-element multiply/add/ReLU), exact Radeon metadata,
managed storage and confirmed completion. The strict harness verifies CPU results.
Overflow before ReLU is the fifth case: expected rejection with no GPU dispatch.
The [hardware record](A1990_METAL_VALIDATION.md) preserves the report's hash,
provenance and observed timings. The initial failed report remains unchanged.

At 65,536 elements, the report observes approximately **0.094 ms GPU interval**,
**2.160 ms submit/wait** and **462.787 ms full CLI wall time**. Startup, IO,
compilation, transfers and CPU verification are included in the CLI measurement;
one run does not establish throughput or a CPU speedup. The restricted agent/JVM
test context's two GPU skips remain separate from this owner's hardware run.
This qualifies these workloads on this exact Radeon, not all AMD devices, Intel
UHD 630 execution, full models, ROCm/HIP or cluster capacity. Phone tests stay paused.
