# Build, test and package HyperL

Updated 2026-10-24. Work from the standalone HyperL checkout. This runbook does not
build Meshlit. Maintained JDK 17+ (locally 21), CMake 3.20+, C99 compiler and Python
3.10+ are the base prerequisites. First dependency resolution needs network access.

## Standard validation

```sh
./gradlew --no-daemon test installDist distZip distTar
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
python3 -m unittest discover -s scripts -p 'test_*.py' -v
git diff --check
```

Use `gradlew.bat` on Windows. Record passes, skips and actual hardware separately.
CTest's Metal discovery case is not GPU execution. Quantum SDK doubles are not live
provider acceptance. Optional OpenCL/Metal targets may be unavailable on a CI host.
Do not edit compiled source while a validation run is in flight.

## Focused loops

```sh
./gradlew test --tests '*HyperLHashesTest' --stacktrace
python3 -m unittest discover -s scripts -p 'test_quantum*.py' -v
build/install/hyperl/bin/hyperl run examples/elementwise.json examples/inputs.json
build/install/hyperl/bin/hyperl shard-plan examples/sharding-request.json 50000
```

See [troubleshooting](TROUBLESHOOTING.md) for failure triage. Run changed contracts,
then the standard gate before shipping; repeating unrelated hardware tests does
not improve evidence for a documentation-only change.

## Outputs and platform qualification

| Output | Location / procedure |
|---|---|
| CLI/workbench | `build/install/hyperl/` |
| Portable ZIP/TAR | `build/distributions/` |
| Native libraries | `build/native/`, or configuration subdirectory on Windows |
| Python source | `python/hyperl/`; explicit native loading, no automatic binary download |
| Desktop installers | [Installer workflow](DESKTOP_INSTALLERS.md), per-platform bundled runtime |
| Android CPU probe | `./gradlew -p android/probe assembleDebug lintDebug`; [guide](ANDROID_CROSS_PLATFORM_PROBE.md) |
| Android GPU runner | Separate [Vulkan build/test](ANDROID_VULKAN.md) |

Check free disk space before packaging. Generated outputs, models, private tokens,
keys and raw private device reports do not belong in commits. Preserve prior release
assets; a GitHub push is not a release, installer signature or deployed service.

## Shipping checklist

1. Freeze source and record `git rev-parse HEAD` plus the pending diff.
2. Run required checks and document failure fixes and unavailable targets.
3. Verify doc links, examples, version labels, licensing/provenance and secret exclusion.
4. Commit explicit intended paths and push the working branch without rewriting history.
5. Check CI for that exact commit; old green runs do not qualify new source.
6. Tag/publish only as an explicit release action after its gates. Do not claim alpha.6
   installers published while the release remains alpha.5.

See [progress](PROGRESS_2026_10_24.md), [validation](VALIDATION.md) and
[agent backlog](AGENT_BACKLOG.md) for current state.
