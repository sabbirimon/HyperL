# HyperL

![HyperL — Bounded CPU and GPU compute experiments](docs/assets/hyperl-cover.png)

HyperL explores a common way to describe, validate and eventually execute AI workloads
across CPUs, GPUs and other accelerators. Its goal is to give developers approachable
authoring tools and explicit memory, device and numerical contracts, while letting
qualified native backends use each platform's capabilities.

[Download Studio](https://github.com/sabbirimon/HyperL/releases/tag/v0.1.0-alpha.6) · [User guide](docs/USER_GUIDE.md) · [Native engine](https://github.com/sabbirimon/HyperL/tree/feature/native-mlir-engine) · [Validation](docs/VALIDATION.md)

## Start with a working example

**Desktop Studio.** The alpha.6 release includes Linux x64, macOS ARM64 and Windows x64 archives. Download the archive for your system and compare its SHA-256 with the published checksums before extracting it. [Studio downloads and checksums](https://github.com/sabbirimon/HyperL/releases/tag/v0.1.0-alpha.6)

**Classic CLI.** To run the vector example below, use the alpha.5 CLI distribution and Java 17 or later. From the extracted directory, run these commands on Linux or macOS. [CLI archives](https://github.com/sabbirimon/HyperL/releases/tag/v0.1.0-alpha.5)

```sh
bin/hyperl capabilities
bin/hyperl run examples/elementwise.json examples/inputs.json
```

On Windows, run `bin\hyperl.bat` in place of `bin/hyperl`. The sample multiplies two vectors and applies ReLU. Its expected output is `[0, 6, 12]`. [Installation and programming guide](docs/USER_GUIDE.md)

## Choose an execution path

| If you want to… | Start here |
| --- | --- |
| Write and check a bounded program | Run the CPU reference, inspect memory admission and compare outputs against a known example. [Programming guide](docs/PROGRAMMING_GUIDE.md) |
| Use HyperL from Python or C | Load the C ABI preview or pass supported arrays through the NumPy and PyTorch bridges. [Library interoperability](docs/LIBRARY_INTEROPERABILITY.md) |
| Work with model weights | The native engine branch provides the Rust CLI, read-only SafeTensors mapping and CPU or selected wgpu projection. [Native engine contracts](https://github.com/sabbirimon/HyperL/blob/feature/native-mlir-engine/docs/NATIVE_ENGINE.md) |
| Split a vector workload | Inspect candidate ranges with `shard-plan`. Use `distributed-run` for authenticated dispatch and checked results. [Distributed execution](docs/DISTRIBUTED_EXECUTION.md) |
| Evaluate a GPU | Check the exact workload, driver, precision and device already exercised. [Platform qualifications](docs/PLATFORMS.md) |
| Protect local data | Stream bounded encrypted chunks and verify artifact signatures using your own trusted keys. [Dataset format](docs/DATASET_FORMAT.md) · [Cryptography](docs/CRYPTOGRAPHY.md) |

## Hardware results

![256 attention-weight rows distributed across Intel UHD 630, AMD Radeon Pro 560X and Adreno 730 GPUs](docs/assets/projection-row-split.png)

Two GPUs in an Intel Mac and one GPU in an Android phone computed different rows of the same Qwen2.5 1.5B attention projection. The Intel UHD 630 handled 86 rows, the Radeon Pro 560X handled 85 and the Adreno 730 handled 85. Each row used 1,536 FP32 inputs. The maximum absolute error against the reference was 1.90 × 10⁻⁶. This experiment measured one projection. End-to-end token generation across these GPUs is still unverified. [Measurement record](docs/evidence/intel-amd-adreno-projection-2026-11-24.json)

In a separate CPU experiment, a Samsung phone and a Mac each computed half of a 32,768-element vector. The complete verified run took a median of 180.842 ms over USB and 235.587 ms over Wi-Fi. The local Android reference finished sooner on this workload. [Reproducer and timings](docs/ANDROID_CROSS_PLATFORM_PROBE.md)

## Build, test and contribute

For the classic workbench on this branch, use the Gradle wrapper. Build and test the C runtime separately with CMake.

```sh
./gradlew --no-daemon test installDist
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
```

On Windows use `gradlew.bat`. For a hardware report, include the source revision, workload, device and driver, expected and observed results, and elapsed time including transfers. Failed runs also help maintainers establish support boundaries. [Build and test](docs/BUILD_AND_TEST.md) · [Report a result](docs/TEST_REPORTING.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)

## About and licensing

Created by **Sabbir Hassan Imon (IMON)**. HyperL is an independent project. Meshlit is a separate client project. [Meshlit](https://github.com/sabbirimon/meshlit) · [Documentation index](docs/README.md)

Newly covered work uses the HyperL Community and Enterprise License 1.0. Personal, developer and research use is free, as is qualifying small-business production. Enterprise production terms include a six-calendar-month trial for groups meeting the revenue, budget or employee threshold. Published releases through alpha.5 and earlier Apache-covered code retain their Apache 2.0 grants. [License](LICENSE) · [Eligibility and examples](docs/LICENSING.md) · [License history](docs/LICENSE_HISTORY.md) · [Third-party notices](NOTICE)
