# HyperL

Original open-source AI kernel language, runtime experiments and developer tools.
Apache-2.0 • Experimental `0.1.0-alpha.2` • Created by IMON with Codex assistance.

HyperL is a standalone project. Meshlit is one potential client. The first release
provides a desktop **CLI and GUI**, a portable **C CPU SDK**, typed JSON programs,
source emitters, memory-aware CPU admission, an optional OpenCL GPU bridge and encrypted streaming datasets.
It does not replace CUDA, implement a complete tensor compiler, or qualify every
OS, architecture, accelerator or telecom network. See [architecture and detailed
build plan](docs/ARCHITECTURE_AND_ROADMAP.md) and [validation](docs/VALIDATION.md).

## Install and run

The [installation, usage and programming guide](docs/USER_GUIDE.md) includes
Windows/Linux/macOS commands, program examples, C embedding and troubleshooting.

Download a versioned ZIP/TAR from [Releases](https://github.com/sabbirimon/HyperL/releases).
Install a maintained Java 17+ runtime separately, unpack, then use `bin/hyperl`
on Linux/macOS or `bin\hyperl.bat` on Windows. The same archive contains both modes:

```sh
bin/hyperl capabilities
bin/hyperl run examples/elementwise.json examples/inputs.json
bin/hyperl memory-plan examples/elementwise.json examples/inputs.json
bin/hyperl gui
bin/hyperl emit METAL examples/elementwise.json
```

The [developer UI plan](docs/UI_DESIGN_PLAN.md) and [IDE helpers](docs/ide/README.md)
cover syntax editing, validation, formatting, examples, workspace files and later
SDK/language-server/profiler integration. [Code diagnostics](docs/CODE_DIAGNOSTICS.md)
adds context suggestions, field-level errors, actual CPU sample checking and
explicitly reviewed operation/reference repairs.

The arithmetic example returns `[0, 6, 12]`. The GUI edits program/input JSON, runs
CPU or an explicitly installed GPU bridge, emits source, probes devices and stops
managed local work. It performs no network connection or automatic installation.
`gui` needs a desktop display; headless systems use the CLI/library.

Optional user-local installation (Python 3): `scripts/install.sh ARCHIVE.zip PREFIX SHA256` or
`scripts/install.ps1 -Archive ARCHIVE.zip -Prefix DIRECTORY -Sha256 SHA256`. Use an empty/new prefix
owned by you; no root/admin access or global PATH modification is required. Verify
the release SHA-256 checksums before extracting. Uninstall that chosen prefix only;
keys, datasets and models stay separate and are not removed automatically.

Build from source with a maintained JDK 17+ (local validation uses JDK 21):

```sh
./gradlew test installDist distZip distTar
build/install/hyperl/bin/hyperl help
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
cmake --install build/native --prefix /your/selected/sdk-prefix
```

Gradle resolves the pinned Kotlin/runtime dependencies. It does not require Android
Studio or an Android SDK. Native SDK users need a C99 compiler/CMake and can use
`native/include/hyperl.h` directly without the JVM. See [platform matrix](docs/PLATFORMS.md).

## Memory-aware execution

The CLI and GUI estimate full CPU array/workspace costs and reject jobs exceeding
the selected budget or observed JVM heap headroom. See [memory policy](docs/MEMORY.md).
HBM/GDDR/unified GPU/NUMA/CXL/SSD/NVMe are explicit future adapter tiers; unknown
capacity stays unknown, and automatic kernel spill is unavailable.

## Device GPUs

An optional original OpenCL 1.2 host bridge selects actual GPU devices and fails
when unavailable. It never substitutes a CPU device or installs a vendor driver.
On macOS or Linux with existing OpenCL headers/runtime:

```sh
native/build.sh
build/install/hyperl/bin/hyperl gpu-probe /absolute/path/native/hyperl-opencl
build/install/hyperl/bin/hyperl gpu-run /absolute/path/native/hyperl-opencl 0 examples/elementwise.json examples/inputs.json
```

The current GPU path verifies each request against CPU reference output and rejects
nonfinite/mismatched results. It is a qualification experiment, not a speed claim.
Only elementwise kernels are supported; reduction dispatch is unavailable. A 20 s
process deadline bounds host work, but killing a process is not proof that a driver
has interrupted an accepted GPU kernel. Use the installed, reviewed bridge only.

CUDA, HIP, OpenCL C, Metal and Vulkan GLSL **source** can be emitted. Metal/Vulkan
source has no runtime loader yet. Apple iOS/macOS GPU adapters and Android
Adreno/Mali/Immortalis/other GPUs need native wrappers and actual device tests;
NPUs such as Qualcomm QNN, Apple Neural Engine or Ascend use separate SDK paths.

## Large datasets and IPv6 clusters

```sh
bin/hyperl keygen /private/location/dataset.key
bin/hyperl data-import /source/big-file /destination/dataset /private/location/dataset.key 10737418240
bin/hyperl data-export /destination/dataset /destination/restored /private/location/dataset.key 10737418240
bin/hyperl cluster-plan examples/cluster-ipv6.json
```

Import/export streams bounded 4 MiB chunks, with standard-library AES-256-GCM,
fresh per-file nonces, authenticated manifest/ordinal/length and SHA-256 checks.
Output publishes only after verification; existing outputs are never overwritten.
Keep the 32-byte key private and outside the dataset; losing it loses access.
The explicit quota can be 1 byte–4 TiB, with at most 1,048,576 chunks. These are
format/operation limits, not demonstrated multi-terabyte performance. RAM/RSS also
includes crypto, JVM and file buffers; it is not fixed at one chunk.

Cluster profiles accept bracketed IPv6 literals, IPv4 and DNS **HTTPS** endpoints.
Inventory is configurable up to 1,000,000, active workers to 256, and agent
management is separately default-off. The plan command validates configuration
only: remote transport, enrollment, sharded inventory and scheduler are future work.
No million-node capacity, IPv6 connectivity or distributed large-data compute is
inferred from a valid profile.

`node-probe ORIGIN TOKEN_FILE` makes a bounded authenticated read-only `/node`
request to the existing Meshlit host companion schema. Remote origins need normal
trusted HTTPS; plaintext is limited to numeric IPv4/IPv6 loopback. Redirects and
automatic retry are disabled. This observes host metadata, not model execution.

## Full telecom research

`telecom-plan examples/telecom-5g.json` records a typed research profile and always
reports execution disabled and uncertified. [Telecom research](docs/TELECOM.md)
covers LTE/5G core/RAN, AI/MEC, future 6G, security and standards evidence. No RF
transmitter, subscriber credential handling, live carrier action or network stack
is activated. Full stack research is a separate milestone from AI compute kernels.

## Licensing and provenance

HyperL started as original Meshlit experimental code; see [provenance](NOTICE).
Upstream vendor SDKs retain their own licenses. No NVIDIA closed binary/source,
Odysseus AGPL implementation, radio stack or driver is bundled. Optional research
and independently installed adapters do not grant redistribution or certification.
Report issues through [GitHub](https://github.com/sabbirimon/HyperL/issues).
