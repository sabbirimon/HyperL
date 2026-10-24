# HyperL installation, usage and programming guide

For the detailed alpha.6 tutorial, Python/C examples and diagrams, read
[Programming with HyperL](PROGRAMMING_GUIDE.md). See also
[build/test](BUILD_AND_TEST.md), [troubleshooting](TROUBLESHOOTING.md),
[progress](PROGRESS_2026_10_24.md) and [all documentation](README.md).

Download/install examples target the published Apache-licensed alpha.5 release.
Sections explicitly marked alpha.6 describe unreleased development additions,
with [new community/enterprise rights](LICENSING.md); [earlier grants](LICENSE_HISTORY.md)
remain unchanged. Existing `hyperl/1` graph semantics and ABI-1 structures are retained.

For the separate ARM64 Android Vulkan qualification runner and the actual
Samsung/Adreno 506 results, see [Android GPU build/test](ANDROID_VULKAN.md).
This experiment is not a mobile installer or Meshlit app GPU backend.

For `0.1.0-alpha.5`, language `hyperl/1`, native CPU ABI 1. HyperL is standalone;
you do not need Meshlit, Android Studio, a phone or a cloud account to use its
desktop CLI/GUI and portable C SDK. [Platform matrix](PLATFORMS.md) records what
is tested. This alpha is a developer experiment, not a CUDA-compatible SDK or a
complete AI/telecom runtime. [Architecture/roadmap](ARCHITECTURE_AND_ROADMAP.md)
describes the expansion and performance gates.

## 1. Choose an installation

For the bundled-runtime macOS DMG, Windows MSI and Linux DEB/RPM previews,
see [native desktop installation](DESKTOP_INSTALLERS.md). The steps below are
for the alternative portable archives. [Use cases and twelve recipes](USE_CASES.md)
provide runnable development workflows.

The desktop release ZIP/TAR contains `bin/`, `lib/`, examples, documentation,
install scripts and portable C SDK source under `sdk/`. `bin/hyperl` is the Unix
launcher; `bin/hyperl.bat` is the Windows launcher. Both run the same CLI and GUI.
Install a maintained Java **17 or later** runtime separately. The local build uses
Java 21. The portable archives bundle no JRE, GPU driver, model, Android/iOS app or telecom stack.

The C SDK needs no JVM at runtime. It needs CMake 3.20+ and a C99 compiler to build.
Optional OpenCL execution needs existing reviewed headers/runtime/driver and a
visible GPU. Optional installer scripts need Python 3; manual extraction does not.

### Download and verify

Get a versioned archive and `SHA256SUMS` from
[HyperL Releases](https://github.com/sabbirimon/HyperL/releases). Use the exact hash
for that artifact. Keep keys and datasets outside installation directories.

Linux:

```sh
java -version
sha256sum hyperl-0.1.0-alpha.5.zip
unzip hyperl-0.1.0-alpha.5.zip
cd hyperl-0.1.0-alpha.5
bin/hyperl capabilities
bin/hyperl gui
```

macOS:

```sh
java -version
shasum -a 256 hyperl-0.1.0-alpha.5.zip
unzip hyperl-0.1.0-alpha.5.zip
cd hyperl-0.1.0-alpha.5
bin/hyperl run examples/elementwise.json examples/inputs.json
bin/hyperl gui
```

Windows PowerShell:

```powershell
java -version
Get-FileHash .\hyperl-0.1.0-alpha.5.zip -Algorithm SHA256
Expand-Archive .\hyperl-0.1.0-alpha.5.zip -DestinationPath .\hyperl-download
Set-Location .\hyperl-download\hyperl-0.1.0-alpha.5
.\bin\hyperl.bat capabilities
.\bin\hyperl.bat gui
```

Compare the digest with the release checksum before extraction. A checksum detects
changes relative to the chosen digest; it is not a signing/certification mechanism.
If the extracted Unix launcher lacks an executable bit, run `chmod +x bin/hyperl`.
If multiple JREs exist, select the maintained runtime using its normal `JAVA_HOME`
configuration. No global PATH change is required; the full launcher path works.

### Optional user-local installer

After inspecting the downloaded script and verifying its source, pass the archive,
a **new** selected installation prefix and exact SHA-256:

```sh
python3 scripts/install.py /downloads/hyperl-0.1.0-alpha.5.zip /your/tools/hyperl-0.1 'ACTUAL_64_HEX_SHA256'
```

PowerShell equivalent:

```powershell
python .\scripts\install.py C:\Downloads\hyperl-0.1.0-alpha.5.zip C:\YourTools\HyperL 'ACTUAL_64_HEX_SHA256'
```

The placeholder is not a real checksum. The installer checks hashes, paths, special
entries, collisions and archive bounds. It creates only the chosen new prefix and
does not download dependencies, require admin/root, alter global PATH or overwrite
an existing installation. Stage cleanup runs on failure. Choose a parent you own.

Uninstall by removing only that chosen prefix after closing its programs. Keep
keys, datasets, source projects and models separately. Updates use a new prefix
and verified version; retain the previous prefix until the new version works.

### Build from source

```sh
git clone https://github.com/sabbirimon/HyperL.git
cd HyperL
git checkout v0.1.0-alpha.5
./gradlew --no-daemon test installDist distZip distTar
build/install/hyperl/bin/hyperl capabilities
```

On Windows, use `.\gradlew.bat` and
`.\build\install\hyperl\bin\hyperl.bat`. Gradle resolves pinned dependencies on
the first build. `--offline` works only after required artifacts are cached.
Archive outputs are under `build/distributions/`. Tests include reference semantics,
GUI panel actions, storage integrity and transport rejection; optional hardware
tests explicitly skip when a compiler/bridge/device is unavailable.

## 2. CLI commands

Run `hyperl help` for the installed command list. Replace `hyperl` below with your
launcher path if you have not added a user-local PATH entry.

| Command | Effect |
|---|---|
| `memory-plan PROGRAM INPUTS [BUDGET_BYTES]` | Observed heap/environment, full graph array estimate and admission decision; see [memory policy](MEMORY.md) |
| `library` / `recipe ID` / `recipe-run ID INPUTS [BUDGET]` | Inspect twelve recipes, export a workspace or execute on CPU; see [use cases](USE_CASES.md) |
| `capabilities` | Actual OS/architecture/JRE and implemented tool status; does not invent accelerators |
| `validate PROGRAM.json` | Checks `hyperl/1`, names, dependencies and operations |
| `run PROGRAM.json INPUTS.json` | Executes finite f32 vectors on CPU reference |
| `emit TARGET PROGRAM.json` | Prints generated backend source; does not compile/load it |
| `gui` | Opens desktop workbench; requires a display |
| `gpu-probe EXECUTABLE` | Runs the explicitly supplied local OpenCL bridge's bounded device probe |
| `gpu-run EXECUTABLE INDEX PROGRAM.json INPUTS.json` | Executes a supported elementwise GPU request and verifies CPU-reference agreement |
| `keygen KEYFILE` | Creates a new random 32-byte AES key file; never overwrites |
| `data-import SOURCE DATASET KEYFILE MAX_BYTES` | Encrypts a local file as bounded chunks and authenticated manifest |
| `data-export DATASET DESTINATION KEYFILE MAX_BYTES` | Verifies/decrypts chunks into a new output, published after full integrity checks |
| `sum-precise INPUTS.json [MEMORY_BUDGET_BYTES]` (alpha.6) | Separate compensated CPU reduction of exactly one `x` vector, rounded once to finite f32 |
| `data-rekey DATASET NEW_DATASET OLD_KEYFILE NEW_KEYFILE MAX_BYTES` (alpha.6) | Verifies the entire source and writes HLM2 under a different key without an intermediate plaintext export |
| `cluster-plan PROFILE.json` | Validates IPv6/capacity configuration; does not enroll/dispatch nodes |
| `shard-plan REQUEST.json [OBSERVATION_TIME_MS]` | Estimates bounded elementwise partitions from supplied observations; no discovery, reservation or execution |
| `worker-token NEW_PRIVATE_TOKEN_FILE` | Creates a new private worker token without overwriting |
| `worker WORKER_CONFIG.json` | Starts an explicitly configured bounded compute listener; non-loopback requires TLS |
| `distributed-plan CONFIG.json PROGRAM.json INPUTS.json` | Contacts approved workers and computes bounded placements without running kernels |
| `distributed-run CONFIG.json PROGRAM.json INPUTS.json` | Executes actual split/map/gather through configured CPU or checked OpenCL/Metal workers |
| `node-probe ORIGIN TOKEN_FILE` | Authenticated read-only host metadata observation, no inference qualification |
| `telecom-plan PROFILE.json` | Validates a research profile, reports stack unavailable/uncertified |

Usage/input/backend failures exit with status 2 and a bounded error. Supported
operation output is actual computation. No command silently falls back to a CPU
when a requested GPU is unavailable. No network action occurs for math/emission/
dataset commands or `shard-plan`. `node-probe` and the separate distributed
commands explicitly contact configured hosts; `worker` opens its configured listener.

## 3. Write a HyperL program

Programs are declarative JSON, not shell/CUDA source or arbitrary Python. Each
instruction creates a new immutable value. Save this as `preprocess.json`:

```json
{
  "format": "hyperl/1",
  "inputs": ["x", "w"],
  "instructions": [
    {"output": "product", "operation": "multiply", "inputs": ["x", "w"]},
    {"output": "positive", "operation": "relu", "inputs": ["product"]}
  ],
  "output": "positive"
}
```

Save `values.json`:

```json
{"x": [-1, 2, 3], "w": [2, 3, 4]}
```

Run:

```sh
hyperl validate preprocess.json
hyperl run preprocess.json values.json
```

The result is `[0, 6, 12]`: multiply element by element, then replace negative
values with zero. Input arrays are copied for reference execution. `add` similarly
adds corresponding elements. There is **no implicit broadcasting**: add/multiply
inputs must have equal length.

To reduce to `[18]`, add one CPU instruction and select its output:

```json
{"output": "total", "operation": "sum", "inputs": ["positive"]}
```

Place it at the end of `instructions`, with a comma after the preceding item, and
change the program's `output` to `total`. Sum follows input order with f32 addition;
it is not a parallel/native GPU reduction. Source emitters reject reductions.

### Format and numerical rules

- Exactly `format: "hyperl/1"`; unsupported versions fail.
- 1–8 inputs, 1–64 instructions. Names match `[A-Za-z][A-Za-z0-9_]{0,31}`.
- Inputs to a step must already exist; outputs cannot redefine values.
- `add`/`multiply` take two inputs; `relu`/`sum` take one.
- Input JSON keys exactly match program inputs. Values are finite numeric f32
  arrays, 1–262,144 elements each; quoted numbers, NaN/Infinity and empty vectors fail.
- Retained vector budget: 1,048,576 float elements across input/intermediate arrays.
  This is approximately 4 MiB of vector payload, not a process-RSS guarantee.
  Parsing, output copies, runtime and crypto have separate memory costs.
- Nonfinite arithmetic results fail. Emitted fusion exposes an intermediate
  overflow marker even when a later ReLU would hide it; native callers must reject
  the entire result if any marker/nonfinite value exists.
- Work checks cancellation at instructions and every 1,024 reference elements.
  No training/autograd, tensors/attention, pointers, files, network or crypto keys
  are expressed in the language yet.

## 4. Use the GUI

Launch `hyperl gui`. The left editor contains program JSON above input JSON; the
right panel shows actual results/source/errors. Select `CPU_REFERENCE`, edit the
two inputs and click **Run**. Select a source target and click **Emit source**.

For OpenCL, type an absolute path to your installed bridge, choose GPU index from
**Probe GPU**, then select `OPENCL_GPU` and **Run**. Empty/unavailable executable,
missing GPU, malformed inputs or mismatched output fail visibly. Only one managed
workbench task runs at a time; Run/Emit/Probe disable while it is active.

**Stop** requests coroutine/native-process cancellation and waits for local cleanup.
Closing the window cancels managed jobs. Process termination does not prove a
driver interrupted an accepted GPU kernel. This GUI does not control an entire
remote cluster or independently launched CLI processes; those have their own
controls. Dataset and node commands currently use the CLI.

The repository's GUI test operates actual Swing panel buttons headlessly. That
does not prove native window interaction on every OS.

The thin header contains **Theme**, **Focus** and **Glass**. Graphite and Aurora
provide darker surfaces; Paper provides a light appearance. Focus hides the environment
row and output panel to expand the editor; toggling it restores the previous divider.
Theme changes keep edited code, input data and execution output. Theme/glass settings
are saved in `~/.hyperl/ui.properties`; they are display preferences, not runtime or
network permissions. `-Dhyperl.ui.settings=/path/to/ui.properties` overrides this file
for a test or isolated profile. Glass is a painted highlight, not OS backdrop blur.

## 5. Source emitters and accelerators

```sh
hyperl emit LLVM_CPU preprocess.json > kernel.c
hyperl emit CUDA preprocess.json > kernel.cu
hyperl emit ROCM_HIP preprocess.json > kernel.hip.cpp
hyperl emit OPENCL_SPIRV preprocess.json > kernel.cl
hyperl emit METAL preprocess.json > kernel.metal
hyperl emit VULKAN_SPIRV preprocess.json > kernel.comp
```

`LLVM_CPU` currently means C99 source, not LLVM IR. `OPENCL_SPIRV` emits OpenCL C,
and `VULKAN_SPIRV` emits GLSL compute source, not SPIR-V binaries. Emission never
installs a compiler/driver, launches a kernel or proves device qualification.
Input arguments/buffer bindings sort names: `w` precedes `x` in this example.
Output and count bindings follow inputs. Native consumers must validate shapes,
finite inputs, bounds, aliasing, buffer ownership and all returned values.

### Optional OpenCL GPU bridge

From a source checkout, with existing OpenCL support:

```sh
native/build.sh
hyperl gpu-probe /absolute/path/HyperL/native/hyperl-opencl
hyperl gpu-run /absolute/path/HyperL/native/hyperl-opencl 0 preprocess.json values.json
```

From a binary release, the source is under `sdk/native/`; build there instead.
Linux needs reviewed OpenCL headers/library/pkg-config already installed. macOS
links its system framework where present. Windows needs a separately reviewed
OpenCL header/import-library build; the shell script does not install it.

The bridge enumerates **GPU** devices only and runs generated elementwise kernels.
It verifies every request against CPU output within the stated numerical tolerance,
so this mode includes verification cost. It has no sum/tensor/model dispatch.
Host work uses a 20 s deadline, bounded files and private temporary directories.
No JIT-compiled source from a web page is automatically accepted by this frontend.

This local Mac's probe reported no OpenCL GPU; successful bridge compilation is
not GPU execution proof. Apple mobile GPUs need Metal, Android GPUs generally
need qualified Vulkan/OEM paths, and NPUs require separate providers. See PLATFORMS.md.

### Optional macOS Metal host

Metal supports compatible Intel Macs (including AMD/Intel GPUs) and Apple silicon.
HyperL now provides an original source host preview with explicit device-name
selection, managed/shared buffer handling and mandatory CPU result verification.
Four bounded cases now pass on Radeon Pro 560X in the owner's Terminal, with CPU
comparison, confirmed completion and expected overflow rejection. See the
[hardware record](A1990_METAL_VALIDATION.md) and
[Metal build/CLI/GUI and Radeon Terminal test guide](MACOS_METAL.md). Use corrected
source after alpha.4; the published alpha.4 archives precede the JSON contract fix.
Source emission, hardware discovery and CPU checks do not establish GPU execution.

## 6. Embed the portable C SDK

This is the experimental C API foundation. The owner requested the full HyperL SDK
later, after standalone runtime qualification; see the dedicated roadmap milestone.

Build/install into your chosen prefix:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
cmake --install build/native --prefix /your/tools/hyperl-sdk
```

For a release, run CMake from its `sdk/` directory. The public header declares
`HYPERL_ABI_VERSION`, status/operation enums and `hl_execute`. Example:

```c
#include <hyperl.h>
#include <stdio.h>
int main(void) {
    float x[] = {-1, 2, 3}, w[] = {2, 3, 4}, result[3];
    struct hl_vector inputs[] = {{x, 3}, {w, 3}};
    struct hl_step steps[] = {{HL_MULTIPLY, 0, 1}, {HL_RELU, 2, 0}};
    size_t written = 0;
    enum hl_status status = hl_execute(inputs, 2, steps, 2, 3,
        result, 3, &written, NULL, NULL);
    if (status != HL_OK) return 1;
    printf("%g %g %g\n", result[0], result[1], result[2]);
    return 0;
}
```

Values 0 and 1 refer to inputs; each step creates the next value, 2 then 3. This
API uses validated numeric indices rather than JSON names. Link the installed
`hyperl_cpu` static library and include path using your platform's compiler. Caller
output must have enough capacity and must not overlap input buffers. Caller owns
output/cancellation context; the library owns/frees its intermediates. On failure,
`output_length` is zero and output must be discarded. A cancellation callback can
return nonzero; it must itself remain bounded. This is C CPU execution, not a GPU SDK.

The C ABI can support Swift/C++/Rust/JNI/other bindings. Those bindings and target
packages still need their own tests. Direct/kernel/bare-metal privilege adapters
are unimplemented; linking a C library does not grant hardware access.

## 7. Large data storage

Keep data and keys outside the installation. Generate a new key file:

```sh
hyperl keygen /private/keys/dataset.key
hyperl data-import /data/source.bin /data/dataset /private/keys/dataset.key 10737418240
hyperl data-export /data/dataset /data/restored.bin /private/keys/dataset.key 10737418240
```

The last argument is the explicit operation quota in bytes (10 GiB in this example).
There is no automatic RAM-to-storage spill. Data streams in 4 MiB chunks with
AES-256-GCM and SHA-256; the manifest is authenticated/encrypted too. Existing
destinations are rejected. A temporary output publishes after the entire dataset
verifies, avoiding accepted truncated/tampered exports. Ordinary process failures
clean staging; abrupt OS crashes can leave private staging that needs owner review.

Alpha.5 writes `HLM1`; alpha.6 writes typed `HLM2` records and still reads authentic
HLM1 datasets. Records have at most 512 plaintext bytes, authenticated order and a
final authenticated total/hash footer. HLM2 uses an authenticated key ID and grouped
chunk directories. Metadata decoding does not buffer the whole manifest. Nonces
are freshly random; keys are random 32-byte AES keys, not password-derived keys.
[Format, publication boundaries and key rotation](DATASET_FORMAT.md).

Alpha.6 rotation writes a new dataset and leaves the source/key unchanged:

```sh
hyperl keygen /private/keys/new.key
hyperl data-rekey /data/dataset /data/rotated /private/keys/dataset.key /private/keys/new.key 10737418240
```

Export publishes verified plaintext with atomic no-overwrite hard-link creation
on the selected filesystem. A filesystem without hard links fails explicitly.
Import/rekey reserve a new directory and publish the authenticated manifest last;
this is a commit marker, not an atomic whole-directory rename or fsync guarantee.

Back up the 32-byte key securely; never paste it into prompts, source, JSON programs
or logs. POSIX-created keys/directories receive private permissions; review Windows
ACLs in the chosen private parent. No KMS, distributed key exchange,
secure erasure or storage snapshot guarantee is implemented. The quota upper bound
is 4 TiB, not a tested multi-terabyte performance claim. This is local storage IO;
distributed large-data operators/shuffle/offload are future work.

The JVM API also supports Ed25519 signatures over bounded artifact manifests and
streamed SHA-256 verification. It requires a separately trusted public key and
does not encrypt the artifact or authorize execution. `keygen` creates an AES
dataset key, not an Ed25519 signing key. See [Cryptography](CRYPTOGRAPHY.md) for
the API, metadata-only verification distinction, limits and replay precautions.

## 8. IPv6 nodes and configuration

`examples/cluster-ipv6.json` contains an IPv6 **documentation address**, not a real
node. `cluster-plan` validates settings without connecting. Replace endpoints only
with owner-enrolled hosts. Inventory ceiling can be 1–1,000,000, active worker
ceiling 1–256 and no greater than inventory; agent management defaults off. The
current inline endpoint list is limited to 10,000 and does not implement a scalable
directory, scheduler or demonstrated simultaneous capacity.

To observe an existing schema-1 Meshlit node companion, keep its endpoint token in
a private text file and explicitly request the selected origin:

```sh
hyperl node-probe 'https://[YOUR_ACTUAL_IPV6_ADDRESS]:8443' /private/keys/node-token.txt
```

Use an actual valid IPv6 address, normal trusted HTTPS and a host you approve.
Credentials are read from that file, not CLI literals or URL userinfo. The request
is `GET /node`, read-only, bounded to 16 KiB, with connection/request timeouts,
no redirects and no automatic retry. It records source/time and observed metadata;
it never sets inference-qualified. Numeric `http://[::1]:PORT` and
`http://127.0.0.1:PORT` are allowed for private local fixtures/tunnels. No other
plaintext host is accepted. IPv6 link-local zone IDs are currently rejected.

The existing companion protocol is reused rather than inventing another discovery
format. Published alpha.5 does not start a worker or dispatch kernels. Alpha.6
development adds separate explicit `worker`, `distributed-plan` and
`distributed-run` commands for bounded vector shards. See
[distributed execution](DISTRIBUTED_EXECUTION.md) for adapters, tokens, TLS,
memory ownership and cancellation. Enrollment, remote datasets, independent
device qualification and fleet scale remain later.

### Sharding admission example

The checked-in [sharding request](../examples/sharding-request.json) is a
**synthetic FIXTURE**, with invented observations and a fixed clock. Its device
names do not identify hardware that was tested. Run it with the explicit clock:

```sh
hyperl shard-plan examples/sharding-request.json 50000
```

It describes 131,072-element multiply/ReLU work, two hypothetical CPU workers
with different compute/copy/link speeds, and small permitted working-memory
budgets. These budgets represent assigned capacity, not each device's total RAM.
Both directions of each link and shared source ingress/egress ceilings are
declared. All timestamps are `50000`; omitting the explicit clock evaluates this
historical fixture against current time and should report stale evidence.

The planner returns candidate index ranges, per-domain memory admission,
transfer-inclusive estimates, exclusions and a decision. Its output always states
`executionActivated: false`, `reservationGuaranteed: false` and
`timingsAreEstimates: true`. A sharding candidate is not a speed measurement,
remote execution or a resource lease. A single-device preference can be a valid
outcome when network overhead exceeds the expected compute benefit.

For a real request, the embedding application must collect current observations,
establish identity/owner permission, qualify each exact backend and graph, and
bind benchmark measurements to the graph hash and element count. Observations
must be no older than 30 seconds and cannot be from the future. Setting
`qualified` or `authenticated` in JSON does not perform those checks. Memory
aliases that share physical RAM must use one domain ID. The fixture's canonical
program hash is `119c15f34d7576a08f98064ba8e090f933e2cda0658f1602f59ce6d623273077`;
recompute it with `HyperLShardingPlanner.programSha256` if the graph changes.

This H0 planner admits only independent f32 `add`, `multiply` and `relu` graphs:
2–262,144 elements, up to eight workers, sixteen physical memory domains and
64 directed links, within the language's retained-array limit. It rejects
ordered `sum` rather than changing its numerical order. Authenticated dispatch,
worker admission and checked split/map/gather already exist in the separate
`distributed-plan` / `distributed-run` path, which uses its own configuration and
placement contract. This H0 request/plan is not automatically passed to that
executor. The executor additionally supports a single terminal ordered sum using
exact CPU workers and coordinator reduction; `shard-plan` does not admit it.

The existing distributed path attempts authenticated cancellation; acknowledgment
means requested, not proven termination. Live measured adaptive load balancing,
full LLM/tensor/model sharding, collectives and complex-number math remain future
work. The separate `sum-precise` CPU operation is implemented and does not change
the ordered f32 sharding contract. See [distributed execution](DISTRIBUTED_EXECUTION.md)
and [the staged execution contract](SHARDING_EXECUTION_CONTRACT.md).

## 9. Telecom and extensions

```sh
hyperl telecom-plan examples/telecom-5g.json
```

Output remains research-only, disabled and uncertified, even if profile flags claim
owner approval or a hash. See TELECOM.md for full-stack research layers, standard
editions and private lab milestones. There is no carrier stack, radio transmission,
subscriber-authentication service or ISO/3GPP certification in this release.

To add an execution backend, implement `HyperLBackend`, assign a target/runtime
revision, and explicitly register it in `HyperLRuntime`. Qualification descriptors
do not load code or grant privileges. Preserve version rejection, numerical
contracts, memory/shape bounds, cancellation, real failure states, license notices
and precise OS/ABI/model/driver evidence. Reuse reviewed upstream runtimes through
thin adapters; see UPSTREAM_REUSE.md. Never label an installed SDK as executed AI.

## 10. Troubleshooting and useful applications

| Symptom | Check |
|---|---|
| Java not found / unsupported class version | Install/select maintained Java 17+; inspect `java -version` and JAVA_HOME |
| GUI unavailable on server | Use CLI; Swing needs a desktop display |
| Offline build missing artifact | Run a normal pinned-dependency build once; offline mode cannot fetch uncached packages |
| Invalid DAG/name/version | Run `validate`, check instruction order/unique outputs/format and limits |
| Shape/nonfinite failure | Equal vector lengths for binary operations; finite f32 range including intermediates |
| Backend unavailable / reduction emission failure | Use supported CPU operations; emissions/native backends have distinct coverage |
| No GPU devices | Verify actual vendor/driver/API exposure; do not infer availability from GPU model name |
| OpenCL deadline/failure | Inspect installed bridge/driver and smaller supported inputs; no silent fallback or action replay |
| Dataset decrypt/hash failure | Correct private key and complete unmodified dataset; never accept partially verified output |
| Existing destination / quota | Choose a new destination or explicit supported quota; source/output are not overwritten |
| Node TLS/auth/HTTP failure | Correct approved origin, trusted CA and endpoint-bound token; redirects/plaintext are rejected |

Current use cases: AI vector preprocessing experiments, cross-backend source and
correctness comparisons, C embedding, encrypted local dataset handling and bounded
host capability observations and experimental bounded split/map/gather. Future
tensor/model/vision/DSP/fleet-scale/telecom
use cases need actual backends, measurements and acceptance. HyperL can reduce
duplicate integration work while keeping provenance, device limits and human
controls visible; speed and universal compatibility are outcomes to demonstrate.

## Memory-aware jobs

See [MEMORY.md](MEMORY.md) for the implemented CPU preflight, GUI Memory plan,
optional `run` byte budget, bounded dataset headroom and future HBM/unified
memory/SSD/NVMe allocation interfaces. The full developer SDK remains later.

## Developer workbench and IDE helpers

The workbench provides syntax-highlighted/foldable JSON editors, line numbers,
undo/redo, literal Find next, semantic/shape Validate, Format JSON and two examples.
Open/Save workspace uses bounded `hyperl-workspace/1` documents containing only
program and input JSON; opening never executes them. Overwriting an existing file
requires the visible file choice and confirmation. Export output saves the currently
displayed result/source to an explicitly selected local file. Unsaved editor changes
require confirmation before replacing an example/workspace or closing the window.
Ctrl/Cmd+Enter runs the selected backend; Stop cancels managed local work.

```sh
bin/hyperl workspace-new my-example.hyperl.json
bin/hyperl workspace-validate my-example.hyperl.json
```

[IDE helpers](ide/README.md) include a local JSON schema and optional VS Code
workspace/process task. They require your independently installed IDE/CLI and
manual task invocation; no plugin or language server is silently installed.
[UI design plan](UI_DESIGN_PLAN.md) records the visual system, performance and
accessibility gates, native profiler/cluster workspace and full SDK/IDE roadmap.

## Advanced diagnostics and suggested fixes

Use **Analyze** to inspect syntax, dependencies, shapes, memory and actual CPU
behavior on the supplied vectors. **Review fix** previews one bounded operation/
reference correction and requires explicit selection/confirmation; stale editor
text is rejected. `hyperl diagnose PROGRAM INPUTS [BUDGET_BYTES]` provides the
same structured diagnostics and context suggestions for scripts. See
[CODE_DIAGNOSTICS.md](CODE_DIAGNOSTICS.md) for scope, exit codes and sample-only
evidence. It is a HyperL analyzer, with full language-server/AI tooling staged later.

## Alpha.4 layout and deployment plans

The desktop workbench groups controls into Execution, Source emission and GPU setup
tabs. Select the tab for Run/Stop/budget, target source generation or reviewed bridge
configuration. The footer explains the selected path. Local font resources provide
Inter/JetBrains Mono typography with system fallback and no runtime download.
The editor starts with 70% of the horizontal split, constrained by the output pane's
minimum width. Drag the divider to change the allocation. Compact margins preserve
code space. The header and four status cards now use thin horizontal strips. Toggle **Glass** in the header for solid panel surfaces, or launch with
`JAVA_OPTS=-Dhyperl.ui.glass=false` on Linux/macOS (PowerShell:
`$env:JAVA_OPTS='-Dhyperl.ui.glass=false'`). The subtle glass-inspired tint/sheens are
static cross-platform painting; they do not use Apple's native Liquid Glass or blur
the desktop. Code and output remain opaque in either mode.

[Native performance architecture](NATIVE_PERFORMANCE_ARCHITECTURE.md),
[enterprise services](ENTERPRISE_AND_CLUSTER_PLAN.md),
[single-phone mode](MOBILE_AND_EDGE_PLAN.md) and
[Python-style developer experience](DEVELOPER_EXPERIENCE_PLAN.md) record later
implementation plans. They do not change the current execution/platform contracts.

The alpha.4 [Python/native library preview](LIBRARY_INTEROPERABILITY.md) contains
small reusable recipes and explicit framework bridges. It needs a separately built
reviewed shared C library and selected Python environment; the full SDK remains later.

See the [software stack overview](SOFTWARE_STACK.md) for layers and deployment profiles,
and [NVIDIA coverage](NVIDIA_COMPATIBILITY.md) for architecture/library qualification.
