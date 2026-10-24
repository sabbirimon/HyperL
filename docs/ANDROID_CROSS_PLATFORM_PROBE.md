# Android and desktop CPU qualification probe

Prepared 2026-10-24 (Asia/Dhaka). This guide describes the standalone foreground
probe and its reproducible physical-device test. Qualification results are recorded
separately below; building or installing the APK is not an execution result.

The probe runs portable HyperL CPU code on Android and sends a disjoint vector range
to an explicitly provisioned desktop CPU worker. It uses the existing
[bounded shard protocol](DISTRIBUTED_EXECUTION.md), independently of Meshlit and the
new `shard-plan` cost estimator. The test uses a fixed half-and-half split, not the
estimator's placement algorithm.

The dated APK hashes and measurements in this guide are preserved snapshots.
Later negotiation, cryptography or [quantum Python](QUANTUM_EXTENSION_PLAN.md)
source additions do not retroactively become capabilities of that tested APK.

## 1. Exact scope

| Item | Contract |
|---|---|
| Android application | `org.hyperl.probe`, activity `.MainActivity`; API 26 or later, foreground only |
| Android implementation | Six unchanged portable Kotlin source files compiled into the probe: `HyperL`, `HyperLAdmission`, `HyperLCodec`, `HyperLContract`, `HyperLLibrary`, `PreciseReduction` |
| Desktop worker | Explicit `CPU_REFERENCE` worker named `mac-probe`; current runtime/numerical contract checked before each remote shard |
| Workload | `weighted_relu`: elementwise multiply followed by ReLU; 32,768 deterministic f32 elements |
| Android range | `[0, 16384)` |
| Desktop range | `[16384, 32768)` |
| Repetitions | One warm-up, then five measured repetitions; both branch requests are launched concurrently in each repetition |
| Correctness | Android gathers both halves and compares every raw f32 bit with a complete Android CPU reference run |
| Limits | 180-second overall probe timeout, 10-second shard/request deadline, bounded capabilities/metadata and 8 MiB frames |
| Lifecycle | Stop or leaving the foreground cancels local work, disconnects active requests and attempts authenticated remote cancellation |

The inputs for index `i` are `x = (i % 257 - 128) / 8` and
`w = (i % 13 - 6) / 4`. These binary fractions and their products are exactly
representable for this workload. SHA-256 of all 32,768 gathered results encoded as
network-order f32 must be:

```text
00f1b8b9a53d80c11b22c106528c085fef5fcc14ac49b0a7fccc31009b931822
```

The host harness independently computes this expected hash. A repeated hash alone
is insufficient: it also checks the fresh run/job IDs, ranges, worker process and
instance identity, completed-shard delta, sample counts and finite bounded timings.

This is actual bounded vector work if those checks pass. It is not model-layer
sharding, tensor parallelism, GPU/NPU execution, shared memory, automatic resource
pooling or a production cluster scheduler. It does not establish faster execution
than a single device. Concurrent task launch also does not establish the exact
overlap of physical kernel execution across unsynchronized device clocks.

## 2. Build and install on the selected Android device

Prerequisites are JDK 21, an installed Android SDK with the project's selected
compile SDK, an owner-authorized ADB connection and the root desktop distribution.
Current project versions are recorded in [the probe project](../android/probe/README.md)
and its Gradle files. From the repository root:

```sh
./gradlew installDist
./gradlew -p android/probe assembleDebug
```

Set `ANDROID_HOME` or an untracked `android/probe/local.properties` to the installed
SDK. The produced APK is:

```text
android/probe/build/outputs/apk/debug/hyperl-android-probe-debug.apk
```

Select the intended phone explicitly. Replace `PHONE_SERIAL` below with its ADB
serial; do not use an unqualified install command when another phone is connected.

```sh
"$ANDROID_HOME/platform-tools/adb" devices -l
"$ANDROID_HOME/platform-tools/adb" -s PHONE_SERIAL install -r android/probe/build/outputs/apk/debug/hyperl-android-probe-debug.apk
"$ANDROID_HOME/platform-tools/adb" -s PHONE_SERIAL shell am start -n org.hyperl.probe/.MainActivity
```

`install -r` updates this separate probe without uninstalling or clearing Meshlit
or another app. The debug APK is for controlled qualification and is not a signed
production/store release. Record the actual APK hash, source revision and desktop
runtime revision with the reports used for a qualification claim.

Press **Run local CPU checks** first if desired. It checks the sample weighted-ReLU
result `[0, 6, 12]` and compensated-sum result `1`. This local report has no remote
shards and cannot establish two-device execution. No test begins on app launch.

## 3. Provision a temporary desktop worker

Use [the owner-controlled harness](../scripts/qualify_android_probe.py). It does
not install an APK or tap the phone. Replace `PHONE_SERIAL` and `PRIVATE_IPV4`
with the selected phone and this desktop's actual private LAN address. Use a new
output directory for each qualification attempt so a retry preserves prior results.

```sh
python3 scripts/qualify_android_probe.py \
  --cli "$PWD/build/install/hyperl/bin/hyperl" \
  --adb "$ANDROID_HOME/platform-tools/adb" \
  --serial PHONE_SERIAL \
  --lan-address PRIVATE_IPV4 \
  --output /tmp/hyperl-android-probe-run
```

The harness creates a temporary private directory, random worker token and keystore
password, and a two-day TLS certificate with SANs for `127.0.0.1` and the selected
LAN address. It starts one CPU worker with a 16,384-element shard maximum, 64 MiB
configured admission budget, one job slot and a 256 MiB JVM heap ceiling. The worker
listens on an automatically selected port, using TLS on `0.0.0.0`; only an approved
test network should reach it. The harness does not change OS firewalls or trust.

The debug ADB owner path installs the token, public certificate and per-run connection
configuration into the probe's private files. The private signing key stays on the
desktop. Android creates a private trust store for that public certificate and keeps
ordinary HTTPS hostname verification. No global Android certificate is installed.
Each split run requires a canonical fresh UUID `runId`; missing/invalid IDs fail
before execution. The harness generates and correlates this ID automatically.
The app atomically consumes each prepared ID once, including failed or cancelled
attempts. Repeated Start presses show a session-used message without overwriting
the prior report. Configure `wired` or `wireless` again before a retry. The bounded
private session ledger holds up to 256 IDs and fails closed when full.

The harness refuses to replace an existing ADB reverse mapping on phone port 18091.
Keep other tasks' mappings intact. Its `ready` response lists four commands:
`wired`, `wireless`, `collect` and `stop`.

## 4. Run the two transports separately

### Wired ADB reverse

1. Keep the phone connected over the selected USB/ADB session and leave the probe
   in the foreground.
2. Enter `wired` in the harness. It requires `adb get-devpath` to report an actual
   `usb:` device path, then prepares a fresh run ID and the origin
   `https://127.0.0.1:18091`. Its ADB reverse mapping forwards that phone port to
   the temporary desktop TLS worker.
3. Press **Run Samsung + Mac split** on the phone. Wait for the app's terminal
   **Passed**, **Failed** or **Stopped** result.
4. Enter `collect` in the harness. It verifies the same USB device path still exists
   and requires `networkObservation.boundToObservedWifi=false`. On a validated pass it writes
   `wired-adb-reverse.json` to the chosen output directory.

This route includes Android networking, TLS, ADB forwarding, USB and host overhead.
It is not a measurement of raw USB bandwidth. A loopback address on the phone does
not mean the remote shard executes on the phone; the report is correlated with the
actual desktop process and completed-shard counter.
ADB reverse alone is not USB evidence because wireless ADB can also forward ports;
the observed USB path before and after the run is a separate required check.

### Direct Wi-Fi LAN with TLS

1. Connect the selected phone and desktop to a mutually reachable approved LAN.
   The phone's active network must be Wi-Fi without a VPN. The app does not change
   Wi-Fi credentials, VPNs or system settings.
2. Enter `wireless` in the same harness. It selects the desktop's numeric private
   LAN HTTPS address and actual worker port, using a new run ID.
3. Press **Run Samsung + Mac split**, wait for completion, then enter `collect`.
   A validated pass writes `wireless-lan.json`.

For this mode, the app observes `TRANSPORT_WIFI`, rejects a VPN active network and
binds its connection to that Android network. The report records
`boundToObservedWifi` and the interface name. The request uses the desktop LAN
address directly; the still-present reverse mapping is not its destination. The
certificate must match the LAN address. A transport label without these observations
and host correlation is not physical-route evidence.
The host validator requires `boundToObservedWifi=true` for a wireless pass and
rejects reports whose network observation contradicts the requested mode.

The probe's optional `loopback-plaintext` developer mode accepts only the specific
numeric loopback HTTP origin. The supplied harness uses TLS for both measured
transports; plaintext results must never be combined with or labeled as TLS results.
The client supports additional private/ULA numeric addresses, but this harness's
LAN certificate/provisioning path is IPv4; this test does not qualify IPv6.

## 5. Read the measurements correctly

The app writes reports atomically with format `hyperl-android-probe/1`. Each split
has a private `files/reports/<runId>.json` file; the app updates that file as its
own run progresses. The consumed-session ledger prevents a later attempt from
reusing that ID. `files/probe-result.json` is a convenience copy of the latest
local or split result. The harness collects the exact prepared UUID's report, so
a subsequent local check cannot replace the evidence it is waiting to collect.
Rows 0–5 contain one warm-up and five measured repetitions.
The harness report uses `hyperl-android-desktop-qualification/1`, adds host OS and
architecture, correlates the worker identity/counters and sets `crossDeviceVerified`
only after its independent pass checks.

| Field | What it measures |
|---|---|
| `localComputeMs` | Android branch interval, including its input slicing and CPU execution |
| `remoteComputeMs` | Desktop interval around admitted `compute.execute`, including its coroutine/timeout wrapper; excludes request validation/decode, response encoding and transfer |
| `remoteRoundTripMs` | Android-observed execute POST helper, including its request hashing/upload and response reads; excludes preceding capabilities GET, program/frame construction and subsequent response digest/decode |
| `splitMs` | Both branch orchestration intervals through receipt of complete results; includes capability/request work and waits for the slower branch |
| `fullReferenceMs` | Complete same-workload CPU reference execution on Android after the split |
| `comparisonAndHashMs` | Gather, exact f32 comparison, result-byte encoding and SHA-256 interval |
| `endToEndVerifiedMs` | Split plus complete reference, gather/comparison and result hashing, ending when the row is produced |
| `sentPayloadBytes` / `receivedPayloadBytes` | Application shard-frame sizes; exclude HTTP/TLS/USB/Wi-Fi overhead and capability-query traffic |
| `remoteShards` | Six remote shards on full success, including warm-up |
| `measuredRemoteShards` | Five remote shards after excluding warm-up |

The initial source sampled `comparisonAndHashMs` before calling the digest. Before
physical split qualification, the timer was corrected to sample after SHA-256.
Preserve that distinction if interpreting an earlier APK's report. Do not reinterpret
partial stage timings as complete user-perceived completion time.

Summaries use the five measured rows only: median is the third sorted value and
nearest-rank p95 is the fifth value, which is also the maximum. Five observations
do not establish a stable production tail-latency distribution. Keep the individual
samples, failed attempts, cold/warm distinction and exact transport configuration.
Do not derive one-way network delay by subtracting worker time across different
clocks, or call payload/time a sustained link-capacity benchmark.

The probe records Android ABI/process architecture, runtime and heap/system memory;
the desktop worker protocol does not carry OS/architecture, so the harness supplies
host observations. Device names such as `mac-probe` are not architecture evidence.
Memory figures are observed/admitted CPU values, not pooled RAM/VRAM reservations.
No power, thermal, energy or reward valuation is measured by this test.

## 6. Failures, Stop and cleanup

Any wrong token/certificate/hostname, incompatible worker/runtime, malformed or
oversized frame, wrong digest/range/job identity, nonfinite value, deadline or
bitwise mismatch must leave a failed/cancelled report. The client does not redirect,
retry another worker or silently substitute local execution for the remote half.
The harness cannot publish `crossDeviceVerified=true` from a local-only or stale run.

Press **Stop** or leave the foreground to cancel. Local cancellation disconnects
active HTTP work and attempts a bounded authenticated cancellation request. HTTP
`202` means cancellation was requested; it does not confirm remote termination.
Remote completion can remain uncertain after a link failure. A cancellation test
is a separate acceptance result from successful split computation.

Enter `stop` in the harness when finished. Its cleanup stops the exact temporary
worker, removes its test credentials/configuration from the probe and removes its
owned phone-port-18091 reverse mapping. The temporary desktop key directory is
deleted on exit; evidence reports remain in the chosen output directory. If the
phone disconnects, host shutdown still proceeds and the harness reports that phone
cleanup needs a reconnect. The probe remains installed and its last result remains
available; cleanup does not erase unrelated apps or tests.

## 7. Qualification record

### Rejected attempts and corrections

The first controlled wired attempt was **rejected by the host evidence validator**.
Two completed phone runs reused one prepared `runId`; their report timestamps were
different, and the host completed-shard delta did not equal the required six. The
phone's exact arithmetic results do not override this session-correlation failure.
No timings from that attempt are published as qualified transport measurements.

The harness now preserves a rejected attempt with `crossDeviceVerified=false`, the
observed shard delta and a bounded `qualificationError` instead of discarding its
diagnostics when verification throws. The app now atomically consumes each run ID
once and preserves prior reports when setup is missing or a session is already used.
The session-guard APK assembled and passed lint in seven seconds; a physical
missing-configuration check verified that the existing report was preserved.
The duplicate-session/setup UI checks were performed on that 1,830,099-byte APK:

```text
12529a00100e2f51cec7c6a7d2802347a31444746930188f3151079d7fc5af64
```

That APK subsequently passed controlled wired qualification. A later Wi-Fi
collection attempt failed its run-ID check after a local CPU
check replaced the single latest-report file. No Wi-Fi timings from that attempt
are qualified. The app now retains each split report under its consumed UUID, and
the harness collects that exact session path. The final APK retains the session
guard code unchanged; the guard UI checks above were not repeated on the final
APK. Its separate report-retention check compared the saved wired split report
byte-for-byte after a later local check and passed.

At this historical qualification checkpoint, the harness verification suite had
four cases and the full Python script suite had eighteen passing cases. Those
counts remain part of the recorded checkpoint; see [current validation](VALIDATION.md)
for later source/test additions. Fixture tests remain separate from phone evidence.

### Final APK and local CPU check — 2026-10-24

The final 1,830,571-byte debug APK passed assembly/lint in seven seconds and has
SHA-256:

```text
7e850367855b8bd5d3479768b80ec5f2454b7e7395cfa89de4fa4ec8d2aa0c56
```

The selected Samsung SM-A207F, Android API 30, `aarch64`/64-bit process passed the
local CPU check using this APK: weighted ReLU `[0, 6, 12]`, compensated sum `1`,
zero remote shards and a recorded combined check interval of **3.048073 ms**.
This is one local correctness check, not a repeated performance benchmark.
An earlier metric-corrected APK also passed locally in 3.97901 ms; the timings
below use only the final APK's physical split reports.

### Final physical wired and Wi-Fi TLS results

Both transports pass independent host qualification using the final APK and the
same Darwin `x86_64` desktop worker process/instance. For each mode, the host records
exactly six completed remote shards, including warm-up, with distinct job IDs.
Each device computes its own 16,384-element range. All twelve gathered results
match the independent expected output hash. Wired qualification verifies the same
USB path before/after and `boundToObservedWifi=false`. Direct LAN qualification
verifies `boundToObservedWifi=true` on `wlan0`; it does not use ADB reverse as its
compute route. Harness shutdown and test credential/mapping cleanup succeeded.

| Transport / measured repetition | Split ms | Complete verified ms | Execute round-trip ms |
|---|---:|---:|---:|
| Wired 1 | 162.256042 | 198.552083 | 97.416927 |
| Wired 2 | 153.181302 | 188.741042 | 98.742448 |
| Wired 3 | 146.831562 | 175.976875 | 92.644427 |
| Wired 4 | 139.523125 | 176.284531 | 88.932240 |
| Wired 5 | 140.935261 | 180.842344 | 94.955365 |
| **Wired median** | **146.831562** | **180.842344** | **94.955365** |
| Wired nearest-rank p95 / maximum | 162.256042 | 198.552083 | 98.742448 |
| Wi-Fi 1 | 293.008698 | 326.621354 | 234.585990 |
| Wi-Fi 2 | 178.506250 | 209.694896 | 119.836042 |
| Wi-Fi 3 | 187.082343 | 217.441822 | 122.285937 |
| Wi-Fi 4 | 536.524479 | 565.332083 | 486.456615 |
| Wi-Fi 5 | 205.884427 | 235.587395 | 137.330885 |
| **Wi-Fi median** | **205.884427** | **235.587395** | **137.330885** |
| Wi-Fi nearest-rank p95 / maximum | 536.524479 | 565.332083 | 486.456615 |

The excluded warm-up split intervals are 544.117552 ms wired and 915.953125 ms
Wi-Fi. The full Android reference, run after each measured split, has median
19.758959 ms wired and 16.725729 ms Wi-Fi. For this small workload, both split
paths are slower than their local references. The measurements establish correct
physical CPU contributions and expose transport/orchestration cost. The reference
is a different timing scope and follows each split; it is not an independent
randomized speed benchmark. The observed Wi-Fi variation is preserved rather than
removed as an outlier. No speedup or raw-link bandwidth is claimed.

[Sanitized evidence](evidence/android-desktop-2026-10-24.json) records the final
APK, individual samples, local check and correction history. Original private
report hashes, before sanitization, are:

| Report | SHA-256 |
|---|---|
| Wired | `0b0cb9dc71d7cacea40ce02fc109afb2233ca7599ceffd0ef9dd1526f8d902ef` |
| Wi-Fi | `5451d6de7a8ce1ba7b2739a9d7508bc8d2921d1655615f34fcc1a6e1a61f67a0` |
| Local | `792601ee018838b0a886bb70c8375c7c9c210c71f38830ad82b4c57dd5eacf5d` |

These successful transport cases do not close H1 general planner integration,
durable recovery, coordinator epoch/fencing or production scheduling gates.

This probe can qualify a narrow physical ARM Android + desktop CPU vector path.
Full models, GPU/NPU mixed execution, automatic placement, capacity pooling,
durable coordinator recovery, Internet/NAT operation, raw network speed, production
scale and model-performance improvements require their own evidence.

## Licensing

The APK bundles LICENSE, NOTICE, licence history and the prior Apache text. New
probe/orchestration code and unchanged portable foundations retain their respective
licence scopes. Viewing the licence screen does not record acceptance. See
[licence history](LICENSE_HISTORY.md); interoperability does not relicense HyperL
code or automatically grant a port into Meshlit.
