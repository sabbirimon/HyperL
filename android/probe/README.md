# Standalone HyperL Android probe

A small foreground-only qualification app, separate from Meshlit. It compiles six
unchanged portable HyperL source files through a generated Android source set:
`HyperL`, `HyperLAdmission`, `HyperLCodec`, `HyperLContract`, `HyperLLibrary` and
`PreciseReduction`. It does not include JVM GUI, worker server, JDK HttpClient,
model weights, GPU adapters or automatic background work.

From the HyperL checkout with JDK21 and an installed Android SDK:

```sh
./gradlew -p android/probe assembleDebug
```

Set `ANDROID_HOME` or private untracked `android/probe/local.properties` to the SDK.
AGP9.2.1, Kotlin2.4.10, Gradle9.4.1 and compileSdk37 are selected. Minimum API26;
physical acceptance is separate from successful assembly. The debug APK is for
owner-controlled testing, not a production store release.

The 2026-10-24 APK evidence below is a preserved build snapshot. Later HyperL
capability negotiation, cryptography and quantum Python APIs are not automatically
included in this six-source CPU probe. See the [current validation record](../../docs/VALIDATION.md)
for broader source tests and the [quantum prototype](../../docs/QUANTUM_EXTENSION_PLAN.md)
for its separate simulator/provider boundaries.

## Start and evidence

Install on the selected phone without clearing existing application data. Open
**HyperL Probe**, then press **Run local CPU checks** or **Run Samsung + Mac split**.
No run starts on launch. Local checks execute weighted ReLU and compensated sum.
The split test executes 32,768 deterministic multiply/ReLU elements: Android
handles `[0,16384)`, the configured Mac CPU worker handles `[16384,32768)`.
The two branches start concurrently; Android gathers and compares every f32 bit
against its complete CPU reference. One warm-up and five measured repetitions
report individual timings, median and nearest-rank p95 (maximum for five samples).
No measured speedup, energy/reward value, full model sharding or GPU claim follows.

Reports live at app-private `files/probe-result.json`; each split additionally
preserves its own `files/reports/<runId>.json`. The host collects the exact staged
UUID path so a later local run cannot replace its evidence. Reports use format
`hyperl-android-probe/1`, status `running`, `passed`, `failed` or `cancelled`.
They include mode, runId, startedAtEpochMs/finishedAtEpochMs, real Android ABI/process
architecture, heap/system memory, remote process/instance/job identity, exact
split ranges, payload byte counts and result hashes. `measurements[0]` is warm-up;
indices1–5 are measured. `remoteShards` includes warm-up (6 on full success), while
`measuredRemoteShards` excludes it (5). Remote OS/architecture is not in the worker
protocol: correlate with the host harness instead of assuming it from a name.

Stop or leaving the foreground cancels local work, disconnects active HTTP and
attempts bounded authenticated remote cancellation. A `202` acknowledges a request,
not remote termination. Failure/cancellation never returns a passed report.

ADB selectors for the owner-controlled harness:

- Package/activity: `org.hyperl.probe/.MainActivity`.
- Split button resource: `org.hyperl.probe:id/start_split`.
- Content description: `Start Samsung and Mac split test`.
- Local: `org.hyperl.probe:id/start_local`; Stop: `org.hyperl.probe:id/stop_probe`.

## Explicit connection provisioning

Start a separately configured HyperL worker with ID `mac-probe`, CPU_REFERENCE,
capacity for 16,384-element shards and at least a 4 MiB admission budget. Put its
random token into app-private `files/worker.token` using the debug owner harness,
never an Intent, source, logs or a report. Put the trusted public certificate in
`files/worker-ca.pem`. The probe creates a private TLS trust store containing only
this certificate and retains ordinary HTTPS hostname verification. It never
changes Android's global trust store.

App-private `files/connection.json` chooses exactly one transport and requires a
fresh canonical UUID `runId` before the split test can start:

```json
{
  "origin": "https://127.0.0.1:18091",
  "transport": "wired-adb-reverse",
  "workerId": "mac-probe",
  "runId": "00000000-0000-4000-8000-000000000001"
}
```

The shown runId is an example; generate a fresh UUID per actual measurement.
For wired testing, configure `adb reverse tcp:18091 tcp:HOST_PORT` on the selected
phone. For Wi-Fi, set transport `wireless-lan` and use the Mac's actual numeric
private LAN HTTPS address/port; its certificate must include that address in SAN.
No Wi-Fi passwords or system settings are changed by the app. Compare wired and
wireless runs with the same graph, input, worker and TLS configuration. USB tunnel
results include ADB forwarding overhead; they do not establish raw USB bandwidth.

The app accepts only numeric private IPv4/ULA IPv6 or loopback destinations, no
DNS names, URL credentials, query/fragment/path, redirects or arbitrary requests.
The explicit optional `loopback-plaintext` transport only permits
`http://127.0.0.1:18091`; it is not the default and must be labeled separately from
HTTPS measurements. Missing certificates/tokens or invalid configuration fail.
Transport support is not permission to connect to an unapproved worker.

## License

The APK includes the repository LICENSE, NOTICE, license history and prior Apache
text for inspection. Current Community/Enterprise terms and previous Apache grants
retain their respective scopes. Opening the license viewer records no acceptance.
Kotlin/coroutines/serialization retain their Apache-2.0 licenses. See the root
license history; this Android probe does not relicense copied portable foundations.

## Physical qualification — 2026-10-24

The final standalone debug APK assembled and passed Android lint on macOS/JDK21.
The last combined build/lint took 7 seconds: zero errors and 16 warnings covering
target/dependency metadata, optional icon/extraction rules and programmatic UI
internationalization. The APK is 1,830,571 bytes with SHA-256:

`7e850367855b8bd5d3479768b80ec5f2454b7e7395cfa89de4fa4ec8d2aa0c56`

LICENSE/NOTICE/history/prior-Apache assets were inspected in the archive. The APK
was installed on Samsung Galaxy A20s (SM-A207F, Android 11/API30) without clearing
application data. The real process reported ARM64/aarch64 and 64-bit execution.
Xiaomi was not operated. This is a debug qualification app, not a production APK.

The final local test passed: weighted ReLU returned `[0,6,12]`, compensated sum
returned `1`, and the observed local check took 3.048073 ms. Remote shards were zero.
Both final cross-device sessions passed every element's raw f32 comparison, and
the independent Mac harness verified exactly six completed remote shards per
session: one warm-up plus five measured runs. Samsung computed the first half;
the Mac's x86_64 CPU worker computed the second half.

| Observed transport | Split median / p95 | Verified total median / p95 | Remote round trip median / p95 |
| --- | --- | --- | --- |
| TLS through USB ADB reverse | 146.832 / 162.256 ms | 180.842 / 198.552 ms | 94.955 / 98.742 ms |
| TLS directly through Wi-Fi LAN | 205.884 / 536.524 ms | 235.587 / 565.332 ms | 137.331 / 486.457 ms |

Each p95 is the maximum of only five measured samples, not a fleet percentile.
The full-phone reference medians were 19.759 ms and 16.726 ms respectively.
Transfer, framing and validation dominated this small workload; splitting was
slower than local execution. These results demonstrate correct heterogeneous
execution and transport operation. They do not demonstrate speedup, large-model
sharding, GPU/NPU execution, reward economics or production fleet reliability.

The Wi-Fi connection was bound to Android's observed Wi-Fi network (`wlan0`), with
no active VPN. The host independently verified the USB device path and ADB reverse
mapping for the wired run. Configuration labels alone are not route evidence.
The reviewed public evidence is
[`android-desktop-2026-10-24.json`](../../docs/evidence/android-desktop-2026-10-24.json).
The operator also saved the raw harness reports as `wired-adb-reverse.json` and
`wireless-lan.json`; session IDs were `09d4488a-aabd-4205-838d-77b1fd43e439` and
`41866f51-56a3-46b2-8258-36b785442660`. After collection the Mac worker stopped and
test connection credentials/forwarding were cleaned up. The host verified an
empty ADB reverse list and removal of the private token, certificate and connection
configuration. Only the consumed-session ledger and reports remained. The app
remains installed.

### Fixes and evidence boundaries

- AGP9 source-directory API differences were resolved by syncing the six portable
  files and registering them in Android's Kotlin source set; JVM-only sources are
  excluded.
- `comparisonAndHashMs` now ends after computing the output hash, so its name
  matches the timed work.
- A split requires a fresh canonical UUID. It is atomically consumed before any
  dispatch, including attempts that subsequently fail or are cancelled. A bounded
  ledger retains up to 256 consumed IDs. Repeated Start shows a waiting-for-host
  message and preserves completed evidence. Missing configuration, certificate or
  token shows setup guidance before computation starts.
- Physical setup-only and duplicate-Start checks on the session-guard build
  preserved the previous report byte-for-byte. The final APK retains that guard.
- Every split writes its own UUID report before updating the latest report. On the
  final APK, a successful local test deliberately replaced the latest report;
  the earlier wired UUID report remained byte-identical and the host collected
  it successfully. This is the evidence used above.
- An early unstaged run had no UUID, a later repeated session exceeded the expected
  host count, and another latest report was replaced by a local test. Those runs
  were excluded from qualification. The final two sessions used the same final
  APK and passed the host's exact identity, output-hash and six-job checks.

The final APK has not changed since those successful physical runs. No further
Android-device operation is required to interpret these results. Remaining
production work includes broader hardware/network coverage, longer workloads,
release signing/distribution and backend/model execution beyond the bounded CPU
probe implemented here.
