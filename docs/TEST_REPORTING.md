# Share HyperL test results

HyperL welcomes reproducible test results, failed runs and design feedback from
individual developers, research labs, hardware vendors and large organizations.
Open a [GitHub issue](https://github.com/sabbirimon/HyperL/issues/new/choose) with
this report, or link a pull request that adds a redacted evidence file. A local
build, discovered device and successful run are different facts; say which you
observed. Keep raw measurements and failed attempts rather than reporting only
an attractive average.

## Existing measured checkpoints

| Dated test | Actual result | Scope and source |
| --- | --- | --- |
| Local JVM/native/Python, 2026-10-24 | 89 JVM cases: 87 pass, two unavailable-GPU skips, zero failures; 53 Python cases and three CTest cases pass | [Validation](VALIDATION.md); remote CI applies to its own exact commit |
| Mac CPU baseline, 2026-10-08 | 65,536-element f32 multiply/ReLU median: Python API 10,132.175 µs, native ABI 875.063 µs, NumPy pipeline 311.647 µs and preallocated C 18.528 µs | [Method and raw samples](CPU_BENCHMARK.md); paths have different setup/allocation scope |
| Samsung ARM64 + Mac x86-64 CPU split, 2026-10-24 | All 32,768 output elements match reference bits; five measured split medians: 146.832 ms wired TLS and 205.884 ms direct Wi-Fi TLS | [Probe and sanitized samples](ANDROID_CROSS_PLATFORM_PROBE.md); slower than the local reference for this workload |

These checkpoints do not establish enterprise throughput, full-model sharding,
production isolation or generic GPU/NPU speed. The [validation ledger](VALIDATION.md)
records further checks, rejected attempts and remaining gates.

## Run and document one test

1. Choose a supported path: the [build/test runbook](BUILD_AND_TEST.md),
   [CPU benchmark](CPU_BENCHMARK.md), [Android/desktop probe](ANDROID_CROSS_PLATFORM_PROBE.md),
   or a focused contract test. Obtain permission before using organizational
   hardware, networks or datasets.
2. Record the exact Git commit, release tag, build type, OS, compiler/JDK, driver,
   firmware and runtime versions. Name every actual device and its CPU, GPU/NPU,
   RAM, memory tier and link. A catalog entry is not proof that an adapter ran.
3. Save the workload shape, operator and precision, fixed input seed or sanitized
   input manifest, program/artifact hash, memory limits, worker count and topology.
   Use a known answer or independent reference and record exact output hashes or
   numerical error bounds before interpreting timings.
4. Keep warm-ups separate from measured repetitions. Share individual samples,
   median, p95 definition, units and timing boundary. State whether transfer,
   allocation, admission, compilation, queueing and result verification are
   included. Record peak or observed memory, energy and network bytes only when
   actual instruments supplied them.
5. Preserve failures: exit status, rejection reason, timeout/cancellation behavior,
   wrong output, worker loss and cleanup result. Mark unrun checks as unrun and
   unavailable metrics as unknown.

## Report template

Copy this into a new issue and fill only measured fields:

```text
Test title and date:
HyperL commit/tag and build type:
Goal and expected result:
Hardware: device/CPU/GPU/NPU, RAM/VRAM, memory domains, link topology:
OS, toolchain, runtime, driver and firmware versions:
Workload: operator, shape, precision, seed/input hash, limits, worker count:
Command or numbered reproduction steps:
Correctness: independent reference, exact hash or error bound, pass/fail:
Warm-up count and raw measured samples with units:
Timing scope and summary method:
Measured peak/observed memory, network and energy (or unknown):
Failures, retries, Stop/cancellation and cleanup:
Logs or redacted evidence links:
What remains untested:
```

For large testbeds, add tenant/policy scope, concurrent users/jobs, scheduler,
identity method, storage locality, fault-injection procedure and data residency
only if those services were actually deployed. Current HyperL source does not
supply a production enterprise control plane; see the
[enterprise delivery plan](ENTERPRISE_AND_CLUSTER_PLAN.md).

Remove tokens, private keys, internal hostnames/IPs, customer data, proprietary
model weights and personal prompts. Use synthetic or approved public inputs when
possible. Send security concerns privately to
[sabbirimon@gmail.com](mailto:sabbirimon@gmail.com) rather than a public issue.
Feature requests and usability feedback are welcome even without a
benchmark; describe the use case, affected hardware and desired behavior.
