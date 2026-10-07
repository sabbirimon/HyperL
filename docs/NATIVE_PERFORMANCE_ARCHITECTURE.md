# Native performance architecture decision

Owner-approved direction, 2026-10-08 (Asia/Dhaka): **C/C++ for kernels and accelerator
adapters; Rust for cluster services.** Use a stable C ABI between layers. This is
an architectural decision and ordered plan, not a shipped Rust control plane or a
new performance result. The current desktop editor remains Kotlin/JVM; the native
CPU preview is C99. Full SDK/native optimization work is later.

| Layer | Selected direction | Why / boundary |
|---|---|---|
| Program semantics | Versioned IR independent of host language | Preserve numerical, precision, memory, cancellation and version rules across implementations |
| Native kernels | C99 ABI, optimized C/C++ implementations and qualified LLVM/AOT/SIMD paths | Explicit buffers, allocation/lifetime and target-specific code; conformance before fast paths |
| Accelerator host adapters | C/C++ with narrowly scoped platform wrappers | Public vendor SDKs, transfer/compile/dispatch/device-loss handling; Apple may use Objective-C++/Swift where native APIs require it |
| Cluster control and worker services | Rust preferred | Ownership/concurrency model, explicit deadlines/backpressure and typed protocol state; FFI/unsafe code remains reviewed and bounded |
| Desktop developer UI today | Kotlin/JVM 17+ | Already working editing/diagnostics; not the production low-latency kernel engine |
| Future desktop package | Native CLI first; evaluate a thin native UI separately | Java is not a permanent platform requirement; measured package/startup/RSS/responsiveness and licensing determine UI framework |
| Mobile clients | Native Android/iOS UI/bindings to the C ABI | No desktop Java GUI dependency; follow each OS's lifecycle and platform API rules |

A language switch alone cannot guarantee faster inference. First remove repeated
parsing, avoid copies, reuse bounded buffers, fuse supported operators and compile/cache
against exact target/driver identity. Keep UI, scheduling, JSON and telemetry out of
timed native kernels. Preserve finite intermediate rejection and ordered f32 sums;
parallel reductions/fast-math need an explicit precision contract, not silent drift.

The ABI must specify caller ownership, capacities/strides, supported operators,
error codes, finite values, cancellation polls, cleanup and binary compatibility.
Rust wrappers expose reviewed safe interfaces; C/C++ calls and raw pointers remain
unsafe boundaries. No panic/exception may unwind across the ABI. GPU driver state
and remote stop have explicit cleanup/acknowledgment limitations.

Desktop options to measure: keep the current UI as a tools package; compare a small
native C++ UI prototype against a Rust-hosted/native shell only after the runtime
is stable. Qt is one candidate, not a chosen/bundled dependency; its exact modules
and distribution license obligations need review. Do not require a browser engine,
web server or new language solely for a gradient or a settings screen.

Acceptance before migration: same example/edit/save/analyze/export/stop workflow;
compatible workspace files; cold/warm startup, idle/peak RSS, installed size and input
latency on Windows/Linux/macOS; signed packaging, accessibility/IME/HiDPI, install/
uninstall and error recovery. Measure actual end-to-end native execution/transfer
costs separately. Publish hardware/compiler/driver/precision/sample details and CPU
baseline. No speedup, lower latency or smaller footprint is claimed from the choice.

Next packages: C ABI conformance/ownership corpus -> optimized native CPU path and
native CLI -> one qualified GPU adapter -> versioned Rust worker/control schema and
3–5-host pilot -> thin native desktop UI decision -> SDK bindings and profiler.
The current alpha stays usable while these separately validated packages develop.

Primary engineering references: [LLVM target-independent code generation](https://llvm.org/docs/CodeGenerator.html),
[Rust FFI boundaries](https://doc.rust-lang.org/nomicon/ffi.html),
[Rust performance/ownership design](https://rust-lang.org/),
[Qt licensing by module](https://doc.qt.io/qt-6/licensing.html).
These explain mechanisms/options; they are not evidence of HyperL performance.
