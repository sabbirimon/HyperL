# Single-phone and edge-device plan

Owner clarification, 2026-10-08 (Asia/Dhaka): HyperL must remain useful on a single
phone as well as enterprise clusters. Standalone desktop Java CLI/GUI is implemented;
Android/iOS native app installation remains future work. An ARM64 native Vulkan
qualification experiment now runs listed kernels on the owner's Adreno 506;
[actual results and limits](ANDROID_VULKAN.md) are separate from app integration.
The separate Meshlit Android project can become a client; it is not proof that the
standalone HyperL release can already be installed as a phone app.

| Mode | Planned behavior | Required evidence |
|---|---|---|
| Offline local CPU | Native C ABI/runtime with bounded preprocessing, vision/security kernels and later model execution | ARM64 Android NDK and Apple native builds, numerical/cancellation/memory tests; tensor/model engine must exist before model claims |
| Local GPU/NPU | Vulkan/OpenCL where supported, Apple Metal/Core ML/ANE and vendor-qualified NPU adapters | Actual drivers, operator/precision support, buffer ownership, transfer costs, device loss and CPU agreement; source emission alone is insufficient |
| Remote client | Submit approved jobs to a workstation/private cluster over authenticated IPv6/IPv4 | Explicit enrollment/server trust, scoped short-lived credentials, status/cancel/results, offline/reconnect handling; no automatic external upload |
| Optional phone worker | Opt-in bounded workloads with leases, quotas and battery/thermal policy | Foreground/background rules, process suspension/death, lease expiry and persistent recovery; no always-on availability assumption |

Keep the mobile UI small: editor/examples or task selection, capabilities, input
preview, local/remote target, observed memory/thermal state, output and Stop. No
Kubernetes, desktop Swing or datacenter dependencies in the default phone install.
Share versioned IR, errors and conformance with the full SDK later; use platform
native UI/bindings only when they improve measured package/startup/runtime costs.

Non-root Android and normal iOS app sandboxes are the defaults. Privileged Android
plugins are separate owner-installed and qualified components; they cannot be a
requirement for CPU mode. iOS must use supported native APIs/entitlements. Root does
not grant missing GPU operators or bypass adapter correctness requirements.

Budgets include native buffers, JVM/Swift/JNI bridges if present, unified-memory
sharing, display/decoder/model costs and driver overhead. Allocation failure still
needs handling. Defaults favor low concurrency and battery/thermal stability;
no automatic overclocking. Foreground execution is explicit; background work obeys
OS lifecycle constraints and gives honest paused/stopped state. Storage quotas and
encrypted temporary data are explicit and cancellable, with keys in platform key
stores when integrated. Never place keys in programs, AI prompts or logs.

Acceptance: actual single-phone CPU output, install/uninstall/update, permissions,
low-memory interruption, rotation/background/resume, cancellation and offline use;
then qualified GPU/NPU checks, then one remote endpoint with revoke/cancel/timeout
and network changes. The owner resumed single-phone GPU testing: eleven finite
outputs/two expected rejections pass on Galaxy A20s / Adreno 506 through the
native ADB-shell runner. Meshlit app-UID/JNI/lifecycle and two-phone cluster tests
remain pending. Desktop CI does not establish these hardware results.

See [enterprise/cluster services](ENTERPRISE_AND_CLUSTER_PLAN.md),
[platform matrix](PLATFORMS.md) and [SDK/runtime roadmap](ARCHITECTURE_AND_ROADMAP.md).
