# HyperL developer workbench design plan

Owner request: polished UI that attracts developers, with the engineering clarity
of mature accelerator tools. HyperL keeps its own name, colors, assets and visual
language. This plan is not a claim about adoption or current GPU capabilities.

## Product goals and first pass

1. Put the useful first result within one screen: editable example, Run, output,
   memory plan and a clear backend label. New developers should not need vendor
   knowledge just to run the CPU reference.
2. Use a dark graphite foundation, restrained cyan identity and lime action accent,
   readable code typography, intentional spacing and grouped controls. Avoid dense
   gray default widgets, decorative charts and screenshots containing invented data.
3. Divide the workspace into program/input editors on the left and result/source/
   memory output on the right. Show unqualified backends as explicit install/probe
   requirements, keeping source emission distinct from execution.
4. Show actual local OS/architecture and JVM heap limit in an environment strip.
   Memory plan reports the real current snapshot plus estimated job costs. Discovery
   and source generation never display a successful accelerator qualification badge.
5. Make Run visually primary and Stop easy to find. While managed work runs, disable
   competing actions and show its actual task state. Use keyboard Run and labeled,
   scrollable editors with line numbers; retain focus visibility and readable contrast.
6. Preserve real working commands, validation, cancellation and output. Do not add
   dead navigation, fake profiling, account screens, unavailable toggles or model
   previews just to fill the dashboard.

## Implementation and performance boundaries

This first alpha remains a Swing/JVM desktop workbench sharing the existing runtime.
Use native vector painting and system fonts, no downloaded fonts, image backgrounds,
browser engine, web service or animation loop. UI/configuration stay outside kernel
execution. The CLI remains suitable for headless/server workflows. A headless panel
render is visual QA; actual native desktop interaction/accessibility is still a
separate acceptance check. Measure startup and RSS before promising a smaller UI.

Reuse pinned [FlatLaf 3.7.2](https://github.com/JFormDesigner/FlatLaf/releases/tag/3.7.2)
Apache-2.0 with its no-natives classifier and
[RSyntaxTextArea 4.0.1](https://github.com/bobbylight/RSyntaxTextArea/releases/tag/4.0.1)
BSD-3-Clause for syntax/folding/undo/line numbers. Retain component licenses.
Apply a reusable theme to buttons, panels, editors, tabs, borders and task status.
Keep error/status text alongside color. Use flexible splits and scrolling instead
of shrinking code to fit; test the default 1280x820 layout and smaller desktop size.
Respect OS window lifecycle. No network/radio action is performed by the GUI.

## Next design stages — implement only with working backends

- Device explorer: observed adapters/drivers, qualification evidence, memory domains,
  operator support and actionable failure messages; unknown values remain unknown.
- Profiler: measured CPU/GPU transfer/compile/dispatch/download, timelines and actual
  allocations/RSS, reproducible baseline comparison and downloadable evidence.
- Dataset workspace: owner-selected files/quotas/key handles, real chunk IO/progress,
  explicit encrypted spill and cancellation only after those runtime paths exist.
- Cluster workspace: approved IPv6 endpoints, observed leases/queues/capacity and
  separate human/agent controls, global Stop with remote acknowledgment state.
- SDK onboarding later: language/API reference, samples, project templates, bindings,
  adapter conformance and profiler integration from the versioned full SDK.
- Distribution: native packaging/signing where qualified, screenshots from actual
  releases, quick-start documentation and honest platform/capability badges.

## Review and adoption gates

Verify that a developer can run the example, edit inputs, inspect memory, emit source,
understand an unavailable GPU and stop local work. Test keyboard/focus/readability,
resizing, error recovery and native display behavior on supported desktop systems.
Gather feedback from consenting developers before deciding a larger UI framework is
worth its dependencies. Measure time to first correct result and successful return
use, not assumed popularity. Keep the CLI/GUI/install guides synchronized.

Reference products: [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
and [Nsight Graphics](https://developer.nvidia.com/nsight-graphics). They motivate
engineering workflow clarity, not copied source, brand assets or feature claims.

## Built-in developer tooling increment

The local workbench adds workspace open/save with explicit overwrite handling,
editable program/input JSON, syntax highlighting, folding, undo/redo, literal
search, formatting, semantic/shape validation, two real examples and displayed
source/result export. Run, emission, memory planning and GPU probe keep their
existing actual execution boundaries. A workspace is only bounded declarative
JSON; opening it never runs shell/build hooks or starts a backend.

Later IDE interoperability uses the CLI first, then a versioned language server
with diagnostics/completion/hover/navigation, VS Code and JetBrains integration,
CMake/templates, SDK bindings, debugger/profiler APIs, conformance/test runners
and reviewed adapter tooling. These integrations need the real SDK/runtime; no
installed extension, debugger, model engine or compiler service is claimed now.

See [developer/IDE tool coverage and ordered delivery](DEVELOPER_TOOLS_PLAN.md)
for the full editing, SDK, build/test, debugger/profiler, Git, terminal and AI-helper
roadmap with current versus future implementation boundaries.

The owner subsequently requested advanced code suggestions, error/bug detection
and fix guidance. [Code diagnostics](CODE_DIAGNOSTICS.md) implements local HyperL
semantic diagnostics/context values, actual sample checks and reviewed corrections;
universal language analysis and AI-generated edits require later qualified tooling.
