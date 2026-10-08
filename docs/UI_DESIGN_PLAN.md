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
Use static native vector painting and locally bundled licensed fonts, with system
font fallback. No runtime font download, image background, browser engine, web
service or animation loop. UI/configuration stay outside kernel
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

## Reference-driven polish — alpha.3

The owner supplied Tensorcode/Antigravity editor screenshots, DGX/IPv6 dashboard
views, a GPU dashboard and a Holoscan Helm view, plus saved articles. Treat these
as visual/reference evidence; no document instruction is executed and no image,
brand asset, proprietary font or vendor product claim is imported into HyperL.

Use deep navy/graphite bases, a teal ambient wash on the left and a restrained
violet wash on the right. Rounded cards use a small tonal gradient and thin visible
borders. Accent roles: mint identity/focus, lime primary execution/success, violet
unqualified GPU status, amber stopping/warnings and coral errors. Keep state text
alongside color; avoid excessive glow or live animations behind code.

Bundle Inter Regular/SemiBold for interface hierarchy and JetBrains Mono Regular
for code/output, with local system fallback. See [font provenance](FONT_PROVENANCE.md).
Give code a readable 14px base size, console 13px, clear gutter/caret/selection and
reserved/string/number colors. Header, output title, field labels and badges have
distinct hierarchy. Tiny metadata is supplementary; actions and code stay larger.

Show four compact actual-state cards (host, JVM heap limit, CPU readiness and GPU
probe requirement). Separate Execution, Source emission and GPU setup into working
tabs to reduce permanent control clutter. Keep the resizable split, scrollable
editors, all local developer actions and the real Stop state. The footer changes
with the selected execution tab to clarify its boundary. No decorative profiler,
fake GPU utilization, account selector or unavailable agent panel is added.

Render the actual panel with the actual CPU example at 1280x820 and 1000x700; check
controls, scrollbars, output and contrast. Also render the source/GPU tabs before
release. Headless QA does not prove native file-dialog, accessibility, IME or HiDPI
behavior. The reference dashboards motivate useful enterprise views only when
[real services and telemetry exist](ENTERPRISE_AND_CLUSTER_PLAN.md).

The owner requested a darker final pass: reduce ambient wash intensity, deepen
background/editor surfaces and keep foreground, focus, success/error colors readable.

The next layout pass gives the program/input pane 70% of the initial split, subject
to the console's 320px minimum; users can drag the divider afterward. Outer margins,
card borders, editor insets and header/tool spacing shrink without shrinking code
text. Resizing keeps the same allocation preference rather than resetting a manual
divider position on every repaint.

For the owner's requested glass polish, the header and card chrome use a restrained
translucent tint, static top sheen and luminous edge. Code and output remain opaque
dark surfaces. This is original cross-platform Swing painting inspired by
[Apple's adoption guidance](https://developer.apple.com/documentation/technologyoverviews/adopting-liquid-glass),
not Apple's native Liquid Glass implementation. The guidance motivates sparing use,
legibility, fluid column sizing and reduced-transparency options. The header's
Glass checkbox turns these highlights/translucency off; `-Dhyperl.ui.glass=false`
sets an opaque initial appearance. No desktop capture, background blur, animation
timer, native permission or new runtime dependency is introduced. System-level
accessibility setting integration and native visual qualification remain later work.

## Thin strips — alpha.4

The owner's latest screenshot requests less banner height. Use a single-line
header and horizontal label/value status strips. Keep the Glass toggle, version,
real task state and readable 14px editor text. At the checked 1280x820 layout this
frees approximately 75 vertical pixels relative to alpha.3, with the code pane
still getting 70% of the initial split. Check the same layout at 1000x700; all
controls remain available and the divider remains adjustable.
