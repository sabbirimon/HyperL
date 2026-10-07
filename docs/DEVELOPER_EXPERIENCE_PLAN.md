# Developer experience: Python-style ease, native execution

Owner requirement, 2026-10-08 (Asia/Dhaka): easy to start and program, comparable to
Python's approachable workflow. This is a language/API and onboarding plan, not a
complete SDK or text-language compiler. The alpha.3 increment adds a small explicit
[Python/C ABI preview and recipes](LIBRARY_INTEROPERABILITY.md), separate from the full SDK. Current alpha programs use bounded
`hyperl/1` JSON; the full SDK is explicitly a later milestone.

## Authoring goals

- A simple Python-like surface with small readable expressions, named inputs/outputs,
  sensible defaults, helpful type/shape errors and no unnecessary boilerplate.
- A small standard Python preview now, expanding to full notebook/script APIs later;
  add a standalone text authoring option
  so Python does not become mandatory on every deployed worker.
- Compile these frontends into the same versioned typed IR and native runtime;
  interpretive authoring does not put Python, JSON or UI into the timed kernel loop.
- Keep JSON as a inspectable interchange/debugging format, rather than the only
  comfortable way to write substantial programs. Avoid separate numerical semantics.
- Discover observed capabilities and make device/precision selection explicit when
  it changes correctness, cost or compatibility. Unsupported targets give a useful
  error; no implicit GPU-to-CPU substitute.

Conceptual syntax only, not executable in this alpha:

```python
# Proposed HyperL authoring syntax / API shape
x = input("x", dtype="f32")
w = input("w", dtype="f32")
positive = relu(x * w)
output(positive)
```

The first compiler subset should lower named inputs, add/multiply/ReLU/ordered sum
to existing bounded IR before adding tensors, broadcasting, loops or model imports.
Type/shape/precision/cost validation is shared with native bindings. Arbitrary Python
is executable host code; it is not accepted as a safe remote kernel format. A native
standalone parser should accept a deliberately bounded grammar, not `eval` scripts.

## Workflow and API requirements

| Area | Required developer outcome | Delivery stage |
|---|---|---|
| First correct result | Install, load example, run, inspect output and edit inputs in minutes | CLI/GUI examples implemented; onboarding time must be measured with users |
| Authoring | Clear operators, typed values, shape defaults and concise Python-like code | Frontend/full SDK later |
| Diagnostics | Source line/column/path, cause, suggested edit and explainable limitations | JSON path/sample checks now; text-language/LSP source maps later |
| Completion | Useful operations/arguments/types from actual schema/capabilities | Context suggestions now; semantic LSP/IDE completion later |
| Packaging | CLI + optional GUI, reproducible examples, no cloud/account requirement for local CPU | Desktop implemented; native/mobile installers later |
| SDK use | Small Python/C++/Rust/Swift examples with consistent ownership/errors | C ABI and small Python preview now; full bindings/package/version policy later |
| Debug/profile | Reproducible errors, numerical checks and measured transfer/execution/memory | Reference/sample checks now; native debugger/profiler later |
| Remote jobs | Same artifact from local example to explicit cluster submission | Authenticated durable cluster API/services later |

Use standard package/install workflows for each qualified SDK binding and avoid
requiring every developer to build GPU drivers or learn all vendor APIs. Provide
minimal quick starts, language/reference cookbook, tested examples, migrations and
troubleshooting. Pin reproducible environments without bundling proprietary SDKs
or hiding platform-specific requirements. Offer editor/CLI help and offline docs.

Optional AI assistance later must describe the selected program/context, explain
errors, preview diffs and invoke only scoped tools; developer confirmation precedes
applying the suggested change. Existing reviewed typo repair is local and bounded,
not generative AI autocomplete or proof of universal bug detection.

## Acceptance and implementation order

1. Keep existing CLI/editor examples, diagnostics, workspace files and no-execution
   on open behavior reliable; improve docs and collect onboarding feedback.
2. Stabilize native compiler/runtime/C ABI and conformance, with explicit performance
   measurements; implement the small text frontend and reference source mapping.
3. Deliver the full SDK and Python binding sharing the same IR/numerical contracts;
   qualify package install and small end-to-end programs on selected OS/architectures.
4. Semantic language server, VS Code/JetBrains integrations, templates and measured
   native debugger/profiler. Extend tensor/model language only after runtime support.

Test usable examples, invalid/ambiguous shapes, useful diagnostics, target failures,
precision changes, memory ownership, cancellation and reproducible exports. Measure
installation/time-to-first-result, cold start/RSS and real kernel/transfer performance
separately. Ease of use and speed are goals, not demonstrated adoption or performance.

See [native C/C++ and Rust architecture](NATIVE_PERFORMANCE_ARCHITECTURE.md),
[developer tool roadmap](DEVELOPER_TOOLS_PLAN.md) and
[enterprise services](ENTERPRISE_AND_CLUSTER_PLAN.md).

Prefer the standard Python API for initial adoption; a new text language must add
real value beyond syntax resemblance. Advanced C/C++/Rust bindings remain available
for native integration as the full SDK develops.
