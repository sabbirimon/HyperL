# IDE helpers included in the alpha

Open `hyperl.code-workspace` in an independently installed VS Code. This local
workspace maps `*.hyperl.json` documents to the bundled draft-7 schema for JSON
suggestions, structure checks, formatting and folding. Create a workspace with
`hyperl workspace-new FILE.hyperl.json`, then edit it. A manually invoked task
prompts for your installed HyperL launcher and runs `workspace-validate` on the
current file using a process task. No task executes on open, and no extension,
compiler, driver or language server is installed. Configure your own executable;
a workspace does not authorize arbitrary build hooks or downloads.

Schema assistance is structural only. It cannot prove cross-field DAG ownership,
shape equality, finite f32 conversion, memory admission or backend execution. The
HyperL validator/runtime remains authoritative. JetBrains users can manually add
this JSON schema and an external tool calling `workspace-validate`; no JetBrains
plugin is shipped. Actual IDE UI sessions/tasks are unrun acceptance checks.

The built-in HyperL workbench already supports syntax/folding/undo, literal search,
semantic/shape validation, JSON formatting, example selection, workspace open/save,
source emission/export, memory policy, explicit GPU probes and cancellable CPU work.
See [UI plan](../UI_DESIGN_PLAN.md) and [user guide](../USER_GUIDE.md).

The later complete SDK/IDE stages include a language server, semantic completion,
navigation/refactoring, versioned project/binding generators, real compiler/model
imports, breakpoints/debugger adapters, measured native profiling/timelines,
conformance/test tools and adapter packaging. Security/scope, backend availability,
license notices and human control remain part of those working interfaces.

Primary configuration references: [VS Code JSON support](https://code.visualstudio.com/docs/languages/json)
and [process tasks](https://code.visualstudio.com/docs/debugtest/tasks).
