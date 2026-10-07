# Code suggestions, error detection and reviewed fixes

The workbench **Analyze** action and CLI `diagnose` analyze `hyperl/1` programs
locally. No LLM account, network request or downloaded compiler is required.

```sh
bin/hyperl diagnose examples/elementwise.json examples/inputs.json
bin/hyperl diagnose PROGRAM.json INPUTS.json MEMORY_BUDGET_BYTES
```

Reports contain `valid`, coded issues with a JSON field path, concrete suggestions,
context-aware completion values, a memory plan and `sampleCpuChecked`. CLI exits
2 for an invalid report, while retaining the JSON diagnostics on stdout.

## Actual checks

- JSON syntax, language version, unknown fields/build hooks, bounded unique input
  declarations, ordered DAG dependencies, instruction/operand counts and output names.
- Supported operation suggestions and legal input/output references at each step.
  Bounded edit distance proposes possible operation/reference typos, with explicit
  caution that a similar spelling cannot prove the developer's intended computation.
- Exactly matching input bindings, finite f32 values, vector shape and memory
  admission. Suggestions explain missing arguments, unsupported broadcasting,
  retained-vector limits and explicit budget/partitioning choices.
- Actual CPU reference execution on supplied inputs after static checks pass.
  An overflowing intermediate is reported at its instruction, including when a
  later ReLU would otherwise hide the infinity. CPU sample checking does not
  execute a native/GPU compiler, shell, files, radio or remote provider.
- Selected backend boundaries, such as an unimplemented OpenCL reduction.
  The analyzer does not silently switch the developer's chosen execution backend.

## Review fix

After Analyze, **Review fix** lists only proposed operation/reference corrections.
The dialog shows the field and old/new value; the developer selects and explicitly
confirms one edit. Only that field changes. The repair is bound to a SHA-256 of the
exact analyzed editor text, validates the existing value, permits only supported
operation names or prior defined references and rejects a stale editor. Reanalyze
and test after every edit. No package installation, build script, external code,
root activation or provider action is triggered by a repair.

Other errors receive guidance for manual correction, because resizing inputs,
changing precision, removing evaluation steps, choosing a new backend or enlarging
memory use can change semantics/behavior. They are never silently auto-fixed.

## Scope and later IDE/SDK integration

This analyzer understands HyperL's current bounded f32 JSON DAG, not arbitrary
C/C++/Python projects or general vulnerability discovery. Successful sample checks
cover only the supplied values; they do not prove all numerical inputs, model
quality, accelerator correctness or absence of bugs. Context completions are
reported values and typo/reference repairs, not an AI generative autocomplete
service or an installed language server.

Later stages expose ranged diagnostics, code actions and completion through a
versioned language server to the built-in editor, VS Code and JetBrains. Qualified
compiler/provider adapters add their own diagnostics, debugger/profiler and tensor/
model conformance. Optional AI suggestions will need owner-configured providers,
secret isolation, explicit context sharing and reviewed diffs/tests. The full SDK
remains a later milestone. See [developer tools](DEVELOPER_TOOLS_PLAN.md).
