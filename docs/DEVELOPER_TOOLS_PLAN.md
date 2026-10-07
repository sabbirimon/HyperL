# Built-in developer and IDE tooling

Owner intent: HyperL should be easy to code with, attractive to developers and
useful as standalone CLI/GUI software, with a full SDK later. Prioritize a reliable
first correct result and usable local tools before introducing entire IDE services.

| Tool area | Current workbench increment | Next qualified implementation |
|---|---|---|
| Editor | JSON syntax, folding, line numbers, undo/redo and literal find | Context suggestions/repairs now; later semantic completion, inline ranges, navigation and safe refactoring through a versioned language server |
| Programs/projects | Two runnable examples, bounded declarative workspace open/save and unsaved-change protection | Project templates, multiple source files and dependency/runtime manifests; no auto execution on open |
| Validation | Actual program/DAG, finite input/shape/memory checks, structured issue paths, typo/reference repairs and real CPU sample checking; structural IDE JSON schema | Machine-readable ranged diagnostics, shared compatibility corpus and precise compiler errors |
| Formatting | Validated program/input JSON formatting | Language-server formatting and SDK generated code formatters |
| Build/run | Actual CPU reference, explicit OpenCL bridge, separate source emission and export | Native compiler/cache and tensor/model provider integrations with hashes and numerical conformance |
| Test tools | Existing JVM, C CTest and installer suites from the CLI/build | GUI test panel and adapter/operator conformance runner using real outcomes, no fake pass badges |
| Memory | Job admission budget, actual heap/environment snapshot and full-DAG array estimate | Allocator arenas, domains/NUMA/HBM, pressure/leases, measured allocations and explicit encrypted spill |
| Debugger | Validation/failure messages and managed local Stop | Actual native breakpoints, buffer inspection and debugger protocol backed by qualified compiler/runtime |
| Profiler | No timeline/throughput values invented | Native timings, memory/RSS, transfer/compile/dispatch/download, trace export and reproducible baseline |
| IDE integration | Local VS Code schema/workspace and manually invoked CLI process task | Tested VS Code/JetBrains plugins, shared language server, CMake and qualified SDK/binding support |
| Files/source export | Explicit local file choices, bounded writes, temporary cleanup and overwrite confirmation | Multi-file artifact bundles, provenance and signed release/build manifests |
| Version control | Source is a normal Git project usable with existing installed Git/IDE tools | Optional reviewed Git diff/history panel; commit/push remain explicit owner actions |
| Terminal/package tools | Use existing independently installed terminal/build tools manually | Scoped executable/workspace allowlists, command previews, bounded jobs and package/license provenance |
| API/AI helpers | No chat/provider accounts or automatic code changes added | Optional owner-configured assistant/provider, reviewable diffs, tests and separate agent grants; secrets never in source/prompts |
| Cluster/remote development | CLI profile validation and explicitly authenticated node probe | Approved workspaces/nodes, leases, remote build caches, result provenance and Stop acknowledgments |
| Documentation/onboarding | Packaged install/programming/memory/UI guides and examples | Searchable offline API reference, tutorials, generated SDK docs and migration/conformance guides |

## Implementation order

1. Deliver and visually verify this genuine local workbench increment. Test editor
   actions, workspace round-trip/rejection, backend failure, Stop and controls.
2. Validate native desktop focus, keyboard, resizing, file dialogs and accessibility
   on supported platforms. Use real screenshots and user feedback; never assert
   developer adoption from a color scheme alone.
3. Add source-location diagnostics and a language server shared by built-in editor
   and external IDE plugins. Keep language version and ABI contracts stable.
4. Build full SDK after a native backend and packaging stabilize: headers/libraries,
   compiler/runtime APIs, C++/Rust/Python/Java/Swift bindings as qualified, templates,
   adapter interfaces, documentation and conformance kits.
5. Implement actual debugger/profiler, tensor/model tools, native memory/fabric and
   clusters one backend at a time. Add UI panels only when they have working data.
6. Qualify extension/plugin and package tooling, reproducible artifact provenance,
   update rollback and a maintainable release policy before claiming LTS support.

Developer ease must preserve ownership: opening a project cannot execute scripts,
install packages, activate root, accept a remote agent, transmit radio traffic or
run tools against an unapproved host. Compiler/provider tools require explicit
configuration and bounded/cancellable work. Unknown device or profiling state stays
unknown. A high-quality UI explains a failure and the next concrete step.

[Code diagnostics](CODE_DIAGNOSTICS.md) records the implemented bounded local
analysis, contextual suggestions, sample overflow detection and review-only fixes.
