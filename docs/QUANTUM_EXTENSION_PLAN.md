<!-- SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0 -->
<!-- Copyright 2026 Sabbir Hassan Imon and respective contributors. -->
# Quantum and hybrid execution extension plan

Status: initial Python circuit/simulator and optional SDK bridges implemented; roadmap researched 2026-10-24.
**No physical QPU has been qualified.** The broader registry, ledger and budgets below remain planned.
Quantum contracts are separate from finite `f32` vector execution. No Kotlin quantum target is advertised.

## Implemented prototype boundary

`python/hyperl/quantum.py` defines bounded circuits, local simulation and expiring provider/target permission checks.
`python/hyperl/quantum_providers.py` converts those circuits for IBM SamplerV2, local Cirq, approved Google Engine and Azure IonQ JSON submission.
Optional SDK imports are lazy. Owners install and qualify their chosen SDK versions and supply configured clients; HyperL does not create accounts or store credentials.
IBM requires `qiskit` and `qiskit-ibm-runtime`; local Google simulation requires `cirq-core`; Engine additionally requires `cirq-google`; Azure requires `azure-quantum`.
Remote functions return native asynchronous jobs, leaving persistence, status, result retrieval and cancellation to the caller.
Each remote call requires a canonical UUID `request_id`, exact provider/target, expiry, shot allowance and `allow_remote=True`.
The UUID is correlation metadata; the wrapper has no durable ledger or monetary budget enforcement and never retries a submit call itself.
Provider SDK transport behavior remains caller configuration. An ambiguous submission requires reconciliation before another call.
Tests using SDK doubles verify bridge contracts only; they do not demonstrate provider acceptance, pricing, QPU access or hardware execution.
IBM's bridge follows [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2) and [job tags](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-environment-options); Azure's route follows [IonQ.submit](https://learn.microsoft.com/en-us/python/azure-quantum/azure.quantum.target.ionq?view=qsharp-py).
Google requires explicit physical qubits and a compilation gateset; [target-gateset compilation](https://quantumai.google/reference/python/cirq/optimize_for_target_gateset) and device validation precede [Engine submission](https://quantumai.google/cirq/google/engine).

## Provider interfaces to evaluate

The first four SDK routes above exist as source bridges; additional provider/workload coverage below is prospective.
All routes remain subject to SDK licensing, account access and independent qualification.
An available SDK, resource estimator or simulator does not establish access to a physical QPU.

| Provider route | Verified interface and proposed boundary |
| --- | --- |
| IBM Quantum | Qiskit Runtime offers Estimator and Sampler primitives; distinguish expectation estimates from sampled registers and pin the selected primitive/API revision. [Runtime primitives](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-runtime-primitives). |
| Google Quantum AI | Cirq's Engine interface manages circuit execution, jobs and calibrations. Engine access is explicitly restricted in current documentation; begin with local simulation and qualify cloud access separately. [Quantum Engine API](https://quantumai.google/cirq/google/engine). |
| Microsoft Azure Quantum | QDK provides development and simulation tools with Azure submission routes. Qualify each workspace/provider/target and exact QIR profile; QIR is an intermediate representation, not universal hardware compatibility. [QDK](https://learn.microsoft.com/en-us/azure/quantum/qdk-main-overview), [QIR](https://learn.microsoft.com/en-us/azure/quantum/concepts-qir), [target-profile submission](https://learn.microsoft.com/en-us/azure/quantum/quickstart-microsoft-qiskit). |
| AWS Braket | Hybrid Jobs combines classical resources with QPU or simulator tasks. Preserve the parent hybrid job and child quantum-task identities, accounting and cancellation separately. [Hybrid Jobs](https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs.html). |
| Pasqal | Cloud batches execute parameterized pulse sequences against selected backends. Model analog sequence programs separately; preserve batch/job IDs and target device specifications. [Batches and jobs](https://docs.pasqal.com/cloud/batches/), [Cloud SDK](https://docs.pasqal.com/cloud/quick-start/). |
| Quantinuum | Nexus separates compilation and execution jobs, including hardware and simulator backends. Bind the accepted program format and backend configuration to each submission. [Nexus jobs](https://docs.quantinuum.com/nexus/user_guide/concepts/jobs.html). |

## Versioned program and result contracts

- Introduce a separate future quantum job contract; never serialize qubits, amplitudes or counts as ordinary HyperL `f32` vector work.
- Program kinds should distinguish gate circuits, analog/pulse sequences, annealing/problem models and hybrid workflows. Unimplemented kinds remain unavailable.
- Every artifact needs a format/version/profile, digest, byte length, parameter schema and validated resource bounds; unsupported instructions fail before submission.
- Inputs distinguish classical parameter bindings, measurement specifications, observables and provider-specific configuration; preserve units and bit/register ordering.
- Results distinguish samples/counts, probabilities, expectation estimates with uncertainty, analog measurements and simulation-only state data.
- State vectors are available only when an explicitly qualified simulator exposes them; never promise state-vector extraction from a QPU.
- Record backend ID, hardware/simulator/resource-estimator kind, SDK/compiler revisions, transpilation options, circuit mapping, calibration references and available timestamps.
- Bound artifact size, qubits/atoms, circuit depth, parameters, shots, result bytes and concurrent jobs before allocation or upload.
- Gate-set conversion, QIR profiles, dynamic control, measurements and analog constraints require individual qualification; no automatic cross-provider translation claim.

## Approved provider registry

Use extensible namespaced provider and adapter IDs with versioned schemas, rather than adding a vendor label that implies execution support.
Each locally approved registry entry must declare:

1. Pinned adapter/package provenance, license review, exact API/version range and independently tested program/result kinds.
2. Allowed HTTPS origins, account/workspace/project, target IDs and credential references held outside source, program JSON, prompts and logs.
3. Explicit capabilities and limits, simulator/QPU classification, calibration freshness policy and supported cancellation/reconciliation behavior.
4. Allowed processing/storage regions, data classifications, retention/deletion terms and transfer destinations, including result buckets and logs.
5. Owner-approved submission rights and budgets; read/status permission must not silently grant paid submission or provider switching.

New providers require a reviewed adapter and conformance evidence. Unknown extensions fail closed.
Access, availability and prices are refreshed from the configured provider; advertisements cannot approve themselves.
Data-residency checks cover actual execution, storage and support/telemetry paths; an endpoint region alone is insufficient evidence.

## Durable asynchronous lifecycle

Proposed local states: `DRAFT -> VALIDATED -> APPROVED -> SUBMITTING -> QUEUED -> RUNNING -> SUCCEEDED | FAILED | CANCELLED`.
Also preserve `SUBMISSION_UNKNOWN`, `CANCEL_REQUESTED` and local `DEADLINE_EXCEEDED` without inventing a remote terminal result.
Provider-specific states map explicitly, while the original provider state and observation timestamp remain available.

- Before a network write, atomically journal an immutable request digest, approval/budget identity, target, stable idempotency key and submission intent.
- A durable ledger with a single submission owner and compare-and-set/fencing prevents two coordinators from submitting the same intent concurrently.
- Use provider idempotency only where its scope and retention guarantees are documented and qualified; retain the same key and request on a supported retry.
- Persist provider job IDs immediately. Recover by retrieving that job or reconciling the original submission identity after a crash or lost acknowledgement.
- Without safe reconciliation or provider idempotency, an uncertain submission stays `SUBMISSION_UNKNOWN`; do not retry a paid create call automatically.
- A local ledger alone cannot guarantee exactly-once execution at a remote provider. A deliberate rerun is a new approved intent and budget allocation.
- Poll with bounded exponential backoff and jitter; validate any webhook signature, account/job binding and duplicate delivery before applying state changes.
- Reject stale state regressions. Persist progress, terminal status, result integrity and accounting independently so interrupted result downloads can resume safely.
- Cancellation is a request until confirmed. Local timeout, app exit or credential revocation does not prove the remote computation stopped or charges ceased.
- Retain unresolved job and budget records across restart; reconcile final charges/results even when the requesting agent has stopped.

## Budgets, permissions and hybrid scheduling

- Require explicit credential configuration and owner approval for provider, target, data destination and paid execution scope before first submission.
- Enforce per-job and aggregate limits for shots, iterations, parameter sweeps, queue/wall time, classical CPU/GPU time, storage and monetary exposure.
- Reserve budget before submission and keep it reserved for unknown outcomes; distinguish estimates, reserved exposure and reported charges.
- Include compilation, mitigation overhead, retries, hybrid containers, storage and transfer in estimates where applicable; preserve pricing version and currency.
- Provider hard caps and cancellation guarantees vary. Where spend cannot be bounded reliably, require a separately approved exposure policy or block automatic submission.
- Recheck budgets and owner policy at every adaptive iteration. Exhausted or revoked approval stops new work and initiates qualified cancellation of existing work.
- Treat QPU tasks as asynchronous workflow nodes alongside classical HyperL preprocessing/optimization/postprocessing, with typed result edges.
- Independent circuits or parameter evaluations may be distributed when mathematically valid; do not split an arbitrary entangled circuit across unrelated QPUs.
- Device pairing, NVLink, shared RAM or a high-end NIC does not establish quantum connectivity or permission to run a QPU job.

## Qualification, reproducibility and evidence

- Start with bounded local simulator fixtures, then provider sandbox/emulator tests, and finally a separately approved small QPU qualification run.
- Test crash-before/after-submit, lost acknowledgements, duplicate coordinators, expired credentials, budget exhaustion, cancellation races and partial result recovery.
- Validate counts against requested/completed shots, register widths, endianness, missing outcomes and declared result type; distinguish raw counts from mitigated estimates.
- Use predefined statistical acceptance criteria, confidence intervals and adequate shots for probabilistic results; a random bitstring need not match a reference exactly.
- Record seeds where honored, algorithm/compiler/artifact hashes, parameter values, shot counts, noise/mitigation settings, topology/calibration snapshots and provider job IDs.
- Preserve raw and processed results with bounded storage and integrity metadata. Redact secrets and apply the approved retention policy to inputs and measurement data.
- Reproducing a QPU experiment means reproducing its configuration and statistical test; device drift prevents a promise of identical outcomes.
- Measure end-to-end compile, queue, transfer, classical and quantum time, uncertainty and total cost against an appropriate classical baseline.
- Do not claim speedup, quantum advantage, fault tolerance or distributed quantum execution from a successful API call, simulator run or provider product name.

## Delivery order

1. Specify versioned quantum schemas and registry policy; review semantics and bounds without altering the existing vector ABI.
2. Implement one explicit local simulator adapter plus lifecycle/ledger conformance tests; clearly label all simulated evidence.
3. Add one owner-approved provider adapter with durable recovery, budgets, residency checks and an independently recorded QPU qualification.
4. Extend provider coverage through the same gates, then evaluate bounded hybrid optimization workflows and measured scheduling improvements.

The remaining platform design is future work. No remote quantum jobs were submitted and no provider credentials were requested or stored for this change.
