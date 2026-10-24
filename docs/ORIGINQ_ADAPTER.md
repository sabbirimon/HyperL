# Origin Quantum adapter

Verified 2026-10-24. `hyperl.quantum_originq` implements a narrow optional adapter
for OriginQ's official `pyqpanda3` interface. Contract fixtures do not establish
cloud access, QPU execution, fidelity, provider availability or quantum speedup.

## Version and sources

The first adapter requires **pyqpanda3 0.4.0**, matching the version identified in
OriginQ's accessible API reference. The publisher's package page now lists 0.5.0;
that release is deliberately rejected until its SDK contract is qualified.
[Publisher package metadata](https://pypi.org/project/pyqpanda3/).

The verified calls are `QProg(qubits)`, indexed H/X/RX/RZ/CNOT, terminal `measure`,
`QCloudService.backend(name)` and `QCloudBackend.run(program, shots, options)`.
The returned `QCloudJob` is asynchronous; no result retrieval is performed here.
[OriginQ gates](https://github.com/OriginQ/pyqpanda3-skill/blob/main/resources/api/core/gates.md),
[program construction](https://github.com/OriginQ/pyqpanda3-skill/blob/main/resources/api/core/circuit.md),
[cloud service and 0.4.0 options](https://github.com/OriginQ/pyqpanda3-skill/blob/main/resources/api/qcloud/service.md),
[job handle](https://github.com/OriginQ/pyqpanda3-skill/blob/main/resources/api/qcloud/job.md).

These are provider-authored interface references, not instructions imported into
HyperL. No upstream implementation is copied and no REST endpoint is invented.
The SDK remains optional and is not installed or imported when this module loads.

## Execution boundary

`export_program(circuit)` constructs a local SDK program with all-qubit terminal
measurements. Shared `Circuit.validate()` bounds the supported subset: up to
16 qubits and 4,096 H/X/RX/RZ/CX gates. No mid-circuit measurement, pulse program,
arbitrary source, remote URL or runtime plugin is accepted as a circuit.

`submit(circuit, shots, client=..., permission=..., target=..., request_id=...)`
requires the owner's `OriginQClient(api_key, endpoint)` configuration, an exact
`originq` provider / target permission with unexpired remote opt-in, and a canonical
UUID. The configuration is inert; permission is checked before SDK loading, before
service construction and after backend selection. One `run` call
uses the approved shot count, mapping enabled, optimization and readout correction
disabled. Logical-to-physical mapping remains provider work and needs QPU testing.

The selected backend's `name()` must match the approved target. The receipt contains
the local UUID, target, shots, SDK version, bounded provider job ID and unpolled SDK
job. The UUID is **not provider-side idempotency**; `batch_id` is not
misrepresented as an idempotency key. The caller must persist its own submission
ledger and reconcile an uncertain result before sending another request. Adapter
submission errors redact SDK exception text and never retry or poll automatically.
SDK-internal retries, blocking, cancellation and transport behavior are not qualified.

## Credentials and transport

The adapter never retrieves credentials or selects a default target. The owner
supplies a secret directly to `OriginQClient`, outside circuit/report/log data;
its representation hides the key. The documented SDK constructor exposes an HTTP
default. HyperL rejects opaque SDK clients and non-HTTPS configurations, then calls
`QCloudService(api_key=..., url=...)` with the explicit HTTPS URL only after approval.
URL credentials, query strings and fragments are rejected. The owner must obtain
an approved compatible secure endpoint; no alternative hostname is guessed.

This enforces the SDK constructor's HTTPS URL argument. It does not certify the
SDK's internal TLS verification, redirects, retries or timeout/cancellation behavior.
Those need inspection and an isolated live transport test before production use.
Credentials remain in process memory for the configured client lifetime; no secret
vault, native process sandbox or provider cost accounting is implemented here.

## What has been tested

`python3 -m unittest discover -s scripts -p 'test_quantum_originq.py' -v` checks
eleven stub SDK contracts: exact gate/measurement order, scope/expiry/shot rejection,
lazy dependency failure, backend identity, one submission, no polling, missing job
identity, HTTPS configuration, error redaction and expiry between backend selection and dispatch.
These fixtures execute no cloud request.
Actual SDK construction/simulation and owner-authorized paid/free cloud/QPU jobs
need separate evidence before advertising provider qualification or supported chips.
