# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Optional quantum SDK bridges, with no implicit clients or credential lookup.

Submission functions return the provider's native asynchronous job handle. The
caller owns job persistence, result retrieval, cancellation and cost approval.
request_id is correlation metadata, NOT a guarantee of remote idempotency. This
module does not retry; SDK transport policy remains caller configuration. After
an ambiguous submit failure, reconcile with the provider before any new call.
Optional SDKs are imported only when needed, after remote permission checks.
"""
from __future__ import annotations

from typing import Any

from .quantum import Circuit, ProviderPermission, MAX_SIMULATOR_WORK, validate_job_key

_MAX_COMPILED_OPERATIONS = 65_536


def _authorize(provider: str, target: str, circuit: Circuit, shots: int,
               permission: ProviderPermission, request_id: str) -> None:
    validate_job_key(request_id)
    permission.check(provider, target, circuit, shots)


def to_qiskit(circuit: Circuit) -> Any:
    """Build a measured Qiskit circuit; qubit zero is the least significant bit."""
    circuit.validate()
    from qiskit import QuantumCircuit
    program = QuantumCircuit(circuit.qubits)
    for gate in circuit.gates:
        if gate.name in ("h", "x"):
            getattr(program, gate.name)(gate.targets[0])
        elif gate.name in ("rx", "rz"):
            getattr(program, gate.name)(gate.angle, gate.targets[0])
        elif gate.name == "cx":
            program.cx(*gate.targets)
        else:
            raise ValueError("Unsupported Qiskit gate")
    program.measure_all()
    return program


def submit_ibm(circuit: Circuit, shots: int, *, client: Any,
               permission: ProviderPermission, target: str, request_id: str) -> Any:
    """Submit through an explicitly configured QiskitRuntimeService.

    Uses SamplerV2 job mode, not an automatically created session. Retrieve the
    provider job ID immediately; job tags are searchable but do not deduplicate.
    Results stay provider-native: job.result()[0].data.meas.get_counts().
    """
    _authorize("ibm", target, circuit, shots, permission, request_id)
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    program = to_qiskit(circuit)
    backend = client.backend(target)
    if backend.name != target:
        raise ValueError("IBM backend identity does not match approved target")
    compiled = generate_preset_pass_manager(backend=backend, optimization_level=1).run(program)
    if compiled.size() > _MAX_COMPILED_OPERATIONS:
        raise ValueError("Compiled IBM circuit exceeds operation limit")
    sampler = SamplerV2(mode=backend, options={"environment": {"job_tags": [request_id]}})
    _authorize("ibm", target, circuit, shots, permission, request_id)
    return sampler.run([compiled], shots=shots)


def _cirq_program(circuit: Circuit, qubits: tuple[Any, ...]) -> Any:
    import cirq
    if len(qubits) != circuit.qubits or len(set(qubits)) != circuit.qubits:
        raise ValueError("An explicit distinct qubit mapping is required")
    program = cirq.Circuit()
    for gate in circuit.gates:
        if gate.name == "h":
            operation = cirq.H(qubits[gate.targets[0]])
        elif gate.name == "x":
            operation = cirq.X(qubits[gate.targets[0]])
        elif gate.name == "rx":
            operation = cirq.rx(gate.angle)(qubits[gate.targets[0]])
        elif gate.name == "rz":
            operation = cirq.rz(gate.angle)(qubits[gate.targets[0]])
        elif gate.name == "cx":
            operation = cirq.CNOT(qubits[gate.targets[0]], qubits[gate.targets[1]])
        else:
            raise ValueError("Unsupported Cirq gate")
        program.append(operation)
    # Cirq histogram's first measured qubit is the most significant bit.
    program.append(cirq.measure(*reversed(qubits), key="m"))
    return program


def run_cirq_simulator(circuit: Circuit, shots: int, *, seed: int = 0) -> dict[str, int]:
    """Run local Cirq state-vector sampling, never a cloud service or physical QPU.

    Returns counts in q[n-1]...q[0] order. The simulator is bounded by Circuit's
    16-qubit and shared amplitude-work limits; this function is synchronous and
    does not promise cancellation.
    """
    circuit.validate()
    if type(shots) is not int or not 1 <= shots <= 4096:
        raise ValueError("shots must be an integer in 1..4096")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (1 << circuit.qubits) * max(1, len(circuit.gates)) > MAX_SIMULATOR_WORK:
        raise ValueError("local simulator amplitude-work budget exceeded")
    import cirq
    program = _cirq_program(circuit, tuple(cirq.LineQubit.range(circuit.qubits)))
    result = cirq.Simulator(seed=seed).run(program, repetitions=shots)
    histogram = result.histogram(key="m")
    counts = {format(int(value), f"0{circuit.qubits}b"): int(count) for value, count in histogram.items()}
    if (len(counts) > min(shots, 1 << circuit.qubits) or sum(counts.values()) != shots or
            any(len(bits) != circuit.qubits or set(bits) - {"0", "1"} or count <= 0
                for bits, count in counts.items())):
        raise ValueError("Cirq returned invalid measurement counts")
    return counts


def submit_google_engine(circuit: Circuit, shots: int, *, client: Any,
                         permission: ProviderPermission, target: str, request_id: str,
                         physical_qubits: tuple[Any, ...], target_gateset: Any) -> Any:
    """Submit to a caller-configured, access-approved cirq_google.Engine.

    The caller supplies physical placement and a target compilation gateset;
    device validation rejects unsupported placement/gates. No automatic routing
    or assumption of publicly available Google QPU access is made.
    """
    _authorize("google-engine", target, circuit, shots, permission, request_id)
    import cirq
    if target_gateset is None:
        raise ValueError("An explicit Google target gateset is required")
    if not isinstance(physical_qubits, tuple) or any(not isinstance(q, cirq.GridQubit) for q in physical_qubits):
        raise ValueError("physical_qubits must be an explicit tuple of GridQubit positions")
    program = _cirq_program(circuit, physical_qubits)
    compiled = cirq.optimize_for_target_gateset(program, gateset=target_gateset,
                                               ignore_failures=False, max_num_passes=1)
    if sum(1 for _ in compiled.all_operations()) > _MAX_COMPILED_OPERATIONS:
        raise ValueError("Compiled Google circuit exceeds operation limit")
    device = client.get_processor(target).get_device()
    device.validate_circuit(compiled)
    _authorize("google-engine", target, circuit, shots, permission, request_id)
    return client.run_sweep(program=compiled, processor_id=target,
                            program_id=f"hl-{request_id}", job_id=f"hl-{request_id}",
                            repetitions=shots)


def to_ionq_json(circuit: Circuit) -> dict[str, Any]:
    """Convert only this contract's QIS gates to Azure IonQ circuit JSON.

    This is not QIR, Pulser, native IonQ gates or a generic Azure target payload.
    Measurements are implicit for all qubits in the provider's circuit format.
    """
    circuit.validate()
    operations = []
    for gate in circuit.gates:
        if gate.name == "cx":
            operation = {"gate": "cnot", "control": gate.targets[0], "target": gate.targets[1]}
        elif gate.name in ("h", "x", "rx", "rz"):
            operation = {"gate": gate.name, "target": gate.targets[0]}
            if gate.name in ("rx", "rz"):
                operation["rotation"] = gate.angle
        else:
            raise ValueError("Unsupported IonQ gate")
        operations.append(operation)
    return {"qubits": circuit.qubits, "circuit": operations}


def submit_azure_ionq(circuit: Circuit, shots: int, *, client: Any,
                      permission: ProviderPermission, target: str, request_id: str) -> Any:
    """Submit via a caller-configured Azure Quantum Workspace, IonQ JSON only.

    Return the native Job; name is correlation metadata, not idempotency. Keep
    native results: the ideal IonQ simulator can return a probability histogram
    rather than sampled integer counts. Never silently round it into counts.
    """
    _authorize("azure-ionq", target, circuit, shots, permission, request_id)
    from azure.quantum.target import IonQ
    payload = to_ionq_json(circuit)
    selected = client.get_targets(name=target)
    if not isinstance(selected, IonQ) or selected.name != target:
        raise ValueError("Approved target must be an Azure IonQ target with matching identity")
    if (selected.input_data_format != "ionq.circuit.v1" or
            selected.output_data_format != "ionq.quantum-results.v1"):
        raise ValueError("Only the documented Azure IonQ JSON format is supported")
    _authorize("azure-ionq", target, circuit, shots, permission, request_id)
    return selected.submit(circuit=payload, name=f"hyperl-{request_id}", shots=shots)
