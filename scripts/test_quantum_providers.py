# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""SDK contract doubles only: these tests never contact or qualify a QPU."""
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.quantum import Circuit, Gate, ProviderPermission
from hyperl.quantum_providers import (
    run_cirq_simulator, submit_azure_ionq, submit_google_engine, submit_ibm,
    to_ionq_json, to_qiskit,
)

KEY = "12345678-1234-4234-8234-123456789abc"


class FakeQiskitCircuit:
    def __init__(self, n):
        self.n = n
        self.ops = []

    def h(self, q): self.ops.append(("h", q))
    def x(self, q): self.ops.append(("x", q))
    def rx(self, a, q): self.ops.append(("rx", a, q))
    def rz(self, a, q): self.ops.append(("rz", a, q))
    def cx(self, a, b): self.ops.append(("cx", a, b))
    def measure_all(self): self.ops.append(("measure_all",))


class FakeGridQubit:
    def __init__(self, row, col): self.row, self.col = row, col
    def __hash__(self): return hash((self.row, self.col))
    def __eq__(self, other): return isinstance(other, FakeGridQubit) and (self.row, self.col) == (other.row, other.col)


class FakeCirqCircuit:
    def __init__(self): self.ops = []
    def append(self, operation): self.ops.append(operation)
    def all_operations(self): return iter(self.ops)


def fake_cirq():
    return SimpleNamespace(
        Circuit=FakeCirqCircuit, GridQubit=FakeGridQubit,
        H=lambda q: ("h", q), X=lambda q: ("x", q),
        rx=lambda a: lambda q: ("rx", a, q), rz=lambda a: lambda q: ("rz", a, q),
        CNOT=lambda a, b: ("cx", a, b), measure=lambda *q, key: ("measure", q, key),
        optimize_for_target_gateset=Mock(side_effect=lambda program, **kwargs: program),
        LineQubit=SimpleNamespace(range=lambda n: list(range(n))), Simulator=Mock(),
    )


class QuantumProviderContracts(unittest.TestCase):
    def setUp(self):
        self.bell = Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))

    def permission(self, provider_id, target_id, **changes):
        values = dict(provider=provider_id, target=target_id, max_shots=100,
                      expires_at_epoch_ms=int(time.time() * 1000) + 60_000, allow_remote=True)
        values.update(changes)
        return ProviderPermission(**values)

    def test_qiskit_conversion_retains_rotation_units_and_measures_all(self):
        circuit = Circuit(2, (Gate("x", (0,)), Gate("rx", (1,), math.pi),
                              Gate("rz", (0,), -0.3), Gate("cx", (1, 0))))
        with patch.dict(sys.modules, {"qiskit": SimpleNamespace(QuantumCircuit=FakeQiskitCircuit)}):
            converted = to_qiskit(circuit)
        self.assertEqual(converted.ops, [("x", 0), ("rx", math.pi, 1), ("rz", -0.3, 0), ("cx", 1, 0), ("measure_all",)])

    def test_ibm_compiles_exact_backend_and_submits_once_with_correlation_tag(self):
        client = Mock()
        backend = SimpleNamespace(name="ibm-test")
        client.backend.return_value = backend
        compiled = Mock()
        compiled.size.return_value = 8
        manager = Mock()
        manager.run.return_value = compiled
        factory = Mock(return_value=manager)
        sampler = Mock()
        sampler_factory = Mock(return_value=sampler)
        modules = {"qiskit": SimpleNamespace(QuantumCircuit=FakeQiskitCircuit),
                   "qiskit.transpiler": SimpleNamespace(generate_preset_pass_manager=factory),
                   "qiskit_ibm_runtime": SimpleNamespace(SamplerV2=sampler_factory)}
        with patch.dict(sys.modules, modules):
            job = submit_ibm(self.bell, 20, client=client, permission=self.permission("ibm", "ibm-test"),
                             target="ibm-test", request_id=KEY)
        factory.assert_called_once_with(backend=backend, optimization_level=1)
        sampler_factory.assert_called_once_with(mode=backend, options={"environment": {"job_tags": [KEY]}})
        sampler.run.assert_called_once_with([compiled], shots=20)
        self.assertIs(job, sampler.run.return_value)

    def test_remote_permissions_and_bad_request_ids_fail_before_client_use(self):
        for provider, submit, target, extra in [
            ("ibm", submit_ibm, "ibm-test", {}),
            ("google-engine", submit_google_engine, "processor", {"physical_qubits": (), "target_gateset": None}),
            ("azure-ionq", submit_azure_ionq, "ionq.simulator", {}),
        ]:
            for changes in ({"allow_remote": False}, {"expires_at_epoch_ms": 1},
                            {"target": "other"}, {"provider": "other"}, {"max_shots": 1}):
                client = Mock()
                with self.assertRaises((ValueError, PermissionError)):
                    submit(self.bell, 20, client=client, permission=self.permission(provider, target, **changes),
                           target=target, request_id=KEY, **extra)
                self.assertEqual(client.mock_calls, [])
            with self.assertRaises(ValueError):
                submit(self.bell, 20, client=Mock(), permission=self.permission(provider, target),
                       target=target, request_id="invalid/key", **extra)

    def test_google_validates_physical_mapping_before_native_async_submit(self):
        cirq = fake_cirq()
        client = Mock()
        qubits = (FakeGridQubit(1, 2), FakeGridQubit(1, 3))
        gateset = object()
        with patch.dict(sys.modules, {"cirq": cirq}):
            job = submit_google_engine(self.bell, 20, client=client,
                permission=self.permission("google-engine", "processor"), target="processor", request_id=KEY,
                physical_qubits=qubits, target_gateset=gateset)
        compiled = cirq.optimize_for_target_gateset.call_args.args[0]
        self.assertEqual(compiled.ops[-1], ("measure", tuple(reversed(qubits)), "m"))
        self.assertEqual(cirq.optimize_for_target_gateset.call_args.kwargs,
                         dict(gateset=gateset, ignore_failures=False, max_num_passes=1))
        client.get_processor.assert_called_once_with("processor")
        client.get_processor.return_value.get_device.return_value.validate_circuit.assert_called_once_with(compiled)
        client.run_sweep.assert_called_once_with(program=compiled, processor_id="processor",
            program_id=f"hl-{KEY}", job_id=f"hl-{KEY}", repetitions=20)
        self.assertIs(job, client.run_sweep.return_value)

    def test_google_invalid_device_never_submits_or_falls_back(self):
        cirq = fake_cirq()
        client = Mock()
        client.get_processor.return_value.get_device.return_value.validate_circuit.side_effect = ValueError("Not supported")
        with patch.dict(sys.modules, {"cirq": cirq}), self.assertRaises(ValueError):
            submit_google_engine(self.bell, 20, client=client,
                permission=self.permission("google-engine", "processor"), target="processor", request_id=KEY,
                physical_qubits=(FakeGridQubit(1, 2), FakeGridQubit(1, 3)), target_gateset=object())
        client.run_sweep.assert_not_called()

    def test_local_cirq_counts_use_q0_least_significant_and_reject_invalid_counts(self):
        cirq = fake_cirq()
        cirq.Simulator.return_value.run.return_value.histogram.return_value = {1: 20}
        with patch.dict(sys.modules, {"cirq": cirq}):
            counts = run_cirq_simulator(self.bell, 20, seed=13)
            self.assertEqual(counts, {"01": 20})
            program = cirq.Simulator.return_value.run.call_args.args[0]
            self.assertEqual(program.ops[-1], ("measure", (1, 0), "m"))
            cirq.Simulator.return_value.run.return_value.histogram.return_value = {4: 20}
            with self.assertRaises(ValueError): run_cirq_simulator(self.bell, 20)
        for shots in (0, 4097, True):
            with self.assertRaises(ValueError): run_cirq_simulator(self.bell, shots)

    def test_cirq_work_budget_is_checked_before_optional_import_or_allocation(self):
        oversized = Circuit(16, (Gate("h", (0,)),) * 257)
        oversized.validate()
        with patch.dict(sys.modules, {"cirq": None}), self.assertRaisesRegex(ValueError, "work budget"):
            run_cirq_simulator(oversized, 1)

    def test_ionq_json_is_not_qir_and_preserves_cnot_direction_and_radians(self):
        payload = to_ionq_json(Circuit(2, (Gate("rx", (0,), 0.25), Gate("cx", (1, 0)))))
        self.assertEqual(payload, {"qubits": 2, "circuit": [
            {"gate": "rx", "target": 0, "rotation": 0.25},
            {"gate": "cnot", "control": 1, "target": 0}]})

    def test_azure_requires_exact_ionq_json_target_and_keeps_native_results(self):
        class IonQ:
            name = "ionq.simulator"
            input_data_format = "ionq.circuit.v1"
            output_data_format = "ionq.quantum-results.v1"
        target = IonQ()
        target.submit = Mock()
        client = Mock()
        client.get_targets.return_value = target
        with patch.dict(sys.modules, {"azure.quantum.target": SimpleNamespace(IonQ=IonQ)}):
            result = submit_azure_ionq(self.bell, 20, client=client,
                permission=self.permission("azure-ionq", target.name), target=target.name, request_id=KEY)
            self.assertIs(result, target.submit.return_value)
            target.submit.assert_called_once_with(circuit=to_ionq_json(self.bell), name=f"hyperl-{KEY}", shots=20)
            target.submit.reset_mock()
            target.input_data_format = "qir.v1"
            with self.assertRaises(ValueError):
                submit_azure_ionq(self.bell, 20, client=client,
                    permission=self.permission("azure-ionq", target.name), target=target.name, request_id=KEY)
            target.submit.assert_not_called()

    def test_submit_error_is_not_retried(self):
        class IonQ:
            name = "ionq.simulator"
            input_data_format = "ionq.circuit.v1"
            output_data_format = "ionq.quantum-results.v1"
        target = IonQ()
        target.submit = Mock(side_effect=TimeoutError("acknowledgement lost"))
        client = Mock()
        client.get_targets.return_value = target
        with patch.dict(sys.modules, {"azure.quantum.target": SimpleNamespace(IonQ=IonQ)}), self.assertRaises(TimeoutError):
            submit_azure_ionq(self.bell, 20, client=client,
                permission=self.permission("azure-ionq", target.name), target=target.name, request_id=KEY)
        self.assertEqual(target.submit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
