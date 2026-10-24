# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""OriginQ adapter contract fixtures only: no SDK, network or QPU execution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, call, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.quantum import Circuit, Gate, ProviderPermission
from hyperl import quantum_originq as originq


class ProgramFixture:
    def __init__(self, qubits):
        self.qubits = qubits
        self.operations = []

    def __lshift__(self, operation):
        self.operations.append(operation)
        return self


class OptionsFixture:
    def __init__(self):
        self.settings = {}

    def set_mapping(self, value):
        self.settings["mapping"] = value

    def set_optimization(self, value):
        self.settings["optimization"] = value

    def set_amend(self, value):
        self.settings["amend"] = value


def fixture_sdk(service=None):
    core = SimpleNamespace(QProg=ProgramFixture)
    for name in ("H", "X", "RX", "RZ", "CNOT", "measure"):
        setattr(core, name, lambda *args, name=name: (name, args))
    return core, SimpleNamespace(QCloudOptions=OptionsFixture, QCloudService=Mock(return_value=service))


class OriginQContractTest(unittest.TestCase):
    def setUp(self):
        self.circuit = Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))
        self.permission = ProviderPermission("originq", "approved-fixture", max_shots=128,
            expires_at_epoch_ms=int(time.time() * 1000) + 60_000, allow_remote=True)
        self.client = originq.OriginQClient("fixture-secret", "https://originq.invalid")
        self.service = Mock()
        self.service.backend.return_value.name.return_value = "approved-fixture"
        self.service.backend.return_value.run.return_value.job_id.return_value = "job-fixture-1"
        self.request_id = str(uuid4())

    def submit(self, **changes):
        values = dict(circuit=self.circuit, shots=64, client=self.client,
                      permission=self.permission, target="approved-fixture", request_id=self.request_id)
        values.update(changes)
        return originq.submit(**values)

    def test_import_does_not_import_optional_sdk(self):
        script = """import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith('pyqpanda3'):
        raise RuntimeError('unexpected SDK import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import hyperl.quantum_originq
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
            text=True, timeout=10, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "python")})
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_permission_rejects_before_sdk_or_client_use(self):
        with patch.object(originq, "_load_sdk") as load:
            for permission in (replace(self.permission, allow_remote=False),
                               replace(self.permission, provider="another-provider"),
                               replace(self.permission, target="another-target"),
                               replace(self.permission, expires_at_epoch_ms=1)):
                with self.subTest(permission=permission), self.assertRaises((ValueError, PermissionError)):
                    self.submit(permission=permission)
            with self.assertRaises((ValueError, PermissionError)):
                self.submit(shots=129)
            load.assert_not_called()
            self.assertEqual([], self.service.mock_calls)

    def test_invalid_request_and_circuit_do_not_load_sdk(self):
        with patch.object(originq, "_load_sdk") as load:
            for request_id in ("", "not-a-uuid", self.request_id.upper(), None):
                with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                    self.submit(request_id=request_id)
            with self.assertRaises(ValueError):
                self.submit(circuit=Circuit(2, (Gate("cx", (0, 2)),)))
            load.assert_not_called()
            self.assertEqual([], self.service.mock_calls)

    def test_all_supported_gates_and_measurement_order(self):
        circuit = Circuit(2, (Gate("h", (0,)), Gate("x", (1,)),
            Gate("rx", (1,), 0.25), Gate("rz", (0,), -0.5), Gate("cx", (1, 0))))
        with patch.object(originq, "_load_sdk", return_value=fixture_sdk(self.service)):
            program = originq.export_program(circuit)
        self.assertEqual(2, program.qubits)
        self.assertEqual([("H", (0,)), ("X", (1,)), ("RX", (1, 0.25)),
                          ("RZ", (0, -0.5)), ("CNOT", (1, 0)),
                          ("measure", (0, 0)), ("measure", (1, 1))], program.operations)

    def test_exact_target_shots_options_one_submission_no_polling(self):
        sdk = fixture_sdk(self.service)
        with patch.object(originq, "_load_sdk", return_value=sdk):
            receipt = self.submit()
        sdk[1].QCloudService.assert_called_once_with(api_key="fixture-secret", url="https://originq.invalid")
        self.service.backend.assert_called_once_with("approved-fixture")
        backend = self.service.backend.return_value
        backend.run.assert_called_once()
        program, shots, options = backend.run.call_args.args
        self.assertEqual(64, shots)
        self.assertEqual({"mapping": True, "optimization": False, "amend": False}, options.settings)
        self.assertEqual(("measure", (1, 1)), program.operations[-1])
        self.assertIs(receipt.job, backend.run.return_value)
        self.assertEqual([call.job_id()], receipt.job.mock_calls)
        self.assertEqual("job-fixture-1", receipt.provider_job_id)
        self.assertEqual(self.request_id, receipt.request_id)

    def test_secure_explicit_configuration_without_sdk_or_network(self):
        with patch.object(originq, "_load_sdk") as load:
            for url in ("http://originq.invalid", "https://user:secret@originq.invalid",
                        "https://@originq.invalid",
                        "https://originq.invalid?key=secret", "https://originq.invalid/#x",
                        "https://", "https://originq.invalid:99999", "https://originq.invalid\\evil"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    originq.OriginQClient("fixture-secret", url)
            with self.assertRaises(TypeError):
                self.submit(client=self.service)
            load.assert_not_called()
        self.assertNotIn("fixture-secret", repr(self.client))
        self.assertEqual([], self.service.mock_calls)

    def test_backend_identity_mismatch_prevents_submission(self):
        self.service.backend.return_value.name.return_value = "unapproved-backend"
        with patch.object(originq, "_load_sdk", return_value=fixture_sdk(self.service)):
            with self.assertRaises(originq.OriginQUnavailable):
                self.submit()
        self.service.backend.return_value.run.assert_not_called()

    def test_missing_job_identity_is_uncertain_without_resubmission(self):
        self.service.backend.return_value.run.return_value.job_id.return_value = ""
        with patch.object(originq, "_load_sdk", return_value=fixture_sdk(self.service)):
            with self.assertRaises(originq.OriginQSubmissionError):
                self.submit()
        self.service.backend.return_value.run.assert_called_once()

    def test_expiry_during_backend_selection_prevents_run(self):
        with patch.object(originq, "_load_sdk", return_value=fixture_sdk(self.service)), \
             patch.object(ProviderPermission, "check", side_effect=[None, None, PermissionError("expired")]):
            with self.assertRaises(PermissionError):
                self.submit()
        self.service.backend.return_value.run.assert_not_called()

    def test_uncertain_submit_is_redacted_and_never_retried(self):
        self.service.backend.return_value.run.side_effect = RuntimeError("secret-token-in-provider-error")
        with patch.object(originq, "_load_sdk", return_value=fixture_sdk(self.service)):
            with self.assertRaises(originq.OriginQSubmissionError) as raised:
                self.submit()
        self.assertNotIn("secret-token", str(raised.exception))
        self.service.backend.return_value.run.assert_called_once()

    def test_missing_or_unqualified_sdk_has_explicit_failure(self):
        with patch.object(originq.metadata, "version", side_effect=originq.metadata.PackageNotFoundError):
            with self.assertRaises(originq.OriginQUnavailable):
                originq.export_program(self.circuit)
        with patch.object(originq.metadata, "version", return_value="0.5.0"), \
             patch.object(originq, "import_module") as load:
            with self.assertRaises(originq.OriginQUnavailable):
                originq.export_program(self.circuit)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
