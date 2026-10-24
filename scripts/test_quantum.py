# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Real stdlib simulation tests; permission fixtures do not submit a provider job."""
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.quantum import (Circuit, Gate, ProviderPermission, QuantumCancelled,
                           simulate, validate_job_key)


class QuantumSimulationTest(unittest.TestCase):
    def test_zero_state_x_and_little_endian_measurement(self):
        self.assertEqual({"00": 17}, simulate(Circuit(2, ()), 17)["counts"])
        result = simulate(Circuit(3, (Gate("x", (0,)),)), 32)
        self.assertEqual({"001": 32}, result["counts"])
        self.assertIn("q0 is least significant", result["bitOrder"])
        self.assertFalse(result["hardwareExecution"])
        self.assertEqual("simulator", result["executionKind"])
        self.assertIn("not native 128-bit", result["precision"])

    def test_hadamard_interference(self):
        result = simulate(Circuit(1, (Gate("h", (0,)), Gate("h", (0,)))), 100)
        self.assertEqual({"0": 100}, result["counts"])
        self.assertAlmostEqual(1.0, result["probabilities"]["0"])

    def test_actual_bell_distribution(self):
        circuit = Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))
        result = simulate(circuit, 4096, seed=42)
        self.assertEqual({"00", "11"}, set(result["counts"]))
        self.assertEqual(4096, sum(result["counts"].values()))
        self.assertAlmostEqual(0.5, result["probabilities"]["00"])
        self.assertAlmostEqual(0.5, result["probabilities"]["11"])
        self.assertLess(abs(result["counts"]["00"] - 2048), 256)

    def test_rotations_and_phase_interference(self):
        self.assertEqual({"1": 64}, simulate(Circuit(1, (Gate("rx", (0,), math.pi),)), 64)["counts"])
        half = simulate(Circuit(1, (Gate("rx", (0,), math.pi / 2),)), 64)
        self.assertAlmostEqual(0.5, half["probabilities"]["0"])
        self.assertAlmostEqual(0.5, half["probabilities"]["1"])
        phase = Circuit(1, (Gate("h", (0,)), Gate("rz", (0,), math.pi), Gate("h", (0,))))
        self.assertEqual({"1": 64}, simulate(phase, 64)["counts"])
        cancel_rotation = Circuit(1, (Gate("rx", (0,), 0.625), Gate("rx", (0,), -0.625)))
        self.assertEqual({"0": 64}, simulate(cancel_rotation, 64)["counts"])

    def test_control_target_order_nonadjacent_and_control_off(self):
        on = Circuit(3, (Gate("x", (2,)), Gate("cx", (2, 0))))
        self.assertEqual({"101": 64}, simulate(on, 64)["counts"])
        off = Circuit(3, (Gate("x", (0,)), Gate("cx", (2, 0))))
        self.assertEqual({"001": 64}, simulate(off, 64)["counts"])

    def test_high_qubit_hadamard_builds_ghz_state(self):
        circuit = Circuit(3, (Gate("h", (2,)), Gate("cx", (2, 1)), Gate("cx", (1, 0))))
        result = simulate(circuit, 256, seed=3)
        self.assertEqual({"000", "111"}, set(result["counts"]))
        self.assertAlmostEqual(0.5, result["probabilities"]["000"])
        self.assertAlmostEqual(0.5, result["probabilities"]["111"])

    def test_seed_reproducibility_and_no_global_random_mutation(self):
        circuit = Circuit(4, tuple(Gate("h", (q,)) for q in range(4)))
        before = random.getstate()
        first = simulate(circuit, 200, seed=76)
        second = simulate(circuit, 200, seed=76)
        third = simulate(circuit, 200, seed=77)
        self.assertEqual(first["counts"], second["counts"])
        self.assertNotEqual(first["counts"], third["counts"])
        self.assertEqual(before, random.getstate())
        json.dumps(first, allow_nan=False)
        self.assertAlmostEqual(1.0, math.fsum(first["probabilities"].values()))

    def test_circuit_shape_gate_and_finite_angle_rejections(self):
        invalid = [Circuit(True, ()), Circuit(0, ()), Circuit(17, ()), Circuit(1.0, ()),
                   Circuit(1, []), Circuit(1, (None,)), Circuit(1, (Gate("H", (0,)),)),
                   Circuit(1, (Gate("x", [0]),)), Circuit(1, (Gate("x", (True,)),)),
                   Circuit(1, (Gate("x", (1,)),)), Circuit(1, (Gate("x", (-1,)),)),
                   Circuit(1, (Gate("cx", (0, 0)),)), Circuit(2, (Gate("cx", (0,)),)),
                   Circuit(1, (Gate("x", (0,), 0),)), Circuit(1, (Gate("rx", (0,)),)),
                   Circuit(1, (Gate("rz", (0,), True),)), Circuit(1, (Gate("rx", (0,), "1"),)),
                   Circuit(1, (Gate("rx", (0,), float("nan")),)),
                   Circuit(1, (Gate("rz", (0,), float("inf")),)),
                   Circuit(1, (Gate("rx", (0,), 10**1000),)),
                   Circuit(1, (Gate("x", (0,)),) * 4097)]
        for circuit in invalid:
            with self.subTest(circuit=circuit), self.assertRaises(ValueError):
                circuit.validate()
        Circuit(1, (Gate("rx", (0,), 0),)).validate()
        with self.assertRaises(FrozenInstanceError):
            Gate("x", (0,)).name = "h"

    def test_amplitude_work_admitted_before_allocation_and_maximum_width(self):
        circuit = Circuit(16, (Gate("h", (0,)),) * 257)
        circuit.validate()  # Provider circuit limits are separate from local CPU work.
        with self.assertRaisesRegex(ValueError, "work budget"):
            simulate(circuit, 1)
        result = simulate(Circuit(16, ()), 1)
        self.assertEqual(65536, result["stateVectorElements"])
        self.assertEqual({"0" * 16: 1}, result["counts"])

    def test_shots_seed_and_input_type_bounds(self):
        circuit = Circuit(1, ())
        for shots in (True, 0, -1, 65537, 1.0, "4"):
            with self.subTest(shots=shots), self.assertRaises(ValueError):
                simulate(circuit, shots)
        for seed in (False, -1, 1 << 53, 1.0, "4"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                simulate(circuit, 1, seed)
        with self.assertRaises(ValueError):
            simulate({"qubits": 1}, 1)
        self.assertEqual({"0": 65536}, simulate(circuit, 65536)["counts"])

    def test_cancellation_before_and_during_work_never_returns_partial_counts(self):
        with self.assertRaises(QuantumCancelled):
            simulate(Circuit(1, ()), 10, cancelled=lambda: True)
        calls = 0
        def stop_during():
            nonlocal calls
            calls += 1
            return calls >= 4
        with self.assertRaises(QuantumCancelled):
            simulate(Circuit(16, (Gate("h", (0,)),)), 100, cancelled=stop_during)
        self.assertEqual(4, calls)
        with self.assertRaises(ValueError):
            simulate(Circuit(1, ()), 1, cancelled=True)
        with self.assertRaises(ValueError):
            simulate(Circuit(1, ()), 1, cancelled=lambda: 1)


class QuantumPermissionTest(unittest.TestCase):
    circuit = Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))

    def permission(self, **updates):
        values = dict(provider="test-provider", target="test-target", max_shots=100,
                      expires_at_epoch_ms=2000, allow_remote=True)
        values.update(updates)
        return ProviderPermission(**values)

    def test_remote_defaults_deny_and_exact_owner_scope_succeeds(self):
        with self.assertRaises(PermissionError):
            ProviderPermission("test-provider", "test-target").check("test-provider", "test-target", self.circuit, 10, 1000)
        self.permission().check("test-provider", "test-target", self.circuit, 100, 1000)
        self.permission().check("test-provider", "test-target", self.circuit, 1, 1999)

    def test_scope_expiry_limits_and_bool_confusion_rejected(self):
        for provider, target, shots, now in (("other", "test-target", 1, 1000),
                ("test-provider", "other", 1, 1000), ("test-provider", "test-target", 101, 1000),
                ("test-provider", "test-target", 1, 2000)):
            with self.subTest(provider=provider, target=target, shots=shots, now=now), self.assertRaises(PermissionError):
                self.permission().check(provider, target, self.circuit, shots, now)
        for fields in ({"allow_remote": 1}, {"expires_at_epoch_ms": 0}):
            with self.subTest(fields=fields), self.assertRaises(PermissionError):
                self.permission(**fields).check("test-provider", "test-target", self.circuit, 1, 1000)
        for fields in ({"max_shots": True}, {"max_shots": 65537}, {"expires_at_epoch_ms": True},
                       {"provider": "test provider"}, {"target": "bad\nvalue"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.permission(**fields).check("test-provider", "test-target", self.circuit, 1, 1000)
        for shots, now in ((True, 1000), (1, True), (0, 1000), (1, -1)):
            with self.subTest(shots=shots, now=now), self.assertRaises(ValueError):
                self.permission().check("test-provider", "test-target", self.circuit, shots, now)
        with self.assertRaises(ValueError):
            self.permission().check("test-provider", "test-target", Circuit(17, ()), 1, 1000)

    def test_job_key_is_bounded_canonical_uuid_not_provider_job_proof(self):
        key = "01234567-89ab-4def-8123-456789abcdef"
        self.assertEqual(key, validate_job_key(key))
        for invalid in (key.upper(), key.replace("-", ""), "../secret", "x" * 1024, True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_job_key(invalid)


if __name__ == "__main__":
    unittest.main()
