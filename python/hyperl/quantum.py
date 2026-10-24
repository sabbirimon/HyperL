# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Bounded quantum circuits and an actual local, standard-library simulator.

This is classical state-vector simulation using Python complex numbers (two
binary64 components). It is neither QPU execution nor native 128-bit precision.
ProviderPermission is a caller-owned authorization guard, not remote attestation.
No network, provider SDK, credentials, dynamic code or background work is used.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import math
import random
import re
import time
from typing import Callable
import uuid

MAX_QUBITS = 16
MAX_GATES = 4096
MAX_SHOTS = 65_536
MAX_SIMULATOR_WORK = 16_777_216
MAX_JSON_INTEGER = (1 << 53) - 1
BIT_ORDER = "q0 is least significant; count keys are q[n-1]...q0"
_PROVIDER = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_ARITY = {"h": 1, "x": 1, "rx": 1, "rz": 1, "cx": 2}


def _integer(value: int, minimum: int, maximum: int, description: str) -> None:
    # bool is an int subclass; never treat True as one qubit, shot or timestamp.
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{description} must be an integer in {minimum}..{maximum}")


def _provider_target(provider: str, target: str) -> None:
    if not isinstance(provider, str) or not _PROVIDER.fullmatch(provider):
        raise ValueError("provider must be an explicit bounded lowercase identifier")
    if (not isinstance(target, str) or not 1 <= len(target) <= 512 or
            any(ord(character) < 0x21 or ord(character) > 0x7e for character in target)):
        raise ValueError("target must be an explicit bounded printable identifier")


@dataclass(frozen=True)
class Gate:
    name: str
    targets: tuple[int, ...]
    angle: float | None = None


@dataclass(frozen=True)
class Circuit:
    qubits: int
    gates: tuple[Gate, ...]

    def validate(self) -> None:
        """Validate exact circuit syntax; all qubits are measured at the end.

        For cx, targets are (control, target). Rotation angles are in radians.
        The local simulator applies its separate amplitude-work budget before
        allocating a state vector; provider adapters still share these bounds.
        """
        _integer(self.qubits, 1, MAX_QUBITS, "qubits")
        if type(self.gates) is not tuple or len(self.gates) > MAX_GATES:
            raise ValueError("gates must be a tuple of at most 4096 Gate values")
        for gate in self.gates:
            if not isinstance(gate, Gate) or type(gate.name) is not str or gate.name not in _ARITY:
                raise ValueError("supported gates are h, x, rx, rz and cx")
            if type(gate.targets) is not tuple or len(gate.targets) != _ARITY[gate.name]:
                raise ValueError("gate targets must be a tuple with the exact gate arity")
            for target in gate.targets:
                _integer(target, 0, self.qubits - 1, "gate target")
            if len(set(gate.targets)) != len(gate.targets):
                raise ValueError("gate targets must be distinct")
            if gate.name in ("rx", "rz"):
                if type(gate.angle) not in (int, float):
                    raise ValueError("rotation requires a finite numeric angle in radians")
                try:
                    finite = math.isfinite(gate.angle)
                except (OverflowError, TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ValueError("rotation requires a finite numeric angle in radians")
            elif gate.angle is not None:
                raise ValueError("only rx and rz accept angles")


@dataclass(frozen=True)
class ProviderPermission:
    provider: str
    target: str
    max_shots: int = 4096
    expires_at_epoch_ms: int = 0
    allow_remote: bool = False

    def check(self, provider: str, target: str, circuit: Circuit, shots: int,
              now_ms: int | None = None) -> None:
        """Require owner opt-in, exact provider/target, future expiry and bounds.

        Call immediately before each submission. This object carries no credential
        and does not grant an SDK permission to choose another target or exceed
        an account's limits. Already accepted provider jobs need separate cancel.
        """
        _provider_target(self.provider, self.target)
        _provider_target(provider, target)
        _integer(self.max_shots, 1, MAX_SHOTS, "permission max_shots")
        _integer(self.expires_at_epoch_ms, 0, MAX_JSON_INTEGER, "permission expiry")
        _integer(shots, 1, MAX_SHOTS, "shots")
        if not isinstance(circuit, Circuit):
            raise ValueError("circuit must be a Circuit")
        circuit.validate()
        if now_ms is None:
            now_ms = time.time_ns() // 1_000_000
        _integer(now_ms, 0, MAX_JSON_INTEGER, "current epoch milliseconds")
        if type(self.allow_remote) is not bool or not self.allow_remote:
            raise PermissionError("remote quantum submission is not enabled by the owner")
        if self.provider != provider or self.target != target:
            raise PermissionError("provider or target differs from the approved scope")
        if self.expires_at_epoch_ms <= now_ms:
            raise PermissionError("remote quantum permission has expired or has no expiry")
        if shots > self.max_shots:
            raise PermissionError("requested shots exceed the approved limit")


def validate_job_key(value: str) -> str:
    """Validate a caller's bounded canonical UUID correlation/idempotency key.

    This key alone does not implement provider idempotency or prevent replay.
    Provider adapters must separately verify their actual job identifier/status.
    """
    if type(value) is not str or len(value) != 36:
        raise ValueError("job key must be a canonical UUID")
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError("job key must be a canonical UUID") from error
    if canonical != value:
        raise ValueError("job key must be a canonical UUID")
    return value


class QuantumCancelled(RuntimeError):
    """No completed simulator result is published after owner cancellation."""


def simulate(circuit: Circuit, shots: int, seed: int = 0, *,
             cancelled: Callable[[], bool] | None = None) -> dict:
    """Execute a circuit locally, then draw seeded independent all-qubit shots.

    Counts and probabilities use q0 as the least-significant bit. Sampling uses
    a private Python Random instance, never process-global random state. Results
    are JSON-safe, finite, bounded and explicitly labeled simulation. A supplied
    cancellation callback returns bool; True raises QuantumCancelled without a
    partial result. The callback is checked during gate work and measurement.
    """
    if not isinstance(circuit, Circuit):
        raise ValueError("circuit must be a Circuit")
    circuit.validate()
    _integer(shots, 1, MAX_SHOTS, "shots")
    _integer(seed, 0, MAX_JSON_INTEGER, "seed")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be a callable returning bool")
    dimension = 1 << circuit.qubits
    work = dimension * max(1, len(circuit.gates))
    if work > MAX_SIMULATOR_WORK:
        raise ValueError("local simulator amplitude-work budget exceeded")

    def check_cancelled() -> None:
        if cancelled is not None:
            stop = cancelled()
            if type(stop) is not bool:
                raise ValueError("cancellation callback must return bool")
            if stop:
                raise QuantumCancelled("local quantum simulation cancelled")

    check_cancelled()
    started = time.perf_counter_ns()
    state = [0j] * dimension
    state[0] = 1 + 0j
    inv_sqrt_two = math.sqrt(0.5)
    for gate in circuit.gates:
        check_cancelled()
        if gate.name == "cx":
            control, target = (1 << gate.targets[0]), (1 << gate.targets[1])
            for index in range(dimension):
                if index % 4096 == 0:
                    check_cancelled()
                if index & control and not index & target:
                    partner = index | target
                    state[index], state[partner] = state[partner], state[index]
            continue
        bit = 1 << gate.targets[0]
        if gate.name in ("rx", "rz"):
            half_angle = float(gate.angle) / 2
            cosine, sine = math.cos(half_angle), math.sin(half_angle)
        pair_count = 0
        for base in range(0, dimension, 2 * bit):
            for offset in range(bit):
                if pair_count % 4096 == 0:
                    check_cancelled()
                pair_count += 1
                low, high = base + offset, base + offset + bit
                a, b = state[low], state[high]
                if gate.name == "h":
                    state[low], state[high] = (a + b) * inv_sqrt_two, (a - b) * inv_sqrt_two
                elif gate.name == "x":
                    state[low], state[high] = b, a
                elif gate.name == "rx":
                    state[low], state[high] = cosine * a - 1j * sine * b, -1j * sine * a + cosine * b
                else:  # validated rz
                    state[low], state[high] = complex(cosine, -sine) * a, complex(cosine, sine) * b

    probabilities = []
    for index, amplitude in enumerate(state):
        if index % 4096 == 0:
            check_cancelled()
        probability = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
        if not math.isfinite(probability) or probability < 0:
            raise ArithmeticError("simulator produced a nonfinite probability")
        probabilities.append(probability)
    norm = math.fsum(probabilities)
    if not math.isfinite(norm) or abs(norm - 1) > 1e-9:
        raise ArithmeticError("simulator state normalization check failed")
    # Correct binary64 accumulation drift, not an invalid/non-unitary circuit.
    probabilities = [value / norm for value in probabilities]
    cumulative = []
    running = 0.0
    for index, probability in enumerate(probabilities):
        if index % 4096 == 0:
            check_cancelled()
        running += probability
        cumulative.append(running)
    # Normalize the cumulative sum as a whole. Setting only its last value to 1
    # could make it decrease after a roundoff overshoot, violating bisect's order.
    cumulative = [value / running for value in cumulative]
    cumulative[-1] = 1.0
    generator = random.Random(seed)
    counts = Counter()
    for shot in range(shots):
        if shot % 1024 == 0:
            check_cancelled()
        index = bisect_right(cumulative, generator.random())
        counts[format(index, f"0{circuit.qubits}b")] += 1
    distribution = {}
    for index, probability in enumerate(probabilities):
        if index % 4096 == 0:
            check_cancelled()
        if probability > 0:
            distribution[format(index, f"0{circuit.qubits}b")] = probability
    check_cancelled()
    return {
        "format": "hyperl-quantum-simulator/1",
        "backend": "STDLIB_STATEVECTOR_SIMULATOR",
        "executionKind": "simulator",
        "hardwareExecution": False,
        "precision": "complex with binary64 real and imaginary components; not native 128-bit precision",
        "bitOrder": BIT_ORDER,
        "qubits": circuit.qubits,
        "gates": len(circuit.gates),
        "shots": shots,
        "seed": seed,
        "counts": dict(sorted(counts.items())),
        "probabilities": distribution,
        "stateNormBeforeNormalization": norm,
        "amplitudeWorkBudget": MAX_SIMULATOR_WORK,
        "estimatedAmplitudeWork": work,
        "stateVectorElements": dimension,
        "elapsedMs": (time.perf_counter_ns() - started) / 1_000_000,
    }
