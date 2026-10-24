# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Explicit OriginQ pyqpanda3 adapter; no SDK import or cloud work on import."""
from dataclasses import dataclass, field
from importlib import import_module, metadata
from typing import Any
from urllib.parse import urlsplit

from .quantum import Circuit, ProviderPermission, validate_job_key


PROVIDER = "originq"
SDK_VERSION = "0.4.0"


class OriginQUnavailable(RuntimeError):
    """The documented optional SDK interface is unavailable."""


class OriginQSubmissionError(RuntimeError):
    """Submission may have reached OriginQ; reconcile before resubmitting."""


@dataclass(frozen=True, eq=False)
class OriginQClient:
    """Owner configuration; construction performs no SDK import or network call."""

    api_key: str = field(repr=False, compare=False)
    endpoint: str

    def __post_init__(self):
        if type(self.api_key) is not str or not 1 <= len(self.api_key) <= 4096:
            raise ValueError("An explicit bounded OriginQ API key is required")
        if (type(self.endpoint) is not str or not 1 <= len(self.endpoint) <= 2048 or
                any(ord(char) < 0x21 or ord(char) > 0x7e for char in self.endpoint) or
                "\\" in self.endpoint):
            raise ValueError("An explicit HTTPS OriginQ endpoint is required")
        try:
            url = urlsplit(self.endpoint)
            valid = (url.scheme == "https" and bool(url.hostname) and url.username is None
                     and url.password is None and not url.query and not url.fragment
                     and (url.port is None or 1 <= url.port <= 65535))
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("OriginQ endpoint must use HTTPS without URL credentials, query or fragment")


@dataclass(frozen=True)
class OriginQSubmission:
    """Local correlation metadata and the unpolled official SDK job handle."""

    request_id: str
    target: str
    shots: int
    sdk_version: str
    provider_job_id: str
    job: Any = field(repr=False, compare=False)


def _load_sdk():
    try:
        installed = metadata.version("pyqpanda3")
    except metadata.PackageNotFoundError:
        raise OriginQUnavailable("Optional pyqpanda3==0.4.0 SDK is not installed") from None
    if installed != SDK_VERSION:
        raise OriginQUnavailable("OriginQ adapter requires the documented pyqpanda3==0.4.0 interface")
    try:
        core = import_module("pyqpanda3.core")
        cloud = import_module("pyqpanda3.qcloud")
    except (ImportError, OSError):
        raise OriginQUnavailable("OriginQ SDK core/cloud bindings are unavailable on this platform") from None
    for module, names in ((core, ("QProg", "H", "X", "RX", "RZ", "CNOT", "measure")),
                          (cloud, ("QCloudOptions", "QCloudService"))):
        if any(not callable(getattr(module, name, None)) for name in names):
            raise OriginQUnavailable("OriginQ SDK does not expose the documented interface")
    return core, cloud


def _program(circuit, core):
    program = core.QProg(circuit.qubits)
    for gate in circuit.gates:
        if gate.name == "h":
            operation = core.H(gate.targets[0])
        elif gate.name == "x":
            operation = core.X(gate.targets[0])
        elif gate.name == "rx":
            operation = core.RX(gate.targets[0], gate.angle)
        elif gate.name == "rz":
            operation = core.RZ(gate.targets[0], gate.angle)
        elif gate.name == "cx":
            operation = core.CNOT(gate.targets[0], gate.targets[1])
        else:
            raise ValueError("Unsupported OriginQ gate")
        program << operation
    for index in range(circuit.qubits):
        program << core.measure(index, index)
    return program


def export_program(circuit: Circuit):
    """Build a local QProg with terminal measurements; never create a cloud client."""
    if type(circuit) is not Circuit:
        raise TypeError("A HyperL Circuit is required")
    circuit.validate()
    core, _ = _load_sdk()
    return _program(circuit, core)


def submit(circuit: Circuit, shots: int, *, client, permission: ProviderPermission,
           target: str, request_id: str) -> OriginQSubmission:
    """Submit once through an explicitly configured HTTPS SDK client.

    The caller owns credentials, the approved endpoint and a durable request
    ledger. request_id is local correlation, not provider-side
    idempotency. SDK-internal network behavior is outside this adapter's control.
    """
    if (type(circuit) is not Circuit or type(permission) is not ProviderPermission
            or type(client) is not OriginQClient):
        raise TypeError("HyperL Circuit, ProviderPermission and OriginQClient are required")
    validate_job_key(request_id)
    permission.check(PROVIDER, target, circuit, shots)
    core, cloud = _load_sdk()
    program = _program(circuit, core)
    options = cloud.QCloudOptions()
    # Allow topology mapping; do not silently enable readout correction or optimization.
    options.set_mapping(True)
    options.set_optimization(False)
    options.set_amend(False)
    permission.check(PROVIDER, target, circuit, shots)
    try:
        service = cloud.QCloudService(api_key=client.api_key, url=client.endpoint)
        backend = service.backend(target)
        selected_name = backend.name()
    except Exception:
        raise OriginQUnavailable("OriginQ backend selection failed; no run was submitted by this adapter") from None
    if type(selected_name) is not str or selected_name != target:
        raise OriginQUnavailable("OriginQ selected backend does not match the approved target")
    # Backend selection may take time; recheck expiry before the costly operation.
    permission.check(PROVIDER, target, circuit, shots)
    try:
        job = backend.run(program, shots, options)
        provider_job_id = job.job_id()
        if (type(provider_job_id) is not str or not 1 <= len(provider_job_id) <= 512 or
                any(ord(char) < 0x21 or ord(char) > 0x7e for char in provider_job_id)):
            raise ValueError("Unusable provider job identity")
    except Exception:
        # Provider errors can contain tokens/URLs; do not reproduce their text.
        raise OriginQSubmissionError(
            "OriginQ submission outcome is unknown; reconcile the local request before any retry"
        ) from None
    return OriginQSubmission(request_id, target, shots, SDK_VERSION, provider_job_id, job)
