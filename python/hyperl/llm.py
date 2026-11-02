# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Explicit native llama.cpp model execution and real layer/KV sharding.

No executable/model downloads, shell, public RPC or backend substitution. The
upstream proof-of-concept RPC protocol belongs behind owner-managed isolation.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import secrets
import tempfile
import socket
import subprocess
import threading
import time
import uuid

LLAMA_REVISION = "4df29be4f4c3673f428170fda944a5b19f743bb8"
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_ENDPOINT = re.compile(r"127\.0\.0\.1:([0-9]{1,5})\Z")


def file_digest(path, *, max_bytes=16 * 1024**4, cancelled=lambda: False):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= max_bytes:
        raise ValueError("explicit bounded regular absolute artifact path required")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while data := stream.read(4 * 1024 * 1024):
            if cancelled():
                raise InterruptedError("artifact verification cancelled")
            total += len(data)
            if total > max_bytes:
                raise ValueError("artifact grew beyond quota")
            digest.update(data)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or total != after.st_size:
            raise ValueError("artifact changed during verification")
    return digest.hexdigest()


def _verify(path, digest, maximum):
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or file_digest(path, max_bytes=maximum) != digest:
        raise ValueError("artifact identity/digest mismatch")


def _integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("integer outside supported bounds")
    return value


def _endpoint(value):
    match = _ENDPOINT.fullmatch(value) if isinstance(value, str) else None
    if not match or not 1 <= int(match[1]) <= 65535:
        raise ValueError("raw native RPC must use explicit numeric IPv4 loopback; use an authenticated tunnel for remote hosts")
    return value


def _port():
    with socket.socket() as selected:
        selected.bind(("127.0.0.1", 0))
        return selected.getsockname()[1]


class _Process:
    def __init__(self, command, *, rpc_debug=False):
        self.layers = {}
        self.buffers = {}
        self.graph_calls = 0
        self.recompute_calls = 0
        self.kv_layers = {}
        self.kv_buffers = {}
        self._lock = threading.Lock()
        # Native log content can include prompts. Parse counters, then discard it.
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("LLAMA_ARG_", "GGML_RPC_", "HF_")) and key not in {"GGML_BACKEND_PATH"}}
        if rpc_debug:
            environment["GGML_RPC_DEBUG"] = "1"
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=environment, start_new_session=os.name != "nt")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        stream = self.process.stdout
        while data := stream.readline(8193):
            if len(data) > 8192:
                continue
            line = data.decode("utf-8", errors="replace")
            kv_layer = re.search(r"layer\s+(\d+): dev = (\S+)", line)
            kv_buffer = re.search(r"(RPC\d+\[[^\]]+\]|CPU)\s+KV buffer size =\s+([0-9.]+) MiB", line)
            layer = re.search(r"layer\s+(\d+) assigned to device (\S+)", line)
            buffer = re.search(r"(RPC\d+\[[^\]]+\]|CPU(?:_Mapped)?)\s+model buffer size =\s+([0-9.]+) MiB", line)
            with self._lock:
                if kv_layer:
                    self.kv_layers[int(kv_layer[1])] = kv_layer[2].rstrip(",")
                if kv_buffer:
                    self.kv_buffers[kv_buffer[1]] = float(kv_buffer[2])
                if layer:
                    self.layers[int(layer[1])] = layer[2].rstrip(",")
                if buffer:
                    self.buffers[buffer[1]] = float(buffer[2])
                if "[graph_compute]" in line:
                    self.graph_calls += 1
                if "[graph_recompute]" in line:
                    self.recompute_calls += 1

    def evidence(self):
        with self._lock:
            return dict(process_id=self.process.pid, assigned_layers=dict(self.layers), model_buffers_mib=dict(self.buffers), kv_layers=dict(self.kv_layers), kv_buffers_mib=dict(self.kv_buffers),
                        graph_calls=self.graph_calls, recompute_calls=self.recompute_calls)

    def close(self):
        if self.process.poll() is None:
            if os.name == "nt":
                self.process.terminate()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.process.stdout.close()


@dataclass(frozen=True)
class LlamaSettings:
    executable: str
    executable_sha256: str
    model: str
    model_sha256: str
    backend: str = "CPU"  # CPU, GPU, or RPC. TPU uses the separate explicit JAX adapter.
    rpc: tuple = ()
    devices: tuple = ()
    proportions: tuple = ()
    split_mode: str = "layer"
    context: int = 2048
    threads: int = 2
    max_model_bytes: int = 32 * 1024**3
    startup_seconds: int = 120
    request_seconds: int = 120
    load_mode: str = "mmap"
    upstream_revision: str = LLAMA_REVISION

    def validate(self):
        if self.upstream_revision != LLAMA_REVISION:
            raise ValueError("unqualified native revision; use the pinned source or qualify a new adapter revision")
        _integer(self.context, 64, 131072); _integer(self.threads, 1, 256)
        _integer(self.max_model_bytes, 1, 16 * 1024**4)
        _integer(self.startup_seconds, 1, 600); _integer(self.request_seconds, 1, 600)
        if self.backend not in {"CPU", "GPU", "RPC"} or self.split_mode != "layer" or self.load_mode not in {"none", "mmap"}:
            raise ValueError("unsupported native backend/split/load contract")
        if len(self.devices) > 16 or any(not isinstance(d, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:.\[\]-]{0,127}", d) for d in self.devices):
            raise ValueError("invalid explicit device selection")
        if self.backend == "CPU" and (self.rpc or self.devices or self.proportions):
            raise ValueError("CPU execution cannot hide configured accelerator work")
        if self.backend == "GPU" and (not self.devices or self.rpc):
            raise ValueError("GPU backend requires exact local device names")
        if self.backend == "RPC":
            if not 1 <= len(self.rpc) <= 16 or len(set(self.rpc)) != len(self.rpc):
                raise ValueError("one bounded native cell needs 1..16 unique approved RPC routes")
            for endpoint in self.rpc:
                _endpoint(endpoint)
            if self.devices:
                raise ValueError("RPC cell selects exactly device zero per configured route")
            if self.split_mode != "layer":
                raise ValueError("RPC row/tensor collectives are not qualified; use layer/KV sharding")
        count = len(self.rpc) if self.backend == "RPC" else len(self.devices)
        if self.proportions and (len(self.proportions) != count or any(type(p) not in (int, float) or not math.isfinite(p) or p <= 0 for p in self.proportions)):
            raise ValueError("positive finite per-device split proportions required")


class RpcWorker:
    """Foreground native worker, loopback only, explicit device, no inferred GPU fallback."""
    def __init__(self, executable, sha256, *, port, device="CPU", threads=2):
        _verify(executable, sha256, 512 * 1024 * 1024)
        _integer(port, 1, 65535); _integer(threads, 1, 256)
        if not isinstance(device, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", device):
            raise ValueError("exact native device required")
        self.endpoint = f"127.0.0.1:{port}"
        self.native = _Process([executable, "--host", "127.0.0.1", "--port", str(port), "--device", device, "--threads", str(threads)], rpc_debug=True)
        # Socket readiness is not device qualification; the client handshake checks the route.
        deadline = time.monotonic() + 15
        try:
            while True:
                if self.native.process.poll() is not None:
                    raise RuntimeError("configured native RPC worker failed; no fallback")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native RPC worker startup deadline")
                    time.sleep(.05)
        except BaseException:
            self.close()
            raise

    def evidence(self):
        return self.native.evidence()

    def close(self):
        self.native.close()


class LlamaSession:
    """One explicitly configured model process with bounded context and deterministic completion.

    KV and weights stay in the selected native backend. Requests are serialized;
    cancellation kills this owned process. Remote RPC cleanup can remain uncertain.
    """
    def __init__(self, settings, *, cancelled=lambda: False):
        settings.validate()
        self.settings = settings
        _verify(settings.executable, settings.executable_sha256, 512 * 1024 * 1024)
        _verify(settings.model, settings.model_sha256, settings.max_model_bytes)
        self.id = str(uuid.uuid4())
        self.port = _port()
        self._gate = threading.Lock()
        self.closed = False
        command = [settings.executable, "--model", settings.model, "--host", "127.0.0.1", "--port", str(self.port),
                   "--ctx-size", str(settings.context), "--threads", str(settings.threads), "--threads-batch", str(settings.threads),
                   "--parallel", "1", "--batch-size", "128", "--ubatch-size", "128", "--offline", "--fit", "off",
                   "--no-context-shift", "--no-warmup", "--no-webui", "--no-ui-mcp-proxy", "--verbosity", "5",
                   "--cache-type-k", "f32", "--cache-type-v", "f32", "--flash-attn", "off", "--no-repack", "--load-mode", settings.load_mode]
        if settings.backend == "CPU":
            command += ["--device", "none", "--n-gpu-layers", "0"]
        else:
            devices = [f"RPC{i}" for i in range(len(settings.rpc))] if settings.backend == "RPC" else list(settings.devices)
            if settings.backend == "RPC":
                probe = subprocess.run([settings.executable, "--offline", "--rpc", ",".join(settings.rpc), "--list-devices"],
                    capture_output=True, text=True, timeout=30,
                    env={k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_ARG_", "GGML_RPC_", "HF_")) and k != "GGML_BACKEND_PATH"})
                listed = re.findall(r"^\s+(RPC\d+): (127\.0\.0\.1:\d+) ", probe.stdout, re.MULTILINE)
                if probe.returncode != 0 or len(probe.stdout) + len(probe.stderr) > 262144 or listed != list(zip(devices, settings.rpc)):
                    raise RuntimeError("RPC route/device inventory mismatch; expose one exact device per route")
                command += ["--rpc", ",".join(settings.rpc)]
            command += ["--device", ",".join(devices), "--n-gpu-layers", "999", "--split-mode", settings.split_mode]
            if settings.proportions:
                command += ["--tensor-split", ",".join(str(p) for p in settings.proportions)]
        self._private = tempfile.TemporaryDirectory(prefix="hyperl-model-session-")
        self._api_key = secrets.token_urlsafe(32)
        key_file = Path(self._private.name) / "api-key"
        key_file.write_text(self._api_key + "\n"); key_file.chmod(0o600)
        command += ["--api-key-file", str(key_file)]
        try:
            self.native = _Process(command)
        except BaseException:
            self._private.cleanup(); raise
        started = time.monotonic()
        try:
            while True:
                if cancelled():
                    raise InterruptedError("native model startup cancelled")
                if self.native.process.poll() is not None:
                    raise RuntimeError("configured native model failed to start; inspect the pinned runtime/configuration")
                try:
                    if self._http("GET", "/health", None, timeout=1).get("status") == "ok":
                        break
                except (OSError, ValueError, RuntimeError):
                    pass
                if time.monotonic() - started > settings.startup_seconds:
                    raise TimeoutError("native model startup deadline")
                time.sleep(.05)
            if settings.backend != "CPU":
                evidence = self.native.evidence()
                required = {f"RPC{i}" for i in range(len(settings.rpc))} if settings.backend == "RPC" else set(settings.devices)
                observed = set(evidence["assigned_layers"].values())
                if not required <= observed:
                    raise RuntimeError("requested model shards were not assigned to every configured device; expected=" + str(sorted(required)) + "; observed=" + str(sorted(observed)))
            self.startup_seconds = time.monotonic() - started
        except BaseException:
            self.close()
            raise

    def _http(self, method, path, body, *, timeout):
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            data = None if body is None else json.dumps(body, allow_nan=False).encode()
            client.request(method, path, data, {"Content-Type": "application/json", "Authorization": "Bearer " + self._api_key})
            response = client.getresponse()
            value = response.read(4 * 1024 * 1024 + 1)
            if response.status != 200 or len(value) > 4 * 1024 * 1024:
                raise RuntimeError("native request rejected or response exceeded its bound")
            return json.loads(value)
        finally:
            client.close()

    def complete(self, prompt, *, max_tokens=32, seed=0, cancelled=lambda: False):
        if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 64 * 1024:
            raise ValueError("prompt must contain 1..65536 UTF-8 bytes")
        _integer(max_tokens, 1, 4096); _integer(seed, 0, 2**31 - 1)
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("model session busy; no unbounded request queue")
        stop = threading.Event()
        reason = []
        started = time.monotonic()
        def watchdog():
            while not stop.wait(.05):
                if cancelled() or time.monotonic() - started > self.settings.request_seconds:
                    reason.append("cancelled or deadline exceeded")
                    self.close()
                    break
        watcher = threading.Thread(target=watchdog, daemon=True)
        try:
            if self.closed or cancelled():
                raise InterruptedError("model session closed/cancelled")
            watcher.start()
            tokens = self._http("POST", "/tokenize", dict(content=prompt, add_special=True), timeout=10).get("tokens")
            if not isinstance(tokens, list) or len(tokens) + max_tokens > self.settings.context:
                raise ValueError("prompt plus output exceeds reserved context; context shifting is disabled")
            result = self._http("POST", "/completion", dict(prompt=prompt, n_predict=max_tokens, temperature=0, seed=seed,
                                cache_prompt=False, return_tokens=True, n_probs=0, stream=False), timeout=self.settings.request_seconds)
            if reason or cancelled():
                raise InterruptedError("native generation cancelled; no result published")
            output = result.get("tokens")
            if (not isinstance(output, list) or len(output) > max_tokens + 4 or any(type(t) is not int or t < 0 for t in output)
                    or not isinstance(result.get("content"), str)):
                raise RuntimeError("native result violates the completion contract")
            return dict(format="hyperl-llm-result/1", session=self.id, backend=self.settings.backend, model_sha256=self.settings.model_sha256,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), tokens=output, text=result["content"],
                        probabilities=result.get("completion_probabilities", result.get("probs", [])),
                        startup_seconds=self.startup_seconds, request_seconds=time.monotonic() - started, native=self.native.evidence())
        except (OSError, RuntimeError) as error:
            if reason or cancelled():
                raise InterruptedError("native generation cancelled; remote completion can be uncertain") from error
            raise
        finally:
            stop.set()
            if watcher.ident is not None:
                watcher.join(timeout=6)
            self._gate.release()

    def close(self):
        self.closed = True
        self.native.close()
        self._private.cleanup()
