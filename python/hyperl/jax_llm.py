# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Explicit JAX f32 decoder with tensor-parallel weights and sharded KV ownership.

The requested backend is mandatory. CPU tests are not TPU qualification. Importing
this module does not import JAX, initialize a distributed runtime or install drivers.
"""
from dataclasses import dataclass
import hashlib
import json
import itertools
import shutil
from pathlib import Path
import threading
import uuid
from .llm import _integer, file_digest


@dataclass(frozen=True)
class DecoderConfig:
    vocabulary: int
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    context: int = 2048
    rms_epsilon: float = 1e-5
    rope_theta: float = 10000.0

    def validate(self):
        for value in (self.vocabulary, self.hidden, self.intermediate, self.layers, self.heads, self.kv_heads, self.context):
            _integer(value, 1, 131072)
        if self.layers > 256 or self.hidden % self.heads or self.heads % self.kv_heads or (self.hidden // self.heads) % 2:
            raise ValueError("invalid decoder/GQA/rotary dimensions")
        import math
        if not math.isfinite(self.rms_epsilon) or not 0 < self.rms_epsilon < 1 or not math.isfinite(self.rope_theta) or not 1 < self.rope_theta <= 1e9:
            raise ValueError("invalid numerical decoder contract")

    def shapes(self):
        self.validate()
        h = self.hidden; k = self.kv_heads * (h // self.heads); m = self.intermediate
        shapes = {"embedding": (self.vocabulary, h), "norm": (h,), "head": (h, self.vocabulary)}
        for i in range(self.layers):
            shapes.update({f"{i}.attention_norm": (h,), f"{i}.ffn_norm": (h,), f"{i}.q": (h, h), f"{i}.k": (h, k),
                           f"{i}.v": (h, k), f"{i}.o": (h, h), f"{i}.gate": (h, m), f"{i}.up": (h, m), f"{i}.down": (m, h)})
        return shapes


class JaxDecoder:
    """Single-sequence Llama-style dense/GQA decoder; explicit contiguous f32 checkpoint.

    Embedding/vocabulary, attention heads and FFN dimensions partition over the
    named tp mesh. Norms replicate. XLA supplies reductions for row-parallel output
    projections. KV partitions by KV head. No CPU fallback, quantization, MoE or
    implicit precision change. Configure multi-controller JAX before constructing.
    """
    def __init__(self, config, weights, *, backend, device_ids, budget_bytes=4 * 1024**3):
        config.validate()
        if backend not in {"cpu", "gpu", "tpu"}:
            raise ValueError("select an explicit JAX backend")
        _integer(budget_bytes, 1024, 1024**4)
        ids = tuple(device_ids)
        if not 1 <= len(ids) <= 256 or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("explicit unique global JAX device IDs required")
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        try:
            available = {device.id: device for device in jax.devices(backend)}
        except RuntimeError as error:
            raise RuntimeError("requested JAX backend unavailable; no fallback") from error
        if any(i not in available for i in ids) or any(available[i].platform != backend for i in ids):
            raise RuntimeError("requested JAX device unavailable; no fallback")
        count = len(ids)
        if any(size % count for size in (config.heads, config.kv_heads, config.intermediate, config.vocabulary)):
            raise ValueError("tensor-parallel mesh must divide query/KV heads, FFN and vocabulary; no replicated substitute")
        shapes = config.shapes()
        if set(weights) != set(shapes):
            raise ValueError("checkpoint tensor names do not match decoder contract")
        # Conservative application admission, separate from XLA process/driver arenas.
        weight_bytes = sum(int(np.prod(shape)) * 4 for shape in shapes.values())
        cache_bytes = config.layers * config.context * config.kv_heads * (config.hidden // config.heads) * 4 * 2
        if weight_bytes * 2 + cache_bytes * 2 + config.context * config.heads * 16 > budget_bytes:
            raise ValueError("model, host staging and KV exceed configured application budget")
        self.jax = jax; self.jnp = jnp; self.np = np; self.config = config; self.backend = backend
        self.mesh = Mesh(np.array([available[i] for i in ids]), ("tp",))
        self.replicated = NamedSharding(self.mesh, P())
        self.cache_sharding = NamedSharding(self.mesh, P(None, None, "tp", None))
        self.shardings = {}
        self.weights = {}
        identity = hashlib.sha256(json.dumps(config.__dict__, sort_keys=True).encode())
        for name in sorted(shapes):
            array = weights[name]
            if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != shapes[name] or not array.flags.c_contiguous:
                raise ValueError("checkpoint requires exact contiguous f32 tensor: " + name)
            flat = array.reshape(-1)
            for start in range(0, flat.size, 262144):
                chunk = flat[start:start + 262144]
                if not np.isfinite(chunk).all():
                    raise ValueError("nonfinite checkpoint tensor")
                identity.update(chunk.tobytes())
            axis = 0 if name == "embedding" or name.endswith((".o", ".down")) else 1 if name == "head" or name.endswith((".q", ".k", ".v", ".gate", ".up")) else None
            spec = P("tp", None) if axis == 0 else P(None, "tp") if axis == 1 else P()
            sharding = NamedSharding(self.mesh, spec)
            self.shardings[name] = sharding
            self.weights[name] = jax.make_array_from_callback(array.shape, sharding, lambda index, a=array: np.array(a[index], copy=True))
        self.identity = identity.hexdigest()
        self.id = str(uuid.uuid4())
        self._gate = threading.Lock()
        self._compiled = jax.jit(self._step, in_shardings=(self.shardings, self.cache_sharding, self.cache_sharding, self.replicated, self.replicated),
                                 out_shardings=(self.replicated, self.cache_sharding, self.cache_sharding))
        self._reset()

    @classmethod
    def load(cls, directory, *, manifest_sha256, backend, device_ids, budget_bytes=4 * 1024**3):
        """Read owner-selected .npy tensors via an exact trusted manifest; no pickle."""
        import numpy as np
        root = Path(directory)
        path = root / "manifest.json"
        if not root.is_absolute() or root.is_symlink() or file_digest(path, max_bytes=256 * 1024) != manifest_sha256:
            raise ValueError("checkpoint manifest identity mismatch")
        manifest = json.loads(path.read_bytes())
        if manifest.get("format") != "hyperl-llama-f32/1":
            raise ValueError("unsupported model checkpoint")
        config = DecoderConfig(**manifest["config"]); shapes = config.shapes()
        if set(manifest["tensors"]) != set(shapes):
            raise ValueError("checkpoint tensor directory mismatch")
        # Admit from shapes before mapping any payload.
        total = sum(int(np.prod(s)) * 4 for s in shapes.values())
        if total * 2 > budget_bytes:
            raise ValueError("checkpoint exceeds application quota")
        tensors = {}
        for name, info in manifest["tensors"].items():
            filename = info["file"]
            if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npy"):
                raise ValueError("checkpoint paths must be local basenames")
            target = root / filename
            if file_digest(target, max_bytes=budget_bytes) != info["sha256"]:
                raise ValueError("checkpoint tensor digest mismatch")
            tensors[name] = np.load(target, mmap_mode="r", allow_pickle=False, max_header_size=8192)
        return cls(config, tensors, backend=backend, device_ids=device_ids, budget_bytes=budget_bytes)

    def _reset(self):
        cfg = self.config
        shape = (cfg.layers, cfg.context, cfg.kv_heads, cfg.hidden // cfg.heads)
        self.keys = self.jax.make_array_from_callback(shape, self.cache_sharding, lambda index: self.np.zeros(tuple(len(range(*s.indices(n))) for s, n in zip(index, shape)), self.np.float32))
        self.values = self.jax.make_array_from_callback(shape, self.cache_sharding, lambda index: self.np.zeros(tuple(len(range(*s.indices(n))) for s, n in zip(index, shape)), self.np.float32))
        self.position = 0; self.logits = None; self.failed = False

    def _step(self, weights, keys, values, token, position):
        jnp = self.jnp; cfg = self.config; d = cfg.hidden // cfg.heads
        def matmul(a, b):
            return jnp.matmul(a, b, precision=self.jax.lax.Precision.HIGHEST)
        def norm(x, scale):
            return x * self.jax.lax.rsqrt(jnp.mean(x * x) + cfg.rms_epsilon) * scale
        def rotary(x):
            frequencies = position / (cfg.rope_theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))
            angles = jnp.concatenate((frequencies, frequencies))
            rotated = jnp.concatenate((-x[..., d // 2:], x[..., :d // 2]), axis=-1)
            return x * jnp.cos(angles) + rotated * jnp.sin(angles)
        x = weights["embedding"][token]
        for i in range(cfg.layers):
            prefix = str(i) + "."
            a = norm(x, weights[prefix + "attention_norm"])
            q = rotary((matmul(a, weights[prefix + "q"])).reshape(cfg.heads, d))
            k = rotary((matmul(a, weights[prefix + "k"])).reshape(cfg.kv_heads, d))
            v = (matmul(a, weights[prefix + "v"])).reshape(cfg.kv_heads, d)
            keys = keys.at[i, position].set(k); values = values.at[i, position].set(v)
            all_k = jnp.repeat(keys[i], cfg.heads // cfg.kv_heads, axis=1)
            all_v = jnp.repeat(values[i], cfg.heads // cfg.kv_heads, axis=1)
            scores = jnp.einsum("hd,thd->ht", q, all_k, precision=self.jax.lax.Precision.HIGHEST) / jnp.sqrt(jnp.float32(d))
            scores = jnp.where(jnp.arange(cfg.context)[None, :] <= position, scores, -jnp.inf)
            probabilities = self.jax.nn.softmax(scores, axis=-1)
            attention = jnp.einsum("ht,thd->hd", probabilities, all_v, precision=self.jax.lax.Precision.HIGHEST).reshape(cfg.hidden)
            x = x + matmul(attention, weights[prefix + "o"])
            a = norm(x, weights[prefix + "ffn_norm"])
            x = x + matmul(self.jax.nn.silu(matmul(a, weights[prefix + "gate"])) * matmul(a, weights[prefix + "up"]), weights[prefix + "down"])
        return matmul(norm(x, weights["norm"]), weights["head"]), keys, values

    def _token(self, token, cancelled):
        _integer(token, 0, self.config.vocabulary - 1)
        if self.failed or self.position >= self.config.context:
            raise ValueError("decoder failed or KV context exhausted; reset explicitly")
        if cancelled():
            self.failed = True
            raise InterruptedError("decoder cancelled before dispatch")
        logits, keys, values = self._compiled(self.weights, self.keys, self.values,
            self.jax.device_put(self.np.int32(token), self.replicated), self.jax.device_put(self.np.int32(self.position), self.replicated))
        logits.block_until_ready()
        if cancelled():
            self.failed = True
            raise InterruptedError("decoder cancelled after device completion; no state published")
        if not bool(self.np.isfinite(self.np.asarray(logits)).all()):
            self.failed = True
            raise ValueError("nonfinite decoder result")
        self.logits = logits; self.keys = keys; self.values = values; self.position += 1

    def generate(self, tokens, *, max_tokens=16, reset=True, cancelled=lambda: False):
        prompt = tuple(itertools.islice(tokens, self.config.context + 1))
        _integer(max_tokens, 1, 4096)
        if (not prompt and (reset or self.logits is None)) or any(type(t) is not int or not 0 <= t < self.config.vocabulary for t in prompt):
            raise ValueError("explicit bounded token IDs required; tokenizer is model-specific")
        if len(prompt) + max_tokens + (0 if reset else self.position) > self.config.context:
            raise ValueError("prompt and generation exceed KV reservation")
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("decoder session busy")
        try:
            if reset:
                self._reset()
            for token in prompt:
                self._token(token, cancelled)
            result = []
            for _ in range(max_tokens):
                next_token = int(self.np.asarray(self.jnp.argmax(self.logits)))
                self._token(next_token, cancelled)
                result.append(next_token)
            return dict(format="hyperl-jax-llm-result/1", session=self.id, model_sha256=self.identity, backend=self.backend,
                        tokens=result, kv_position=self.position, devices=[dict(id=d.id, platform=d.platform, process=d.process_index) for d in self.mesh.devices.flat],
                        sharded_weights=sum(not w.is_fully_replicated for w in self.weights.values()),
                        kv_spec=str(self.cache_sharding.spec), physical_tpu_tested=self.backend == "tpu")
        except BaseException:
            self.failed = True
            raise
        finally:
            self._gate.release()

    def reset(self):
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("decoder session busy")
        try:
            self._reset()
        finally:
            self._gate.release()

    def save_checkpoint(self, directory, *, max_bytes, cancelled=lambda: False):
        """Commit a complete token-boundary KV/logits snapshot to a new private directory."""
        _integer(max_bytes, 1, 1024**4)
        if self.jax.process_count() != 1:
            raise ValueError("multi-controller checkpoint publication needs a separate coordinated manifest")
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("cannot checkpoint an active token")
        root = Path(directory); created = False
        try:
            if self.failed or self.position < 1 or self.logits is None or not root.is_absolute() or root.is_symlink():
                raise ValueError("checkpoint needs a valid token boundary and new absolute directory")
            payload = self.config.layers * self.position * self.config.kv_heads * (self.config.hidden // self.config.heads) * 8 + self.config.vocabulary * 4
            if payload + self.config.layers * 1024 + 16384 > max_bytes:
                raise ValueError("checkpoint staging quota insufficient")
            root.mkdir(mode=0o700); created = True
            files = {}; written = 0
            for name, array in itertools.chain([("logits.npy", self.logits)], ((f"{kind}-{i}.npy", data[i, :self.position]) for i in range(self.config.layers) for kind, data in (("keys", self.keys), ("values", self.values)))):
                if cancelled():
                    raise InterruptedError("checkpoint cancelled")
                path = root / name
                with path.open("xb") as output:
                    self.np.save(output, self.np.asarray(array), allow_pickle=False)
                written += path.stat().st_size
                if written > max_bytes:
                    raise ValueError("checkpoint quota exceeded")
                files[name] = file_digest(path, max_bytes=max_bytes, cancelled=cancelled)
            manifest = dict(format="hyperl-llama-kv/1", model_sha256=self.identity, config=self.config.__dict__,
                backend=self.backend, dtype="f32", position=self.position, files=files)
            encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            if len(encoded) > 256 * 1024 or written + len(encoded) > max_bytes or cancelled():
                raise InterruptedError("checkpoint cancelled or metadata quota exhausted")
            with (root / "manifest.json").open("xb") as output:
                output.write(encoded)
            return dict(manifest_sha256=hashlib.sha256(encoded).hexdigest(), position=self.position, bytes=written + len(encoded))
        except BaseException:
            if created:
                shutil.rmtree(root)
            raise
        finally:
            self._gate.release()

    def restore_checkpoint(self, directory, *, manifest_sha256, max_bytes, cancelled=lambda: False):
        _integer(max_bytes, 1, 1024**4)
        if self.jax.process_count() != 1:
            raise ValueError("multi-controller restore requires a coordinated checkpoint")
        if not self._gate.acquire(blocking=False):
            raise RuntimeError("cannot replace active KV state")
        try:
            root = Path(directory); path = root / "manifest.json"
            if not root.is_absolute() or root.is_symlink() or file_digest(path, max_bytes=256 * 1024) != manifest_sha256:
                raise ValueError("checkpoint manifest identity mismatch")
            meta = json.loads(path.read_bytes()); position = _integer(meta["position"], 1, self.config.context)
            if (meta.get("format") != "hyperl-llama-kv/1" or meta.get("model_sha256") != self.identity
                    or meta.get("config") != self.config.__dict__ or meta.get("backend") != self.backend or meta.get("dtype") != "f32"):
                raise ValueError("checkpoint model/precision/backend contract mismatch")
            expected = {"logits.npy": (self.config.vocabulary,)}
            for i in range(self.config.layers):
                for kind in ("keys", "values"):
                    expected[f"{kind}-{i}.npy"] = (position, self.config.kv_heads, self.config.hidden // self.config.heads)
            if set(meta["files"]) != set(expected):
                raise ValueError("checkpoint component directory mismatch")
            total = path.stat().st_size; arrays = {}
            for name, shape in expected.items():
                target = root / name; total += target.stat().st_size
                if total > max_bytes or file_digest(target, max_bytes=max_bytes, cancelled=cancelled) != meta["files"][name]:
                    raise ValueError("checkpoint component quota/digest mismatch")
                array = self.np.load(target, mmap_mode="r", allow_pickle=False, max_header_size=8192)
                if array.dtype != self.np.float32 or array.shape != shape or not array.flags.c_contiguous or not self.np.isfinite(array).all():
                    raise ValueError("checkpoint component dtype/shape/nonfinite mismatch")
                arrays[name] = array
            if cancelled():
                raise InterruptedError("checkpoint restore cancelled")
            self._reset()
            for i in range(self.config.layers):
                self.keys = self.keys.at[i, :position].set(arrays[f"keys-{i}.npy"])
                self.values = self.values.at[i, :position].set(arrays[f"values-{i}.npy"])
            self.logits = self.jax.device_put(arrays["logits.npy"], self.replicated)
            self.keys.block_until_ready(); self.values.block_until_ready(); self.logits.block_until_ready()
            if cancelled():
                raise InterruptedError("checkpoint restore cancelled after device transfer")
            self.position = position; self.failed = False
        except BaseException:
            self.failed = True
            raise
        finally:
            self._gate.release()

    def residency(self):
        return dict(model_sha256=self.identity, backend=self.backend, weights={name: dict(shape=list(value.shape), spec=str(value.sharding.spec),
                    local_shards=[dict(device=s.device.id, shape=list(s.data.shape)) for s in value.addressable_shards]) for name, value in self.weights.items()},
                    kv=dict(shape=list(self.keys.shape), spec=str(self.keys.sharding.spec), position=self.position))


@dataclass(frozen=True)
class JaxSettings:
    directory: str
    manifest_sha256: str
    backend: str
    device_ids: tuple
    budget_bytes: int = 4 * 1024**3
    request_seconds: int = 120
    checkpoint_directory: str | None = None
    checkpoint_sha256: str | None = None

    def validate(self):
        from .llm import _DIGEST
        if not Path(self.directory).is_absolute() or not _DIGEST.fullmatch(self.manifest_sha256) or self.backend not in {"cpu", "gpu", "tpu"}:
            raise ValueError("explicit checkpoint identity and JAX backend required")
        if not 1 <= len(self.device_ids) <= 256 or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("bounded unique JAX device IDs required")
        if (self.checkpoint_directory is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint directory and trusted digest must be supplied together")
        _integer(self.budget_bytes, 1024, 1024**4); _integer(self.request_seconds, 1, 600)


class JaxModelSession:
    def __init__(self, settings, *, cancelled=lambda: False):
        settings.validate(); self.settings = settings; self.closed = False
        if cancelled():
            raise InterruptedError("JAX model startup cancelled")
        self.decoder = JaxDecoder.load(settings.directory, manifest_sha256=settings.manifest_sha256, backend=settings.backend,
            device_ids=settings.device_ids, budget_bytes=settings.budget_bytes)
        if cancelled():
            self.close(); raise InterruptedError("JAX model startup cancelled after device transfer")

    def complete(self, prompt, *, max_tokens=32, seed=0, cancelled=lambda: False):
        import time
        if self.closed or type(seed) is not int or seed != 0:
            raise ValueError("JAX session closed or unsupported sampling; this contract is greedy seed=0")
        started = time.monotonic()
        if self.settings.checkpoint_directory is not None:
            self.decoder.restore_checkpoint(self.settings.checkpoint_directory, manifest_sha256=self.settings.checkpoint_sha256,
                max_bytes=self.settings.budget_bytes, cancelled=cancelled)
        result = self.decoder.generate(prompt, max_tokens=max_tokens, reset=self.settings.checkpoint_directory is None, cancelled=lambda: cancelled() or time.monotonic() - started > self.settings.request_seconds)
        result["checkpoint_manifest_sha256"] = self.settings.manifest_sha256
        result["request_seconds"] = time.monotonic() - started
        return result

    def close(self):
        self.closed = True
        for array in (*self.decoder.weights.values(), self.decoder.keys, self.decoder.values):
            array.delete()
        if self.decoder.logits is not None:
            self.decoder.logits.delete()
        # JAX/XLA may retain allocator arenas; this is logical buffer release, not
        # a hardware interruption or process-RSS reclamation guarantee.
