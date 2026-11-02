# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Streaming immutable model artifacts and explainable residency plans.

No model deserialization, pickle execution, remote downloads, quantization or LLM
inference. GPU/TPU placements are plans until a qualified model backend stages them.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import itertools
import math
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import uuid


class ModelStore:
    """Content-addressed disk chunks with a transactional indexed manifest.

    A private owner-selected root stores opaque weight bytes. Import/export are
    streaming, cancellable and quota bounded. File manifests publish after full
    integrity verification; failed imports may leave reusable unreferenced blobs.
    SQLite commits metadata, not a cross-filesystem fsync/durable checkpoint claim.
    """
    def __init__(self, root, chunk_bytes=4 * 1024 * 1024):
        if not isinstance(chunk_bytes, int) or not 4096 <= chunk_bytes <= 64 * 1024 * 1024:
            raise ValueError("chunk size must be 4 KiB..64 MiB")
        self.root = Path(root)
        if not self.root.is_absolute() or self.root.is_symlink():
            raise ValueError("explicit absolute private storage root required")
        self.root.mkdir(mode=0o700, parents=False, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(mode=0o700, exist_ok=True)
        if self.blobs.is_symlink():
            raise ValueError("blob directory cannot be a symlink")
        self.chunk_bytes = chunk_bytes
        self.db = sqlite3.connect(self.root / "manifest.sqlite3")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS artifacts (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, chunks INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS chunks (artifact TEXT NOT NULL REFERENCES artifacts(id),
                ordinal INTEGER NOT NULL, digest TEXT NOT NULL, bytes INTEGER NOT NULL,
                PRIMARY KEY(artifact,ordinal));
        """)

    def close(self):
        self.db.close()

    def ingest(self, source, *, name=None, max_bytes, cancelled=lambda: False):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("source must be a regular file")
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 16 * 1024**4:
            raise ValueError("explicit 1 byte..16 TiB quota required")
        size = source.stat().st_size
        if not 1 <= size <= max_bytes:
            raise ValueError("source exceeds quota or is empty")
        label = source.name if name is None else name
        if not isinstance(label, str) or not 1 <= len(label) <= 256 or any(ord(c) < 32 for c in label):
            raise ValueError("invalid artifact name")
        identifier = str(uuid.uuid4())
        total = 0
        count = 0
        overall = hashlib.sha256()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            # The provisional row is invisible to other connections until commit.
            self.db.execute("INSERT INTO artifacts VALUES(?,?,0,'',0)", (identifier, label))
            with source.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError("source must be regular")
                while True:
                    if cancelled():
                        raise InterruptedError("model import cancelled")
                    data = stream.read(self.chunk_bytes)
                    if not data:
                        break
                    total += len(data)
                    if total > max_bytes or count >= 10_000_000:
                        raise ValueError("model import quota/chunk limit exceeded")
                    digest = hashlib.sha256(data).hexdigest()
                    overall.update(data)
                    self._put_blob(digest, data)
                    self.db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (identifier, count, digest, len(data)))
                    count += 1
                after = os.fstat(stream.fileno())
                if total != size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("source changed during import")
            self.db.execute("UPDATE artifacts SET bytes=?,sha256=?,chunks=? WHERE id=?", (total, overall.hexdigest(), count, identifier))
            if cancelled():
                raise InterruptedError("model import cancelled")
            self.db.commit()
            return identifier
        except BaseException:
            self.db.rollback()
            raise

    def _put_blob(self, digest, data):
        directory = self.blobs / digest[:2]
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink():
            raise ValueError("blob directory cannot be a symlink")
        destination = directory / digest
        if destination.exists():
            self._read_blob(digest, len(data))
            return
        fd, temporary = tempfile.mkstemp(prefix=".staging-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            try:
                os.link(temporary, destination)  # atomic no-overwrite publication
            except FileExistsError:
                self._read_blob(digest, len(data))
        finally:
            os.unlink(temporary)

    def _read_blob(self, digest, size):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not 1 <= size <= 64 * 1024 * 1024:
            raise ValueError("invalid chunk manifest")
        path = self.blobs / digest[:2] / digest
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("model chunk missing or length mismatch")
        with path.open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("model chunk integrity failure")
        return data

    def info(self, identifier):
        if str(uuid.UUID(identifier)) != identifier:
            raise ValueError("invalid artifact identity")
        row = self.db.execute("SELECT name,bytes,sha256,chunks FROM artifacts WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("unknown artifact")
        name, size, digest, count = row
        if not 1 <= size <= 16 * 1024**4 or not 1 <= count <= 10_000_000 or len(digest) != 64:
            raise ValueError("invalid artifact manifest")
        return dict(id=identifier, name=name, bytes=size, sha256=digest, chunks=count)

    def iter_chunks(self, identifier, *, max_bytes, cancelled=lambda: False):
        info = self.info(identifier)
        if info["bytes"] > max_bytes:
            raise ValueError("model read exceeds quota")
        total = 0
        overall = hashlib.sha256()
        count = 0
        for ordinal, digest, size in self.db.execute("SELECT ordinal,digest,bytes FROM chunks WHERE artifact=? ORDER BY ordinal", (identifier,)):
            if cancelled():
                raise InterruptedError("model read cancelled")
            if ordinal != count or count >= info["chunks"] or total + size > info["bytes"]:
                raise ValueError("model manifest order/size mismatch")
            data = self._read_blob(digest, size)
            total += len(data)
            overall.update(data)
            count += 1
            yield data  # chunk is verified; whole-artifact acceptance requires exhausting iterator
        if total != info["bytes"] or count != info["chunks"] or overall.hexdigest() != info["sha256"]:
            raise ValueError("model manifest integrity failure")

    def export(self, identifier, destination, *, max_bytes, cancelled=lambda: False):
        destination = Path(destination)
        if not destination.is_absolute() or destination.exists() or destination.is_symlink():
            raise ValueError("new absolute destination required")
        fd, temporary = tempfile.mkstemp(prefix=".hyperl-export-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as out:
                for data in self.iter_chunks(identifier, max_bytes=max_bytes, cancelled=cancelled):
                    out.write(data)
            if cancelled():
                raise InterruptedError("model export cancelled")
            os.link(temporary, destination)
        finally:
            os.unlink(temporary)


class ModelChunkCache:
    """Optional bounded RAM LRU. Disk remains authoritative; cache keys are hashes."""
    def __init__(self, budget_bytes):
        if not isinstance(budget_bytes, int) or not 0 <= budget_bytes <= 1024**3:
            raise ValueError("cache budget must be 0..1 GiB")
        self.budget = budget_bytes
        self.used = 0
        self.data = OrderedDict()

    def remember(self, data):
        data = bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if len(data) <= self.budget:
            old = self.data.pop(digest, None)
            if old is not None:
                self.used -= len(old)
            while self.used + len(data) > self.budget:
                self.used -= len(self.data.popitem(last=False)[1])
            self.data[digest] = data
            self.used += len(data)
        return digest

    def get(self, digest):
        data = self.data.pop(digest, None)
        if data is not None:
            self.data[digest] = data
        return data


@dataclass(frozen=True)
class MemoryTier:
    id: str
    node: str
    kind: str  # GPU, TPU, NPU, RAM, STORAGE
    available_bytes: int
    bandwidth_bytes_per_second: float
    physical_domain: str  # aliases/unified memory must share this key
    backend: str = "unqualified"


@dataclass(frozen=True)
class ModelLayer:
    name: str
    weight_bytes: int
    kv_bytes: int = 0  # caller estimates exact precision/context/concurrency cost


@dataclass(frozen=True)
class Workload:
    scenario: str = "auto"
    prompt_tokens: int = 512
    concurrent_requests: int = 1
    max_new_tokens: int = 256
    training: bool = False
    reserve_fraction: float = 0.15
    allow_storage_offload: bool = False
    allow_host_offload: bool = True
    max_transfer_seconds: float = 60.0
    activation_bytes: int = 0
    training_state_multiplier: float = 4.0


def plan_model(layers, tiers, workload=Workload(), *, manual=None):
    """Deterministic advisory layer residency with exact byte accounting and overrides.

    Capacity/bandwidth are caller observations, not discovery or reservations. KV,
    activation and training state are reserved in compute memory. Offloaded weights
    require accelerator staging for the largest assigned layer and account for at
    least one full weight transfer per pass. No inference execution is activated.
    """
    layers = list(itertools.islice(layers, 4097))
    tiers = list(itertools.islice(tiers, 1025))
    if not 1 <= len(layers) <= 4096 or not 1 <= len(tiers) <= 1024 or len({l.name for l in layers}) != len(layers) or len({t.id for t in tiers}) != len(tiers):
        raise ValueError("bounded unique layers and tiers required")
    scenarios = {"auto", "decode", "prefill", "batch", "training"}
    if (workload.scenario not in scenarios or not 1 <= workload.prompt_tokens <= 10_000_000 or not 1 <= workload.concurrent_requests <= 1_000_000
            or not 1 <= workload.max_new_tokens <= 10_000_000
            or not math.isfinite(workload.reserve_fraction) or not 0 <= workload.reserve_fraction < 1
            or not math.isfinite(workload.max_transfer_seconds) or workload.max_transfer_seconds <= 0
            or not isinstance(workload.activation_bytes, int) or not 0 <= workload.activation_bytes <= 16 * 1024**4
            or not math.isfinite(workload.training_state_multiplier) or not 1 <= workload.training_state_multiplier <= 32):
        raise ValueError("invalid scenario policy")
    scenario = workload.scenario
    reason = "manual scenario override"
    if scenario == "auto":
        scenario = "training" if workload.training else "batch" if workload.concurrent_requests >= 8 else "prefill" if workload.prompt_tokens >= 4096 else "decode"
        reason = "training flag" if workload.training else "concurrency >= 8" if scenario == "batch" else "prompt >= 4096 tokens" if scenario == "prefill" else "interactive generation default"
    if workload.training and scenario != "training":
        raise ValueError("training cannot use an inference memory policy")
    for layer in layers:
        if not isinstance(layer.name, str) or not 1 <= len(layer.name) <= 256 or not isinstance(layer.weight_bytes, int) or not 1 <= layer.weight_bytes <= 16 * 1024**4 or not isinstance(layer.kv_bytes, int) or not 0 <= layer.kv_bytes <= 16 * 1024**4:
            raise ValueError("invalid layer size")
    for tier in tiers:
        if (not tier.id or not tier.node or not tier.physical_domain or tier.kind not in {"GPU", "TPU", "NPU", "RAM", "STORAGE"}
                or not isinstance(tier.available_bytes, int) or not 0 <= tier.available_bytes <= 16 * 1024**4
                or not math.isfinite(tier.bandwidth_bytes_per_second) or tier.bandwidth_bytes_per_second <= 0):
            raise ValueError("invalid tier observation")
    # Ambiguous aliases reject instead of summing shared RAM and unified VRAM.
    if len({t.physical_domain for t in tiers}) != len(tiers):
        raise ValueError("physical memory aliases must be collapsed into one tier before planning")
    by_id = {t.id: t for t in tiers}
    manual = dict(manual or {})
    if set(manual) - {l.name for l in layers} or set(manual.values()) - by_id.keys():
        raise ValueError("unknown manual layer/tier")
    usable = {t.id: int(t.available_bytes * (1 - workload.reserve_fraction)) for t in tiers}
    remaining = dict(usable)
    accelerators = [t for t in tiers if t.kind in {"GPU", "TPU", "NPU"}]
    if not accelerators:
        accelerators = [t for t in tiers if t.kind == "RAM"]
    if not accelerators:
        raise ValueError("storage alone cannot execute a model")
    # A single execution backend family per plan: incompatible TPU/GPU stacks do
    # not become interoperable merely because their byte counts fit.
    families = {t.backend for t in accelerators}
    if len(families) != 1 or "unqualified" in families:
        raise ValueError("select one explicit compatible model backend family per pool")
    state_bytes = sum(l.kv_bytes for l in layers) + workload.activation_bytes
    if scenario == "training":
        state_bytes += math.ceil(sum(l.weight_bytes for l in layers) * (workload.training_state_multiplier - 1))
    execution = sorted(accelerators, key=lambda t: (-t.bandwidth_bytes_per_second, t.id))
    # Reserve state by bounded partitions; weights are still placed as whole layers.
    state = {}
    left = state_bytes
    for tier in execution:
        take = min(left, remaining[tier.id]); state[tier.id] = take; remaining[tier.id] -= take; left -= take
    if left:
        raise ValueError("KV/activation/training state does not fit execution memory; lower context/concurrency or add capacity")
    placements = []
    staging = {t.id: 0 for t in execution}
    transfer_seconds = 0.0
    # Place large/pinned layers first to reduce fragmentation; report execution order below.
    for layer in sorted(layers, key=lambda l: (l.name not in manual, -l.weight_bytes, l.name)):
        candidates = execution + sorted((t for t in tiers if t not in execution and t.kind == "RAM" and workload.allow_host_offload), key=lambda t: (-t.bandwidth_bytes_per_second, t.id))
        if workload.allow_storage_offload and scenario != "training":
            candidates += sorted((t for t in tiers if t.kind == "STORAGE"), key=lambda t: (-t.bandwidth_bytes_per_second, t.id))
        chosen = None
        for tier in candidates:
            if layer.name in manual and tier.id != manual[layer.name]:
                continue
            if remaining[tier.id] < layer.weight_bytes:
                continue
            target = tier if tier in execution else next((t for t in execution if remaining[t.id] >= max(0, layer.weight_bytes - staging[t.id])), None)
            if target is None:
                continue
            cost = 0.0 if tier == target else layer.weight_bytes / min(tier.bandwidth_bytes_per_second, target.bandwidth_bytes_per_second)
            if transfer_seconds + cost > workload.max_transfer_seconds:
                continue
            chosen = tier, target, cost
            break
        if chosen is None:
            raise ValueError("layer cannot fit policy/staging/transfer budget: " + layer.name)
        tier, target, cost = chosen
        remaining[tier.id] -= layer.weight_bytes
        if tier != target:
            extra = max(0, layer.weight_bytes - staging[target.id]); remaining[target.id] -= extra; staging[target.id] += extra
        transfer_seconds += cost
        placements.append(dict(layer=layer.name, storage_tier=tier.id, execution_tier=target.id, weight_bytes=layer.weight_bytes,
                               transfer_bytes_per_pass=0 if tier == target else layer.weight_bytes))
    order = {layer.name: index for index, layer in enumerate(layers)}
    placements.sort(key=lambda p: order[p["layer"]])
    return dict(format="hyperl-model-residency/1", process_id=str(uuid.uuid4()), scenario=scenario, reason=reason,
                placements=placements, used_bytes={key: usable[key] - remaining[key] for key in usable}, state_bytes=state,
                staging_bytes=staging, estimated_weight_transfer_seconds_per_pass=transfer_seconds,
                execution_ready=False, reservations_guaranteed=False,
                limits="Advisory whole-layer placement. No LLM backend, tensor-parallel collectives, inter-layer activation transfer cost or speedup qualification. Bandwidth estimates exclude contention; token decoding can repeat weight transfers.")


@dataclass(frozen=True)
class LLMProfile:
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    vocabulary: int
    weight_bits: int = 16
    cache_bits: int = 16
    tied_embeddings: bool = True
    quantization_group: int = 128
    attention_workspace: str = "dense"  # dense score matrix or explicitly requested linear/fused estimate


def estimate_llm(profile, workload=Workload()):
    """Size estimator for a gated decoder with GQA; never reads or executes weights.

    Real checkpoint/adapter sizes override this estimate. Includes packed weights,
    2-byte group scales below 16 bits, KV for prompt + generation + concurrency,
    and conservative activation workspace. Other model architectures require exact
    caller-supplied ModelLayer sizes. No kernel/precision support is inferred.
    """
    dimensions = (profile.hidden_size, profile.intermediate_size, profile.layers,
                  profile.attention_heads, profile.kv_heads, profile.vocabulary, profile.quantization_group)
    if any(not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1_000_000 for value in dimensions):
        raise ValueError("invalid LLM dimensions")
    if (profile.layers > 4094 or profile.hidden_size % profile.attention_heads
            or profile.attention_heads % profile.kv_heads or profile.weight_bits not in {4, 8, 16, 32}
            or profile.cache_bits not in {16, 32} or profile.attention_workspace not in {"dense", "linear"}
            or not 1 <= workload.prompt_tokens <= 10_000_000 or not 1 <= workload.max_new_tokens <= 10_000_000
            or not 1 <= workload.concurrent_requests <= 1_000_000):
        raise ValueError("unsupported LLM estimate contract")
    h = profile.hidden_size
    head = h // profile.attention_heads
    kv_dimension = profile.kv_heads * head
    def packed(parameters):
        return (parameters * profile.weight_bits + 7) // 8 + ((parameters + profile.quantization_group - 1) // profile.quantization_group * 2 if profile.weight_bits < 16 else 0)
    embedding = packed(profile.vocabulary * h)
    decoder = packed(2 * h * h + 2 * h * kv_dimension + 3 * h * profile.intermediate_size + 2 * h)
    kv = 2 * kv_dimension * (workload.prompt_tokens + workload.max_new_tokens) * workload.concurrent_requests * (profile.cache_bits // 8)
    result = [ModelLayer("embedding", embedding)]
    result += [ModelLayer(f"decoder-{index}", decoder, kv) for index in range(profile.layers)]
    if not profile.tied_embeddings:
        result.append(ModelLayer("output", embedding))
    tokens = workload.prompt_tokens * workload.concurrent_requests
    activations = tokens * (6 * h + 2 * profile.intermediate_size) * 4
    if profile.attention_workspace == "dense":
        activations += workload.concurrent_requests * profile.attention_heads * workload.prompt_tokens**2 * 4
    if any(layer.weight_bytes > 16 * 1024**4 or layer.kv_bytes > 16 * 1024**4 for layer in result) or activations > 16 * 1024**4:
        raise ValueError("estimated model exceeds bounded planning contract")
    return result, activations
