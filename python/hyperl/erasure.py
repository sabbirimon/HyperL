# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Native verified-erasure correction for immutable model chunks, not hardware ECC."""
import ctypes as C
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from .llm import _integer, _verify, _DIGEST


class ErasureCodec:
    def __init__(self, library, sha256, *, budget_bytes=256 * 1024 * 1024):
        _verify(library, sha256, 128 * 1024 * 1024)
        _integer(budget_bytes, 4096, 1024**3)
        self.budget = budget_bytes; self.library = C.CDLL(library)
        self.pointer = C.POINTER(C.c_uint8); self.callback = C.CFUNCTYPE(C.c_int, C.c_void_p)
        self.library.hl_erasure_encode.argtypes = [C.c_uint, C.c_uint, C.c_size_t, C.POINTER(self.pointer), C.POINTER(self.pointer), self.callback, C.c_void_p]
        self.library.hl_erasure_recover.argtypes = [C.c_uint, C.c_uint, C.c_size_t, C.POINTER(self.pointer), C.c_uint32, C.POINTER(self.pointer), self.callback, C.c_void_p]
        self.library.hl_erasure_encode.restype = C.c_int; self.library.hl_erasure_recover.restype = C.c_int

    def _admit(self, k, m, size):
        _integer(k, 1, 16); _integer(m, 1, 4); _integer(size, 1, 4 * 1024 * 1024)
        if (3 * k + 5 * m) * size > self.budget:
            raise ValueError("erasure working payload exceeds configured budget")

    def _cancel(self, cancelled):
        errors = []
        def check(_):
            try:
                return int(bool(cancelled()))
            except BaseException as error:
                errors.append(error); return 1
        return self.callback(check), errors

    def _result(self, status, errors):
        if errors:
            raise errors[0]
        if status == 3:
            raise InterruptedError("erasure operation cancelled; discard partial output")
        if status != 0:
            raise ValueError("erasure request invalid or exceeds redundancy; no recovery accepted")

    def encode(self, data, parity_count=2, *, cancelled=lambda: False):
        values = list(itertools.islice(data, 17))
        if not values:
            raise ValueError("at least one data shard required")
        size = len(values[0]); self._admit(len(values), parity_count, size)
        if any(not isinstance(v, bytes) or len(v) != size for v in values):
            raise ValueError("equal immutable byte shards required")
        inputs = [(C.c_uint8 * size).from_buffer_copy(v) for v in values]
        outputs = [(C.c_uint8 * size)() for _ in range(parity_count)]
        sources = (self.pointer * len(inputs))(*(C.cast(x, self.pointer) for x in inputs))
        targets = (self.pointer * len(outputs))(*(C.cast(x, self.pointer) for x in outputs))
        callback, errors = self._cancel(cancelled)
        self._result(self.library.hl_erasure_encode(len(inputs), parity_count, size, sources, targets, callback, None), errors)
        return [bytes(x) for x in outputs]

    def recover(self, fragments, data_count, parity_count, *, cancelled=lambda: False):
        fragments = list(itertools.islice(fragments, 21))
        if len(fragments) != data_count + parity_count:
            raise ValueError("fragment count mismatch")
        present = [v for v in fragments if v is not None]
        if len(present) < data_count:
            raise ValueError("insufficient verified fragments; recovery budget exceeded")
        size = len(present[0]); self._admit(data_count, parity_count, size)
        if any(not isinstance(v, bytes) or len(v) != size for v in present):
            raise ValueError("equal immutable byte fragments required")
        buffers = [(C.c_uint8 * size).from_buffer_copy(v) if v is not None else None for v in fragments]
        outputs = [(C.c_uint8 * size)() if v is None else None for v in fragments]
        sources = (self.pointer * len(fragments))(*(C.cast(v, self.pointer) if v is not None else self.pointer() for v in buffers))
        targets = (self.pointer * len(fragments))(*(C.cast(v, self.pointer) if v is not None else self.pointer() for v in outputs))
        mask = sum(1 << i for i, value in enumerate(fragments) if value is not None)
        callback, errors = self._cancel(cancelled)
        self._result(self.library.hl_erasure_recover(data_count, parity_count, size, sources, mask, targets, callback, None), errors)
        return [fragments[i] if value is None else bytes(value) for i, value in enumerate(outputs)]


class ProtectedModelStore:
    """Add immutable k+m redundancy to an existing ModelStore.

    Expected protection manifest digest is supplied independently for every repair.
    Hash failures become erasures; <=m per stripe can be corrected. Model export
    still performs its original whole-artifact check. Same-disk parity does not
    survive loss of that disk; place independently copied stores in failure domains.
    """
    def __init__(self, store, codec):
        self.store = store; self.codec = codec
        store.db.executescript("""
            CREATE TABLE IF NOT EXISTS protection (artifact TEXT PRIMARY KEY, header TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS protection_groups (artifact TEXT NOT NULL, ordinal INTEGER NOT NULL,
                metadata TEXT NOT NULL, PRIMARY KEY(artifact,ordinal));
            CREATE TABLE IF NOT EXISTS repair_events (id TEXT PRIMARY KEY, artifact TEXT NOT NULL,
                at_ms INTEGER NOT NULL, repaired INTEGER NOT NULL, manifest TEXT NOT NULL);
        """)

    @staticmethod
    def _canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def protect(self, artifact, *, data_count=4, parity_count=2, max_bytes, cancelled=lambda: False):
        _integer(data_count, 1, 16); _integer(parity_count, 1, 4)
        info = self.store.info(artifact)
        if info["bytes"] > max_bytes:
            raise ValueError("artifact exceeds protection quota")
        if self.store.db.execute("SELECT 1 FROM protection WHERE artifact=?", (artifact,)).fetchone():
            raise ValueError("immutable protection manifest already exists")
        header = self._canonical(dict(format="hyperl-erasure-cauchy/1", artifact=artifact, artifact_sha256=info["sha256"],
            bytes=info["bytes"], chunks=info["chunks"], k=data_count, m=parity_count))
        digest = hashlib.sha256(header.encode()); group = []; ordinal = 0
        def stripe(values, index):
            size = max(map(len, values)); self.codec._admit(data_count, parity_count, size)
            data = [v + bytes(size - len(v)) for v in values] + [bytes(size)] * (data_count - len(values))
            parity = self.codec.encode(data, parity_count, cancelled=cancelled)
            metadata = dict(size=size, data=[dict(length=len(v), sha256=hashlib.sha256(v).hexdigest()) for v in values],
                            parity=[hashlib.sha256(v).hexdigest() for v in parity])
            for key, value in zip(metadata["parity"], parity):
                self.store._put_blob(key, value)
            text = self._canonical(metadata); digest.update(text.encode())
            self.store.db.execute("INSERT INTO protection_groups VALUES(?,?,?)", (artifact, index, text))
        try:
            self.store.db.execute("BEGIN IMMEDIATE")
            for chunk in self.store.iter_chunks(artifact, max_bytes=max_bytes, cancelled=cancelled):
                group.append(chunk)
                if len(group) == data_count:
                    stripe(group, ordinal); ordinal += 1; group = []
            if group:
                stripe(group, ordinal)
            if cancelled():
                raise InterruptedError("protection cancelled")
            self.store.db.execute("INSERT INTO protection VALUES(?,?,?)", (artifact, header, digest.hexdigest()))
            self.store.db.commit()
            return dict(artifact=artifact, manifest_sha256=digest.hexdigest(), data_shards=data_count, parity_shards=parity_count,
                        corruption_detection="trusted SHA-256", repair_budget_per_stripe=parity_count)
        except BaseException:
            self.store.db.rollback(); raise

    def _restore(self, digest, data):
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("reconstructed chunk does not match trusted manifest")
        directory = self.store.blobs / digest[:2]
        directory.mkdir(mode=0o700, exist_ok=True)
        target = directory / digest
        if directory.is_symlink() or target.is_symlink():
            raise ValueError("repair cannot replace symlink targets")
        try:
            self.store._read_blob(digest, len(data)); return
        except (ValueError, OSError):
            pass
        fd, staging = tempfile.mkstemp(prefix=".repair-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
            os.replace(staging, target)  # one verified content-addressed chunk commits atomically
        finally:
            if os.path.exists(staging):
                os.unlink(staging)

    def repair(self, artifact, *, expected_manifest_sha256, max_bytes, cancelled=lambda: False):
        if not isinstance(expected_manifest_sha256, str) or not _DIGEST.fullmatch(expected_manifest_sha256):
            raise ValueError("independently trusted protection digest required")
        repaired = 0
        try:
            self.store.db.execute("BEGIN IMMEDIATE")
            row = self.store.db.execute("SELECT header,digest FROM protection WHERE artifact=?", (artifact,)).fetchone()
            if row is None or row[1] != expected_manifest_sha256:
                raise ValueError("protection manifest identity mismatch")
            header = json.loads(row[0]); digest = hashlib.sha256(row[0].encode())
            if header["format"] != "hyperl-erasure-cauchy/1" or header["artifact"] != artifact or header["bytes"] > max_bytes:
                raise ValueError("protection format/identity/quota mismatch")
            k, m = header["k"], header["m"]; _integer(k, 1, 16); _integer(m, 1, 4)
            expected_groups = (header["chunks"] + k - 1) // k
            count = 0
            for ordinal, text in self.store.db.execute("SELECT ordinal,metadata FROM protection_groups WHERE artifact=? ORDER BY ordinal", (artifact,)):
                if ordinal != count or count >= expected_groups or len(text) > 8192:
                    raise ValueError("protection group order/size mismatch")
                digest.update(text.encode()); count += 1
            if count != expected_groups or digest.hexdigest() != expected_manifest_sha256:
                raise ValueError("protection manifest integrity failure")
            for _, text in self.store.db.execute("SELECT ordinal,metadata FROM protection_groups WHERE artifact=? ORDER BY ordinal", (artifact,)):
                if cancelled():
                    raise InterruptedError("repair cancelled")
                group = json.loads(text); size = group["size"]; self.codec._admit(k, m, size)
                refs = group["data"]; parity = group["parity"]
                if not 1 <= len(refs) <= k or len(parity) != m:
                    raise ValueError("protection stripe shape mismatch")
                fragments = []; bad = []
                for i, info in enumerate(refs):
                    _integer(info["length"], 1, size)
                    try:
                        data = self.store._read_blob(info["sha256"], info["length"])
                        fragments.append(data + bytes(size - len(data)))
                    except (ValueError, OSError):
                        fragments.append(None); bad.append(i)
                fragments += [bytes(size)] * (k - len(refs))
                for p, key in enumerate(parity):
                    try:
                        fragments.append(self.store._read_blob(key, size))
                    except (ValueError, OSError):
                        fragments.append(None); bad.append(k + p)
                if not bad:
                    continue
                result = self.codec.recover(fragments, k, m, cancelled=cancelled)
                # Validate every reconstructed digest before publishing any stripe member.
                restored = []
                for index in bad:
                    key = refs[index]["sha256"] if index < k else parity[index - k]
                    data = result[index][:refs[index]["length"]] if index < k else result[index]
                    if hashlib.sha256(data).hexdigest() != key:
                        raise ValueError("erasure result failed the trusted chunk digest")
                    restored.append((key, data))
                for key, data in restored:
                    if cancelled():
                        raise InterruptedError("repair cancelled before chunk commit")
                    self._restore(key, data); repaired += 1
            # Whole-artifact integrity remains the publication gate after correction.
            for _ in self.store.iter_chunks(artifact, max_bytes=max_bytes, cancelled=cancelled):
                pass
            event = str(uuid.uuid4())
            self.store.db.execute("INSERT INTO repair_events VALUES(?,?,?,?,?)", (event, artifact, int(time.time() * 1000), repaired, expected_manifest_sha256))
            self.store.db.commit()
            return dict(event=event, artifact=artifact, repaired_fragments=repaired, whole_artifact_verified=True,
                        hardware_ecc=False, erasure_format="hyperl-erasure-cauchy/1")
        except BaseException:
            self.store.db.rollback(); raise
