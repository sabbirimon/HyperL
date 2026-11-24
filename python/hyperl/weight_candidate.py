# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Build a separate research candidate by applying one bounded numerical edit."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .llm import file_digest
from .model_formats import _Budget, _open, _safetensors


def clip_candidate(source, destination, tensor, limit, expected_sha256, max_bytes, cancelled=lambda: False):
    import numpy as np
    if not re.fullmatch("[0-9a-f]{64}", expected_sha256) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Expected a source SHA-256 and a positive finite clipping limit")
    destination = Path(destination).absolute()
    if destination.suffix != ".safetensors" or os.path.lexists(destination):
        raise ValueError("Choose a new .safetensors destination")
    # Inspect only the verified private snapshot. The source may be concurrently replaced.
    with tempfile.TemporaryDirectory(dir=destination.parent) as stage:
        candidate = Path(stage) / "candidate.safetensors"
        digest = hashlib.sha256()
        stream, size = _open(source)
        with stream, candidate.open("xb") as out:
            if size > max_bytes:
                raise ValueError("Checkpoint exceeds the output budget")
            remaining = size
            while remaining:
                if cancelled():
                    raise InterruptedError("Candidate cancelled")
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("Source truncated during copy")
                out.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            if stream.read(1) or digest.hexdigest() != expected_sha256:
                raise ValueError("Source digest or size changed")
        manifest = _safetensors(candidate, _Budget())
        entry = manifest["tensors"].get(tensor)
        if not entry or entry["dtype"] not in {"F32", "F16"}:
            raise ValueError("Choose a named F32 or F16 tensor")
        dtype = np.dtype("<f4" if entry["dtype"] == "F32" else "<f2")
        if limit > np.finfo(dtype).max or dtype.type(limit) == 0:
            raise ValueError("Limit outside the tensor dtype range")
        start, end = entry["data_offsets"]
        remaining = end - start
        changed = 0
        with candidate.open("r+b") as weights:
            weights.seek(size - manifest["payload_bytes"] + start)
            while remaining:
                if cancelled():
                    raise InterruptedError("Candidate cancelled")
                position = weights.tell()
                block = weights.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("Truncated candidate tensor")
                values = np.frombuffer(block, dtype=dtype)
                if not np.isfinite(values).all():
                    raise ValueError("Selected tensor contains nonfinite values")
                clipped = values.clip(-dtype.type(limit), dtype.type(limit))
                changed += int(np.count_nonzero(clipped != values))
                weights.seek(position)
                weights.write(clipped.tobytes())
                remaining -= len(block)
        if not changed:
            raise ValueError("The proposed edit would change no values")
        sha = file_digest(candidate, max_bytes=max_bytes, cancelled=cancelled)
        if cancelled():
            raise InterruptedError("Candidate cancelled")
        os.link(candidate, destination)
    return dict(operation="clip", tensor=tensor, limit=limit, values_changed=changed,
                source_sha256=expected_sha256, candidate_sha256=sha, evaluated=False)

def proposed_candidate(source, destination, proposal, expected_sha256, max_bytes):
    """Apply a data-only proposal bound to the owner's expected source digest."""
    with Path(proposal).open("rb") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("Proposal exceeds 8 KiB")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != "hyperl.weight-proposal.v1":
        raise ValueError("Unknown proposal schema")
    if data.get("source_sha256") != expected_sha256 or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("Proposal source digest mismatch")
    common = {"schema", "operation", "source_sha256"}
    if data.get("operation") == "abstain" and set(data) == common:
        return dict(operation="abstain", evaluated=False)
    if data.get("operation") != "clip" or set(data) != common | {"tensor", "limit"}:
        raise ValueError("Expected one bounded clip or abstain operation")
    if not isinstance(data["tensor"], str) or not 1 <= len(data["tensor"]) <= 1024 or type(data["limit"]) not in (int, float):
        raise ValueError("Invalid proposal tensor or limit")
    return clip_candidate(source, destination, data["tensor"], data["limit"], expected_sha256, max_bytes)
