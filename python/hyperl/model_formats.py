# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Bounded, read-only model inspection. Never loads model code or weight payloads.

Registry entries are developer-reviewed data, not executable plugin declarations.
Inspection does not authorize execution, allocate devices or qualify sharding.
"""
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

REGISTRY_VERSION = "hyperl-model-formats/1"
_REGISTRY = {
    "gguf": {"adapter": "explicit llama.cpp adapter", "versions": [2, 3]},
    "safetensors": {"adapter": "architecture-specific conversion or runtime adapter", "versions": [], "container_versioning": "unversioned"},
    "safetensors-index": {"adapter": "architecture-specific conversion or runtime adapter", "versions": [], "container_versioning": "unversioned"},
    "hyperl-llama-f32/1": {"adapter": "explicit HyperL JAX decoder", "versions": [1]},
    "hyperl-llama-kv/1": {"adapter": "matching HyperL JAX session restore", "versions": [1]},
    "onnx": {"adapter": "ONNX adapter required", "versions": []},
    "tflite": {"adapter": "TFLite adapter required", "versions": []},
    "coreml": {"adapter": "CoreML adapter required", "versions": []},
    "pytorch": {"adapter": "trusted architecture-specific conversion required; pickle is not executed", "versions": []},
}
_WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "U32": 4, "I32": 4,
           "U64": 8, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
_MAX_FILE = 16 * 1024**4
_MAX_HEADER = 16 * 1024**2


def registry():
    return dict(format=REGISTRY_VERSION, entries={key: dict(value, versions=list(value["versions"])) for key, value in _REGISTRY.items()})


def _integer(value, minimum=0, maximum=_MAX_FILE):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid bounded integer")
    return value


def _json(data):
    text = data.decode("utf-8")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 16:
                raise ValueError("metadata nesting exceeds limit")
        elif char in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite metadata")
        return parsed
    def constant(_):
        raise ValueError("nonfinite metadata")
    def integer(value):
        if len(value.lstrip("-")) > 20:
            raise ValueError("metadata integer exceeds 20 digits")
        return int(value)
    result = json.loads(text, object_pairs_hook=pairs, parse_int=integer, parse_float=number, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("metadata must be an object")
    return result


class _Budget:
    def __init__(self):
        self.remaining = 64 * 1024**2
    def read(self, stream, count):
        if count > self.remaining:
            raise ValueError("aggregate inspection metadata exceeds 64 MiB")
        self.remaining -= count
        data = stream.read(count)
        if len(data) != count:
            raise ValueError("truncated model metadata")
        return data


def _open(path):
    # O_NOFOLLOW prevents replacing the final component with a symlink at open.
    if Path(path).is_symlink():
        raise ValueError("artifact symlinks are not inspected")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    stream = os.fdopen(fd, "rb")
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_FILE:
        stream.close()
        raise ValueError("inspection requires a bounded regular model file")
    return stream, info.st_size


def _child(root, filename):
    if not isinstance(filename, str) or not 1 <= len(filename) <= 512 or "\\" in filename or ":" in filename:
        raise ValueError("invalid local artifact path")
    relative = Path(filename)
    if relative.is_absolute() or any(part in ("..", ".") for part in relative.parts) or any(ord(c) < 32 for c in filename):
        raise ValueError("artifact path escapes inspection directory")
    target = root
    for part in relative.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("artifact symlinks are not inspected")
    if not target.is_file() or root.resolve() not in target.resolve().parents:
        raise ValueError("artifact file is missing or outside inspection directory")
    return target


def _safetensors(path, budget):
    stream, size = _open(path)
    with stream:
        length = struct.unpack("<Q", budget.read(stream, 8))[0]
        _integer(length, 2, min(_MAX_HEADER, size - 8))
        header = _json(budget.read(stream, length))
    if len(header) > 100001:
        raise ValueError("too many tensor records")
    payload = size - 8 - length
    tensors, ranges, dtypes = {}, [], set()
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in descriptor.items()):
                raise ValueError("safetensors metadata must contain strings")
            continue
        if not 1 <= len(name) <= 1024 or any(ord(c) < 32 for c in name):
            raise ValueError("invalid tensor name")
        if not isinstance(descriptor, dict) or set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("unsupported tensor descriptor")
        dtype, shape, offsets = descriptor["dtype"], descriptor["shape"], descriptor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _WIDTHS or not isinstance(shape, list) or len(shape) > 32:
            raise ValueError("unsupported tensor dtype or shape")
        elements = 1
        for dimension in shape:
            elements *= _integer(dimension, 0, 2**31)
            if elements > _MAX_FILE:
                raise ValueError("tensor shape exceeds inspection limit")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid tensor offsets")
        start, end = [_integer(value, 0, payload) for value in offsets]
        if end < start or end - start != elements * _WIDTHS[dtype]:
            raise ValueError("tensor shape/offset byte mismatch")
        ranges.append((start, end)); tensors[name] = descriptor; dtypes.add(dtype)
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("tensor offsets overlap or leave holes")
        cursor = end
    if cursor != payload:
        raise ValueError("unindexed tensor payload bytes")
    return dict(tensors=tensors, dtypes=sorted(dtypes), payload_bytes=payload, file_bytes=size)


def _index(path, document, budget):
    if set(document) - {"metadata", "weight_map"} or not isinstance(document.get("weight_map"), dict) or not 1 <= len(document["weight_map"]) <= 100000:
        raise ValueError("unsupported safetensors index")
    mapping = document["weight_map"]
    if any(not isinstance(name, str) or not 1 <= len(name) <= 1024 or not isinstance(file, str) for name, file in mapping.items()):
        raise ValueError("invalid shard mapping")
    files = set(mapping.values())
    if len(files) > 128:
        raise ValueError("at most 128 shards may be inspected")
    found, total, payload, dtypes = {}, 0, 0, set()
    for filename in sorted(files):
        if not filename.endswith(".safetensors"):
            raise ValueError("index requires safetensors shards")
        inspected = _safetensors(_child(path.parent, filename), budget)
        total += inspected["file_bytes"]; payload += inspected["payload_bytes"]
        if total > _MAX_FILE:
            raise ValueError("aggregate shards exceed 16 TiB")
        for name in inspected["tensors"]:
            if name in found or mapping.get(name) != filename:
                raise ValueError("duplicate or incorrectly indexed shard tensor")
            found[name] = filename
        dtypes.update(inspected["dtypes"])
    if found != mapping:
        raise ValueError("shard tensor mapping is incomplete")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("invalid index metadata")
    if "total_size" in metadata and _integer(metadata["total_size"]) != payload:
        raise ValueError("declared shard payload size mismatch")
    return dict(tensor_count=len(found), shard_count=len(files), payload_bytes=payload, dtypes=sorted(dtypes))


def _provenance(value, context):
    fields = {"importer", "sources", "source_format", "output_dtype", "tokenizer", "original_context", "source_dtypes"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("unsupported checkpoint provenance")
    expected = {"importer": "hyperl-safetensors-llama/1", "source_format": "safetensors", "output_dtype": "float32",
                "tokenizer": "not imported; caller supplies matching token IDs"}
    if any(value[key] != required for key, required in expected.items()):
        raise ValueError("unknown checkpoint conversion contract")
    if _integer(value["original_context"], 1, 131072) < context:
        raise ValueError("converted context exceeds source contract")
    dtypes = value["source_dtypes"]
    if not isinstance(dtypes, list) or not 1 <= len(dtypes) <= 3 or any(dtype not in ("F32", "F16", "BF16") for dtype in dtypes) or dtypes != sorted(set(dtypes)):
        raise ValueError("unsupported conversion dtypes")
    sources = value["sources"]
    if not isinstance(sources, dict) or not 2 <= len(sources) <= 130 or "config.json" not in sources:
        raise ValueError("conversion source identities missing")
    for filename, digest in sources.items():
        if (not isinstance(filename, str) or not 1 <= len(filename) <= 512 or Path(filename).name != filename
                or "\\" in filename or ":" in filename or any(ord(char) < 32 for char in filename)
                or not (filename in ("config.json", "model.safetensors.index.json") or filename.endswith(".safetensors"))
                or not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)):
            raise ValueError("invalid conversion source identity")
    if not any(filename.endswith(".safetensors") for filename in sources):
        raise ValueError("conversion weight source missing")


def _jax(path, document):
    # This trusted module's dataclass checks do not initialize JAX or load weights.
    from .jax_llm import DecoderConfig
    if path.stat().st_size > 256 * 1024:
        raise ValueError("checkpoint manifest exceeds 256 KiB")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration missing")
    try:
        profile = DecoderConfig(**config); shapes = profile.shapes()
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported checkpoint configuration") from error
    if document["format"] == "hyperl-llama-f32/1":
        if not {"format", "config", "tensors"} <= set(document) or set(document) - {"format", "config", "tensors", "provenance"} or not isinstance(document.get("tensors"), dict) or set(document["tensors"]) != set(shapes):
            raise ValueError("checkpoint tensor directory mismatch")
        if "provenance" in document:
            _provenance(document["provenance"], profile.context)
        records = document["tensors"].values()
        if any(not isinstance(record, dict) or set(record) != {"file", "sha256"} for record in records):
            raise ValueError("invalid checkpoint tensor record")
        entries = [(record["file"], record["sha256"]) for record in records]
    else:
        if set(document) != {"format", "config", "model_sha256", "backend", "dtype", "position", "files"} or document.get("dtype") != "f32" or document.get("backend") not in {"cpu", "gpu", "tpu"}:
            raise ValueError("unsupported checkpoint state")
        _integer(document["position"], 1, profile.context)
        if not isinstance(document.get("model_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", document["model_sha256"]):
            raise ValueError("checkpoint model identity missing")
        expected = {"logits.npy"} | {f"{kind}-{i}.npy" for i in range(profile.layers) for kind in ("keys", "values")}
        if not isinstance(document.get("files"), dict) or set(document["files"]) != expected:
            raise ValueError("checkpoint state directory mismatch")
        entries = list(document["files"].items())
    names = set(); total = 0
    for filename, digest in entries:
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npy") or filename in names:
            raise ValueError("checkpoint requires unique local NPY basenames")
        names.add(filename)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("checkpoint checksum missing")
        component = _child(path.parent, filename)
        total += _integer(component.stat().st_size, 1)
        if total > _MAX_FILE:
            raise ValueError("checkpoint files exceed 16 TiB inspection contract")
    return dict(component_count=len(entries))


def _result(kind, level, **details):
    return dict(format="hyperl-model-inspection/1", registry=REGISTRY_VERSION, detected_format=kind,
                validation=level, details=details, execution_ready=False, sharding_qualified=False,
                adapter=_REGISTRY.get(kind, {}).get("adapter", "no registered adapter"),
                opaque_storage_eligible=True, payload_values_checked=False, payload_integrity_verified=False,
                limits="Inspection does not execute models, verify publisher provenance, qualify numerical behavior or reserve memory. Opaque storage retains its separate quotas and integrity checks. Inspection is not an immutable file identity guarantee; paths can change concurrently.")


def inspect_model(source):
    path = Path(source)
    if path.is_symlink():
        raise ValueError("selected artifacts cannot be symlinks")
    if path.is_dir():
        if path.suffix.lower() in {".mlpackage", ".mlmodelc"}:
            return _result("coreml", "extension-hint-only")
        path = _child(path, "manifest.json")
    budget = _Budget()
    stream, size = _open(path)
    with stream:
        prefix = budget.read(stream, min(size, 24))
        if prefix.startswith(b"GGUF"):
            if len(prefix) != 24:
                raise ValueError("truncated GGUF header")
            version, tensors, metadata = struct.unpack("<IQQ", prefix[4:])
            if version not in (2, 3):
                return _result("gguf", "unsupported-version", version=version)
            _integer(tensors, 1, 1000000); _integer(metadata, 0, 1000000)
            return _result("gguf", "header-only", version=version, tensor_count=tensors, metadata_count=metadata)
        if path.suffix.lower() == ".json":
            if size > _MAX_HEADER:
                raise ValueError("JSON model metadata exceeds 16 MiB")
            stream.seek(0); document = _json(budget.read(stream, size))
            if "weight_map" in document:
                return _result("safetensors-index", "structure-validated", **_index(path, document, budget))
            if document.get("format") in ("hyperl-llama-f32/1", "hyperl-llama-kv/1"):
                return _result(document["format"], "manifest-validated-only", **_jax(path, document))
            return _result("unknown", "unsupported-format")
    if path.suffix.lower() == ".safetensors":
        inspected = _safetensors(path, budget)
        return _result("safetensors", "structure-validated", tensor_count=len(inspected["tensors"]),
                       dtypes=inspected["dtypes"], payload_bytes=inspected["payload_bytes"])
    kind = {".onnx": "onnx", ".tflite": "tflite", ".mlmodel": "coreml", ".pt": "pytorch", ".pth": "pytorch"}.get(path.suffix.lower(), "unknown")
    return _result(kind, "extension-hint-only" if kind != "unknown" else "unsupported-format")
