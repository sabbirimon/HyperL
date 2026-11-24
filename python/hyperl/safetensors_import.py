# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Explicit local standard-Llama safetensors -> hyperl-llama-f32/1 conversion.

No torch/pickle, remote code, downloads, tokenizer execution or device startup.
Only dense bias-free SiLU/GQA, untied embeddings and unscaled split-half RoPE.
F16/BF16 values are explicitly widened to f32; inference still takes token IDs.
Format reference: https://github.com/safetensors/safetensors#format
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile

from .jax_llm import DecoderConfig
from .llm import _integer, file_digest

HEADER_LIMIT = 4 * 1024 * 1024
AGGREGATE_HEADER_LIMIT = 16 * 1024 * 1024
DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2)}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(data):
    # Bound nesting before the JSON decoder builds objects (including on Python
    # versions without an interpreter integer-string limit).
    depth = 0; quoted = False; escaped = False
    for char in data:
        if quoted:
            if escaped: escaped = False
            elif char == 92: escaped = True
            elif char == 34: quoted = False
        elif char == 34: quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > 32: raise ValueError("JSON nesting exceeds bound")
        elif char in (93, 125): depth -= 1
    def integer(text):
        if len(text) > 20: raise ValueError("JSON integer exceeds bound")
        return int(text)
    def floating(text):
        if len(text) > 64: raise ValueError("JSON float exceeds bound")
        result = float(text)
        if not math.isfinite(result): raise ValueError("nonfinite JSON float")
        return result
    value = json.loads(data, object_pairs_hook=_object,
                       parse_int=integer, parse_float=floating,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _local(root, name):
    if (not isinstance(name, str) or not 1 <= len(name) <= 255 or Path(name).name != name
            or "\\" in name or ":" in name or any(ord(char) < 32 for char in name) or name in {".", ".."}):
        raise ValueError("source paths must be confined local basenames")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("source must be a regular file, never a symlink")
    return path


def _read_json(root, name):
    path = _local(root, name)
    if not 1 <= path.stat().st_size <= HEADER_LIMIT:
        raise ValueError("JSON metadata exceeds bound")
    with path.open("rb") as stream:
        data = stream.read(HEADER_LIMIT + 1)
    if len(data) > HEADER_LIMIT:
        raise ValueError("JSON metadata grew beyond bound")
    return _json(data), hashlib.sha256(data).hexdigest()


def decoder_config(value, context=None):
    """Reject architecture extensions instead of silently changing their math."""
    allowed = {"model_type", "architectures", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
               "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "rms_norm_eps", "rope_theta",
               "hidden_act", "attention_bias", "mlp_bias", "tie_word_embeddings", "rope_scaling", "head_dim",
               "bos_token_id", "eos_token_id", "pad_token_id", "torch_dtype", "dtype", "transformers_version",
               "_name_or_path", "use_cache", "initializer_range", "attention_dropout", "pretraining_tp"}
    if set(value) - allowed:
        raise ValueError("unsupported config fields: " + ", ".join(sorted(set(value) - allowed)))
    if value.get("model_type") != "llama" or value.get("architectures", ["LlamaForCausalLM"]) != ["LlamaForCausalLM"]:
        raise ValueError("only standard LlamaForCausalLM is supported")
    if value.get("hidden_act", "silu") != "silu" or value.get("rope_scaling") is not None:
        raise ValueError("only SiLU and unscaled RoPE are supported")
    if any(value.get(key, False) is not False for key in ("attention_bias", "mlp_bias", "tie_word_embeddings")):
        raise ValueError("biases and tied embeddings are not supported")
    if type(value.get("pretraining_tp", 1)) is not int or value.get("pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp must be 1")
    original_context = value["max_position_embeddings"]
    _integer(original_context, 1, 131072)
    selected_context = original_context if context is None else context
    _integer(selected_context, 1, original_context)
    cfg = DecoderConfig(value["vocab_size"], value["hidden_size"], value["intermediate_size"],
                        value["num_hidden_layers"], value["num_attention_heads"],
                        value.get("num_key_value_heads", value["num_attention_heads"]), selected_context,
                        value.get("rms_norm_eps", 1e-6), value.get("rope_theta", 10000.0))
    cfg.validate()
    if "head_dim" in value and (type(value["head_dim"]) is not int or value["head_dim"] != cfg.hidden // cfg.heads):
        raise ValueError("custom head_dim is unsupported")
    return cfg


def tensor_mapping(config):
    """Destination -> (HF source name, transpose), exact decoder names/shapes."""
    mapping = {"embedding": ("model.embed_tokens.weight", False), "norm": ("model.norm.weight", False),
               "head": ("lm_head.weight", True)}
    for i in range(config.layers):
        prefix = f"model.layers.{i}."
        mapping[f"{i}.attention_norm"] = (prefix + "input_layernorm.weight", False)
        mapping[f"{i}.ffn_norm"] = (prefix + "post_attention_layernorm.weight", False)
        for short in ("q", "k", "v", "o"):
            mapping[f"{i}.{short}"] = (prefix + "self_attn." + short + "_proj.weight", True)
        for short in ("gate", "up", "down"):
            mapping[f"{i}.{short}"] = (prefix + "mlp." + short + "_proj.weight", True)
    return mapping


def _header(stream, file_size, remaining_headers=AGGREGATE_HEADER_LIMIT):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("truncated safetensors prefix")
    length = struct.unpack("<Q", raw)[0]
    if not 2 <= length <= min(HEADER_LIMIT, remaining_headers) or length + 8 > file_size:
        raise ValueError("invalid or oversized safetensors header")
    raw = stream.read(length)
    if not raw.startswith(b"{"):
        raise ValueError("safetensors header must start with object")
    value = _json(raw)
    metadata = value.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("safetensors metadata must be string pairs")
    if not 1 <= len(value) <= 4096:
        raise ValueError("tensor count out of bounds")
    spans = []
    for name, info in value.items():
        if not 1 <= len(name) <= 256 or not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"} or not isinstance(info["dtype"], str) or info["dtype"] not in DTYPES:
            raise ValueError("only exact F32/F16/BF16 tensor descriptors are supported")
        shape = info["shape"]
        if not isinstance(shape, list) or not 1 <= len(shape) <= 2:
            raise ValueError("only vector/matrix tensors supported")
        for dim in shape:
            _integer(dim, 1, 131072)
        offsets = info["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("invalid tensor offsets")
        start, end = offsets
        for offset in offsets:
            _integer(offset, 0, file_size - 8 - length)
        if end - start != math.prod(shape) * DTYPES[info["dtype"]][1]:
            raise ValueError("tensor shape and byte span differ")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("overlapping tensors or holes in safetensors buffer")
        cursor = end
    if cursor != file_size - 8 - length:
        raise ValueError("unindexed safetensors trailing data")
    return value, 8 + length


def convert(source, destination, *, max_bytes, context=None, chunk_bytes=4 * 1024 * 1024, cancelled=lambda: False):
    """Convert into a new directory; max_bytes bounds source and output separately.

    No tokenizer is imported. Return the output manifest SHA-256 for explicit
    JaxDecoder.load trust. Owner must provision model-specific token IDs/tokenizer.
    Source and destination must be absolute, without symlink path components.
    """
    import numpy as np
    _integer(max_bytes, 1024, 1024**4)
    _integer(chunk_bytes, 1024, 64 * 1024 * 1024)
    root = Path(source); output = Path(destination)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("absolute real source directory required")
    if not output.is_absolute() or output.resolve() != output or output.exists() or not output.parent.is_dir():
        raise ValueError("new absolute destination in an existing real directory required")
    config_data, config_hash = _read_json(root, "config.json")
    cfg = decoder_config(config_data, context)
    shapes = cfg.shapes(); mapping = tensor_mapping(cfg)
    # Reserve bounded metadata/NPY headers before creating any output.
    projected = sum(math.prod(shape) * 4 + 256 for shape in shapes.values()) + 256 * 1024
    if projected > max_bytes:
        raise ValueError("expanded f32 checkpoint exceeds output quota")
    source_hashes = {"config.json": config_hash}
    index = root / "model.safetensors.index.json"
    if index.exists():
        index_data, index_hash = _read_json(root, index.name)
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or set(weight_map) != {item[0] for item in mapping.values()}:
            raise ValueError("shard index must contain exactly the required Llama tensors")
        if any(not isinstance(name, str) for name in weight_map.values()):
            raise ValueError("invalid shard filenames")
        filenames = sorted(set(weight_map.values()))
        source_hashes[index.name] = index_hash
    else:
        filenames = ["model.safetensors"]; weight_map = None
    if not 1 <= len(filenames) <= 128 or any(not name.endswith(".safetensors") for name in filenames):
        raise ValueError("invalid bounded safetensors shard set")
    source_total = sum(_local(root, name).stat().st_size for name in [*source_hashes, *filenames])
    if source_total > max_bytes:
        raise ValueError("source checkpoint exceeds quota")
    staging = None
    try:
        with ExitStack() as stack:
            tensors = {}; source_files = {}; remaining_headers = AGGREGATE_HEADER_LIMIT
            for name in filenames:
                if cancelled():
                    raise InterruptedError("conversion cancelled")
                path = _local(root, name)
                source_hashes[name] = file_digest(path, max_bytes=max_bytes, cancelled=cancelled)
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                stream = stack.enter_context(os.fdopen(fd, "rb"))
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                    raise ValueError("bounded regular source required")
                source_files[name] = (stream, info)
                header, base = _header(stream, info.st_size, remaining_headers)
                remaining_headers -= base - 8
                for tensor, descriptor in header.items():
                    if tensor in tensors or (weight_map is not None and weight_map.get(tensor) != name):
                        raise ValueError("duplicate tensor or mismatched shard index")
                    tensors[tensor] = (stream, base, descriptor)
            if set(tensors) != {item[0] for item in mapping.values()}:
                raise ValueError("source tensors differ from exact decoder contract")
            for name, (original, transpose) in mapping.items():
                expected = shapes[name][::-1] if transpose else shapes[name]
                if tuple(tensors[original][2]["shape"]) != expected:
                    raise ValueError("tensor shape mismatch: " + original)
            staging = Path(tempfile.mkdtemp(prefix=".hyperl-import-", dir=output.parent))
            manifest = {"format": "hyperl-llama-f32/1", "config": asdict(cfg), "tensors": {},
                        "provenance": {"importer": "hyperl-safetensors-llama/1", "sources": source_hashes,
                                       "source_format": "safetensors", "output_dtype": "float32",
                                       "tokenizer": "not imported; caller supplies matching token IDs",
                                       "original_context": config_data["max_position_embeddings"],
                                       "source_dtypes": sorted({v[2]["dtype"] for v in tensors.values()})}}
            tile = max(1, math.isqrt(chunk_bytes // 16))
            written = 0
            for name, (original, transpose) in mapping.items():
                stream, base, info = tensors[original]
                array = np.memmap(stream, mode="r", dtype=DTYPES[info["dtype"]][0],
                                  shape=tuple(info["shape"]), offset=base + info["data_offsets"][0])
                target = staging / (name + ".npy")
                result = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=shapes[name])
                try:
                    rows = array.shape[0]; columns = array.shape[1] if array.ndim == 2 else 1
                    for row in range(0, rows, tile):
                        for col in range(0, columns, tile):
                            if cancelled():
                                raise InterruptedError("conversion cancelled")
                            selection = (slice(row, row + tile), slice(col, col + tile)) if array.ndim == 2 else slice(row, row + tile)
                            block = array[selection]
                            block = (block.astype(np.uint32) << 16).view(np.float32) if info["dtype"] == "BF16" else block.astype(np.float32)
                            if not np.isfinite(block).all():
                                raise ValueError("nonfinite source tensor: " + original)
                            if transpose:
                                result[col:col + tile, row:row + tile] = block.T
                            else:
                                result[selection] = block
                    result.flush()
                finally:
                    array._mmap.close(); result._mmap.close()
                written += target.stat().st_size
                if written > max_bytes:
                    raise ValueError("output quota exceeded")
                manifest["tensors"][name] = {"file": target.name, "sha256": file_digest(target, max_bytes=max_bytes, cancelled=cancelled)}
            for name, digest in source_hashes.items():
                if file_digest(_local(root, name), max_bytes=max_bytes, cancelled=cancelled) != digest:
                    raise ValueError("source changed during conversion")
            for stream, before in source_files.values():
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("mapped source changed during conversion")
            encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
            if len(encoded) > 256 * 1024 or written + len(encoded) > max_bytes:
                raise ValueError("manifest/output quota exceeded")
            (staging / "manifest.json").write_bytes(encoded)
            if cancelled():
                raise InterruptedError("conversion cancelled before publication")
            if output.exists():
                raise ValueError("destination already exists")
            staging.rename(output); staging = None
            return {"directory": str(output), "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                    "format": manifest["format"], "bytes": written + len(encoded), "tokenizer_imported": False}
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="absolute local HF directory with config.json and safetensors")
    parser.add_argument("destination", type=Path, help="new absolute output directory")
    parser.add_argument("--max-bytes", type=int, required=True, help="source/output disk quota, each independently bounded")
    parser.add_argument("--context", type=int, help="explicit context reduction; cannot exceed source context")
    args = parser.parse_args()
    try:
        print(json.dumps(convert(args.source, args.destination, max_bytes=args.max_bytes, context=args.context), sort_keys=True))
    except (ValueError, KeyError, OSError, InterruptedError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
