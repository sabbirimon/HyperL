# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""GGUF metadata and tensor views. For generation use hyperl.llm.LlamaSession."""

import struct
import mmap
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian


class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ2_XXS = 15


class TensorInfo:
    def __init__(self, name: str, shape: Tuple[int, ...], ggml_type: int, offset: int):
        self.name = name
        self.shape = shape
        self.ggml_type = ggml_type
        self.offset = offset


class GGUFModel:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Model file not found: {file_path}")

        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.data_offset = 0

        self._file = open(self.path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse()

    def _read_str(self, pos: int) -> Tuple[str, int]:
        length = struct.unpack_from("<Q", self._mmap, pos)[0]
        pos += 8
        s = self._mmap[pos:pos + length].decode("utf-8", errors="replace")
        return s, pos + length

    def _read_val(self, pos: int, val_type: int) -> Tuple[Any, int]:
        if val_type == GGUFValueType.UINT8:
            return struct.unpack_from("<B", self._mmap, pos)[0], pos + 1
        elif val_type == GGUFValueType.INT8:
            return struct.unpack_from("<b", self._mmap, pos)[0], pos + 1
        elif val_type == GGUFValueType.UINT16:
            return struct.unpack_from("<H", self._mmap, pos)[0], pos + 2
        elif val_type == GGUFValueType.INT16:
            return struct.unpack_from("<h", self._mmap, pos)[0], pos + 2
        elif val_type == GGUFValueType.UINT32:
            return struct.unpack_from("<I", self._mmap, pos)[0], pos + 4
        elif val_type == GGUFValueType.INT32:
            return struct.unpack_from("<i", self._mmap, pos)[0], pos + 4
        elif val_type == GGUFValueType.FLOAT32:
            return struct.unpack_from("<f", self._mmap, pos)[0], pos + 4
        elif val_type == GGUFValueType.UINT64:
            return struct.unpack_from("<Q", self._mmap, pos)[0], pos + 8
        elif val_type == GGUFValueType.INT64:
            return struct.unpack_from("<q", self._mmap, pos)[0], pos + 8
        elif val_type == GGUFValueType.FLOAT64:
            return struct.unpack_from("<d", self._mmap, pos)[0], pos + 8
        elif val_type == GGUFValueType.BOOL:
            return bool(self._mmap[pos]), pos + 1
        elif val_type == GGUFValueType.STRING:
            return self._read_str(pos)
        elif val_type == GGUFValueType.ARRAY:
            sub_type, count = struct.unpack_from("<IQ", self._mmap, pos)
            pos += 12
            arr = []
            for _ in range(count):
                v, pos = self._read_val(pos, sub_type)
                arr.append(v)
            return arr, pos
        else:
            raise ValueError(f"Unsupported GGUF metadata type: {val_type}")

    def _parse(self):
        magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", self._mmap, 0)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        self.version = version
        pos = 24

        # Read Metadata KV
        for _ in range(kv_count):
            key, pos = self._read_str(pos)
            val_type = struct.unpack_from("<I", self._mmap, pos)[0]
            pos += 4
            val, pos = self._read_val(pos, val_type)
            self.metadata[key] = val

        # Read Tensor Infos
        for _ in range(tensor_count):
            name, pos = self._read_str(pos)
            ndim = struct.unpack_from("<I", self._mmap, pos)[0]
            pos += 4
            shape = []
            for _ in range(ndim):
                d = struct.unpack_from("<Q", self._mmap, pos)[0]
                shape.append(d)
                pos += 8
            ggml_type = struct.unpack_from("<I", self._mmap, pos)[0]
            pos += 4
            offset = struct.unpack_from("<Q", self._mmap, pos)[0]
            pos += 8
            self.tensors[name] = TensorInfo(name, tuple(shape), ggml_type, offset)

        # Align to 32 bytes (default GGUF alignment)
        alignment = self.metadata.get("general.alignment", 32)
        self.data_offset = (pos + alignment - 1) & ~(alignment - 1)

    def get_tensor_data(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise KeyError(f"Tensor {name} not found in model")
        info = self.tensors[name]
        start = self.data_offset + info.offset

        if info.ggml_type == GGMLType.F32:
            numel = np.prod(info.shape)
            return np.frombuffer(self._mmap, dtype=np.float32, count=numel, offset=start).reshape(info.shape[::-1])
        elif info.ggml_type == GGMLType.F16:
            numel = np.prod(info.shape)
            return np.frombuffer(self._mmap, dtype=np.float16, count=numel, offset=start).astype(np.float32).reshape(info.shape[::-1])
        elif info.ggml_type == GGMLType.Q4_0:
            # Q4_0 raw buffer
            numel = np.prod(info.shape)
            n_blocks = numel // 32
            block_bytes = n_blocks * 18
            return np.frombuffer(self._mmap, dtype=np.uint8, count=block_bytes, offset=start)
        else:
            raise NotImplementedError(f"Tensor type {info.ggml_type} not yet implemented")

    def close(self):
        if hasattr(self, "_mmap") and self._mmap:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_file") and self._file:
            self._file.close()
            self._file = None
