# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Graph construction for the StableHLO MLIR compiler."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import enum


class DType(enum.Enum):
    F32 = "f32"
    BF16 = "bf16"
    F16 = "f16"
    Q4_0 = "q4_0"
    INT8 = "i8"
    INT32 = "i32"


@dataclass
class Type:
    dtype: DType
    shape: Tuple[int, ...]

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def bytes(self) -> int:
        if self.dtype == DType.F32:
            return self.numel() * 4
        elif self.dtype in (DType.F16, DType.BF16):
            return self.numel() * 2
        elif self.dtype == DType.Q4_0:
            return (self.numel() // 32) * 18
        elif self.dtype == DType.INT8:
            return self.numel()
        elif self.dtype == DType.INT32:
            return self.numel() * 4
        return self.numel() * 4


@dataclass
class Value:
    name: str
    type: Type
    producer: Optional['Op'] = None

    def __repr__(self):
        return f"%{self.name}: {self.type.dtype.value}{list(self.type.shape)}"


@dataclass
class Op:
    inputs: List[Value]
    outputs: List[Value]
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def op_name(self) -> str:
        return self.__class__.__name__


class InputOp(Op): pass
class RMSNormOp(Op): pass
class RoPEOp(Op): pass
class GEMVOp(Op): pass
class GEMMOp(Op): pass
class SoftmaxOp(Op): pass
class AttentionOp(Op): pass
class DequantizeQ4Op(Op): pass
class ReturnOp(Op): pass


class Graph:
    def __init__(self, name: str = "hyperl_subgraph"):
        self.name = name
        self.inputs: List[Value] = []
        self.ops: List[Op] = []
        self.outputs: List[Value] = []
        self._var_counter = 0

    def new_value(self, type_: Type, hint: str = "v") -> Value:
        v = Value(name=f"{hint}_{self._var_counter}", type=type_)
        self._var_counter += 1
        return v

    def add_input(self, name: str, type_: Type) -> Value:
        v = Value(name=name, type=type_)
        op = InputOp(inputs=[], outputs=[v])
        v.producer = op
        self.inputs.append(v)
        self.ops.append(op)
        return v

    def rmsnorm(self, x: Value, weight: Optional[Value] = None, eps: float = 1e-5) -> Value:
        out = self.new_value(x.type, "norm")
        inputs = [x] if weight is None else [x, weight]
        op = RMSNormOp(inputs=inputs, outputs=[out], attributes={"eps": eps})
        out.producer = op
        self.ops.append(op)
        return out

    def gemv(self, W: Value, x: Value, bias: Optional[Value] = None) -> Value:
        out_shape = (W.type.shape[0],)
        out = self.new_value(Type(DType.F32, out_shape), "gemv")
        inputs = [W, x] if bias is None else [W, x, bias]
        op = GEMVOp(inputs=inputs, outputs=[out])
        out.producer = op
        self.ops.append(op)
        return out

    def softmax(self, x: Value) -> Value:
        out = self.new_value(x.type, "smax")
        op = SoftmaxOp(inputs=[x], outputs=[out])
        out.producer = op
        self.ops.append(op)
        return out

    def matmul(self, a: Value, b: Value) -> Value:
        if len(a.type.shape) != 2 or len(b.type.shape) != 2 or a.type.shape[1] != b.type.shape[0]:
            raise ValueError("incompatible matrices")
        out = self.new_value(Type(DType.F32, (a.type.shape[0], b.type.shape[1])), "matmul")
        out.producer = GEMMOp([a, b], [out])
        self.ops.append(out.producer)
        return out

    def attention(self, q: Value, k: Value, v: Value, *, causal=False) -> Value:
        if type(causal) is not bool or min(len(t.type.shape) for t in (q, k, v)) < 2:
            raise ValueError("expected attention matrices and boolean causal flag")
        out = self.new_value(Type(DType.F32, (*q.type.shape[:-1], v.type.shape[-1])), "attention")
        out.producer = AttentionOp([q, k, v], [out], {"causal": causal})
        self.ops.append(out.producer)
        return out

    def lower(self, *, max_tensor_bytes=256 << 20):
        from .mlir_engine import lower
        return lower(self, max_tensor_bytes=max_tensor_bytes)

    def set_outputs(self, values: List[Value]):
        self.outputs = values
        self.ops.append(ReturnOp(inputs=values, outputs=[]))


class MemoryPlanner:
    """Computes exact tensor liveness intervals and assigns arena offsets."""
    @staticmethod
    def plan(graph: Graph) -> Tuple[Dict[str, int], int]:
        first_def = {}
        last_use = {}

        for idx, op in enumerate(graph.ops):
            for out in op.outputs:
                first_def[out.name] = idx
                last_use[out.name] = idx
            for inp in op.inputs:
                if inp.name not in first_def:
                    first_def[inp.name] = 0
                last_use[inp.name] = idx

        # Persistent weights/inputs are kept outside the scratch arena
        input_names = {v.name for v in graph.inputs}
        scratch_values = [v for op in graph.ops for v in op.outputs if v.name not in input_names]

        # Greedy interval coloring for minimal memory peak
        offsets: Dict[str, int] = {}
        active_allocations: List[Tuple[int, int, str]] = [] # (start_offset, end_offset, name)
        current_peak = 0

        for val in scratch_values:
            start_step = first_def[val.name]
            # Release memory whose last use was before start_step
            active_allocations = [
                (start, end, name)
                for start, end, name in active_allocations
                if last_use[name] >= start_step
            ]

            size = val.type.bytes()
            # Find first fitting gap or allocate at end
            active_sorted = sorted(active_allocations, key=lambda a: a[0])
            chosen_offset = 0
            for start, end, _ in active_sorted:
                chosen_offset = (chosen_offset + 63) & ~63
                if chosen_offset + size <= start:
                    break
                chosen_offset = max(chosen_offset, end)

            # 64-byte alignment
            chosen_offset = (chosen_offset + 63) & ~63
            offsets[val.name] = chosen_offset
            active_allocations.append((chosen_offset, chosen_offset + size, val.name))
            current_peak = max(current_peak, chosen_offset + size)

        return offsets, current_peak
