# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Compile a static HyperL tensor graph to verified StableHLO MLIR."""
import math
from dataclasses import dataclass

from .ir import (AttentionOp, DType, GEMMOp, GEMVOp, InputOp, ReturnOp,
                 RMSNormOp, SoftmaxOp)


@dataclass(frozen=True)
class Module:
    text: str
    inputs: tuple
    outputs: tuple
    specs: tuple

    def compile(self, device):
        """Load the verified StableHLO module on one explicitly selected PJRT device."""
        import jax
        import numpy as np
        from jax.lib import xla_client

        opts = xla_client.CompileOptions()
        opts.device_assignment = xla_client.DeviceAssignment.create(np.array([[device.id]], np.int32))
        executable = device.client.compile(self.text, compile_options=opts)

        def run(*arrays):
            if len(arrays) != len(self.specs):
                raise ValueError("wrong input count")
            for value, spec in zip(arrays, self.specs):
                if tuple(value.shape) != spec.shape or value.dtype != spec.dtype:
                    raise ValueError("input shape or dtype differs from compiled signature")
            return tuple(executable.execute([jax.device_put(a, device) for a in arrays]))

        return run

    def iree(self, target):
        """Ahead-of-time compile MLIR to an IREE VM flatbuffer using an installed compiler."""
        if target not in {"llvm-cpu", "vulkan-spirv", "metal-spirv", "cuda"}:
            raise ValueError("unsupported IREE target")
        from iree.compiler import compile_str
        return compile_str(self.text, input_type="stablehlo", target_backends=[target])


def lower(graph, *, max_tensor_bytes=256 << 20):
    """Bound logical tensor sizes before tracing. Compiler workspace is separate."""
    import jax
    import jax.numpy as jp

    if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
        raise ValueError("positive tensor budget required")
    if not 1 <= len(graph.inputs) <= 64 or not 2 <= len(graph.ops) <= 256:
        raise ValueError("graph exceeds operation/input bounds")
    if not graph.outputs or not isinstance(graph.ops[-1], ReturnOp):
        raise ValueError("final return required")
    types = {DType.F32: jp.float32, DType.BF16: jp.bfloat16, DType.F16: jp.float16}
    known = set()
    names = set()
    total = 0
    for op in graph.ops:
        if any(id(v) not in known for v in op.inputs):
            raise ValueError("value used before definition")
        for v in op.outputs:
            if v.name in names or v.type.dtype not in types:
                raise ValueError("duplicate value name or unsupported dtype")
            shape = v.type.shape
            if not 1 <= len(shape) <= 4 or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError("expected a static positive shape of rank 1 to 4")
            total += v.type.bytes()
            names.add(v.name)
            known.add(id(v))
        if isinstance(op, AttentionOp):
            q, k, _ = op.inputs
            total += 4 * math.prod(q.type.shape[:-1]) * k.type.shape[-2]
        if total > max_tensor_bytes:
            raise ValueError("logical tensor budget exceeded")
    if [id(v) for op in graph.ops if isinstance(op, InputOp) for v in op.outputs] != [id(v) for v in graph.inputs]:
        raise ValueError("input signature differs from graph")
    if [id(v) for v in graph.outputs] != [id(v) for v in graph.ops[-1].inputs]:
        raise ValueError("return signature differs from graph")

    def compute(*args):
        values = {id(v): arg for v, arg in zip(graph.inputs, args)}
        for op in graph.ops:
            if isinstance(op, InputOp):
                continue
            if isinstance(op, ReturnOp):
                if op is not graph.ops[-1]:
                    raise ValueError("early return")
                return tuple(values[id(v)] for v in op.inputs)
            a = [values[id(v)].astype(jp.float32) for v in op.inputs]
            x = a[0]
            if isinstance(op, (GEMMOp, GEMVOp)):
                rank = 2 if isinstance(op, GEMMOp) else 1
                if x.ndim != 2 or a[1].ndim != rank or x.shape[1] != a[1].shape[0]:
                    raise ValueError("incompatible matrix dimensions")
                y = jp.matmul(x, a[1], precision=jax.lax.Precision.HIGHEST)
                if len(a) == 3:
                    if a[2].shape != y.shape:
                        raise ValueError("bias shape mismatch")
                    y = y + a[2]
            elif isinstance(op, AttentionOp):
                q, k, v = a
                if (q.ndim not in (2, 3, 4) or q.ndim != k.ndim or k.ndim != v.ndim
                    or q.shape[:-2] != k.shape[:-2] or k.shape[:-1] != v.shape[:-1]
                    or q.shape[-1] != k.shape[-1]):
                    raise ValueError("attention shape mismatch")
                scores = jp.matmul(q, k.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
                scores *= q.shape[-1] ** -0.5
                if op.attributes["causal"]:
                    if q.shape[-2] != k.shape[-2]:
                        raise ValueError("causal self-attention requires equal sequence lengths")
                    mask = jp.arange(k.shape[-2]) <= jp.arange(q.shape[-2])[:, None]
                    scores = jp.where(mask, scores, -jp.inf)
                y = jp.matmul(jax.nn.softmax(scores, axis=-1), v, precision=jax.lax.Precision.HIGHEST)
            elif isinstance(op, RMSNormOp):
                eps = op.attributes["eps"]
                if not math.isfinite(eps) or eps <= 0 or (len(a) == 2 and a[1].shape != x.shape):
                    raise ValueError("invalid RMSNorm weight or epsilon")
                y = x * jax.lax.rsqrt(jp.mean(x * x) + eps)
                if len(a) == 2:
                    y *= a[1]
            elif isinstance(op, SoftmaxOp):
                y = jax.nn.softmax(x, axis=None)
            else:
                raise ValueError(f"unsupported operation {type(op).__name__}")
            out, = op.outputs
            if tuple(y.shape) != tuple(out.type.shape):
                raise ValueError("inferred output shape differs from graph declaration")
            values[id(out)] = y.astype(types[out.type.dtype])

    specs = tuple(jax.ShapeDtypeStruct(v.type.shape, types[v.type.dtype]) for v in graph.inputs)
    ir = jax.jit(compute).lower(*specs).compiler_ir(dialect="stablehlo")
    if not ir.operation.verify():
        raise ValueError("MLIR verifier rejected graph")
    return Module(str(ir), tuple(v.name for v in graph.inputs), tuple(v.name for v in graph.outputs), specs)
