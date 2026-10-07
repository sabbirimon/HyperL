"""Optional actual PyTorch elementwise dispatch with mandatory native CPU agreement."""
from .native import NativeCpu, Program, MAX_VECTOR
from .interop import from_torch
import re


class TorchBackend:
    """Explicit cpu/cuda:index only. No autograd, graph compile, reduction or fallback."""
    def __init__(self, reference, device):
        import torch
        if not isinstance(reference, NativeCpu):
            raise TypeError("mandatory configured native CPU reference required")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]{1,3}", device):
            raise ValueError("select cpu or an explicit cuda:index")
        self._torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and (not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count()):
            raise RuntimeError("requested CUDA device unavailable; no CPU fallback")
        self.reference = reference

    def execute(self, program, inputs, cancelled=None):
        torch = self._torch
        if not isinstance(program, Program) or set(inputs) != set(program.inputs):
            raise ValueError("program/input mismatch")
        if any(s.operation == "sum" for s in program.steps):
            raise ValueError("ordered reduction is not qualified for the PyTorch path")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        def check():
            if cancelled is not None and cancelled():
                raise RuntimeError("PyTorch work cancelled at host boundary; accepted device work may still complete")
        values = {}
        cpu_inputs = {}
        for name in program.inputs:
            check()
            value = inputs[name]
            if not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != torch.float32 or value.ndim != 1 or not value.is_contiguous() or value.requires_grad or not 1 <= value.numel() <= MAX_VECTOR:
                raise ValueError("inputs must be bounded contiguous float32 tensors on the selected device, without autograd")
            # This preview deliberately copies input snapshots to CPU for verification.
            cpu_inputs[name] = from_torch(value.detach().to("cpu"))
            values[name] = value.detach().clone()
        expected = self.reference.execute(program, cpu_inputs, cancelled=cancelled)
        with torch.no_grad():
            for step in program.steps:
                check()
                a = values[step.inputs[0]]
                value = a + values[step.inputs[1]] if step.operation == "add" else a * values[step.inputs[1]] if step.operation == "multiply" else torch.relu(a)
                if not torch.isfinite(value).all().item():
                    raise RuntimeError("nonfinite PyTorch intermediate: " + step.output)
                values[step.output] = value
        check()
        result = values[program.output]
        actual = from_torch(result.detach().to("cpu"))
        if len(actual) != len(expected) or any(a != b for a, b in zip(actual, expected)):
            raise RuntimeError("PyTorch result differs from native CPU reference")
        return result
