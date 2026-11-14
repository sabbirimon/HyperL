"""Optional actual PyTorch elementwise dispatch with mandatory native CPU agreement."""
from .native import NativeCpu, Program, MAX_VECTOR, MAX_RETAINED
from .interop import from_torch
import re


class BackendUnavailable(RuntimeError):
    """Requested PyTorch runtime or device could not be selected."""


def _resolve_device(torch, selector):
    # ROCm uses the CUDA device namespace inside PyTorch, but the SDK requires
    # the user to name the actual runtime so its results cannot be mislabelled.
    match = re.fullmatch(r"(cuda|rocm|xpu):([0-9]{1,3})", selector) if isinstance(selector, str) else None
    if selector == "cpu":
        return "cpu"
    if match:
        target, index = match[1], int(match[2])
        if target == "xpu":
            runtime = getattr(torch, "xpu", None)
            if runtime and runtime.is_available() and index < runtime.device_count():
                return f"xpu:{index}"
        else:
            hip = getattr(torch.version, "hip", None)
            cuda = getattr(torch.version, "cuda", None)
            matches_build = bool(hip) if target == "rocm" else bool(cuda) and not hip
            if matches_build and torch.cuda.is_available() and index < torch.cuda.device_count():
                return f"cuda:{index}"
    else:
        raise ValueError("select cpu, cuda:index, rocm:index, xpu:index")
    raise BackendUnavailable(f"requested {selector} device/runtime unavailable; no CPU fallback")


class TorchBackend:
    """Verified f32 vector dispatch on an explicitly selected PyTorch runtime."""
    def __init__(self, reference, device):
        if not isinstance(reference, NativeCpu):
            raise TypeError("mandatory configured native CPU reference required")
        try:
            import torch
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            raise BackendUnavailable("PyTorch not installed; no CPU fallback") from error
        self.target = device
        self._torch = torch
        self.device = torch.device(_resolve_device(torch, device))
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
        retained_inputs = 0
        for name in program.inputs:
            check()
            value = inputs[name]
            if not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != torch.float32 or value.ndim != 1 or not value.is_contiguous() or value.requires_grad or not 1 <= value.numel() <= MAX_VECTOR:
                raise ValueError("inputs must be bounded contiguous float32 tensors on the selected device, without autograd")
            retained_inputs += value.numel()
            if retained_inputs > MAX_RETAINED or 4 * retained_inputs > self.reference.budget_bytes:
                raise MemoryError("PyTorch input snapshots exceed the selected budget")
        with torch.no_grad():
            for name in program.inputs:
                check()
                snapshot = inputs[name].detach().clone()
                values[name] = snapshot
                cpu_inputs[name] = from_torch(snapshot.to("cpu"))
            expected = self.reference.execute(program, cpu_inputs, cancelled=cancelled)
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
        check()
        if result.device != self.device or len(actual) != len(expected) or any(a != b for a, b in zip(actual, expected)):
            raise RuntimeError("PyTorch result differs from native CPU reference")
        return result
