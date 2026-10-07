"""Explicit copying bridges. Optional frameworks are never imported at package import."""
from array import array
from .native import f32, MAX_VECTOR


def from_numpy(values):
    import numpy as np
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32") or values.ndim != 1 or not values.flags.c_contiguous or not 1 <= values.size <= MAX_VECTOR:
        raise ValueError("expected bounded contiguous 1-D native-endian NumPy float32")
    return f32(values.tolist())


def to_numpy(values):
    import numpy as np
    if not isinstance(values, array) or values.typecode != "f":
        raise TypeError("expected array('f')")
    return np.frombuffer(f32(values), dtype=np.float32).copy()


def from_torch(values):
    import torch
    if not isinstance(values, torch.Tensor) or values.device.type != "cpu" or values.dtype != torch.float32 or values.ndim != 1 or not values.is_contiguous() or values.requires_grad or not 1 <= values.numel() <= MAX_VECTOR:
        raise ValueError("expected bounded contiguous CPU float32 tensor without autograd; no implicit device transfer")
    return f32(values.tolist())


def to_torch(values):
    import torch
    if not isinstance(values, array) or values.typecode != "f":
        raise TypeError("expected array('f')")
    return torch.tensor(f32(values).tolist(), dtype=torch.float32, device="cpu")
