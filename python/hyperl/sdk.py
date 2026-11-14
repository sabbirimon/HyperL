# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""HyperL SDK preview: explicit CPU/accelerator and managed model session entry points."""
from contextlib import contextmanager

from .native import NativeCpu
from .torch_backend import BackendUnavailable, TorchBackend
from .llm import LlamaSettings, LlamaSession
from .jax_llm import JaxSettings, JaxModelSession

SDK_API_VERSION = "hyperl-sdk/1-preview"


def open_compute(target="cpu", *, library=None, bundled=False, budget_bytes=16 * 1024 * 1024):
    """Select native CPU or verified PyTorch; choose exactly one native reference."""
    if type(bundled) is not bool or bundled == (library is not None):
        raise ValueError("select an explicit native library or bundled=True")
    cpu = (NativeCpu.bundled(budget_bytes=budget_bytes) if bundled else
           NativeCpu(library, budget_bytes=budget_bytes))
    return cpu if target == "cpu" else TorchBackend(cpu, target)


@contextmanager
def open_model(settings, *, cancelled=lambda: False):
    """Own one pinned GGUF or JAX session; release it on scope exit."""
    if isinstance(settings, LlamaSettings):
        factory = LlamaSession
    elif isinstance(settings, JaxSettings):
        factory = JaxModelSession
    else:
        raise TypeError("choose explicit LlamaSettings or JaxSettings")
    if not callable(cancelled):
        raise TypeError("cancelled must be callable")
    if cancelled():
        raise InterruptedError("model startup cancelled")
    session = factory(settings, cancelled=cancelled)
    try:
        yield session
    finally:
        session.close()
