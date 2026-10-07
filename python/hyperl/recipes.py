"""Small ready-made f32 preprocessing recipes, not a full model/tensor library."""
from .native import Program, Step

_WEIGHTED = Program(("x", "w"), (Step("product", "multiply", ("x", "w")), Step("positive", "relu", ("product",))), "positive")
_RESIDUAL = Program(("x", "residual"), (Step("combined", "add", ("x", "residual")), Step("positive", "relu", ("combined",))), "positive")
_SUM = Program(("x",), (Step("positive", "relu", ("x",)), Step("total", "sum", ("positive",))), "total")


def weighted_relu(backend, values, weights, *, cancelled=None):
    return backend.execute(_WEIGHTED, {"x": values, "w": weights}, cancelled=cancelled)


def residual_relu(backend, values, residual, *, cancelled=None):
    return backend.execute(_RESIDUAL, {"x": values, "residual": residual}, cancelled=cancelled)


def positive_sum(backend, values, *, cancelled=None):
    return backend.execute(_SUM, {"x": values}, cancelled=cancelled)
