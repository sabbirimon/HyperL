"""Small ready-made f32 preprocessing recipes, not a full model/tensor library."""
from .native import Program, Step

LIBRARY = ("add", "multiply", "relu", "weighted_relu", "residual_relu", "affine",
           "affine_relu", "residual_affine_relu", "dot", "sum", "positive_sum", "squared_norm")


def recipe_program(name):
    """Return an immutable bounded DAG. Reductions require a qualified CPU backend."""
    steps = []
    def step(output, operation, *inputs):
        steps.append(Step(output, operation, inputs))
        return output
    if name in ("add", "multiply"):
        inputs = ("x", "y") if name == "add" else ("x", "w")
        output = step("result", name, *inputs)
    elif name == "relu":
        inputs = ("x",); output = step("result", "relu", "x")
    elif name in ("weighted_relu", "dot"):
        inputs = ("x", "w"); product = step("product", "multiply", "x", "w")
        output = step("result", "relu" if name == "weighted_relu" else "sum", product)
    elif name == "residual_relu":
        inputs = ("x", "residual"); combined = step("combined", "add", "x", "residual")
        output = step("result", "relu", combined)
    elif name in ("affine", "affine_relu", "residual_affine_relu"):
        inputs = ("x", "w", "bias") + (("residual",) if name == "residual_affine_relu" else ())
        product = step("product", "multiply", "x", "w")
        output = step("biased", "add", product, "bias")
        if name == "residual_affine_relu":
            output = step("combined", "add", output, "residual")
        if name != "affine":
            output = step("result", "relu", output)
    elif name in ("sum", "positive_sum", "squared_norm"):
        inputs = ("x",); output = "x"
        if name == "positive_sum":
            output = step("positive", "relu", "x")
        elif name == "squared_norm":
            output = step("squares", "multiply", "x", "x")
        output = step("result", "sum", output)
    else:
        raise ValueError("Unknown ready-made recipe")
    return Program(inputs, tuple(steps), output)


def add(backend, values, other, *, cancelled=None):
    return backend.execute(recipe_program("add"), {"x": values, "y": other}, cancelled=cancelled)


def multiply(backend, values, weights, *, cancelled=None):
    return backend.execute(recipe_program("multiply"), {"x": values, "w": weights}, cancelled=cancelled)


def relu(backend, values, *, cancelled=None):
    return backend.execute(recipe_program("relu"), {"x": values}, cancelled=cancelled)


def affine(backend, values, weights, bias, *, cancelled=None):
    return backend.execute(recipe_program("affine"), {"x": values, "w": weights, "bias": bias}, cancelled=cancelled)


def affine_relu(backend, values, weights, bias, *, cancelled=None):
    return backend.execute(recipe_program("affine_relu"), {"x": values, "w": weights, "bias": bias}, cancelled=cancelled)


def residual_affine_relu(backend, values, weights, bias, residual, *, cancelled=None):
    return backend.execute(recipe_program("residual_affine_relu"), {"x": values, "w": weights, "bias": bias, "residual": residual}, cancelled=cancelled)


def dot(backend, values, weights, *, cancelled=None):
    return backend.execute(recipe_program("dot"), {"x": values, "w": weights}, cancelled=cancelled)


def sum_values(backend, values, *, cancelled=None):
    return backend.execute(recipe_program("sum"), {"x": values}, cancelled=cancelled)


def squared_norm(backend, values, *, cancelled=None):
    return backend.execute(recipe_program("squared_norm"), {"x": values}, cancelled=cancelled)

_WEIGHTED = Program(("x", "w"), (Step("product", "multiply", ("x", "w")), Step("positive", "relu", ("product",))), "positive")
_RESIDUAL = Program(("x", "residual"), (Step("combined", "add", ("x", "residual")), Step("positive", "relu", ("combined",))), "positive")
_SUM = Program(("x",), (Step("positive", "relu", ("x",)), Step("total", "sum", ("positive",))), "total")


def weighted_relu(backend, values, weights, *, cancelled=None):
    return backend.execute(_WEIGHTED, {"x": values, "w": weights}, cancelled=cancelled)


def residual_relu(backend, values, residual, *, cancelled=None):
    return backend.execute(_RESIDUAL, {"x": values, "residual": residual}, cancelled=cancelled)


def positive_sum(backend, values, *, cancelled=None):
    return backend.execute(_SUM, {"x": values}, cancelled=cancelled)
