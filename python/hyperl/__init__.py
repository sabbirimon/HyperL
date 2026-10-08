"""Experimental bounded HyperL Python/C ABI bridge. Full SDK remains later."""
from .native import NativeCpu, Program, Step, f32
from .recipes import (LIBRARY, recipe_program, add, multiply, relu, weighted_relu, residual_relu,
                      affine, affine_relu, residual_affine_relu, dot, sum_values, positive_sum, squared_norm)
from .interop import from_numpy, to_numpy, from_torch, to_torch

__version__ = "0.1.0-alpha.5"
__all__ = ["NativeCpu", "Program", "Step", "f32", "LIBRARY", "recipe_program", "add", "multiply", "relu",
           "weighted_relu", "residual_relu", "affine", "affine_relu", "residual_affine_relu", "dot",
           "sum_values", "positive_sum", "squared_norm", "from_numpy", "to_numpy", "from_torch", "to_torch"]
