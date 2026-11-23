# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Execute translated pointwise ops using the caller's device-resident array framework."""
import operator
from .native import MAX_VECTOR


def execute(report, inputs, *, device, to_host):
    """Take binding-indexed arrays, an explicit DLPack device tuple and a host snapshot callback.

    The owner initializes the framework/device. to_host must synchronize and return a
    NumPy-compatible f32 snapshot. The result stays in the original framework.
    """
    import numpy as np
    if report.get("format") != "hyperl-migration/1" or len(device) != 2:
        raise ValueError("Expected a migration report and (DLDeviceType, device_id)")
    spec = report["interface"]
    operations = {"Add": operator.add, "Sub": operator.sub, "Mul": operator.mul}
    op = operations[spec["operation"]]
    indices = spec["operands"]
    if len(indices) != 2 or set(inputs) != set(indices):
        raise ValueError("Supply the exact operand bindings from the migration report")

    def snapshot(array):
        shape = tuple(array.shape)
        if len(shape) != 1 or not 1 <= shape[0] <= MAX_VECTOR:
            raise ValueError("Expected a bounded nonempty vector")
        if tuple(array.__dlpack_device__()) != tuple(device):
            raise ValueError("Array is on a different device from the selected one")
        # Reject mixed-precision arrays before a callback can coerce their dtype.
        if str(array.dtype).split(".")[-1] != "float32":
            raise ValueError("Expected device f32")
        host = np.asarray(to_host(array))
        if host.shape != shape or host.dtype != np.float32 or not np.isfinite(host).all():
            raise ValueError("Host snapshot must preserve finite f32 shape and dtype")
        return host.copy()

    refs = {key: snapshot(array) for key, array in inputs.items()}
    a, b = indices
    if refs[a].shape != refs[b].shape or type(inputs[a]) is not type(inputs[b]):
        raise ValueError("Operands must share a framework, device, shape and dtype")
    expected = op(refs[a], refs[b])
    if not np.isfinite(expected).all():
        raise ValueError("Result exceeds the finite f32 contract")
    result = op(inputs[a], inputs[b])
    actual = snapshot(result)
    if type(result) is not type(inputs[a]) or not np.array_equal(actual, expected):
        raise ValueError("Framework result differs from the f32 CPU reference")
    return result
