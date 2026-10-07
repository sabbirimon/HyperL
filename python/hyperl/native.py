"""Explicitly configured, version-checked C99 CPU reference; no binary search/download."""
from array import array
from dataclasses import dataclass
from pathlib import Path
import ctypes as c
import math
import re
from collections.abc import Mapping, Sequence

MAX_VECTOR = 262144
MAX_RETAINED = 1048576
OPS = {"add": 1, "multiply": 2, "relu": 3, "sum": 4}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def f32(values):
    """Explicit finite f32 conversion/copy from a sized sequence, never a generator."""
    if not isinstance(values, (Sequence, array)) or isinstance(values, (str, bytes)):
        raise TypeError("f32 requires a sized numeric sequence")
    if not 1 <= len(values) <= MAX_VECTOR:
        raise ValueError("vector length must be 1..262144")
    try:
        result = array("f", values)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("values must be representable finite f32") from exc
    if result.itemsize != 4 or not all(math.isfinite(v) for v in result):
        raise ValueError("values must be representable finite f32")
    return result


@dataclass(frozen=True)
class Step:
    output: str
    operation: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Program:
    inputs: tuple[str, ...]
    steps: tuple[Step, ...]
    output: str

    def __post_init__(self):
        if not isinstance(self.inputs, tuple) or not 1 <= len(self.inputs) <= 8:
            raise ValueError("program requires a tuple of 1..8 inputs")
        if not isinstance(self.steps, tuple) or not 1 <= len(self.steps) <= 64:
            raise ValueError("program requires a tuple of 1..64 steps")
        names = set()
        for name in self.inputs:
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
                raise ValueError("invalid or duplicate input name")
            names.add(name)
        for step in self.steps:
            if not isinstance(step, Step) or not isinstance(step.output, str) or not _NAME.fullmatch(step.output) or step.output in names:
                raise ValueError("invalid or duplicate step output")
            if not isinstance(step.operation, str) or step.operation not in OPS:
                raise ValueError("unsupported operation")
            count = 2 if step.operation in ("add", "multiply") else 1
            if not isinstance(step.inputs, tuple) or len(step.inputs) != count or any(not isinstance(x, str) or x not in names for x in step.inputs):
                raise ValueError("operands must reference earlier values with correct arity")
            names.add(step.output)
        if not isinstance(self.output, str) or self.output not in names:
            raise ValueError("output must name a defined value")

    @classmethod
    def from_dict(cls, data):
        """Import the existing declarative hyperl/1 shape, not executable Python."""
        if not isinstance(data, dict) or set(data) != {"format", "inputs", "instructions", "output"} or data["format"] != "hyperl/1":
            raise ValueError("expected exact hyperl/1 fields")
        if not isinstance(data["inputs"], list) or not 1 <= len(data["inputs"]) <= 8 or not isinstance(data["instructions"], list) or not 1 <= len(data["instructions"]) <= 64:
            raise ValueError("program declaration bounds exceeded")
        steps = []
        for item in data["instructions"]:
            if not isinstance(item, dict) or set(item) != {"output", "operation", "inputs"} or not isinstance(item["inputs"], list) or not 1 <= len(item["inputs"]) <= 2:
                raise ValueError("invalid step fields")
            steps.append(Step(item["output"], item["operation"], tuple(item["inputs"])))
        return cls(tuple(data["inputs"]), tuple(steps), data["output"])

    def lengths(self, values):
        if not isinstance(values, Mapping) or set(values) != set(self.inputs):
            raise ValueError("input keys must exactly match program inputs")
        lengths = {}
        for name in self.inputs:
            value = values[name]
            if not isinstance(value, array) or value.typecode != "f" or value.itemsize != 4 or not 1 <= len(value) <= MAX_VECTOR:
                raise ValueError("input must be a bounded array('f'); convert explicitly with f32/interop")
            lengths[name] = len(value)
        retained = sum(lengths.values())
        for step in self.steps:
            a = lengths[step.inputs[0]]
            if step.operation in ("add", "multiply") and a != lengths[step.inputs[1]]:
                raise ValueError("elementwise shapes must match; no broadcasting")
            lengths[step.output] = 1 if step.operation == "sum" else a
            retained += lengths[step.output]
        if retained > MAX_RETAINED:
            raise ValueError("retained graph exceeds 1048576 f32 values")
        return lengths, retained


class _Vector(c.Structure):
    _fields_ = [("data", c.POINTER(c.c_float)), ("length", c.c_size_t)]


class _Step(c.Structure):
    _fields_ = [("operation", c.c_int), ("a", c.c_size_t), ("b", c.c_size_t)]


_Cancel = c.CFUNCTYPE(c.c_int, c.c_void_p)


class NativeCpu:
    """Loads only the owner's explicit absolute reviewed library path. ABI 1."""
    def __init__(self, library, budget_bytes=16 * 1024 * 1024):
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("choose an existing absolute native library path")
        if not isinstance(budget_bytes, int) or isinstance(budget_bytes, bool) or not 1 <= budget_bytes <= 1024 ** 3:
            raise ValueError("budget must be 1 byte..1 GiB")
        self.budget_bytes = budget_bytes
        self._lib = c.CDLL(str(path.resolve()))
        self._lib.hl_version.argtypes = []
        self._lib.hl_version.restype = c.c_char_p
        if self._lib.hl_version() != b"hyperl-cpu/1":
            raise ValueError("unsupported native ABI/version")
        self._lib.hl_execute.argtypes = [c.POINTER(_Vector), c.c_size_t, c.POINTER(_Step), c.c_size_t, c.c_size_t, c.POINTER(c.c_float), c.c_size_t, c.POINTER(c.c_size_t), _Cancel, c.c_void_p]
        self._lib.hl_execute.restype = c.c_int

    def execute(self, program, inputs, cancelled=None):
        if not isinstance(program, Program):
            raise TypeError("expected Program")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        lengths, retained = program.lengths(inputs)
        estimate = 4 * (retained + sum(lengths[name] for name in program.inputs) + lengths[program.output]) + 65536
        if estimate > self.budget_bytes:
            raise MemoryError("native array/workspace estimate exceeds selected budget")
        # Private copies pin ctypes buffers and prevent caller mutation during native work.
        copies = [f32(inputs[name]) for name in program.inputs]
        buffers = [(c.c_float * len(v)).from_buffer(v) for v in copies]
        vectors = (_Vector * len(copies))(*[_Vector(buf, len(v)) for buf, v in zip(buffers, copies)])
        ids = {name: i for i, name in enumerate(program.inputs)}
        steps = []
        for step in program.steps:
            steps.append(_Step(OPS[step.operation], ids[step.inputs[0]], ids[step.inputs[1]] if len(step.inputs) == 2 else 0))
            ids[step.output] = len(ids)
        native_steps = (_Step * len(steps))(*steps)
        result = array("f", [0.0]) * lengths[program.output]
        out = (c.c_float * len(result)).from_buffer(result)
        count = c.c_size_t(0)
        callback_errors = []
        def check(_):
            try:
                return int(bool(cancelled()))
            except BaseException as exc:
                if not callback_errors:
                    callback_errors.append(exc)
                return 1
        callback = _Cancel(check) if cancelled is not None else _Cancel(0)
        status = self._lib.hl_execute(vectors, len(vectors), native_steps, len(steps), ids[program.output], out, len(result), c.byref(count), callback, None)
        if callback_errors:
            raise RuntimeError("cancellation callback failed; native cleanup completed") from callback_errors[0]
        if status != 0:
            raise RuntimeError({1: "native input invalid", 2: "native memory unavailable", 3: "native nonfinite result", 4: "native work cancelled"}.get(status, "unknown native status"))
        if count.value != len(result):
            raise RuntimeError("native output length mismatch")
        return result
