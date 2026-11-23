# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Parse the complete, bounded CUDA pointwise kernel grammar before lowering to WGSL."""
import re

TOKEN = re.compile(r'\s+|//[^\n]*|/\*[\s\S]*?\*/|[A-Za-z_]\w*|[(){}\[\];,.=*+<>-]', re.ASCII)
NAME = re.compile(r'[A-Za-z_]\w*\Z', re.ASCII)


def lower(source):
    if len(source) > 65536:
        raise ValueError("Portable CUDA kernels are limited to 64 KiB")
    # Reject preprocessing and line splicing. Even a backslash in a // comment
    # changes the following line in C++ translation phase two.
    if "\\" in source or "??" in source:
        raise ValueError("Line splicing and trigraphs require the HIP or SYCL project route")
    tokens = []
    pos = 0
    while pos < len(source):
        match = TOKEN.match(source, pos)
        if not match:
            raise ValueError(f"Unsupported CUDA token on line {source.count(chr(10), 0, pos) + 1}")
        value = match[0]
        if not value.isspace() and not value.startswith(("//", "/*")):
            tokens.append(value)
        pos = match.end()
    tokens.append("<end>")
    cursor = 0

    def take(expected=None):
        nonlocal cursor
        value = tokens[cursor]
        if value == "<end>" or (expected is not None and value != expected):
            raise ValueError(f"Expected {expected or 'identifier'}, found {value} (token {cursor + 1})")
        cursor += 1
        return value

    def name():
        value = take()
        if not NAME.fullmatch(value) or value in {"const","int","unsigned","float","void","if","threadIdx","blockIdx","blockDim","__global__"}:
            raise ValueError(f"Invalid or reserved CUDA identifier: {value}")
        return value

    def integer():
        unsigned = tokens[cursor] == "unsigned"
        if unsigned:
            take("unsigned")
        take("int")
        return unsigned

    take("__global__")
    take("void")
    kernel = name()
    take("(")
    args = []
    while tokens[cursor] in ("const", "float"):
        readonly = tokens[cursor] == "const"
        if readonly:
            take("const")
        take("float")
        take("*")
        args.append((name(), readonly))
        if len(args) > 8:
            raise ValueError("Portable kernels accept up to eight float buffers")
        take(",")
    unsigned_count = integer()
    count = name()
    take(")")
    take("{")
    unsigned_index = integer()
    index = name()
    if len({a for a, _ in args} | {index, count}) != len(args) + 2:
        raise ValueError("Duplicate CUDA identifier")
    for value in "= blockIdx . x * blockDim . x + threadIdx . x ; if (".split():
        take(value)
    take(index)
    take("<")
    take(count)
    take(")")
    block = tokens[cursor] == "{"
    if block:
        take("{")
    bindings = {a: (n, const) for n, (a, const) in enumerate(args)}

    def access(read):
        ident = name()
        if ident not in bindings or bindings[ident][1] != read:
            raise ValueError(f"Expected {'const input' if read else 'writable output'} buffer, found {ident}")
        for value in ("[", index, "]"):
            take(value)
        return bindings[ident][0]

    destination = access(False)
    take("=")
    left = access(True)
    op = take()
    if op not in {"+", "-", "*"}:
        raise ValueError(f"Unsupported CUDA elementwise operation: {op}")
    right = access(True)
    take(";")
    if block:
        take("}")
    take("}")
    if tokens[cursor] != "<end>":
        raise ValueError(f"Extra code after kernel: {tokens[cursor]}")
    fields = [f"@group(0) @binding({n}) var<storage, {'read' if const else 'read_write'}> data{n}: array<f32>;" for n, (_, const) in enumerate(args)]
    fields += [f"@group(0) @binding({len(args)}) var<uniform> length: u32;",
               "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
               f"    if id.x < length {{ data{destination}[id.x] = data{left}[id.x] {op} data{right}[id.x]; }}", "}"]
    return "\n".join(fields) + "\n", {
        "operation": {"+": "Add", "-": "Sub", "*": "Mul"}[op], "operands": [left, right], "destination": destination,
        "sourceEntry": kernel, "entry": "main", "workgroupSize": [64, 1, 1],
        "buffers": [{"name": a, "group": 0, "binding": n, "access": "read" if const else "read_write"} for n, (a, const) in enumerate(args)],
        "lengthUniform": {"name": count, "binding": len(args), "type": "u32"},
        "maximumIndex": 4294967295 if unsigned_count and unsigned_index else 2147483647,
        "dispatchContract": "One-dimensional launch, non-overlapping buffers of at least length f32 values, nonnegative length, representable launch indices. Pad uniform allocation for the target ABI."
    }


def interchange(target, spec):
    """Emit one graph operation with an explicit source-to-model buffer mapping."""
    lhs, rhs = spec["operands"]
    out = spec["destination"]
    op = spec["operation"]
    symbol = {"Add": "+", "Sub": "-", "Mul": "*"}[op]
    if target == "opencl":
        args = [("__global const float *" if b["access"] == "read" else "__global float *") + f"data{b['binding']}" for b in spec["buffers"]]
        args.append("uint count")
        return f"__kernel void hyperl_kernel({', '.join(args)}) {{\n    size_t i = get_global_id(0);\n    if (i < count) data{out}[i] = data{lhs}[i] {symbol} data{rhs}[i];\n}}\n"
    # Graph importers pass precisely the active prefix. A repeated operand is one graph input.
    ids = list(dict.fromkeys([lhs, rhs]))
    spec["graphInputs"] = ids
    spec["graphOutput"] = out
    spec["graphContract"] = "Inputs have identical shape [length]. Assign the returned vector to the active prefix of the destination."
    if target == "stablehlo":
        signature = ", ".join(f"%data{i}: tensor<?xf32>" for i in ids)
        operation = {"Add": "add", "Sub": "subtract", "Mul": "multiply"}[op]
        return f"module {{\n  func.func @main({signature}) -> tensor<?xf32> {{\n    %out = stablehlo.{operation} %data{lhs}, %data{rhs} : tensor<?xf32>\n    return %out : tensor<?xf32>\n  }}\n}}\n"
    if target == "onnx-text":
        signature = ", ".join(f"float[N] data{i}" for i in ids)
        return f'<ir_version: 8, opset_import: ["": 13]>\nhyperl_pointwise ({signature}) => (float[N] result) {{\n  result = {op} (data{lhs}, data{rhs})\n}}\n'
    raise ValueError("Unsupported interchange format")
