# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Offline CUDA source migration. Each run publishes into a new directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from accelerators import STACKS
from cuda_kernel import interchange, lower

HIPIFY = Path(__file__).resolve().parent / "vendor/hipify/hipify-perl"
HIPIFY_SHA = "7993ebf035c2336b90d398a956de72f893367042c3e2f1a5eb3a09c590dd5091"
SHADERS = {"wgsl": ".wgsl", "spirv": ".spv", "metal": ".metal", "hlsl": ".hlsl"}
EXT = {".cu", ".cuh", ".cc", ".cpp", ".c", ".h", ".hpp"}
INTERCHANGE = {"opencl": ".cl", "onnx-text": ".onnxtxt", "stablehlo": ".mlir"}
PORTABLE = {*SHADERS, *INTERCHANGE}
TARGETS = ("hip", "sycl", *SHADERS, *INTERCHANGE)


def perl():
    found = shutil.which("perl")
    if not found and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/usr/bin/perl.exe"
        if candidate.is_file():
            found = str(candidate)
    return found


def sources(source):
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = []
        visited = 0
        for base, dirs, files in os.walk(source):
            visited += len(dirs) + len(files)
            if visited > 20000 or any(Path(base, d).is_symlink() for d in dirs):
                raise ValueError("Too many directory entries or linked input directory")
            paths.extend(Path(base, f) for f in files if Path(f).suffix.lower() in EXT)
            if len(paths) > 128:
                raise ValueError("Choose up to 128 source files")
    else:
        raise ValueError("Choose existing CUDA source")
    if not paths or any(p.is_symlink() or not p.is_file() or p.suffix.lower() not in EXT for p in paths):
        raise ValueError("Expected regular CUDA or C/C++ source files")
    sizes = [p.stat().st_size for p in paths]
    if max(sizes) > 1048576 or sum(sizes) > 8388608:
        raise ValueError("Source limit is 1 MiB per file and 8 MiB per project")
    return sorted(paths)


def migrate(source, output, target, engine=None):
    source, output = Path(source).absolute(), Path(output).absolute()
    if target not in TARGETS or source.is_symlink() or output.is_symlink() or output.exists():
        raise ValueError("Choose a supported target and a new destination directory")
    root = source if source.is_dir() else source.parent
    if source.is_dir() and (output.resolve() == root.resolve() or root.resolve() in output.resolve().parents):
        raise ValueError("Choose a destination outside the source tree")
    paths = sources(source)
    if target in PORTABLE and (len(paths) != 1 or source.is_dir()):
        raise ValueError("Shader targets accept one standalone pointwise CUDA kernel")
    if target == "hip":
        tool = perl()
        if not tool:
            raise ValueError("HIP migration requires Perl 5 on PATH or Perl from Git for Windows")
        if hashlib.sha256(HIPIFY.read_bytes()).hexdigest() != HIPIFY_SHA:
            raise ValueError("Bundled HIPIFY integrity check failed")
    if target == "sycl":
        tool = shutil.which("c2s") or shutil.which("dpct")
        if not tool:
            raise ValueError("Select an installed SYCLomatic c2s or Intel dpct on PATH with CUDA headers configured")
    if target in SHADERS and not engine:
        raise ValueError("Select the HyperL engine executable for shader validation and translation")
    report = {"format": "hyperl-migration/1", "target": target, "compiledForDevice": False, "hardwareExecuted": False, "sources": []}
    deadline = time.monotonic() + 180

    def run(args, cwd, stdout=subprocess.PIPE):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("Migration time budget expired")
        result = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.PIPE, timeout=min(60, remaining))
        diagnostics = result.stderr.decode(errors="replace")
        if result.returncode:
            raise ValueError(f"Translator exited {result.returncode}: {diagnostics[-2000:]}")
        return diagnostics

    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        work = Path(temp)
        stage = work / "translated"
        stage.mkdir()
        inputs = work / "source"
        for p in paths:
            relative = p.relative_to(root)
            raw = p.read_bytes()
            if len(raw) > 1048576 or b"\0" in raw:
                raise ValueError("Expected bounded text input")
            copy = inputs / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(raw)
            record = {"path": relative.as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
            report["sources"].append(record)
            if target == "hip":
                dest = stage / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as stream:
                    diag = run([tool, str(HIPIFY), "-print-stats", str(copy)], work, stream)
                warnings = re.search(r"WARNINGS:\s*(\d+)", diag)
                record["upstreamWarnings"] = int(warnings[1]) if warnings else None
                record["diagnostics"] = diag[:8000].replace(str(inputs), ".")
                if not dest.stat().st_size and raw.strip():
                    raise ValueError("HIPIFY produced empty output")
            elif target in INTERCHANGE:
                _, bindings = lower(raw.decode("utf-8"))
                bindings["entry"] = {"stablehlo": "main", "opencl": "hyperl_kernel", "onnx-text": "hyperl_pointwise"}[target]
                text = interchange(target, bindings)
                (stage / ("kernel" + INTERCHANGE[target])).write_text(text, encoding="utf-8")
                report["interface"] = bindings
            elif target in SHADERS:
                shader, bindings = lower(raw.decode("utf-8"))
                intermediate = work / "kernel.wgsl"
                intermediate.write_text(shader, encoding="utf-8")
                with tempfile.TemporaryFile() as reflection:
                    run([str(Path(engine).resolve()), "translate-shader", str(intermediate), target, str(stage / ("kernel" + SHADERS[target]))], work, reflection)
                    reflection.seek(0)
                    entry = re.search(rb"^entry main (\w+)$", reflection.read(), re.MULTILINE)
                    if entry:
                        bindings["entry"] = entry[1].decode("ascii")
                if target == "metal":
                    bindings["bufferSizesSlot"] = 30
                report["interface"] = bindings
        if target == "sycl":
            units = [inputs / p.relative_to(root) for p in paths if p.suffix.lower() in {".cu", ".cpp", ".c", ".cc"}]
            if not units:
                raise ValueError("SYCL migration needs a translation unit")
            report["diagnostics"] = run([tool, "--in-root=" + str(inputs), "--out-root=" + str(stage), *map(str, units), "--", "-I" + str(inputs)], work)[:8000]
            if not any(p.is_file() for p in stage.rglob("*")):
                raise ValueError("SYCLomatic produced no source files")
        report["translator"] = {"hip": "AMD HIPIFY " + HIPIFY_SHA, "sycl": "SYCLomatic / dpct"}.get(target, "HyperL CUDA subset + Naga 24" if target in SHADERS else "HyperL CUDA subset")
        (stage / "migration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Exclusive directory creation reserves the destination even for concurrent callers.
        output.mkdir()
        try:
            for entry in stage.iterdir():
                entry.rename(output / entry.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return report


def main(argv=None, engine=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source")
    cli.add_argument("new_directory")
    choices = cli.add_mutually_exclusive_group()
    choices.add_argument("--target", choices=TARGETS)
    choices.add_argument("--for-stack", choices=sorted(STACKS))
    cli.add_argument("--engine", default=engine)
    args = cli.parse_args(argv)
    target = args.target or "hip"
    if args.for_stack:
        label, target, next_step, url = STACKS[args.for_stack]
        if target is None:
            cli.exit(2, f"{label} needs an SDK port. {next_step} {url}\n")
    try:
        result = migrate(args.source, args.new_directory, target, args.engine)
        print(f"{result['target']}: {len(result['sources'])} source(s). Review {Path(args.new_directory) / 'migration.json'} before target compilation.")
    except (ValueError, OSError, subprocess.SubprocessError) as err:
        cli.exit(1, str(err) + "\n")


if __name__ == "__main__":
    main()
