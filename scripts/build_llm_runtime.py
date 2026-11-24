#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Build owner-selected pinned local llama.cpp source. No clone/download/driver install."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.llm import LLAMA_REVISION, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    if not args.source.is_absolute() or not args.source.is_dir() or not args.output.is_absolute() or not 1 <= args.parallel <= 32:
        parser.error("absolute local source/output and bounded build parallelism required")
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != LLAMA_REVISION or dirty:
        parser.error("pinned clean source revision required; no automatic source replacement")
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_RPC=ON", "-DGGML_RPC_RDMA=OFF", "-DGGML_NATIVE=OFF",
             "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=ON",
             "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_TOOLS_INSTALL=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_APP=OFF"]
    flags += [f"-DGGML_{name.upper()}={'ON' if args.backend == name else 'OFF'}" for name in ("metal", "cuda", "vulkan")]
    subprocess.run(["cmake", "-S", str(args.source), "-B", str(args.output), *flags], check=True)
    subprocess.run(["cmake", "--build", str(args.output), "--config", "Release", "--target", "llama-server", "ggml-rpc-server", "--parallel", str(args.parallel)], check=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    binaries = {}
    for name in ("llama-server", "ggml-rpc-server"):
        matches = list((args.output / "bin").rglob(name + suffix))
        if len(matches) != 1:
            raise RuntimeError("ambiguous native build output")
        binaries[name] = dict(path=str(matches[0].resolve()), sha256=file_digest(matches[0].resolve()))
    manifest = dict(format="hyperl-native-llm-build/1", source="https://github.com/RunanywhereAI/llama.cpp", revision=commit,
                    license="MIT", backend=args.backend, flags=flags, binaries=binaries,
                    libraries={str(p.relative_to(args.output)): file_digest(p.resolve()) for p in (args.output / "bin").rglob("*")
                               if p.is_file() and p.suffix in (".dll", ".dylib", ".so")},
                    rpc_scope="Experimental trusted-owner protocol; loopback only or authenticated isolated tunnel")
    path = args.output / "hyperl-runtime-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(manifest=str(path), sha256=file_digest(path)), indent=2))


if __name__ == "__main__":
    main()
