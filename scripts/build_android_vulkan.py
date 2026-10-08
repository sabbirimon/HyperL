#!/usr/bin/env python3
"""Build a reviewed ARM64 Android Vulkan qualification bundle, without a phone/downloads."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
from pathlib import Path

NDK_VERSION = "28.2.13676358"
RECIPES = ("add", "multiply", "relu", "weighted_relu", "residual_relu", "affine", "affine_relu", "residual_affine_relu")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args, timeout=60):
    result = subprocess.run(list(map(str, args)), capture_output=True, timeout=timeout)
    if len(result.stdout) > 16*1024*1024 or len(result.stderr) > 65536:
        raise ValueError("Command output limit exceeded")
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[:2048])
    return result.stdout

def executable(path):
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Explicit absolute reviewed executable path required")
    return path

def build(ndk, cli, output):
    executable(cli)
    properties = (ndk / "source.properties").read_text()
    if f"Pkg.Revision = {NDK_VERSION}" not in properties:
        raise ValueError(f"Qualification toolchain pinned to NDK {NDK_VERSION}")
    host = {"Darwin":"darwin-x86_64", "Linux":"linux-x86_64"}.get(platform.system())
    if not host:
        raise ValueError("Build harness currently supports macOS/Linux NDK hosts")
    compiler = executable(ndk / f"toolchains/llvm/prebuilt/{host}/bin/aarch64-linux-android24-clang++")
    glslc = executable(ndk / f"shader-tools/{host}/glslc")
    validator = executable(ndk / f"shader-tools/{host}/spirv-val")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / "native/vulkan_bridge.cpp"
    bridge = output / "hyperl-vulkan"
    command([compiler,"-std=c++17","-O2","-Wall","-Wextra","-Werror","-ffp-contract=off","-static-libstdc++",source,"-lvulkan","-o",bridge])
    # Preserve HyperL and NDK runtime notices beside the statically linked C++ host.
    root=source.parents[1]
    for original,name in ((root/"LICENSE","LICENSE"),(root/"NOTICE","NOTICE"),
                          (ndk/"NOTICE.toolchain","NOTICE.android-toolchain")):
        shutil.copyfile(original,output/name)
    cases = []
    def case(name, program, values, rejection=False):
        folder = output/name; folder.mkdir()
        program_path=folder/"program.json"; program_path.write_text(json.dumps(program))
        inputs_path=folder/"inputs.json"; inputs_path.write_text(json.dumps(values))
        shader=folder/"kernel.comp"; shader.write_bytes(command([cli,"emit","VULKAN_SPIRV",program_path]))
        command([glslc,"--target-env=vulkan1.0","-fshader-stage=compute","-O0",shader,"-o",folder/"kernel.spv"])
        command([validator,"--target-env","vulkan1.0",folder/"kernel.spv"])
        n=len(next(iter(values.values())))
        for i,key in enumerate(sorted(program["inputs"])):
            if len(values[key]) != n: raise ValueError("Input length mismatch")
            (folder/f"input_{i}.bin").write_bytes(struct.pack(f"<{n}f",*values[key]))
        if rejection:
            result=subprocess.run([str(cli),"run",str(program_path),str(inputs_path)],capture_output=True,timeout=30)
            if result.returncode!=2 or b"nonfinite" not in result.stderr: raise ValueError("CPU did not reject overflow case")
        else:
            reference=json.loads(command([cli,"run",program_path,inputs_path]))
            (folder/"reference.json").write_text(json.dumps(reference))
        cases.append({"name":name,"elements":n,"inputs":len(values),"expectedNonfiniteRejection":rejection})
    weighted=None
    for recipe in RECIPES:
        workspace=json.loads(command([cli,"recipe",recipe])); case(recipe,workspace["program"],workspace["inputs"])
        if recipe=="weighted_relu": weighted=workspace["program"]
    for n in (257,65536):
        case(f"weighted_relu_{n}",weighted,{"x":[(i%17)-8 for i in range(n)],"w":[(i%7)-3 for i in range(n)]})
    case("fractional_weighted_relu",weighted,{"x":[-1.25,0.1,3.14159],"w":[2.5,0.3,-0.75]})
    # The bridge is exercised directly to verify overflow cannot be hidden by ReLU.
    # The production CPU preflight would reject this before dispatch.
    case("overflow_before_relu",weighted,{"x":[3.4028234663852886e38],"w":[2]},True)
    case("negative_overflow_before_relu",weighted,{"x":[-3.4028234663852886e38],"w":[2]},True)
    manifest={"format":"hyperl-android-vulkan-bundle/1","abi":"arm64-v8a","minimumAndroidApi":24,
              "ndk":NDK_VERSION,"sourceSha256":digest(source),"compilerVersion":command([compiler,"--version"]).decode()[:2048],
              "shaderCompilerVersion":command([glslc,"--version"]).decode()[:2048],"cases":cases,
              "files":{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk",required=True,type=Path)
    parser.add_argument("--cli",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    try:
        if not args.ndk.is_absolute() or not args.output.is_absolute(): raise ValueError("Absolute paths required")
        manifest=build(args.ndk,args.cli,args.output)
        print(json.dumps({"status":"built-not-device-tested","bundle":str(args.output),"cases":len(manifest["cases"])})); return 0
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        parser.exit(2,f"HyperL Android build failed: {error}\n")

if __name__=="__main__": raise SystemExit(main())
