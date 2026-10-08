#!/usr/bin/env python3
"""Test an explicitly built local HyperL bundle on one owner-authorized Samsung.
No root, downloads, APK install, Wi-Fi pairing, driver changes or CPU fallback.
"""
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import socket
import struct
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

def verify_bundle(folder):
    if not folder.is_absolute() or folder.is_symlink(): raise ValueError("Explicit absolute bundle directory required")
    manifest=json.loads((folder/"manifest.json").read_text())
    if manifest.get("format")!="hyperl-android-vulkan-bundle/1" or manifest.get("abi")!="arm64-v8a": raise ValueError("Bundle contract mismatch")
    files=manifest.get("files",{})
    if not isinstance(files,dict) or not 1<=len(files)<=128: raise ValueError("File count limit")
    for name,expected in files.items():
        path=Path(name)
        if path.is_absolute() or ".." in path.parts or any(not re.fullmatch(r"[a-zA-Z0-9_.-]+",part) for part in path.parts): raise ValueError("Unsafe bundle path")
        target=folder/path
        if target.is_symlink() or not target.is_file() or folder.resolve() not in target.resolve().parents: raise ValueError("Unsafe bundle file")
        if target.stat().st_size>16*1024*1024 or hashlib.sha256(target.read_bytes()).hexdigest()!=expected: raise ValueError("Bundle integrity mismatch")
    if "hyperl-vulkan" not in files: raise ValueError("Bridge missing")
    cases=manifest.get("cases",[])
    if not isinstance(cases,list) or not 1<=len(cases)<=16: raise ValueError("Case count limit")
    names=set()
    for case in cases:
        name=case["name"]
        if not isinstance(name,str) or not re.fullmatch(r"[a-zA-Z0-9_-]+",name) or name in names: raise ValueError("Unsafe case name")
        names.add(name)
        if type(case["elements"]) is not int or not 1<=case["elements"]<=262144 or type(case["inputs"]) is not int or not 1<=case["inputs"]<=8: raise ValueError("Case bounds invalid")
        if type(case["expectedNonfiniteRejection"]) is not bool: raise ValueError("Case expectation missing")
        required=[f"{name}/kernel.spv"]+[f"{name}/input_{i}.bin" for i in range(case["inputs"])]
        if not case["expectedNonfiniteRejection"]: required.append(f"{name}/reference.json")
        if any(f not in files for f in required): raise ValueError("Case artifact not integrity checked")
    return manifest

def select_gpu(probe, exact_name=None):
    if probe.get("format")!="hyperl-vulkan-probe/1" or probe.get("inferenceQualified") is not False: raise ValueError("Probe contract mismatch")
    devices=probe.get("devices")
    if not isinstance(devices,list) or len(devices)>32: raise ValueError("Device count limit")
    matches=[d for d in devices if d.get("hardwareGpu") is True and d.get("computeQueue") is True and (exact_name is None or d.get("name")==exact_name)]
    if len(matches)!=1: raise ValueError("Hardware GPU absent or ambiguous; specify exact --gpu-name; no fallback")
    chosen=matches[0]
    if type(chosen.get("index")) is not int or not 0<=chosen["index"]<32 or not isinstance(chosen.get("name"),str): raise ValueError("Invalid GPU metadata")
    return chosen

def validate_result(metadata, chosen, case, actual, reference):
    if metadata.get("format")!="hyperl-vulkan-result/1" or metadata.get("backend")!="VULKAN_GPU" or metadata.get("device")!=chosen["name"] or metadata.get("gpuCompletionConfirmed") is not True:
        raise ValueError("GPU identity or completion mismatch")
    if metadata.get("elements")!=case["elements"] or metadata.get("storageMode")!="host-visible-coherent": raise ValueError("GPU shape/storage mismatch")
    allocated=metadata.get("allocatedBytes")
    if type(allocated) is not int or not 0<allocated<=32*1024*1024: raise ValueError("Allocation evidence invalid")
    for key in ("compileMs","submitAndWaitMs"):
        v=metadata.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0: raise ValueError("Timing evidence invalid")
    if len(actual)!=case["elements"] or len(reference)!=len(actual): raise ValueError("Result length mismatch")
    if any(not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-6*max(1,abs(b)) for a,b in zip(actual,reference)): raise ValueError("CPU reference mismatch")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb",required=True,type=Path); parser.add_argument("--bundle",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path); parser.add_argument("--serial"); parser.add_argument("--gpu-name")
    parser.add_argument("--adb-port",type=int,default=5037,help="Existing owner-started loopback ADB server; never auto-started by the harness")
    args=parser.parse_args()
    if not args.adb.is_absolute() or not args.adb.is_file() or not os.access(args.adb,os.X_OK): parser.error("Explicit reviewed absolute ADB path required")
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink(): parser.error("Fresh absolute report path required")
    if not 1024<=args.adb_port<=65535: parser.error("ADB port outside bounds")
    report={"format":"hyperl-android-vulkan-qualification/1","timeUtc":datetime.now(timezone.utc).isoformat(),"status":"unrun","scope":"Listed bounded f32 kernels only; not model inference, production readiness or speedup","cases":[]}
    serial=None; remote=None; code=2
    def adb(arguments, check=True, timeout=30):
        # Do not auto-start an ADB daemon from a restricted process. The owner
        # starts the USB-capable loopback server in their normal Terminal.
        with socket.create_connection(("127.0.0.1",args.adb_port),timeout=2): pass
        invocation=[str(args.adb),"-H","127.0.0.1","-P",str(args.adb_port)]+(["-s",serial] if serial else [])+list(map(str,arguments))
        environment=dict(os.environ); environment.pop("ADB_SERVER_SOCKET",None); environment.pop("ANDROID_SERIAL",None)
        result=subprocess.run(invocation,capture_output=True,timeout=timeout,env=environment)
        if len(result.stdout)>16*1024*1024 or len(result.stderr)>65536: raise ValueError("ADB output limit")
        if check and result.returncode: raise ValueError(result.stderr.decode(errors="replace")[:2048])
        return result
    def shell(arguments,check=True):
        # ADB forwards a shell string. Quote every token, including owner-provided GPU indices.
        return adb(["shell"," ".join(shlex.quote(str(a)) for a in arguments)],check)
    def prop(name): return shell(["getprop",name]).stdout.decode().strip()[:128]
    try:
        manifest=verify_bundle(args.bundle); report["bundleManifestSha256"]=hashlib.sha256((args.bundle/"manifest.json").read_bytes()).hexdigest()
        report["sourceSha256"]=manifest["sourceSha256"]; report["ndk"]=manifest["ndk"]
        lines=adb(["devices"]).stdout.decode().splitlines()
        online=[line.split()[0] for line in lines if len(line.split())>=2 and line.split()[1]=="device"]
        if args.serial:
            if args.serial not in online: raise ValueError("Requested phone not online/authorized")
            serial=args.serial
        elif len(online)==1: serial=online[0]
        else: raise ValueError("Need one authorized phone, or explicit --serial; none or multiple available")
        manufacturer=prop("ro.product.manufacturer")
        if manufacturer.lower()!="samsung": raise ValueError("Selected phone is not Samsung; explicit target scope mismatch")
        api=prop("ro.build.version.sdk"); abi=prop("ro.product.cpu.abilist")
        if not api.isdecimal() or int(api)<24 or "arm64-v8a" not in abi.split(","): raise ValueError("Requires Android API 24+ and ARM64")
        report["phone"]={"manufacturer":manufacturer,"model":prop("ro.product.model"),"androidRelease":prop("ro.build.version.release"),"api":int(api),"abis":abi,"soc":prop("ro.soc.model"),"board":prop("ro.product.board")}
        remote="/data/local/tmp/hyperl-vulkan-"+uuid.uuid4().hex
        shell(["mkdir","-m","700",remote]); adb(["push",args.bundle/"hyperl-vulkan",remote+"/hyperl-vulkan"]); shell(["chmod","700",remote+"/hyperl-vulkan"])
        probe=json.loads(shell([remote+"/hyperl-vulkan","--probe"]).stdout); report["probe"]=probe
        chosen=select_gpu(probe,args.gpu_name); report["gpu"]=chosen
        for case in manifest["cases"]:
            name=case["name"]; target=remote+"/"+name; shell(["mkdir",target])
            folder=args.bundle/name
            for filename in ["kernel.spv"]+[f"input_{i}.bin" for i in range(case["inputs"])]: adb(["push",folder/filename,target+"/"+filename])
            start=time.perf_counter(); result=shell(["timeout","20",remote+"/hyperl-vulkan","--run",chosen["index"],case["elements"],case["inputs"],target],False)
            wall=(time.perf_counter()-start)*1000
            if case["expectedNonfiniteRejection"]:
                if result.returncode!=2 or b"Nonfinite output" not in result.stderr: raise ValueError("Native overflow-before-ReLU rejection missing")
                exists=shell(["test","-e",target+"/result.bin"],False)
                if exists.returncode!=1: raise ValueError("Rejected case published an output")
                report["cases"].append({"name":name,"status":"expected-rejection","gpuDispatched":True,"resultPublished":False}); continue
            if result.returncode: raise ValueError(result.stderr.decode(errors="replace")[:2048] or f"GPU exit {result.returncode}")
            metadata=json.loads(result.stdout)
            raw=adb(["exec-out","cat",target+"/result.bin"]).stdout
            if len(raw)!=case["elements"]*4: raise ValueError("Output byte count mismatch")
            actual=struct.unpack(f"<{case['elements']}f",raw); reference=json.loads((folder/"reference.json").read_text())
            validate_result(metadata,chosen,case,actual,reference)
            report["cases"].append({"name":name,"elements":case["elements"],"status":"pass","cpuVerified":True,"sample":actual[:3],"nativeMetrics":metadata,"adbRunAndWaitWallMs":wall})
        report["status"]="passed-listed-cases"; code=0
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
        report["status"]="failed"; report["reason"]=str(error)[:2048]
    finally:
        if remote:
            try:
                result=shell(["rm","-rf",remote],False); report["temporaryDeviceFilesRemoved"]=result.returncode==0
            except (OSError,ValueError,subprocess.TimeoutExpired): report["temporaryDeviceFilesRemoved"]=False
            if not report["temporaryDeviceFilesRemoved"]:
                report["cleanupPath"]=remote; report["status"]="failed"; report["reason"]="Temporary device cleanup failed"; code=2
    with args.output.open("x") as file: json.dump(report,file,indent=2); file.write("\n")
    print(json.dumps({"status":report["status"],"report":str(args.output),"reason":report.get("reason")})); return code

if __name__=="__main__": raise SystemExit(main())
