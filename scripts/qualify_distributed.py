#!/usr/bin/env python3
"""Explicit local qualification: independent JVM workers and trusted TLS. No external services."""
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import argparse
import json
import os
from pathlib import Path
import queue
import re
import secrets
import ssl
import subprocess
import tempfile
import threading
import urllib.request


def qualify(cli, output, bundled=False):
    processes = []
    with tempfile.TemporaryDirectory(prefix="hyperl-distributed-") as temporary:
        root = Path(temporary)
        if os.name != "nt":
            root.chmod(0o700)
        base_env = {**os.environ, "JAVA_OPTS": "", "JAVA_TOOL_OPTIONS": "-Xmx256m"}
        if bundled:
            base_env.pop("JAVA_HOME", None)
            base_env["PATH"] = ""
        def run(args, env=None, expect=True):
            result = subprocess.run([str(cli), *map(str, args)], env=env or base_env,
                                    capture_output=True, text=True, timeout=90)
            if expect and result.returncode:
                raise RuntimeError("HyperL CLI qualification failed: " + result.stderr[-2048:])
            return result
        def config(name, value):
            path = root / (name + ".json")
            path.write_text(json.dumps(value))
            return path
        def start(identifier, tls=None):
            token_path = root / (identifier + ".token")
            run(["worker-token", token_path])
            value = {"id": identifier, "port": 0, "tokenFile": str(token_path),
                     "scratchDirectory": str(root), "maxShardElements": 1024}
            if tls:
                value.update(tls)
            path = config(identifier, value)
            stderr = open(root / (identifier + ".stderr"), "w")
            process = subprocess.Popen([str(cli), "worker", str(path)], env=base_env,
                                       stdout=subprocess.PIPE, stderr=stderr, text=True)
            processes.append((process, stderr))
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True).start()
            line = ready.get(timeout=30)
            match = re.search(r"listening at (https?://127[.]0[.]0[.]1:[0-9]+);", line)
            if not match:
                raise RuntimeError("Worker did not start; review private fixture stderr")
            return {"id": identifier, "origin": match[1], "tokenFile": str(token_path)}
        def observe(endpoint, context=None):
            request = urllib.request.Request(endpoint["origin"] + "/hyperl/capabilities",
                       headers={"Authorization": "Bearer " + Path(endpoint["tokenFile"]).read_text().strip()})
            with urllib.request.urlopen(request, timeout=10, context=context) as response:
                return json.load(response)
        try:
            a, b = start("a"), start("b")
            observations = [observe(a), observe(b)]
            assert len({node["processId"] for node in observations}) == 2
            program = config("program", {"format":"hyperl/1", "inputs":["x","w"],
                "instructions":[{"output":"y","operation":"multiply","inputs":["x","w"]},
                                {"output":"r","operation":"relu","inputs":["y"]}], "output":"r"})
            inputs = config("inputs", {"x":list(range(-123, 3973)), "w":[2]*4096})
            settings = config("cluster", {"workers":[a,b], "maxParallelism":2, "shardElements":512})
            result = json.loads(run(["distributed-run", settings, program, inputs]).stdout)
            local = json.loads(run(["run", program, inputs]).stdout)
            assert result["result"] == local
            assert {part["workerId"] for part in result["shards"]} == {"a","b"}
            assert len(result["shards"]) == 8 and result["cpuVerified"]
            assert not result["plan"]["physicalMemoryPooled"]
            assert all(observe(node)["completedShards"] > 0 for node in [a,b])
            plan = json.loads(run(["distributed-plan", settings, program, inputs]).stdout)
            assert len(plan["shards"]) == 8
            # TLS fixture: private key stays in the worker's temporary PKCS12, outside JSON/CLI literals.
            keytool = Path(os.environ["JAVA_HOME"]) / "bin" / ("keytool.exe" if os.name == "nt" else "keytool")
            key_password = secrets.token_urlsafe(24)
            key_env = {**os.environ, "HYPERL_TEST_KEY_PASSWORD":key_password}
            keystore, cert, trust = root/"worker.p12", root/"worker.pem", root/"trust.p12"
            def key(args, env=key_env):
                subprocess.run([str(keytool), *map(str,args)], env=env, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
            key(["-genkeypair","-alias","worker","-keyalg","RSA","-keysize","2048",
                 "-dname","CN=HyperL local test","-ext","SAN=IP:127.0.0.1","-validity","2",
                 "-storetype","PKCS12","-keystore",keystore,"-storepass:env","HYPERL_TEST_KEY_PASSWORD"])
            key(["-exportcert","-rfc","-alias","worker","-keystore",keystore,
                 "-storepass:env","HYPERL_TEST_KEY_PASSWORD","-file",cert])
            # This trust store contains a public certificate only.
            key(["-importcert","-noprompt","-alias","worker","-storetype","PKCS12","-keystore",trust,
                 "-storepass","fixture-public-trust","-file",cert])
            password_file = root/"key.password"
            password_file.write_text(key_password)
            if os.name != "nt":
                password_file.chmod(0o600)
                keystore.chmod(0o600)
            tls = start("tls", {"tlsKeystoreFile":str(keystore), "tlsPasswordFile":str(password_file)})
            tls_observation = observe(tls, ssl.create_default_context(cafile=str(cert)))
            tls_config = config("tls-cluster", {"workers":[tls], "maxParallelism":1, "shardElements":1024})
            rejected = run(["distributed-run", tls_config, program, inputs], expect=False)
            assert rejected.returncode != 0, "Untrusted certificate was accepted"
            trusted_env = {**base_env, "JAVA_TOOL_OPTIONS": "-Xmx256m -Djavax.net.ssl.trustStore=" + str(trust)
                           + " -Djavax.net.ssl.trustStoreType=PKCS12 -Djavax.net.ssl.trustStorePassword=fixture-public-trust"}
            tls_result = json.loads(run(["distributed-run", tls_config, program, inputs], env=trusted_env).stdout)
            assert tls_result["result"] == local and tls_result["cpuVerified"]
            wrong_host = config("wrong-host", {"workers":[{**tls, "origin":tls["origin"].replace("127.0.0.1", "localhost")}], "maxParallelism":1})
            assert run(["distributed-plan", wrong_host, program, inputs], env=trusted_env, expect=False).returncode != 0, "TLS hostname mismatch was accepted"
            evidence = {"format":"hyperl-distributed-qualification/1", "status":"passed",
                        "scope":"Independent JVM processes on one host, HTTP loopback and trusted TLS loopback only",
                        "workerProcessIds":[node["processId"] for node in observations],
                        "cpuWorkers":2, "shards":8, "elements":4096,
                        "endToEndMs":result["endToEndMs"], "tlsProcessId":tls_observation["processId"],
                        "trustedTlsPassed":True, "untrustedCertificateRejected":True, "hostnameMismatchRejected":True,
                        "remoteHostsTested":False, "mixedGpuNpuTested":False,
                        "speedupClaimed":False}
            evidence["bundledRuntimeWithoutExternalJava"] = bundled
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(evidence, indent=2)+"\n")
            print(json.dumps(evidence, indent=2))
        finally:
            for process, stderr in processes:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process.stdout.close()
                stderr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundled-runtime", action="store_true")
    args = parser.parse_args()
    qualify(args.cli.resolve(), args.output, args.bundled_runtime)
