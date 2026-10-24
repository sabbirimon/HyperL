#!/usr/bin/env python3
"""Owner-requested Samsung/desktop probe harness. No APK install or automatic phone taps.

Starts one temporary authenticated TLS CPU worker, prepares the app's private test
configuration, and records both physical devices' evidence. Enter wired / wireless,
press Start on the probe, then enter collect. Enter stop to remove test credentials,
the exact ADB reverse mapping and this worker. Never changes OS trust or firewalls.
"""
# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import secrets
import ssl
import subprocess
import struct
import tempfile
import threading
import time
import urllib.request
import uuid

PACKAGE = "org.hyperl.probe"


def expected_output_hash():
    # These binary fractions and products are exactly representable in f32.
    values = (max(0.0, ((i % 257 - 128) / 8.0) * ((i % 13 - 6) / 4.0)) for i in range(32768))
    return hashlib.sha256(b"".join(struct.pack(">f", value) for value in values)).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise RuntimeError("Worker redirect rejected")


def require_usb_device_path(value):
    """ADB reverse alone is not USB evidence: wireless ADB supports it too."""
    if not re.fullmatch(r"usb:[^\s\x00-\x1f]{1,200}", value):
        raise ValueError("Wired qualification requires an observed USB ADB transport")
    return value


def verify_passed(report, pending, after):
    """Fail closed before a report can claim actual cross-device success."""
    def require(condition):
        if not condition:
            raise ValueError("Cross-device qualification evidence did not match the controlled run")
    require(report.get("status") == "passed" and report.get("mode") == "split" and report.get("exact") is True)
    require(report.get("runId") == pending["runId"] and report.get("transport") == pending["transport"])
    require(report.get("origin") == pending["origin"] and report.get("remoteWorkerId") == "mac-probe")
    require(report.get("networkObservation", {}).get("boundToObservedWifi") is (pending["transport"] == "wireless-lan"))
    if pending["transport"] == "wired-adb-reverse":
        require_usb_device_path(pending.get("adbDevicePath", ""))
    require(report.get("elements") == 32768 and report.get("localRange") == [0, 16384] and report.get("remoteRange") == [16384, 32768])
    require(report.get("remoteShards") == 6 and report.get("measuredRemoteShards") == 5)
    require(after["instanceId"] == pending["hostInstanceId"] and after["processId"] == pending["hostProcessId"])
    require(after["completedShards"] - pending["hostCompletedBefore"] == 6)
    rows = report.get("measurements")
    require(isinstance(rows, list) and len(rows) == 6)
    jobs, hashes = set(), set()
    for index, row in enumerate(rows):
        require(row.get("iteration") == index and row.get("warmup") is (index == 0) and row.get("exact") is True)
        require(row.get("remoteProcessId") == pending["hostProcessId"] and row.get("remoteInstanceId") == pending["hostInstanceId"])
        job = row.get("jobId")
        require(str(uuid.UUID(job)) == job and job not in jobs)
        jobs.add(job)
        digest = row.get("outputSha256", "")
        require(re.fullmatch(r"[a-f0-9]{64}", digest))
        hashes.add(digest)
        for field in ("localComputeMs", "remoteComputeMs", "remoteRoundTripMs", "splitMs", "fullReferenceMs", "endToEndVerifiedMs"):
            value = row.get(field)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 180000)
        require(0 < row.get("sentPayloadBytes", 0) <= 8 * 1024 * 1024)
        require(0 < row.get("receivedPayloadBytes", 0) <= 8 * 1024 * 1024)
    require(hashes == {expected_output_hash()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--lan-address", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    address = ipaddress.ip_address(args.lan_address)
    if address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified:
        parser.error("Supply this host's private IPv4 LAN address")
    java_home = Path(os.environ["JAVA_HOME"])
    cli, adb = args.cli.resolve(), args.adb.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def phone(*commands, data=None, check=True):
        return subprocess.run([str(adb), "-s", args.serial, *commands], input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, timeout=30)

    def stage(name, data):
        # Names are fixed by this script. Neither credentials nor filenames come from a peer.
        if name not in {"worker.token", "worker-ca.pem", "connection.json"}:
            raise ValueError("Unknown private probe file")
        phone("shell", "run-as", PACKAGE, "sh", "-c", f"'cat > files/{name}'", data=data)

    worker = None
    mapped = False
    with tempfile.TemporaryDirectory(prefix="hyperl-phone-qualification-") as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        token = secrets.token_urlsafe(40)
        password = secrets.token_urlsafe(32)
        token_path, password_path = root / "worker.token", root / "key.password"
        token_path.write_text(token)
        password_path.write_text(password)
        token_path.chmod(0o600)
        password_path.chmod(0o600)
        keystore, cert = root / "worker.p12", root / "worker-ca.pem"
        keytool = java_home / "bin" / ("keytool.exe" if os.name == "nt" else "keytool")
        key_env = {**os.environ, "HYPERL_PROBE_KEY_PASSWORD": password}

        def key(*commands):
            subprocess.run([str(keytool), *map(str, commands)], env=key_env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)

        key("-genkeypair", "-alias", "worker", "-keyalg", "RSA", "-keysize", "2048",
            "-dname", "CN=HyperL temporary phone test", "-ext", f"SAN=IP:127.0.0.1,IP:{address}",
            "-validity", "2", "-storetype", "PKCS12", "-keystore", keystore,
            "-storepass:env", "HYPERL_PROBE_KEY_PASSWORD")
        key("-exportcert", "-rfc", "-alias", "worker", "-keystore", keystore,
            "-storepass:env", "HYPERL_PROBE_KEY_PASSWORD", "-file", cert)
        keystore.chmod(0o600)
        config = root / "worker.json"
        config.write_text(json.dumps({"id": "mac-probe", "bind": "0.0.0.0", "port": 0,
            "tokenFile": str(token_path), "scratchDirectory": str(root),
            "tlsKeystoreFile": str(keystore), "tlsPasswordFile": str(password_path),
            "maxShardElements": 16384, "memoryBudgetBytes": 64 * 1024 * 1024, "maxJobs": 1}))
        stderr = open(root / "worker.stderr", "w")
        try:
            worker = subprocess.Popen([str(cli), "worker", str(config)],
                env={**os.environ, "JAVA_OPTS": "", "JAVA_TOOL_OPTIONS": "-Xmx256m"},
                stdout=subprocess.PIPE, stderr=stderr, text=True)
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(worker.stdout.readline()), daemon=True).start()
            line = ready.get(timeout=30)
            match = re.search(r"listening at https://0[.]0[.]0[.]0:([0-9]+);", line)
            if not match:
                raise RuntimeError("Temporary TLS worker did not become ready")
            port = int(match[1])
            context = ssl.create_default_context(cafile=str(cert))
            opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))

            def observe():
                request = urllib.request.Request(f"https://127.0.0.1:{port}/hyperl/capabilities",
                                                  headers={"Authorization": "Bearer " + token})
                with opener.open(request, timeout=10) as response:
                    data = response.read(262145)
                    if len(data) > 262144:
                        raise RuntimeError("Oversized worker observations")
                    return json.loads(data)

            phone("shell", "run-as", PACKAGE, "mkdir", "-p", "files")
            stage("worker.token", token.encode())
            stage("worker-ca.pem", cert.read_bytes())
            # Do not replace an unrelated task's reverse mapping.
            mappings = phone("reverse", "--list").stdout.decode()
            if any("tcp:18091" in line.split() for line in mappings.splitlines()):
                raise RuntimeError("Phone port 18091 already mapped; remove/reassign its owner first")
            phone("reverse", "tcp:18091", f"tcp:{port}")
            mapped = True
            pending = None
            print(json.dumps({"status": "ready", "hostArchitecture": platform.machine(),
                              "workerId": "mac-probe", "transport": "TLS for both USB and Wi-Fi",
                              "commands": ["wired", "wireless", "collect", "stop"]}), flush=True)
            while True:
                command = input().strip().lower()
                if command in {"stop", "quit", "exit"}:
                    break
                if command in {"wired", "wireless"}:
                    transport = "wired-adb-reverse" if command == "wired" else "wireless-lan"
                    device_path = phone("get-devpath").stdout.decode().strip()
                    if command == "wired":
                        require_usb_device_path(device_path)
                    origin = "https://127.0.0.1:18091" if command == "wired" else f"https://{address}:{port}"
                    run_id = str(uuid.uuid4())
                    phone("shell", "run-as", PACKAGE, "rm", "-f", "files/probe-result.json")
                    stage("connection.json", json.dumps({"origin": origin, "transport": transport,
                        "workerId": "mac-probe", "runId": run_id}).encode())
                    baseline = observe()
                    pending = {"runId": run_id, "transport": transport, "origin": origin,
                               "adbDevicePath": device_path,
                               "hostCompletedBefore": baseline["completedShards"], "hostInstanceId": baseline["instanceId"],
                               "hostProcessId": baseline["processId"], "preparedAtEpochMs": int(time.time() * 1000)}
                    print(json.dumps({"status": "configured", "transport": transport,
                                      "next": "Press Start Samsung and Mac split test, then collect"}), flush=True)
                elif command == "collect":
                    if pending is None:
                        raise RuntimeError("Configure wired or wireless first")
                    # The phone may run a local check afterward. Collect the immutable
                    # prepared session, never whichever result happens to be latest.
                    report_path = "files/reports/" + str(uuid.UUID(pending["runId"])) + ".json"
                    raw = phone("exec-out", "run-as", PACKAGE, "head", "-c", str(128 * 1024 + 1), report_path).stdout
                    if len(raw) > 128 * 1024:
                        raise RuntimeError("Probe report exceeds limit")
                    report = json.loads(raw)
                    if report.get("runId") != pending["runId"]:
                        raise RuntimeError("Probe report belongs to another run")
                    after = observe()
                    completed = after["completedShards"] - pending["hostCompletedBefore"]
                    passed = report.get("status") == "passed"
                    qualification_error = None
                    if report.get("status") not in {"passed", "failed", "cancelled"}:
                        print("Test still running; wait for the app's result", flush=True)
                        continue
                    if passed:
                        try:
                            if pending["transport"] == "wired-adb-reverse":
                                current_path = require_usb_device_path(phone("get-devpath").stdout.decode().strip())
                                if current_path != pending["adbDevicePath"]:
                                    raise ValueError("USB transport changed during qualification")
                            verify_passed(report, pending, after)
                        except (ValueError, TypeError, KeyError) as error:
                            passed = False
                            qualification_error = type(error).__name__ + ": " + str(error)[:240]
                    evidence = {"format": "hyperl-android-desktop-qualification/1", **pending,
                        "hostArchitecture": platform.machine(), "hostSystem": platform.system(),
                        "hostWorkerProcessId": after["processId"], "hostCompletedDuringTest": completed,
                        "physicalDevicesAttempted": 2, "crossDeviceVerified": passed, "tlsConfigured": True, "android": report,
                        "qualificationError": qualification_error,
                        "scope": "Bounded vector split on one Android phone and one desktop; no model layers or GPU/NPU qualification"}
                    target = output / (pending["transport"] + ".json")
                    target.write_text(json.dumps(evidence, indent=2) + "\n")
                    print(json.dumps({"status": "qualification-rejected" if qualification_error else report.get("status"), "report": str(target),
                                      "observedRemoteShards": completed}), flush=True)
                    pending = None
                else:
                    print("Choose wired, wireless, collect or stop", flush=True)
        finally:
            # A disconnected phone must never prevent stopping the host listener.
            cleanup_errors = []
            if worker is not None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
                worker.stdout.close()
            stderr.close()
            commands = [("shell", "run-as", PACKAGE, "rm", "-f", "files/worker.token", "files/worker-ca.pem", "files/connection.json")]
            if mapped:
                commands.append(("reverse", "--remove", "tcp:18091"))
            for command in commands:
                try:
                    if phone(*command, check=False).returncode:
                        cleanup_errors.append("PhoneCleanupFailed")
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)
            if cleanup_errors:
                print("Host worker stopped; reconnect the phone to remove its probe credentials/mapping", flush=True)


if __name__ == "__main__":
    main()
