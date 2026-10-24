"""Evidence validation fixtures, not physical-device results."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

from qualify_android_probe import NoRedirect, verify_passed, expected_output_hash, require_usb_device_path


class AndroidProbeEvidenceTest(unittest.TestCase):
    def fixture(self):
        pending = {"runId": str(uuid.uuid4()), "transport": "wired-adb-reverse",
                   "origin": "https://127.0.0.1:18091", "hostInstanceId": str(uuid.uuid4()),
                   "adbDevicePath": "usb:fixture-port",
                   "hostProcessId": 123, "hostCompletedBefore": 2}
        after = {"instanceId": pending["hostInstanceId"], "processId": 123, "completedShards": 8}
        rows = []
        for index in range(6):
            row = {"iteration": index, "warmup": index == 0, "exact": True,
                   "remoteProcessId": 123, "remoteInstanceId": pending["hostInstanceId"],
                   "jobId": str(uuid.uuid4()), "outputSha256": expected_output_hash(),
                   "sentPayloadBytes": 131072, "receivedPayloadBytes": 65536}
            row.update({key: 1.0 for key in ("localComputeMs", "remoteComputeMs", "remoteRoundTripMs",
                                           "splitMs", "fullReferenceMs", "endToEndVerifiedMs")})
            rows.append(row)
        report = {"status": "passed", "mode": "split", "exact": True,
                  "runId": pending["runId"], "transport": pending["transport"], "origin": pending["origin"],
                  "remoteWorkerId": "mac-probe", "elements": 32768, "localRange": [0, 16384],
                  "networkObservation": {"boundToObservedWifi": False},
                  "remoteRange": [16384, 32768], "remoteShards": 6, "measuredRemoteShards": 5, "measurements": rows}
        return report, pending, after

    def test_success_must_bind_controlled_jobs_identity_transport_and_exact_output(self):
        report, pending, after = self.fixture()
        verify_passed(report, pending, after)
        mutations = [lambda r: r.update(status="running"), lambda r: r.update(exact=False),
                     lambda r: r.update(runId=str(uuid.uuid4())), lambda r: r.update(transport="wireless-lan"),
                     lambda r: r.update(remoteRange=[16383, 32768]), lambda r: r["measurements"].pop(),
                     lambda r: r["measurements"][0].update(remoteProcessId=456),
                     lambda r: r["measurements"][0].update(jobId=r["measurements"][1]["jobId"]),
                     lambda r: r["measurements"][0].update(outputSha256="b" * 64),
                     lambda r: r["measurements"][0].update(localComputeMs=float("nan")),
                     lambda r: r["measurements"][0].update(localComputeMs=True)]
        mutations.append(lambda r: r["networkObservation"].update(boundToObservedWifi=True))
        for change in mutations:
            changed = copy.deepcopy(report)
            change(changed)
            with self.assertRaises(ValueError):
                verify_passed(changed, pending, after)
        with self.assertRaises(ValueError):
            verify_passed(report, pending, {**after, "completedShards": 7})
        with self.assertRaises(ValueError):
            verify_passed(report, pending, {**after, "instanceId": str(uuid.uuid4())})

    def test_wired_requires_usb_and_wireless_requires_observed_wifi(self):
        for invalid in ("", "unknown", "192.168.1.20:5555", "adb-serial._adb-tls-connect._tcp", "usb:", "usb:a\nb"):
            with self.assertRaises(ValueError):
                require_usb_device_path(invalid)
        report, pending, after = self.fixture()
        with self.assertRaises(ValueError):
            verify_passed(report, {**pending, "adbDevicePath": "192.168.1.20:5555"}, after)
        report["transport"] = pending["transport"] = "wireless-lan"
        with self.assertRaises(ValueError):
            verify_passed(report, pending, after)
        report["networkObservation"]["boundToObservedWifi"] = True
        verify_passed(report, pending, after)

    def test_python_optimization_cannot_disable_evidence_checks(self):
        script = "from qualify_android_probe import verify_passed\ntry:\n verify_passed({'status':'passed'}, {}, {})\nexcept ValueError:\n pass\nelse:\n raise RuntimeError('invalid evidence accepted')\n"
        result = subprocess.run([sys.executable, "-O", "-c", script], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_redirects_never_forward_the_worker_credentials(self):
        with self.assertRaises(RuntimeError):
            NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.invalid")


if __name__ == "__main__":
    unittest.main()
