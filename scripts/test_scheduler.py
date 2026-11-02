# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import pathlib
import sys
import tempfile
import threading
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.scheduler import Cell, FleetScheduler


class SchedulerTest(unittest.TestCase):
    def request(self, key="one", profile="model-a"):
        return dict(key=key, pool="combined", profile=profile, payload=dict(prompt="hello", max_tokens=4))

    def cell(self, **changes):
        values = dict(id="a", instance="instance-a", epoch=1, pools=("combined",), profiles=("model-a",), slots=1)
        values.update(changes)
        return Cell(**values)

    def test_durable_idempotency_fairness_real_leases_and_fenced_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "jobs.sqlite"
            scheduler = FleetScheduler(path)
            scheduler.register(self.cell(), now=1000)
            ids = scheduler.submit("a", [self.request("a1"), self.request("a2")], now=1000)
            self.assertEqual(scheduler.submit("a", [self.request("a1")], now=1000), ids[:1])
            other = scheduler.submit("b", [self.request("b1")], now=1000)[0]
            first = scheduler.claim("a", "instance-a", 1, now=1001)
            self.assertEqual(first["id"], ids[0]); self.assertIsNone(scheduler.claim("a", "instance-a", 1, now=1001))
            scheduler.start(first, now=1002)
            digest = scheduler.finish(first, {"tokens": [1, 2]}, now=1003)
            self.assertEqual(digest, scheduler.finish(first, {"tokens": [1, 2]}, now=1004))
            with self.assertRaises(ValueError):
                scheduler.finish(first, {"tokens": [999]}, now=1004)
            second = scheduler.claim("a", "instance-a", 1, now=1005)
            self.assertEqual(second["id"], other)  # fair turn for another tenant
            scheduler.close()
            scheduler = FleetScheduler(path)
            try:
                self.assertEqual(scheduler.job("a", ids[0])["state"], "SUCCEEDED")
                self.assertEqual(scheduler.expire(now=31000), 1)
                self.assertEqual(scheduler.job("b", other)["state"], "UNCERTAIN")
                with self.assertRaises(ValueError):
                    scheduler.finish(second, {"tokens": [1]}, now=31001)
                scheduler.register(self.cell(instance="new", epoch=2), now=31001)
                third = scheduler.claim("a", "new", 2, now=31002)
                self.assertEqual(third["id"], ids[1])
            finally:
                scheduler.close()

    def test_cancel_epoch_and_owner_enrollment_do_not_replay_work(self):
        scheduler = FleetScheduler(":memory:")
        try:
            scheduler.register(self.cell(), now=1000)
            identifier = scheduler.submit("tenant", [self.request()], now=1000)[0]
            receipt = scheduler.claim("a", "instance-a", 1, now=1001)
            scheduler.start(receipt, now=1002)
            scheduler.cancel("tenant", identifier, now=1003)
            self.assertEqual(scheduler.renew(receipt, now=1004)["state"], "CANCEL_REQUESTED")
            with self.assertRaises(ValueError):
                scheduler.finish(receipt, {"tokens": [1]}, now=1005)
            scheduler.finish(receipt, {"stopped": True}, state="CANCELLED", now=1005)
            identifier2 = scheduler.submit("tenant", [self.request("two")], now=1006)[0]
            receipt2 = scheduler.claim("a", "instance-a", 1, now=1007)
            scheduler.register(self.cell(instance="restarted", epoch=2), now=1008)
            self.assertEqual(scheduler.job("tenant", identifier2)["state"], "UNCERTAIN")
            with self.assertRaises(ValueError):
                scheduler.finish(receipt2, {}, now=1009)
            scheduler.revoke("a", now=1010)
            with self.assertRaises(ValueError):
                scheduler.heartbeat(self.cell(instance="restarted", epoch=2), now=1011)
            with self.assertRaises(ValueError):
                scheduler.register(self.cell(instance="restarted", epoch=2), now=1011)
        finally:
            scheduler.close()

    def test_atomic_batches_unknown_profiles_quotas_and_clock_rollback(self):
        scheduler = FleetScheduler(":memory:", max_jobs=2, max_pending_per_tenant=2)
        try:
            scheduler.register(self.cell(), now=1000)
            wrong = self.request(); wrong["payload"] = {"prompt": "different"}
            with self.assertRaises(ValueError):
                scheduler.submit("a", [self.request(), wrong], now=1001)
            self.assertEqual(scheduler.page("a"), [])
            scheduler.submit("a", [self.request("a", profile="unsupported")], now=1001)
            self.assertIsNone(scheduler.claim("a", "instance-a", 1, now=1002))
            with self.assertRaises(ValueError):
                scheduler.submit("a", [self.request("b")], now=999)
            scheduler.submit("a", [self.request("b")], now=1003)
            with self.assertRaises(ValueError):
                scheduler.submit("a", [self.request("c")], now=1004)
        finally:
            scheduler.close()

    def test_owner_registered_model_digest_and_revocation_fence_publication(self):
        scheduler = FleetScheduler(":memory:", audit_key=b"x" * 32)
        try:
            scheduler.require_profiles = True
            scheduler.register_profile("model-a", "a" * 64, "llama.cpp", now=1000)
            scheduler.register(self.cell(), now=1000)
            identifier = scheduler.submit("tenant", [self.request()], now=1001)[0]
            receipt = scheduler.claim("a", "instance-a", 1, now=1002)
            self.assertEqual(receipt["artifact_sha256"], "a" * 64)
            scheduler.start(receipt, now=1003)
            with self.assertRaises(ValueError):
                scheduler.finish(receipt, {"model_sha256": "b" * 64}, now=1004)
            self.assertEqual(scheduler.job("tenant", identifier)["state"], "RUNNING")
            scheduler.revoke_profile("model-a", now=1005)
            with self.assertRaises(ValueError):
                scheduler.finish(receipt, {"model_sha256": "a" * 64}, now=1006)
            self.assertEqual(scheduler.job("tenant", identifier)["state"], "UNCERTAIN")
            with self.assertRaises(ValueError):
                scheduler.register_profile("model-a", "a" * 64, "llama.cpp", now=1007)
        finally:
            scheduler.close()

    def test_two_authority_connections_never_claim_the_same_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "jobs.sqlite"
            scheduler = FleetScheduler(path)
            scheduler.register(self.cell(id="a"), now=1000)
            scheduler.register(self.cell(id="b", instance="instance-b"), now=1000)
            scheduler.submit("tenant", [self.request("one"), self.request("two")], now=1000)
            barrier = threading.Barrier(2)
            result = []
            errors = []
            def claim(cell, instance):
                connection = FleetScheduler(path)
                try:
                    barrier.wait(timeout=5)
                    result.append(connection.claim(cell, instance, 1, now=1001)["id"])
                except Exception as error:
                    errors.append(error)
                finally:
                    connection.close()
            a = threading.Thread(target=claim, args=("a", "instance-a"))
            b = threading.Thread(target=claim, args=("b", "instance-b"))
            a.start(); b.start(); a.join(10); b.join(10)
            scheduler.close()
            self.assertEqual(errors, [])
            self.assertEqual(len(set(result)), 2)


if __name__ == "__main__":
    unittest.main()
