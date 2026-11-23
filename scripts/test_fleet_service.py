# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import asyncio
import pathlib
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.connections import server_tls, client_tls
from hyperl.scheduler import FleetScheduler, Cell
from hyperl.fleet_service import FleetService, FleetClient
from tls_fixtures import certificates


@unittest.skipUnless(shutil.which("openssl"), "explicit OpenSSL test prerequisite unavailable")
class FleetServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_live_authorities_reject_misrouted_tenant(self):
        from hyperl.fleet import shard_for
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            certs = await asyncio.to_thread(certificates, root)
            schedulers = []
            servers = []
            try:
                for index in range(2):
                    ledger = FleetScheduler(root / f"{index}.db", audit_key=b"k"*32)
                    schedulers.append(ledger)
                    ledger.register_profile("model-a", "0"*64, "llama.cpp")
                    servers.append(await FleetService(ledger, server_tls(*certs["server"]),
                        partition_index=index, partition_count=2).start())
                tenant = next(k for k in certs if k.startswith("tenant:"))
                clients = [FleetClient(f"https://127.0.0.1:{server.port}",client_tls(*certs[tenant])) for server in servers]
                right = shard_for(tenant.split(":",1)[1], 2)
                request = {"jobs":[{"key":"test","profile":"model-a","pool":"pool-a","payload":{}}]}
                first = await asyncio.to_thread(clients[right].call, "/v1/jobs/submit", request)
                self.assertEqual(first, await asyncio.to_thread(clients[right].call, "/v1/jobs/submit", request))
                with self.assertRaises((ValueError,RuntimeError)):
                    await asyncio.to_thread(clients[1-right].call, "/v1/jobs/submit", request)
                self.assertEqual(schedulers[1-right].db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],0)
                with self.assertRaises(ValueError):
                    schedulers[right].bind_partition(0,1)
            finally:
                for server in servers:
                    await server.close()
                for ledger in schedulers:
                    ledger.close()

    async def test_mutual_tls_real_dispatch_authority_roles_cancel_and_result_fences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            certs = await asyncio.to_thread(certificates, root)
            scheduler = FleetScheduler(root / "jobs.sqlite", audit_key=b"x" * 32)
            cell = Cell("cell-a", "instance-a", 1, ("combined",), ("model-a",))
            scheduler.register_profile("model-a", "0" * 64, "llama.cpp")
            scheduler.register(cell)
            server = await FleetService(scheduler, server_tls(*certs["server"])).start()
            origin = f"https://127.0.0.1:{server.port}"
            tenant = FleetClient(origin, client_tls(*certs["tenant:tenant-a"]))
            agent = FleetClient(origin, client_tls(*certs["cell:cell-a"]))
            bad = FleetClient(origin, client_tls(*certs["cell:unauthorized"]))
            async def call(client, path, body):
                return await asyncio.to_thread(client.call, path, body)
            try:
                identifier = (await call(tenant, "/v1/jobs/submit", dict(jobs=[dict(key="job", pool="combined", profile="model-a", payload={"prompt": "hello"})])))[0]
                with self.assertRaises(RuntimeError):
                    await call(tenant, "/v1/cells/claim", dict(cell="cell-a", instance="instance-a", epoch=1))
                with self.assertRaises(RuntimeError):
                    await call(bad, "/v1/cells/claim", dict(cell="cell-a", instance="instance-a", epoch=1))
                receipt = await call(agent, "/v1/cells/claim", dict(cell="cell-a", instance="instance-a", epoch=1))
                await call(agent, "/v1/jobs/start", dict(receipt=receipt))
                await call(tenant, "/v1/jobs/cancel", dict(id=identifier))
                self.assertEqual((await call(agent, "/v1/jobs/renew", dict(receipt=receipt)))["state"], "CANCEL_REQUESTED")
                with self.assertRaises(RuntimeError):
                    await call(agent, "/v1/jobs/finish", dict(receipt=receipt, result={"tokens": [1]}, state="SUCCEEDED"))
                await call(agent, "/v1/jobs/finish", dict(receipt=receipt, result={"stopped": True}, state="CANCELLED"))
                self.assertEqual((await call(tenant, "/v1/jobs/get", dict(id=identifier)))["state"], "CANCELLED")
            finally:
                await server.close(); scheduler.close()


if __name__ == "__main__":
    unittest.main()
