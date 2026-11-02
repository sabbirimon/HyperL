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
