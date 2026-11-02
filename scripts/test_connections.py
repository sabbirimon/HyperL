# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import asyncio
import pathlib
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
from hyperl.connections import RpcTunnel, client_tls, server_tls, LinkObservation, admit_link
from tls_fixtures import certificates


class LinkTest(unittest.TestCase):
    def test_measured_links_stale_security_and_tensor_latency(self):
        link = LinkObservation("a", "b", "WIFI", 100000000., 3., 1000, True)
        self.assertAlmostEqual(admit_link(link, now_ms=1001, bytes_per_token=100000)["transfer_seconds_per_token_estimate"], .004)
        with self.assertRaises(ValueError):
            admit_link(link, now_ms=1001, mode="tensor")
        with self.assertRaises(ValueError):
            admit_link(link, now_ms=40000)
        with self.assertRaises(ValueError):
            admit_link(link, now_ms=1001, bytes_per_token=1024**3, max_seconds_per_token=.1)
        self.assertTrue(admit_link(link, now_ms=1001, bytes_per_token=1024**3, max_seconds_per_token=.1, allow_slow_capacity=True)["capacity_only"])


@unittest.skipUnless(shutil.which("openssl"), "explicit OpenSSL CLI test prerequisite unavailable")
class TunnelTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_mutual_tls_binary_route_backpressure_and_wrong_peer_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            certs = await asyncio.to_thread(certificates, pathlib.Path(temporary))
            async def echo(reader, writer):
                try:
                    while data := await reader.read(65536):
                        writer.write(data); await writer.drain()
                finally:
                    writer.close(); await writer.wait_closed()
            native = await asyncio.start_server(echo, "127.0.0.1", 0)
            port = native.sockets[0].getsockname()[1]
            remote = await RpcTunnel(listen_tls=server_tls(*certs["server"]), target_port=port, allowed_peers=("cell:cell-a",)).start()
            local = await RpcTunnel(target_host="127.0.0.1", target_port=remote.port,
                target_tls=client_tls(*certs["cell:cell-a"]), server_hostname="127.0.0.1").start()
            bad = await RpcTunnel(target_host="127.0.0.1", target_port=remote.port,
                target_tls=client_tls(*certs["cell:unauthorized"]), server_hostname="127.0.0.1").start()
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", local.port)
                payload = bytes(range(256)) * 1024
                writer.write(payload); await writer.drain()
                self.assertEqual(await asyncio.wait_for(reader.readexactly(len(payload)), 10), payload)
                writer.close(); await writer.wait_closed()
                self.assertGreaterEqual(remote.forward_bytes, len(payload)); self.assertGreaterEqual(local.return_bytes, len(payload))
                reader, writer = await asyncio.open_connection("127.0.0.1", bad.port)
                writer.write(b"no access"); await writer.drain()
                self.assertEqual(await asyncio.wait_for(reader.read(1), 5), b"")
                writer.close(); await writer.wait_closed()
            finally:
                await local.close(); await bad.close(); await remote.close()
                native.close(); await native.wait_closed()


if __name__ == "__main__":
    unittest.main()
