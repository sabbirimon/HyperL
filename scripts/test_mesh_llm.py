# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Real loopback HTTP/TLS contract tests; fixtures are not model inference proof."""
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import shutil
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.mesh_llm import MeshClient, MeshError, MeshSettings, MAX_RESPONSE, read_document
from tls_fixtures import certificates

CATALOG = {"object": "list", "data": [{"id": "fixture-model", "object": "model", "created": 0, "owned_by": "mesh-llm"}]}
COMPLETION = {"object": "chat.completion", "model": "fixture-model", "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "fixture answer"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


class MeshConfigTest(unittest.TestCase):
    def test_explicit_format_and_scope_no_credential_values(self):
        value = dict(format="hyperl-mesh-llm/1", base_url="http://127.0.0.1:9337/v1", routing_scope="private-mesh")
        self.assertEqual("private-mesh", MeshSettings.from_document(value).routing_scope)
        for changes in [dict(format="future"), dict(api_key="secret"), dict(routing_scope=[]), dict(token_env="token-value!"),
                        dict(allow_remote="true"), dict(timeout_seconds=True), dict(timeout_seconds=float("inf"))]:
            with self.subTest(changes=changes), self.assertRaises(MeshError):
                MeshSettings.from_document(dict(value, **changes))
        for missing in ("base_url", "routing_scope"):
            with self.assertRaises(MeshError):
                MeshSettings.from_document({key: val for key, val in value.items() if key != missing})

    def test_endpoint_rejects_unsafe_transports_queries_and_credentials(self):
        for url in ["http://localhost:9337/v1", "http://192.168.1.3/v1", "ftp://127.0.0.1/v1", "https://example.test/v1",
                    "http://user:secret@127.0.0.1/v1", "http://127.0.0.1/v1?token=secret", "http://127.0.0.1/v1#secret",
                    "http://127.0.0.1/evil", "http://127.0.0.1:0/v1", "http://127.0.0.1:99999/v1", "http://127.0.0.1\r\n/v1"]:
            with self.subTest(url=url), self.assertRaises(MeshError):
                MeshSettings(url, "private-mesh").validate()
        MeshSettings("http://[::1]:9337/v1/", "local-model-only").validate()
        MeshSettings("https://approved.example/v1", "private-mesh", token_env="OWNER_TOKEN", allow_remote=True).validate()

    def test_document_limits_duplicate_keys_and_private_parse_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for data in [b'{"secret":"private-sentinel",', b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":'+b'['*17+b'0'+b']'*17+b'}', b'x'*16385]:
                path.write_bytes(data)
                with self.assertRaises(MeshError) as caught:
                    read_document(path, 16384)
                self.assertNotIn("private-sentinel", str(caught.exception))
            if hasattr(os, "mkfifo"):
                fifo = Path(directory) / "fifo"
                os.mkfifo(fifo)
                with self.assertRaises(MeshError):
                    read_document(fifo)
            with self.assertRaises(MeshError):
                read_document(directory)


class MeshHttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.tasks = set()
        self.writers = set()
        self.response = CATALOG
        self.raw = None
        self.hold = False
        self.received = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.client = MeshClient(MeshSettings(f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1", "private-mesh"))

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        for writer in self.writers.copy():
            writer.close()
        for task in self.tasks.copy():
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task); self.writers.add(writer)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
            body = await reader.readexactly(length)
            self.requests.append((headers, body))
            self.received.set()
            if self.hold:
                await reader.read()
                self.disconnected.set()
                return
            body = json.dumps(self.response).encode()
            raw = self.raw if self.raw is not None else b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            writer.write(raw)
            await writer.drain()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self.writers.discard(writer); self.tasks.discard(task)

    async def test_models_and_unknown_placement_are_explicit(self):
        result = await self.client.models()
        self.assertEqual([dict(id="fixture-model", owned_by="mesh-llm")], result["models"])
        self.assertFalse(result["routing_scope_verified"])
        self.assertFalse(result["sharding_qualified"])
        self.assertFalse(result["model_artifact_verified"])
        self.assertTrue(self.requests[0][0].startswith(b"GET /v1/models HTTP/1.1"))
        self.response = {"object": "list", "data": []}
        self.assertEqual([], (await self.client.models())["models"])

    async def test_chat_uses_wire_contract_and_actual_usage_not_text_counts(self):
        self.response = COMPLETION
        result = await self.client.complete("mesh", [{"role": "user", "content": "hello"}], max_tokens=32, temperature=0)
        payload = json.loads(self.requests[0][1])
        self.assertEqual(dict(model="mesh", messages=[dict(role="user", content="hello")], max_tokens=32, temperature=0, n=1, stream=False), payload)
        self.assertEqual("mesh", result["requested_model"])
        self.assertEqual("fixture-model", result["reported_model"])
        self.assertEqual(7, result["usage"]["total_tokens"])
        self.response = {key: value for key, value in COMPLETION.items() if key != "usage"}
        self.assertIsNone((await self.client.complete("fixture-model", [dict(role="user", content="hi")]))["usage"])

    async def test_env_bearer_proxy_ignored_and_missing_key_fails_before_network(self):
        client = MeshClient(dataclasses.replace(self.client.settings, token_env="HYPERL_TEST_MESH_TOKEN"))
        with patch.dict(os.environ, {"HYPERL_TEST_MESH_TOKEN": "private-test-sentinel", "HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1"}):
            await client.models()
        self.assertIn(b"Authorization: Bearer private-test-sentinel\r\n", self.requests[0][0])
        with patch.dict(os.environ, {"HYPERL_TEST_MESH_TOKEN": "bad\r\nInjected: value"}), self.assertRaises(MeshError):
            await client.models()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(MeshError):
            await client.models()
        self.assertEqual(1, len(self.requests))

    async def test_redirect_and_remote_failure_are_not_followed_or_retried(self):
        for code in (301, 302, 307, 401, 404, 429, 503):
            self.raw = f"HTTP/1.1 {code} private-test-sentinel\r\nLocation: http://127.0.0.1:1/leak\r\nContent-Length: 21\r\n\r\nprivate-test-sentinel".encode()
            before = len(self.requests)
            with self.assertRaises(MeshError) as caught:
                await self.client.models()
            self.assertIn(str(code), str(caught.exception))
            self.assertNotIn("private-test-sentinel", str(caught.exception))
            self.assertEqual(before + 1, len(self.requests))

    async def test_timeout_is_total_and_closes_socket(self):
        self.hold = True
        client = MeshClient(dataclasses.replace(self.client.settings, timeout_seconds=.1))
        with self.assertRaisesRegex(MeshError, "timed out"):
            await client.models()
        await asyncio.wait_for(self.disconnected.wait(), 1)

    async def test_cancellation_closes_socket_and_does_not_retry(self):
        self.hold = True
        task = asyncio.create_task(self.client.models())
        await asyncio.wait_for(self.received.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(self.disconnected.wait(), 1)
        self.assertEqual(1, len(self.requests))

    async def test_chunked_and_connection_close_responses(self):
        body = json.dumps(CATALOG).encode()
        self.raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n" + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
        self.assertEqual("fixture-model", (await self.client.models())["models"][0]["id"])
        self.raw = b"HTTP/1.0 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n\r\n" + body
        self.assertEqual("fixture-model", (await self.client.models())["models"][0]["id"])

    async def test_hostile_json_and_http_framing_are_bounded(self):
        prefix = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        cases = [prefix + b"Content-Length: 1048577\r\n\r\n",
                 prefix + b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
                 prefix + b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}",
                 prefix + b"Content-Encoding: gzip\r\n\r\n{}",
                 prefix + b"Transfer-Encoding: chunked\r\n\r\n100001\r\n",
                 prefix + b"Transfer-Encoding: chunked\r\n\r\n2\r\n{}XX0\r\n\r\n",
                 prefix + b"Content-Length: 100\r\n\r\n{}",
                 prefix + b"X-Large: " + b"x" * 33000 + b"\r\n\r\n{}",
                 prefix + b"\r\n{\"private-test-sentinel\":NaN}",
                 prefix + b"\r\n{\"x\":1,\"x\":2}",
                 prefix + b"\r\n" + b"x" * (MAX_RESPONSE + 1),
                 b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\nprivate-test-sentinel"]
        for raw in cases:
            self.raw = raw
            with self.subTest(size=len(raw)), self.assertRaises(MeshError) as caught:
                await self.client.models()
            self.assertNotIn("private-test-sentinel", str(caught.exception))

    async def test_invalid_inputs_reject_before_request(self):
        for messages in [[], [dict(role="tool", content="x")], [dict(role=[], content="x")], [dict(role="user", content="x", tool_calls=[])],
                         [dict(role="user", content="x" * 65537)], [dict(role="user", content="x" * 65536)] * 3]:
            with self.assertRaises(MeshError):
                await self.client.complete("fixture", messages)
        for option in [dict(max_tokens=True), dict(max_tokens=8193), dict(temperature=float("nan")), dict(temperature=-1)]:
            with self.assertRaises(MeshError):
                await self.client.complete("fixture", [dict(role="user", content="x")], **option)
        self.assertEqual([], self.requests)

    async def test_model_and_completion_validation_refuses_tools_and_fake_success(self):
        for response in [dict(object="list", data=[dict(id="same"), dict(id="same")]), dict(data=[]), dict(object="list", data=[None])]:
            self.response = response
            with self.assertRaises(MeshError):
                await self.client.models()
        for update in [dict(choices=[]), dict(model=None), dict(usage={"total_tokens": -1}), dict(usage={"prompt_tokens": True}),
                       dict(choices=[dict(message=dict(role="assistant", content="", tool_calls=[dict(id="x")]), finish_reason="tool_calls")]),
                       dict(choices=[dict(message=dict(role="assistant", content="x"), finish_reason=[])])]:
            self.response = dict(COMPLETION, **update)
            with self.assertRaises(MeshError):
                await self.client.complete("fixture", [dict(role="user", content="x")])

    async def test_cli_models_and_chat_real_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "mesh.json"
            config.write_text(json.dumps(dict(format="hyperl-mesh-llm/1", **dataclasses.asdict(self.client.settings))))
            request = root / "chat.json"
            request.write_text(json.dumps(dict(model="fixture-model", messages=[dict(role="user", content="hello")], max_tokens=8)))
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "python"))
            for command, extra in [("mesh-models", []), ("mesh-chat", [str(request)])]:
                self.response = CATALOG if command == "mesh-models" else COMPLETION
                process = await asyncio.create_subprocess_exec(sys.executable, "-m", "hyperl.cluster", command, str(config), *extra,
                    env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
                self.assertEqual(0, process.returncode, stderr)
                self.assertEqual("mesh-llm", json.loads(stdout)["backend"])

    @unittest.skipUnless(shutil.which("openssl"), "OpenSSL CLI is required for real local TLS test")
    async def test_https_verifies_server_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            cert, key, ca = (await asyncio.to_thread(certificates, Path(directory), ()))["server"]
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            tls_server = await asyncio.start_server(self.handle, "127.0.0.1", 0, ssl=context)
            try:
                client = MeshClient(MeshSettings(f"https://127.0.0.1:{tls_server.sockets[0].getsockname()[1]}/v1", "private-mesh"))
                with patch.dict(os.environ, {"SSL_CERT_FILE": ca}):
                    self.assertEqual("fixture-model", (await client.models())["models"][0]["id"])
                with patch.dict(os.environ, {"SSL_CERT_FILE": "/nonexistent-hyperl-test-ca"}), self.assertRaises(MeshError):
                    await client.models()
            finally:
                tls_server.close()
                await tls_server.wait_closed()


if __name__ == "__main__":
    unittest.main()
