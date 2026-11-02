# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Authenticated bounded pull service and actual native LLM execution agent."""
import asyncio
import dataclasses
import http.client
import json
import ssl
import threading
import time
from urllib.parse import urlsplit
from .connections import peer_name
from .scheduler import Cell, FleetScheduler, _name
from .llm import LlamaSession, LlamaSettings
from .jax_llm import JaxModelSession, JaxSettings


def _decode(data):
    if len(data) > 1024 * 1024:
        raise ValueError("control frame exceeds 1 MiB")
    depth = 0; quoted = False; escaped = False
    for char in data.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 16:
                raise ValueError("control frame nesting limit")
        elif char in "]}":
            depth -= 1
    value = json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("control frame must be an object")
    return value


class FleetService:
    """One authority shard. CA-issued CN cell:ID or tenant:ID binds every operation.

    Enroll cells locally through FleetScheduler.register before agents connect.
    Each execution cell owns its bounded native workers/model profiles. No request
    can supply an executable path, raw RPC destination, certificate or model file.
    """
    def __init__(self, scheduler, tls, *, host="127.0.0.1", port=0, max_connections=128):
        if not isinstance(scheduler, FleetScheduler) or scheduler.audit.key is None or tls.verify_mode != ssl.CERT_REQUIRED or not 1 <= max_connections <= 1024:
            raise ValueError("durable scheduler and bounded mTLS service required")
        scheduler.require_profiles = True
        self.scheduler = scheduler; self.tls = tls; self.host = host; self.port = port
        self.maximum = max_connections; self.active = 0; self.tasks = set(); self.server = None
        self.reconcile = None; self.closed = False

    async def start(self):
        self.server = await asyncio.start_server(self._handle, self.host, self.port, ssl=self.tls,
            ssl_handshake_timeout=5, limit=16384, backlog=self.maximum)
        self.port = self.server.sockets[0].getsockname()[1]
        self.reconcile = asyncio.create_task(self._reconcile())
        return self

    async def _reconcile(self):
        while True:
            self.scheduler.expire(limit=1000)
            await asyncio.sleep(1)

    async def _handle(self, reader, writer):
        if self.closed or self.active >= self.maximum:
            writer.close(); return
        task = asyncio.current_task(); self.tasks.add(task); self.active += 1
        identity = "unverified"; path = "unparsed"
        try:
            identity = peer_name(writer)
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            if len(header) > 16384:
                raise ValueError("control headers too large")
            lines = header.decode("ascii").split("\r\n")
            method, path, protocol = lines[0].split(" ")
            if method != "POST" or protocol != "HTTP/1.1" or "?" in path or "%" in path:
                raise ValueError("unsupported control method/path/version")
            headers = {}
            for line in lines[1:]:
                if not line:
                    continue
                key, value = line.split(":", 1); key = key.lower()
                if key in headers:
                    raise ValueError("duplicate control header")
                headers[key] = value.strip()
            if "transfer-encoding" in headers or headers.get("content-type") != "application/json":
                raise ValueError("explicit bounded JSON body required")
            size = int(headers.get("content-length", "-1"))
            if not 2 <= size <= 1024 * 1024:
                raise ValueError("control frame size outside bounds")
            body = _decode(await asyncio.wait_for(reader.readexactly(size), 10))
            value = self._dispatch(identity, path, body)
            await self._respond(writer, 200, dict(ok=True, result=value))
        except (ValueError, KeyError, TypeError, OverflowError):
            try:
                self.scheduler.record_denial(identity, path)
                status = 422
            except Exception:
                status = 503
            await self._respond(writer, status, dict(ok=False, error="invalid request or authority/audit unavailable"))
        except (OSError, TimeoutError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        except Exception:
            await self._respond(writer, 503, dict(ok=False, error="authority unavailable; no automatic request replay"))
        finally:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 2)
            except (OSError, TimeoutError, asyncio.TimeoutError):
                pass
            self.tasks.discard(task); self.active -= 1

    def _dispatch(self, identity, path, body):
        if identity.startswith("tenant:"):
            tenant = _name(identity[len("tenant:"):])
            if path == "/v1/jobs/submit":
                return self.scheduler.submit(tenant, body["jobs"])
            if path == "/v1/jobs/get":
                return self.scheduler.job(tenant, body["id"])
            if path == "/v1/jobs/page":
                return self.scheduler.page(tenant, after=body.get("after", 0), limit=body.get("limit", 100))
            if path == "/v1/jobs/cancel":
                self.scheduler.cancel(tenant, body["id"]); return dict(cancellation_requested=True)
        elif identity.startswith("cell:"):
            cell_id = _name(identity[len("cell:"):])
            if path == "/v1/cells/heartbeat":
                cell = Cell(**body["cell"])
                if cell.id != cell_id:
                    raise ValueError("cell certificate mismatch")
                self.scheduler.heartbeat(cell); return dict(renewed=True)
            if path == "/v1/cells/claim":
                if body["cell"] != cell_id:
                    raise ValueError("cell certificate mismatch")
                return self.scheduler.claim(cell_id, body["instance"], body["epoch"])
            receipt = body["receipt"]
            if receipt["cell"] != cell_id:
                raise ValueError("job certificate mismatch")
            if path == "/v1/jobs/start":
                self.scheduler.start(receipt); return dict(started=True)
            if path == "/v1/jobs/renew":
                return self.scheduler.renew(receipt)
            if path == "/v1/jobs/finish":
                return dict(sha256=self.scheduler.finish(receipt, body["result"], state=body.get("state", "SUCCEEDED")))
        raise ValueError("certificate role does not authorize the route")

    async def _respond(self, writer, status, value):
        if writer.is_closing():
            return
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > 2 * 1024 * 1024:
            status = 503; data = b'{"ok":false,"error":"response quota"}'
        writer.write(f"HTTP/1.1 {status} {'OK' if status == 200 else 'Rejected'}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n".encode() + data)
        await asyncio.wait_for(writer.drain(), 5)

    async def close(self):
        self.closed = True
        if self.server:
            self.server.close(); await self.server.wait_closed()
        if self.reconcile:
            self.reconcile.cancel(); await asyncio.gather(self.reconcile, return_exceptions=True)
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class FleetClient:
    def __init__(self, origin, tls, *, timeout=15):
        uri = urlsplit(origin)
        if (uri.scheme != "https" or not uri.hostname or uri.username or uri.password or uri.path not in ("", "/")
                or uri.query or uri.fragment or not tls.check_hostname or tls.verify_mode != ssl.CERT_REQUIRED
                or not 1 <= timeout <= 60):
            raise ValueError("explicit verified HTTPS authority origin required")
        self.host = uri.hostname; self.port = uri.port or 443; self.tls = tls; self.timeout = timeout

    def call(self, path, body):
        if path not in {"/v1/jobs/submit", "/v1/jobs/get", "/v1/jobs/page", "/v1/jobs/cancel", "/v1/jobs/start", "/v1/jobs/renew", "/v1/jobs/finish", "/v1/cells/heartbeat", "/v1/cells/claim"}:
            raise ValueError("unknown control route")
        data = json.dumps(body, allow_nan=False).encode()
        if len(data) > 1024 * 1024:
            raise ValueError("control request quota")
        client = http.client.HTTPSConnection(self.host, self.port, context=self.tls, timeout=self.timeout)
        try:
            client.request("POST", path, data, {"Content-Type": "application/json"})
            response = client.getresponse(); payload = response.read(2 * 1024 * 1024 + 1)
            if response.status != 200 or len(payload) > 2 * 1024 * 1024:
                raise RuntimeError("fleet authority rejected request; do not replay uncertain effects")
            result = json.loads(payload)
            if result.get("ok") is not True:
                raise RuntimeError("fleet authority did not confirm the operation")
            return result["result"]
        finally:
            client.close()


class LlmAgent:
    """A cell agent executes only owner-configured model profiles. No effect replay.

    Failed renewals cancel local inference; late results cannot publish. This agent
    serializes one native session. Multiple agents/cells provide independent bounded
    concurrency; a model session can itself contain several native RPC shards.
    """
    def __init__(self, client, cell, profiles):
        cell.validate()
        if cell.slots != 1 or set(profiles) != set(cell.profiles):
            raise ValueError("this native agent needs one slot and exact enrolled profiles")
        self.client = client; self.cell = cell; self.profiles = dict(profiles)
        for settings in profiles.values():
            if not isinstance(settings, (LlamaSettings, JaxSettings)):
                raise ValueError("explicit registered native/JAX profile required")
            settings.validate()
        self.stopping = threading.Event()
        self.session = None; self.profile_name = None; self.quarantined = False

    def run_once(self):
        if self.stopping.is_set():
            return False
        if self.quarantined:
            raise RuntimeError("cell backend quarantined after uncertain failure; owner restart/revalidation required")
        self.client.call("/v1/cells/heartbeat", dict(cell=dataclasses.asdict(self.cell)))
        receipt = self.client.call("/v1/cells/claim", dict(cell=self.cell.id, instance=self.cell.instance, epoch=self.cell.epoch))
        if receipt is None:
            return False
        self.client.call("/v1/jobs/start", dict(receipt=receipt))
        cancelled = threading.Event(); finished = threading.Event(); authority_lost = threading.Event()
        def renew():
            while not finished.wait(2):
                try:
                    self.client.call("/v1/cells/heartbeat", dict(cell=dataclasses.asdict(self.cell)))
                    response = self.client.call("/v1/jobs/renew", dict(receipt=receipt))
                    if response["state"] == "CANCEL_REQUESTED":
                        cancelled.set(); return
                except Exception:
                    authority_lost.set(); cancelled.set(); return
        heartbeat = threading.Thread(target=renew, daemon=True); heartbeat.start()
        session = None; publishing = False
        try:
            payload = receipt["payload"]
            if set(payload) - {"prompt", "max_tokens", "seed"}:
                raise ValueError("unsupported model job field")
            profile = self.profiles[receipt["profile"]]
            expected = profile.model_sha256 if isinstance(profile, LlamaSettings) else profile.manifest_sha256
            adapter = "llama.cpp" if isinstance(profile, LlamaSettings) else "jax-llama-f32"
            if receipt.get("artifact_sha256") != expected or receipt.get("adapter") != adapter:
                raise ValueError("agent model profile does not match authority-pinned artifact")
            factory = LlamaSession if isinstance(profile, LlamaSettings) else JaxModelSession
            if self.session is not None and (self.profile_name != receipt["profile"] or self.session.closed):
                self.session.close(); self.session = None
            if self.session is None:
                self.session = factory(profile, cancelled=lambda: cancelled.is_set() or self.stopping.is_set())
                self.profile_name = receipt["profile"]
            session = self.session
            result = session.complete(payload["prompt"], max_tokens=payload.get("max_tokens", 32), seed=payload.get("seed", 0),
                                      cancelled=lambda: cancelled.is_set() or self.stopping.is_set())
            if cancelled.is_set() or self.stopping.is_set():
                raise InterruptedError("cell stopped before result publication")
            publishing = True
            self.client.call("/v1/jobs/finish", dict(receipt=receipt, result=result, state="SUCCEEDED"))
        except Exception:
            profile = self.profiles.get(receipt["profile"])
            if isinstance(profile, LlamaSettings) and profile.backend in {"GPU", "RPC"} or isinstance(profile, JaxSettings) and profile.backend in {"gpu", "tpu"}:
                self.quarantined = True
            if self.session is not None:
                self.session.close(); self.session = None
            # Never retry generation/publication after an uncertain transport failure.
            if not authority_lost.is_set() and not publishing:
                state = "CANCELLED" if cancelled.is_set() or self.stopping.is_set() else "FAILED"
                try:
                    self.client.call("/v1/jobs/finish", dict(receipt=receipt, result={"detail": "cell execution stopped or failed; no partial output"}, state=state))
                except Exception:
                    pass
        finally:
            finished.set(); heartbeat.join(timeout=self.client.timeout * 2 + 1)
            if self.stopping.is_set() and self.session is not None:
                self.session.close(); self.session = None
        return True

    def run(self, *, poll_seconds=2):
        if not .1 <= poll_seconds <= 60:
            raise ValueError("bounded pull interval required")
        # Stable cell phase avoids synchronized polling across many independent cells.
        phase = (sum(self.cell.id.encode()) % 100) / 100 * poll_seconds
        self.stopping.wait(phase)
        while not self.stopping.is_set():
            try:
                work = self.run_once()
            except Exception:
                work = False
            if not work:
                self.stopping.wait(poll_seconds)

    def stop(self):
        self.stopping.set()

    def close(self):
        self.stop()
        if self.session is not None:
            self.session.close(); self.session = None
