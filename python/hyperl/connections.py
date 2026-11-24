# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Owner-configured mTLS TCP routes and measured link admission; no discovery/NAT magic."""
import asyncio
from dataclasses import dataclass
import ipaddress
import math
from pathlib import Path
import ssl
from .llm import _endpoint, _integer


def _file(path):
    result = Path(path)
    if not result.is_absolute() or result.is_symlink() or not result.is_file():
        raise ValueError("explicit absolute certificate/key/CA file required")
    return str(result)


def server_tls(certificate, key, ca):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(_file(certificate), _file(key))
    context.load_verify_locations(cafile=_file(ca))
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def client_tls(certificate, key, ca):
    context = ssl.create_default_context(cafile=_file(ca))
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(_file(certificate), _file(key))
    return context


def peer_name(writer):
    certificate = writer.get_extra_info("peercert")
    if not certificate:
        raise ValueError("authenticated client certificate required")
    names = [value for group in certificate.get("subject", ()) for key, value in group if key == "commonName"]
    if len(names) != 1:
        raise ValueError("one explicit certificate common name required for application identity")
    return names[0]


class RpcTunnel:
    """Bounded bidirectional TLS wrapping for one approved native RPC origin.

    Server: mTLS listener -> numeric loopback native RPC. Client: numeric loopback
    listener -> hostname-verified mTLS peer. Peer certificates are allowlisted.
    No redirects, reconnection, file transfer command, shell or raw remote RPC.
    """
    def __init__(self, *, listen_host="127.0.0.1", listen_port=0, target_host="127.0.0.1", target_port,
                 listen_tls=None, target_tls=None, server_hostname=None, allowed_peers=(), max_connections=8, idle_seconds=300):
        _integer(listen_port, 0, 65535); _integer(target_port, 1, 65535)
        _integer(max_connections, 1, 64); _integer(idle_seconds, 1, 600)
        ipaddress.ip_address(listen_host)
        if (listen_tls is None) == (target_tls is None):
            raise ValueError("select exactly one authenticated TLS side")
        if target_tls is not None:
            if listen_host != "127.0.0.1" or not server_hostname or not target_tls.check_hostname or target_tls.verify_mode != ssl.CERT_REQUIRED:
                raise ValueError("tunnel client needs private loopback and verified TLS hostname")
        else:
            _endpoint(f"{target_host}:{target_port}")
            if listen_tls.verify_mode != ssl.CERT_REQUIRED or not allowed_peers:
                raise ValueError("tunnel listener requires mTLS and explicit client identities")
        self.listen_host = listen_host; self.listen_port = listen_port
        self.target_host = target_host; self.target_port = target_port
        self.listen_tls = listen_tls; self.target_tls = target_tls; self.server_hostname = server_hostname
        self.allowed_peers = frozenset(allowed_peers)
        self.maximum = max_connections; self.idle_seconds = idle_seconds
        self.tasks = set(); self.server = None; self.active = 0; self.closed = False
        self.forward_bytes = 0; self.return_bytes = 0; self.failures = 0

    async def start(self):
        kwargs = dict(ssl=self.listen_tls, limit=65536, backlog=self.maximum)
        if self.listen_tls:
            kwargs["ssl_handshake_timeout"] = 5
        self.server = await asyncio.start_server(self._accept, self.listen_host, self.listen_port, **kwargs)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _accept(self, reader, writer):
        if self.closed or self.active >= self.maximum:
            writer.close(); return
        task = asyncio.current_task(); self.tasks.add(task); self.active += 1
        remote = None
        try:
            if self.listen_tls and peer_name(writer) not in self.allowed_peers:
                raise ValueError("TLS peer not authorized for native route")
            args = dict(ssl=self.target_tls, limit=65536)
            if self.target_tls:
                args.update(server_hostname=self.server_hostname, ssl_handshake_timeout=5)
            remote_reader, remote = await asyncio.wait_for(asyncio.open_connection(self.target_host, self.target_port, **args), 10)
            for output in (writer, remote):
                output.transport.set_write_buffer_limits(high=65536, low=16384)
            async def pump(source, destination, forward):
                while data := await asyncio.wait_for(source.read(65536), self.idle_seconds):
                    destination.write(data)
                    await asyncio.wait_for(destination.drain(), self.idle_seconds)
                    if forward:
                        self.forward_bytes += len(data)
                    else:
                        self.return_bytes += len(data)
            outgoing = asyncio.create_task(pump(reader, remote, True))
            incoming = asyncio.create_task(pump(remote_reader, writer, False))
            try:
                done, pending = await asyncio.wait((outgoing, incoming), return_when=asyncio.FIRST_COMPLETED)
                for finished in done:
                    finished.result()
            finally:
                for child in (outgoing, incoming):
                    child.cancel()
                await asyncio.gather(outgoing, incoming, return_exceptions=True)
        except (OSError, ValueError, TimeoutError, asyncio.TimeoutError, asyncio.IncompleteReadError):
            self.failures += 1
        finally:
            for output in (remote, writer):
                if output:
                    output.close()
                    try:
                        await asyncio.wait_for(output.wait_closed(), 2)
                    except (OSError, TimeoutError, asyncio.TimeoutError):
                        pass
            self.tasks.discard(task); self.active -= 1

    async def close(self):
        self.closed = True
        if self.server:
            self.server.close(); await self.server.wait_closed()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True)
class LinkObservation:
    source: str
    target: str
    medium: str
    bytes_per_second: float
    rtt_ms: float
    observed_ms: int
    authenticated: bool


def admit_link(link, *, now_ms, mode="layer", bytes_per_token=0, exchanges_per_token=1,
               max_seconds_per_token=1.0, allow_slow_capacity=False):
    """Explain admission from recent application measurements, not medium labels."""
    if (link.medium not in {"LOCAL", "ETHERNET", "USB_TUNNEL", "WIFI", "WAN", "FABRIC"}
            or not link.source or not link.target or link.source == link.target
            or not link.authenticated or not 0 <= now_ms - link.observed_ms <= 30000
            or not math.isfinite(link.bytes_per_second) or link.bytes_per_second <= 0
            or not math.isfinite(link.rtt_ms) or link.rtt_ms < 0 or mode not in {"layer", "tensor", "replica"}
            or not math.isfinite(max_seconds_per_token) or max_seconds_per_token <= 0):
        raise ValueError("missing, stale, unauthenticated or invalid link observation")
    _integer(bytes_per_token, 0, 1024**4); _integer(exchanges_per_token, 1, 100000)
    estimate = bytes_per_token / link.bytes_per_second + exchanges_per_token * link.rtt_ms / 1000
    if mode == "tensor" and link.rtt_ms > 2:
        raise ValueError("tensor collectives require a separately qualified low-latency island")
    if estimate > max_seconds_per_token and not allow_slow_capacity:
        raise ValueError("link exceeds per-token transfer budget; select independent replicas or explicit capacity mode")
    return dict(mode=mode, medium=link.medium, transfer_seconds_per_token_estimate=estimate,
                capacity_only=estimate > max_seconds_per_token, physical_link_verified=False,
                limit="Application estimates exclude device/queue/collective work; medium is owner-declared, not discovered.")
