# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""Optional, explicitly configured Mesh-LLM HTTP client; no native runtime loading.

Requests can leave the machine through the configured daemon even when its API
binds to loopback. A declared routing scope is not independently verified here.
Cancelling the asyncio task closes the connection; remote work may have started.
"""
import asyncio
from dataclasses import dataclass
import ipaddress
import json
import math
import os
import re
import ssl
import stat
from urllib.parse import urlsplit

from .model_formats import _json

MAX_REQUEST = 128 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_HEADERS = 32 * 1024
UPSTREAM_REVISION = "5f54a094e2c6ec52a1ad71813bad691e312d0deb"


class MeshError(ValueError):
    """Sanitized failure: never includes remote bodies, prompts or credentials."""


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise MeshError("Mesh-LLM integer option is outside its supported bounds")
    return value


def _text(value, maximum, *, empty=False):
    if not isinstance(value, str) or (not empty and not value) or len(value) > maximum:
        raise MeshError("Mesh-LLM text field is invalid or oversized")
    try:
        if len(value.encode("utf-8")) > maximum:
            raise MeshError("Mesh-LLM text field is oversized")
    except UnicodeError:
        raise MeshError("Mesh-LLM text field is not valid Unicode") from None
    return value


def read_document(path, maximum=MAX_REQUEST):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError()
            data = source.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError()
        return _json(data)
    except (OSError, ValueError, TypeError, RecursionError):
        raise MeshError("Mesh-LLM input must be a bounded, valid JSON object") from None


@dataclass(frozen=True)
class MeshSettings:
    base_url: str
    routing_scope: str
    token_env: str | None = None
    allow_remote: bool = False
    timeout_seconds: float = 60

    @classmethod
    def from_document(cls, value):
        keys = {"format", "base_url", "routing_scope", "token_env", "allow_remote", "timeout_seconds"}
        if not isinstance(value, dict) or set(value) - keys or value.get("format") != "hyperl-mesh-llm/1":
            raise MeshError("unsupported Mesh-LLM configuration")
        try:
            result = cls(**{key: item for key, item in value.items() if key != "format"})
        except TypeError:
            raise MeshError("Mesh-LLM configuration requires base_url and routing_scope") from None
        result.validate()
        return result

    def validate(self):
        _text(self.base_url, 2048)
        if not self.base_url.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in self.base_url):
            raise MeshError("Mesh-LLM endpoint must be an ASCII URL without whitespace")
        try:
            url = urlsplit(self.base_url)
            port = url.port
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                    or url.password is not None or url.query or url.fragment or url.path not in {"/v1", "/v1/"}
                    or "%" in url.netloc or "\\" in url.netloc):
                raise ValueError()
            if port is not None:
                _int(port, 1, 65535)
            try:
                local = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                local = False  # Do not trust DNS names such as localhost as a loopback boundary.
        except ValueError:
            raise MeshError("Mesh-LLM endpoint requires an explicit /v1 URL without credentials or query") from None
        if type(self.allow_remote) is not bool:
            raise MeshError("allow_remote must be boolean")
        if url.scheme == "http" and not local:
            raise MeshError("plaintext Mesh-LLM endpoints must use a numeric loopback address")
        if not local and (not self.allow_remote or url.scheme != "https" or not self.token_env):
            raise MeshError("remote Mesh-LLM requires explicit allow_remote, HTTPS and token_env")
        if self.token_env is not None and (not isinstance(self.token_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", self.token_env)):
            raise MeshError("token_env must name an environment variable, never contain a credential")
        if not isinstance(self.routing_scope, str) or self.routing_scope not in {"local-model-only", "private-mesh", "public-mesh"}:
            raise MeshError("declare the operator's Mesh-LLM routing_scope")
        if (type(self.timeout_seconds) not in {int, float} or not math.isfinite(self.timeout_seconds)
                or not 0.1 <= self.timeout_seconds <= 300):
            raise MeshError("Mesh-LLM timeout must be between 0.1 and 300 seconds")
        return url


class MeshClient:
    def __init__(self, settings):
        if not isinstance(settings, MeshSettings):
            raise MeshError("MeshSettings required")
        settings.validate()
        self.settings = settings

    def _evidence(self):
        return dict(backend="mesh-llm", adapter="hyperl-mesh-llm/1",
                    declared_routing_scope=self.settings.routing_scope,
                    routing_scope_verified=False, placement_verified=False,
                    sharding_qualified=False, model_artifact_verified=False)

    async def _request(self, path, document=None):
        # Direct streams intentionally ignore HTTP(S)_PROXY and never follow redirects.
        url = self.settings.validate()
        try:
            payload = b"" if document is None else json.dumps(document, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError):
            raise MeshError("invalid Mesh-LLM request") from None
        if len(payload) > MAX_REQUEST:
            raise MeshError("Mesh-LLM request exceeds 128 KiB")
        token = None
        if self.settings.token_env:
            token = os.environ.get(self.settings.token_env)
            if not token or len(token) > 4096 or not re.fullmatch(r"[\x21-\x7e]+", token):
                raise MeshError("configured Mesh-LLM credential is unavailable or invalid")

        async def exchange():
            context = ssl.create_default_context() if url.scheme == "https" else None
            if context:
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            writer = None
            try:
                reader, writer = await asyncio.open_connection(
                    url.hostname, url.port or (443 if context else 80), ssl=context,
                    server_hostname=url.hostname if context else None, limit=MAX_HEADERS)
                headers = [f"{'GET' if document is None else 'POST'} /v1/{path} HTTP/1.1",
                           f"Host: {url.netloc}", "Accept: application/json", "Accept-Encoding: identity",
                           "Connection: close", "User-Agent: HyperL-MeshLLM/1"]
                if token:
                    headers.append("Authorization: Bearer " + token)
                if document is not None:
                    headers += ["Content-Type: application/json", f"Content-Length: {len(payload)}"]
                writer.write(("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + payload)
                await writer.drain()
                head = await reader.readuntil(b"\r\n\r\n")
                if len(head) > MAX_HEADERS:
                    raise MeshError("Mesh-LLM response headers exceed limit")
                lines = head[:-4].split(b"\r\n")
                match = re.fullmatch(rb"HTTP/1\.[01] ([0-9]{3})(?: [\x20-\x7e]*)?", lines[0])
                if not match:
                    raise MeshError("invalid Mesh-LLM HTTP response")
                status = int(match.group(1))
                if status != 200:
                    # Status only; a remote error may echo prompt or bearer data.
                    raise MeshError(f"Mesh-LLM HTTP {status}; no retry or fallback performed")
                fields = {}
                for line in lines[1:]:
                    name, separator, value = line.partition(b":")
                    name = name.lower()
                    if not separator or not re.fullmatch(rb"[a-z0-9!#$%&'*+.^_`|~-]+", name) or name in fields:
                        raise MeshError("invalid or duplicate Mesh-LLM response header")
                    fields[name] = value.strip()
                if fields.get(b"content-type", b"").split(b";", 1)[0].lower() != b"application/json":
                    raise MeshError("Mesh-LLM response must be JSON")
                if fields.get(b"content-encoding", b"identity").lower() != b"identity":
                    raise MeshError("compressed Mesh-LLM responses are unsupported")
                length, encoding = fields.get(b"content-length"), fields.get(b"transfer-encoding")
                if length is not None and encoding is not None:
                    raise MeshError("ambiguous Mesh-LLM response framing")
                body = bytearray()
                if encoding is not None:
                    if encoding.lower() != b"chunked":
                        raise MeshError("unsupported Mesh-LLM transfer encoding")
                    # Also cap overhead: many zero-sized/non-payload chunks cannot consume unbounded work.
                    for _ in range(16384):
                        line = await reader.readuntil(b"\r\n")
                        if len(line) > 128 or not re.fullmatch(rb"[0-9a-fA-F]{1,8}\r\n", line):
                            raise MeshError("invalid Mesh-LLM chunk framing")
                        size = int(line[:-2], 16)
                        if len(body) + size > MAX_RESPONSE:
                            raise MeshError("Mesh-LLM response exceeds 1 MiB")
                        if size == 0:
                            if await reader.readexactly(2) != b"\r\n":
                                raise MeshError("Mesh-LLM trailers are unsupported")
                            break
                        body.extend(await reader.readexactly(size))
                        if await reader.readexactly(2) != b"\r\n":
                            raise MeshError("invalid Mesh-LLM chunk delimiter")
                    else:
                        raise MeshError("Mesh-LLM response chunk count exceeds limit")
                elif length is not None:
                    if not re.fullmatch(rb"[0-9]{1,8}", length) or int(length) > MAX_RESPONSE:
                        raise MeshError("invalid or oversized Mesh-LLM response length")
                    body.extend(await reader.readexactly(int(length)))
                else:
                    while True:
                        part = await reader.read(min(65536, MAX_RESPONSE + 1 - len(body)))
                        if not part:
                            break
                        body.extend(part)
                        if len(body) > MAX_RESPONSE:
                            raise MeshError("Mesh-LLM response exceeds 1 MiB")
                return _json(bytes(body))
            finally:
                if writer is not None:
                    writer.close()
                    # TLS close_notify from an unresponsive peer must not extend cancellation.
                    try:
                        await asyncio.wait_for(writer.wait_closed(), 0.2)
                    except (OSError, TimeoutError, asyncio.TimeoutError):
                        writer.transport.abort()
        try:
            return await asyncio.wait_for(exchange(), self.settings.timeout_seconds)
        except MeshError:
            raise
        except (TimeoutError, asyncio.TimeoutError):
            raise MeshError("Mesh-LLM request timed out; remote completion is unknown") from None
        except (OSError, ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, RecursionError):
            raise MeshError("Mesh-LLM transport or response failed validation") from None

    async def models(self):
        result = await self._request("models")
        data = result.get("data")
        if result.get("object") != "list" or not isinstance(data, list) or len(data) > 1000:
            raise MeshError("invalid or oversized Mesh-LLM model list")
        models, seen = [], set()
        for item in data:
            if not isinstance(item, dict):
                raise MeshError("invalid Mesh-LLM model entry")
            name = _text(item.get("id"), 1024)
            if name in seen:
                raise MeshError("duplicate Mesh-LLM model ID")
            seen.add(name)
            models.append(dict(id=name, owned_by=_text(item.get("owned_by", ""), 1024, empty=True)))
        return dict(**self._evidence(), models=models)

    async def complete(self, model, messages, *, max_tokens=256, temperature=0.2):
        _text(model, 1024)
        if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
            raise MeshError("Mesh-LLM requires 1 to 128 text messages")
        checked = []
        for item in messages:
            if (not isinstance(item, dict) or set(item) != {"role", "content"}
                    or not isinstance(item.get("role"), str) or item["role"] not in {"system", "user", "assistant"}):
                raise MeshError("Mesh-LLM accepts only system, user and assistant text messages")
            checked.append(dict(role=item["role"], content=_text(item["content"], 65536, empty=True)))
        _int(max_tokens, 1, 8192)
        if type(temperature) not in {int, float} or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise MeshError("Mesh-LLM temperature must be between 0 and 2")
        result = await self._request("chat/completions", dict(model=model, messages=checked,
            max_tokens=max_tokens, temperature=temperature, stream=False, n=1))
        choices = result.get("choices")
        if result.get("object") != "chat.completion" or not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise MeshError("invalid Mesh-LLM completion envelope")
        choice = choices[0]
        message = choice.get("message")
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or message.get("tool_calls") or message.get("function_call")
                or not isinstance(choice.get("finish_reason"), str)
                or choice.get("finish_reason") not in {"stop", "length", "content_filter"}):
            raise MeshError("Mesh-LLM returned an unsupported completion; no tools executed")
        content = _text(message.get("content"), MAX_RESPONSE, empty=True)
        returned_model = _text(result.get("model"), 1024)
        usage = result.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise MeshError("invalid Mesh-LLM usage")
            usage = {key: _int(usage[key], 0, 2**53 - 1) if usage.get(key) is not None else None
                     for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return dict(**self._evidence(), requested_model=model, reported_model=returned_model,
                    content=content, finish_reason=choice["finish_reason"], usage=usage)


def execute_cli(command, config, request=None):
    client = MeshClient(MeshSettings.from_document(read_document(config, 16384)))
    if command == "mesh-models":
        return asyncio.run(client.models())
    value = read_document(request)
    if set(value) - {"model", "messages", "max_tokens", "temperature"} or not {"model", "messages"} <= set(value):
        raise MeshError("Mesh-LLM chat request requires model and messages only, plus optional token/temperature limits")
    return asyncio.run(client.complete(**value))
