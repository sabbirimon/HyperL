# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Explicit local control tools: python -m hyperl.cluster --help. Explicit local artifacts, inference and authenticated fleet services. No automatic downloads."""
import argparse
import dataclasses
import json
from pathlib import Path
from .model_storage import ModelStore, ModelLayer, MemoryTier, Workload, LLMProfile, estimate_llm, plan_model
from .fleet import Device, DeviceDirectory


def read_json(path):
    with Path(path).open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("configuration exceeds 1 MiB")
    # Bound nesting before the recursive JSON decoder.
    depth = 0
    quoted = False
    escaped = False
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
                raise ValueError("configuration nesting limit")
        elif char in "]}":
            depth -= 1
    return json.loads(data)



def profile(value):
    from .llm import LlamaSettings
    from .jax_llm import JaxSettings
    if set(value) != {"adapter", "settings"}:
        raise ValueError("profile requires adapter and settings")
    if value["adapter"] == "llama.cpp":
        settings = LlamaSettings(**value["settings"])
    elif value["adapter"] == "jax-llama-f32":
        settings = JaxSettings(**value["settings"])
    else:
        raise ValueError("unknown model adapter; no substitute")
    settings.validate()
    return settings


def control_client(value):
    from .connections import client_tls
    from .fleet_service import FleetClient
    if value.get("format") != "hyperl-fleet-client/1":
        raise ValueError("unsupported fleet client configuration")
    return FleetClient(value["origin"], client_tls(value["certificate"], value["key_file"], value["ca_file"]))


def execute_command(args):
    import asyncio
    from .llm import LlamaSettings, LlamaSession, RpcWorker, file_digest
    from .jax_llm import JaxModelSession
    from .connections import RpcTunnel, server_tls, client_tls
    from .scheduler import FleetScheduler, Cell
    from .audit import read_key
    from .fleet_service import FleetService, LlmAgent
    if args.command == "audit-key-new":
        from .audit import create_key
        create_key(args.path)
        print(json.dumps(dict(created=True, private_key_path=str(Path(args.path)))))
        return
    if args.command in {"model-protect", "model-repair"}:
        from .erasure import ErasureCodec, ProtectedModelStore
        codec = ErasureCodec(args.library, args.library_sha256)
        store = ModelStore(args.root)
        try:
            protected = ProtectedModelStore(store, codec)
            if args.command == "model-protect":
                result = protected.protect(args.artifact, data_count=args.data_shards, parity_count=args.parity_shards, max_bytes=args.max_bytes)
            else:
                result = protected.repair(args.artifact, expected_manifest_sha256=args.manifest_sha256, max_bytes=args.max_bytes)
            print(json.dumps(result, indent=2))
        finally:
            store.close()
        return
    if args.command == "artifact-hash":
        print(json.dumps(dict(sha256=file_digest(str(Path(args.path).resolve())))))
        return
    value = read_json(args.config)
    if args.command == "llm-run":
        if value.get("format") != "hyperl-model-profile/1":
            raise ValueError("unsupported model profile")
        selected = profile(value["profile"])
        with Path(args.prompt).open("rb") as stream:
            data = stream.read(65537)
        if len(data) > 65536:
            raise ValueError("prompt file exceeds 64 KiB")
        prompt = data.decode() if isinstance(selected, LlamaSettings) else json.loads(data)
        factory = LlamaSession if isinstance(selected, LlamaSettings) else JaxModelSession
        session = factory(selected)
        try:
            print(json.dumps(session.complete(prompt, max_tokens=args.tokens), indent=2))
        finally:
            session.close()
    elif args.command == "rpc-worker":
        if value.get("format") != "hyperl-native-worker/1":
            raise ValueError("unsupported native worker configuration")
        worker = RpcWorker(value["executable"], value["executable_sha256"], port=value["port"], device=value["device"], threads=value.get("threads", 2))
        try:
            print(json.dumps(dict(endpoint=worker.endpoint, scope="trusted-owner loopback RPC")), flush=True)
            import time
            while worker.native.process.poll() is None:
                time.sleep(.5)
        finally:
            worker.close()
    elif args.command == "rpc-tunnel":
        if value.get("format") != "hyperl-rpc-tunnel/1":
            raise ValueError("unsupported route configuration")
        async def tunnel():
            common = dict(listen_host=value.get("listen_host", "127.0.0.1"), listen_port=value["listen_port"],
                target_host=value["target_host"], target_port=value["target_port"], max_connections=value.get("max_connections", 8), idle_seconds=value.get("idle_seconds", 300))
            credentials = (value["certificate"], value["key_file"], value["ca_file"])
            if value["mode"] == "server":
                route = RpcTunnel(**common, listen_tls=server_tls(*credentials), allowed_peers=value["allowed_peers"])
            elif value["mode"] == "client":
                route = RpcTunnel(**common, target_tls=client_tls(*credentials), server_hostname=value["server_hostname"])
            else:
                raise ValueError("explicit tunnel client/server mode required")
            await route.start()
            try:
                print(json.dumps(dict(host=route.listen_host, port=route.port, mode=value["mode"])), flush=True)
                await asyncio.Event().wait()
            finally:
                await route.close()
        asyncio.run(tunnel())
    elif args.command in {"fleet-register", "fleet-serve", "audit-verify"}:
        if value.get("format") != "hyperl-fleet-service/1" or not Path(value["database"]).is_absolute():
            raise ValueError("explicit fleet service database configuration required")
        scheduler = FleetScheduler(value["database"], max_jobs=value.get("max_jobs", 10000000), max_pending_per_tenant=value.get("tenant_queue_quota", 1000000), audit_key=read_key(value["audit_key_file"]))
        try:
            if args.command == "audit-verify":
                expected = read_json(args.checkpoint) if args.checkpoint else None
                print(json.dumps(scheduler.audit.verify(expected), indent=2))
                return
            registered = value.get("model_profiles", {})
            if len(registered) > 1000:
                raise ValueError("bounded model-profile enrollment required")
            for name, record in registered.items():
                scheduler.register_profile(name, record["artifact_sha256"], record["adapter"])
            cells = value.get("cells", [])
            if len(cells) > 1000:
                raise ValueError("enroll at most 1000 owner-selected cells per configuration batch")
            for cell in cells:
                scheduler.register(Cell(**cell), ttl_ms=300000)
            if args.command == "fleet-register":
                print(json.dumps(dict(enrolled=len(cells))))
            else:
                async def serve():
                    server = await FleetService(scheduler, server_tls(value["certificate"], value["key_file"], value["ca_file"]),
                        host=value.get("host", "127.0.0.1"), port=value.get("port", 8443), max_connections=value.get("max_connections", 128)).start()
                    try:
                        print(json.dumps(dict(host=server.host, port=server.port, mutual_tls=True)), flush=True)
                        await asyncio.Event().wait()
                    finally:
                        await server.close()
                asyncio.run(serve())
        finally:
            scheduler.close()
    elif args.command == "fleet-agent":
        if value.get("format") != "hyperl-fleet-agent/1":
            raise ValueError("unsupported cell-agent configuration")
        agent = LlmAgent(control_client(value["client"]), Cell(**value["cell"]), {name: profile(item) for name, item in value["profiles"].items()})
        try:
            agent.run(poll_seconds=value.get("poll_seconds", 2))
        finally:
            agent.close()
    else:
        client = control_client(value)
        if args.command == "fleet-submit":
            jobs = read_json(args.request)
            result = client.call("/v1/jobs/submit", dict(jobs=jobs["jobs"]))
        elif args.command == "fleet-status":
            result = client.call("/v1/jobs/get", dict(id=args.job_id))
        elif args.command == "fleet-cancel":
            result = client.call("/v1/jobs/cancel", dict(id=args.job_id))
        else:
            raise ValueError("unknown cluster operation")
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("model-import")
    ingest.add_argument("root"); ingest.add_argument("source"); ingest.add_argument("max_bytes", type=int)
    export = commands.add_parser("model-export")
    export.add_argument("root"); export.add_argument("artifact"); export.add_argument("destination"); export.add_argument("max_bytes", type=int)
    plan = commands.add_parser("model-plan")
    plan.add_argument("request")
    enroll = commands.add_parser("fleet-enroll")
    enroll.add_argument("database"); enroll.add_argument("request"); enroll.add_argument("now_ms", type=int)
    page = commands.add_parser("fleet-page")
    page.add_argument("database"); page.add_argument("now_ms", type=int); page.add_argument("--cell"); page.add_argument("--pool"); page.add_argument("--after", default=""); page.add_argument("--limit", type=int, default=100)
    for name in ("rpc-worker", "rpc-tunnel", "fleet-register", "fleet-serve", "fleet-agent"):
        command = commands.add_parser(name); command.add_argument("config")
    infer = commands.add_parser("llm-run")
    infer.add_argument("config"); infer.add_argument("prompt"); infer.add_argument("--tokens", type=int, default=32)
    submit = commands.add_parser("fleet-submit")
    submit.add_argument("config"); submit.add_argument("request")
    for name in ("fleet-status", "fleet-cancel"):
        command = commands.add_parser(name); command.add_argument("config"); command.add_argument("job_id")
    digest = commands.add_parser("artifact-hash"); digest.add_argument("path")
    keygen = commands.add_parser("audit-key-new"); keygen.add_argument("path")
    audit = commands.add_parser("audit-verify"); audit.add_argument("config"); audit.add_argument("--checkpoint")
    for name in ("model-protect", "model-repair"):
        command = commands.add_parser(name)
        command.add_argument("root"); command.add_argument("artifact"); command.add_argument("max_bytes", type=int)
        command.add_argument("--library", required=True); command.add_argument("--library-sha256", required=True)
        if name == "model-protect":
            command.add_argument("--data-shards", type=int, default=4); command.add_argument("--parity-shards", type=int, default=2)
        else:
            command.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command in {"llm-run", "rpc-worker", "rpc-tunnel", "fleet-register", "fleet-serve", "fleet-agent", "fleet-submit", "fleet-status", "fleet-cancel", "artifact-hash", "audit-key-new", "audit-verify", "model-protect", "model-repair"}:
        execute_command(args)
        return
    if args.command in {"model-import", "model-export"}:
        store = ModelStore(args.root)
        try:
            if args.command == "model-import":
                identifier = store.ingest(args.source, max_bytes=args.max_bytes)
                print(json.dumps(store.info(identifier)))
            else:
                store.export(args.artifact, args.destination, max_bytes=args.max_bytes)
                print(json.dumps(dict(exported=args.artifact)))
        finally:
            store.close()
    elif args.command == "model-plan":
        request = read_json(args.request)
        if request.get("format") != "hyperl-model-plan-request/1":
            raise ValueError("unsupported model plan request")
        policy = Workload(**request.get("workload", {}))
        if "llm" in request:
            if "layers" in request:
                raise ValueError("supply exact layers or an LLM estimate, not both")
            layers, activations = estimate_llm(LLMProfile(**request["llm"]), policy)
            if "activation_bytes" not in request.get("workload", {}):
                policy = dataclasses.replace(policy, activation_bytes=activations)
        else:
            layers = [ModelLayer(**item) for item in request["layers"]]
        print(json.dumps(plan_model(layers, [MemoryTier(**item) for item in request["tiers"]], policy, manual=request.get("manual")), indent=2))
    else:
        directory = DeviceDirectory(args.database)
        try:
            if args.command == "fleet-enroll":
                request = read_json(args.request)
                if request.get("format") != "hyperl-fleet-enrollment/1":
                    raise ValueError("unsupported fleet enrollment")
                directory.enroll([Device(**item) for item in request["devices"]], args.now_ms)
                for name, cells in request.get("pools", {}).items():
                    directory.set_pool(name, cells)
                print(json.dumps(dict(enrolled=len(request["devices"]))))
            else:
                print(json.dumps([dataclasses.asdict(item) for item in directory.page(args.now_ms, cell=args.cell, pool=args.pool, after=args.after, limit=args.limit)], indent=2))
        finally:
            directory.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, InterruptedError, RuntimeError, KeyError, TypeError) as error:
        raise SystemExit("HyperL control failed: " + str(error)[:512])
