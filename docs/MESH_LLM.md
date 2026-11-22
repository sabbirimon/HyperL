# Optional Mesh-LLM backend

Added 14 November 2026. HyperL can query an operator-configured Mesh-LLM API for
models and send a bounded text-chat request through its Python API or CLI.
This is an external inference adapter, separate from HyperL's native/JAX model
sessions, memory planner, artifact verification and fleet scheduler.

## Placement and useful upstream features

Mesh-LLM belongs on the **HyperL execution/backend side**. Its Mesh service handles
peer discovery and request routing; its Skippy runtime handles inference and
package-backed layer stages. Meshlit remains the chat/bot/settings client.
The reviewed revision is pinned in [provenance](MESH_LLM_PROVENANCE.json).

| Upstream feature | HyperL integration state |
| --- | --- |
| OpenAI-compatible model listing and text completion | Implemented client; real HTTP/TLS contract tests |
| Local or peer model routing through one API | Delegated to explicitly configured external daemon; placement unverified |
| Skippy split-model layer packages | Candidate external backend; not dispatched by HyperL's planner yet |
| Mesh discovery, peer trust and private/public mesh operation | Operator-managed; no automatic joining or publishing |
| Flash-MoE SSD backend/plugin | Research candidate; single-node expert streaming is not distributed expert sharding |
| Automatic routing / Mixture-of-Agents | An explicit `mesh` model request delegates selection to the daemon; no HyperL committee or learning claim |

The daemon's API being on loopback does **not** prove local inference. A private
or unpublished mesh and an invite token do **not** establish owner-only admission.
Configure upstream peer allowlists/trust policy and inspect actual routing before
sending private prompts. HyperL records `declared_routing_scope`, with
`routing_scope_verified`, `placement_verified`, `model_artifact_verified` and
`sharding_qualified` all false. The model name returned by a server is reported
metadata, not proof of the loaded weights or model author.

## Configure an already running daemon

Install and review the pinned upstream separately using its platform instructions.
This adapter does not download or run executables, models, Docker containers or
shell commands. In particular it never invokes `serve --auto`, which upstream
documents as able to download a model and join a public mesh.

For a single local complete model, the reviewed upstream documents
`mesh-llm serve --local-model-only --gguf /absolute/model.gguf --port 9337`.
That bypasses Mesh networking and split planning and is an upstream debugging
mode. It requires a separately installed compatible runtime and enough memory;
it is not a promise of Intel Mac/AMD support or a setup step run by HyperL.

Create `mesh.json` as a regular file:

```json
{
  "format": "hyperl-mesh-llm/1",
  "base_url": "http://127.0.0.1:9337/v1",
  "routing_scope": "local-model-only",
  "timeout_seconds": 60
}
```

Set `routing_scope` to the actual operator-selected `local-model-only`,
`private-mesh` or `public-mesh` mode. This is a declaration, not an enforcement
or attestation switch. Changing it does not change the remote daemon's config.
Use an exact model ID to avoid opting into upstream automatic model selection.

```sh
PYTHONPATH=python python3 -m hyperl.cluster mesh-models mesh.json
PYTHONPATH=python python3 -m hyperl.cluster mesh-chat mesh.json chat.json
```

Example `chat.json` (replace the model with an actual listed ID):

```json
{
  "model": "your-listed-model-id",
  "messages": [{"role": "user", "content": "Explain model layer sharding briefly."}],
  "max_tokens": 256,
  "temperature": 0.2
}
```

Python callers can use `await MeshClient(settings).models()` and
`await client.complete(model, messages, max_tokens=256)`. Import these from
`hyperl.mesh_llm`. Cancelling the asyncio task closes the transport. CLI Ctrl-C
exits with 130. Connection close/timeout is **not acknowledgement that the remote
job stopped**; no uncertain request is retried automatically.

For an owner-controlled remote reverse proxy, set `base_url` to its HTTPS `/v1`
endpoint, `allow_remote` to true, and `token_env` to the **name** of a process
environment variable containing the endpoint's bearer token. Do not store the
token in JSON, command arguments, source or logs. The remote proxy must actually
authenticate that bearer token; an arbitrary value does not secure upstream.
System CA and hostname verification remain enabled. Non-loopback plaintext HTTP,
URL credentials/query/fragments, redirects and environment HTTP proxies are
rejected/ignored. Numeric loopback HTTP is also suitable for a separately managed
authenticated SSH tunnel; the tunnel itself is outside this adapter.

## Bounds and output

- 128 KiB request body, 1 MiB response body, 32 KiB response headers;
  bounded JSON depth 16 and duplicate/nonfinite rejection.
- At most 128 text messages, 64 KiB each, and 1–8,192 requested output tokens.
  This is a request cap, not a guarantee that a server obeys it.
- Total request timeout 0.1–300 seconds, no automatic retry/fallback, no SSE or
  media/tool execution. Responses requesting tools fail visibly.
- Model listing caps 1,000 entries. An empty catalog stays empty; no fake model is
  inserted. Upstream does not support pagination on this compatibility route.
- Missing usage stays null. Present counts come from the provider, not character
  estimates. `requested_model` and `reported_model` are distinct fields.
- Errors expose fixed messages/HTTP status only, not remote bodies, bearer values
  or prompt text. Successful chat output intentionally contains the requested reply.
- No socket or endpoint is opened until a caller invokes a client operation.

## Validation and remaining gates

`python3 -m unittest discover -s scripts -p test_mesh_llm.py -v` exercises real
local TCP HTTP and certificate-verified TLS against controlled protocol fixtures,
CLI listing/chat, bearer delivery, ignored proxies, redirect/non-200 rejection,
malformed/oversized messages, timeout and cancellation disconnect.
Those responses are explicitly test fixtures, not fake production responses and
not evidence of Mesh-LLM model execution.

Pending: installed pinned Mesh/Skippy generation, exact weight/hash identity,
physical multi-host layer split and recovery tests, Intel Mac/AMD performance,
authenticated remote deployment, native catalog/placement telemetry, streaming,
and a planner/scheduler adapter that revalidates placement at dispatch.
There is no replacement of HyperL's native worker protocol or newly qualified
GPU, MoE, mobile, remote-memory or speedup claim.

## License boundary

The reviewed Mesh-LLM root license is Apache-2.0; its native runtimes/plugins and
model files retain their own terms. No upstream source, assets, binaries or models
are copied or bundled here. This original HTTP adapter follows HyperL's existing
Community/Enterprise terms for new material. Mesh-LLM remains independently
installed and licensed; no Apache rights are withdrawn or relabeled.


## Intel Mac runtime check — 2026-11-14

The local host is macOS x86_64. The current `v0.79.0` release has Apple Silicon
macOS host/runtime/CLI assets, with no Intel Mac host archive; the pinned source's
SDK matrix likewise lists aarch64 macOS. Release source
`e8c9e7fba8bf88b3fa65a4187862d1147dce372c` is distinct from this adapter's reviewed
source pin. No incompatible archive was downloaded or executed.

No Mesh/Skippy daemon, Rust compiler/Cargo, just, sccache or Docker is installed.
A real `mesh-models` call against `http://127.0.0.1:9337/v1` correctly failed with
a sanitized transport error and no invented model catalog. This is an observed
missing runtime, not live inference proof or proof that an Intel source port is
impossible. Building and qualifying that larger toolchain/runtime, or connecting
a supported owner-controlled host, remains the next gate.

[Release/platform/probe evidence](evidence/mesh-llm-intel-host-20261114.json).
The existing bundled Meshlit local model remains separate and does not become
Mesh-LLM by exposing a generic compatible endpoint.
