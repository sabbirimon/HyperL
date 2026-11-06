# Conservative MoE residency accounting

Standalone HyperL enhancement under its existing Community and Enterprise licence;
no implementation was copied into Meshlit. Meshlit can consume the existing CLI.

`hyperl-model-plan-request/1` remains the input format. Each exact `layers` entry
may now supply `moe` with exactly these integer fields:

```json
{
  "name": "moe-layer",
  "weight_bytes": 432,
  "kv_bytes": 24,
  "moe": {
    "total_experts": 4,
    "active_experts": 2,
    "expert_weight_bytes": 100,
    "shared_weight_bytes": 24,
    "router_weight_bytes": 8
  }
}
```

This is a tiny synthetic accounting fixture, not a model measurement. Expert
weights must have equal declared byte sizes. `weight_bytes` must equal shared +
router + total experts × bytes per expert. Active experts must be within 1..total;
total is bounded to 4,096. Missing, unknown, fractional, boolean and inconsistent
MoE fields reject. Heterogeneous expert sizes and speculative routes are unsupported.

Every expert remains included in placement, offload transfers and whole-layer
staging. `active_expert_weight_bytes_per_token` is explanatory; it is never
subtracted from resident bytes or used to predict cache hits, speed or execution.
The existing native/backend qualification requirements remain independent.

Optional `workload.cache_bytes` reserves additional execution memory for an
owner-sized cache. It defaults to zero and cannot discount model weights. Outputs
now include `state_breakdown` for KV, activation, additional cache and training
state, alongside the existing per-tier state, staging and total used-byte reports.
These are advisory estimates and not actual allocations or guaranteed reservations.

Use the existing CLI with exact observations in a request file:

```sh
PYTHONPATH=python python3 -m hyperl.cluster model-plan request.json
```

Validation on 2026-11-06:

- Focused model-storage suite: 13 passed, including four new MoE/CLI tests.
- Full Python discovery: 89 tests, 84 passed and 5 skipped because the optional JAX
  environment was not installed. No physical TPU qualification occurred.
- `./gradlew test installDist distZip --console=plain`: successful; unchanged JVM
  test tasks were up to date, and distribution packaging completed.
- Native CMake build succeeded; CTest passed 5/5. The Metal probe is discovery,
  not GPU execution evidence.
- No inference, expert routing, storage throughput or speedup benchmark was run
  for this planner change.
