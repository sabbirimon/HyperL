# Memory-aware HyperL

The standalone alpha observes its JVM heap limit/usage and, when exposed by the
JDK management interface, operating-environment total/free memory. These values
can describe a container rather than a physical host. They are a timestamp-free
point-in-time snapshot, not a reservation, RSS measurement or available accelerator
memory. Missing provider information stays null.

## Working feature

```sh
bin/hyperl capabilities
bin/hyperl memory-plan examples/elementwise.json examples/inputs.json
bin/hyperl run examples/elementwise.json examples/inputs.json 1048576
```

The optional final argument selects a CPU memory budget in bytes (1 byte–1 GiB,
default 16 MiB). It cannot lift the `hyperl/1` format's 1,048,576 retained-float
limit. The GUI's **Memory plan** button displays the same report; Run uses the
default policy. No native GPU memory capacity is inferred from this CPU report.

Planning validates every DAG shape before execution, counts input and retained
vectors, caller input storage, copied inputs and returned output, then adds a
32-byte estimate per array and 64 KiB workspace allowance. This is an estimate,
not a universal JVM layout formula or a bound on total process RSS. Parsed JSON,
UI, native drivers, crypto and unrelated tasks consume additional memory.

CPU execution rejects before copying its input arrays if the estimate exceeds
either the selected budget or observed JVM heap headroom. Headroom is half the remaining
JVM heap after a 32 MiB reserve. Environment free memory is a separate
pressure advisory: free pages do not account for all OS-reclaimable memory, so
that value alone cannot safely determine admission. It also rejects a graph exceeding the retained-vector
format limit. A valid plan is not a promise of future allocation success: other
processes/threads can consume memory immediately afterward. JVM allocation
failure and OS memory pressure still need operational handling; the CLI does not
pretend that it reserved memory or disabled the OS's paging policy.

Encrypted local dataset import/export keeps 4 MiB chunks and independently
bounded authenticated manifest records. Before starting, it checks at least
32 MiB of the same conservative observed headroom for array/crypto work. It
requires explicit destinations and quotas; it never silently spills a kernel,
creates a swapfile, evicts another process, installs a driver or changes clocks.
Disk IO may still fail if storage runs out or permissions change.

## Memory tiers and later SDK

Reports distinguish JVM_HEAP, SYSTEM_RAM, HBM, GDDR, DDR_LPDDR, UNIFIED_GPU, SRAM,
CXL_NUMA and SSD_NVME. Only the actually observed heap/environment has capacity.
The alpha cannot identify RAM technology, memory bus width/rate, NUMA topology,
VRAM, HBM, coherent sharing or whether a filesystem is backed by NVMe.

Later qualified native adapters will expose separate capacity/availability,
allocation domains, shared-pool identities, alignment, locality, coherency,
bandwidth/latency observations and source/time/driver evidence. Shared CPU/GPU
memory must be counted once, and capacity must never be derived from a vendor
name. Native admission must include weights, activations, KV caches, staging,
workspace, pinned transfers and competing reservations.

The full SDK remains a later milestone: caller-controlled arenas, versioned
allocation/buffer ownership, pressure callbacks, release/cancellation, bounded
pools, lifetime-based reuse, out-of-core partitioning, encrypted owner-selected
SSD/NVMe spill and transfer-cost routing. HBM/local GPU memory gets priority only
when actual measurements show benefit and the numerical/ownership contracts hold.
Cluster memory is per allocation domain and node; remote memory is not a free
extension of local RAM. Recompute, quantization and precision changes need explicit
semantics rather than automatic output changes.

## Evidence and references

Memory tests exercise real heap observation and deterministic policy cases for
caller/copy accounting, low budget, low observed free memory, whole-graph limits,
shape mismatch and rejection before input copying. Deterministic snapshots test
policy; they are not hardware observations. Native C ABI retains its existing
fixed vector limit; this JVM policy does not govern arbitrary C callers or GPU VRAM.
See [validation](VALIDATION.md) for the actual run outcome.

Primary interfaces: [JDK Runtime heap methods](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/Runtime.html),
[JDK OperatingSystemMXBean environment observations](https://docs.oracle.com/en/java/javase/21/docs/api/jdk.management/com/sun/management/OperatingSystemMXBean.html),
and [Linux cgroup v2 memory controls](https://docs.kernel.org/admin-guide/cgroup-v2.html).
No cgroup, kernel or memory-controller settings are modified by this alpha.
