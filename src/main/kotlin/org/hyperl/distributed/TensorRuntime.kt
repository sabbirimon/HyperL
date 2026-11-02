// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.Serializable
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

@Serializable data class LogicalDevice(val id: String, val workers: Set<String>)
@Serializable data class TensorPlacement(val slice: TensorSlice, val workerId: String, val backend: HyperLTarget, val transferBytes: Long)
@Serializable data class TensorExecutionPlan(val pool: String, val output: TensorHandle, val placements: List<TensorPlacement>,
    val estimatedTransferBytes: Long, val sharedAddressSpace: Boolean = false)
@Serializable enum class TensorJobState { RUNNING, SUCCEEDED, FAILED, CANCELLED }
@Serializable data class TensorJob(val id: String, val kind: String, val dependencies: List<TensorHandle>, val state: TensorJobState,
    val output: TensorHandle? = null, val detail: String? = null)
@Serializable data class TensorRunResult(val output: TensorHandle, val result: FloatArray, val transfers: TensorTransferStats)
@Serializable data class TensorTransferStats(val requests: Long, val uploadedPayloadBytes: Long, val downloadedPayloadBytes: Long, val cacheHits: Long)

/** One bounded execution cell. A logical device is a placement pool, never an added physical RAM claim.
 * Transactions publish handles only after every shard is checked. No replay after uncertain completion.
 */
class TensorRuntime private constructor(private val settings: DistributedSettings, private val clients: Map<String, ShardClient>,
    private val initial: Map<String, WorkerCapabilities>, private val cacheBudgetBytes: Long) {
    private data class Part(val chunk: TensorChunk, val owners: MutableSet<String>)
    private data class Allocation(val handle: TensorHandle, val parts: List<Part>)
    private val session = UUID.randomUUID().toString()
    private val gate = Mutex()
    private val allocations = mutableMapOf<Pair<String, Long>, Allocation>()
    private val versions = mutableMapOf<String, Long>()
    private val pools = mutableMapOf("combined" to LogicalDevice("combined", initial.keys.toSet()))
    private val ledger = ConcurrentHashMap<String, TensorJob>()
    private val running = ConcurrentHashMap<String, Job>()
    private val permits = Semaphore(settings.maxParallelism)
    private val cache = LinkedHashMap<TensorChunk, FloatArray>(16, 0.75f, true)
    private var cachedBytes = 0L
    private val requests = AtomicLong(); private val uploaded = AtomicLong(); private val downloaded = AtomicLong(); private val hits = AtomicLong()
    @Volatile private var closed = false
    private val chunkElements = minOf(settings.shardElements ?: 4096, initial.values.minOf { it.maxShardElements },
        HyperLContract.MAX_RETAINED_ELEMENTS / (HyperLContract.MAX_INPUTS + HyperLContract.MAX_STEPS))

    companion object {
        suspend fun connect(settings: DistributedSettings, cacheBudgetBytes: Long = minOf(32L * 1024 * 1024, settings.coordinatorBudgetBytes / 4), expectedInstances: Map<String, String> = emptyMap()): TensorRuntime {
            settings.validate(); require(settings.workers.map { it.id }.containsAll(expectedInstances.keys)); expectedInstances.values.forEach(ShardProtocol::uuid); require(cacheBudgetBytes in 0..settings.coordinatorBudgetBytes / 2)
            val clients = mutableMapOf<String, ShardClient>()
            try {
                settings.workers.forEach { clients[it.id] = ShardClient(it, settings.requestTimeoutMs) }
                val offers = withTimeout(settings.jobTimeoutMs) { coroutineScope {
                    settings.workers.map { endpoint -> async { endpoint.id to clients.getValue(endpoint.id).capabilities() } }.awaitAll().toMap()
                } }
                require(expectedInstances.all { (id, instance) -> offers.getValue(id).instanceId == instance }) { "Directory worker-instance fence failed" }
                return TensorRuntime(settings, clients, offers, cacheBudgetBytes)
            } catch (e: Throwable) { clients.values.forEach { it.close() }; throw e }
        }
    }
    fun jobs(): List<TensorJob> = ledger.values.sortedBy { it.id }.map { it.copy(dependencies = it.dependencies.map(::copyHandle), output = it.output?.let(::copyHandle)) }
    fun cancel(jobId: String): Boolean { ShardProtocol.uuid(jobId); return running[jobId]?.let { it.cancel(CancellationException("Cancelled by owner")); true } ?: false }
    fun stats() = TensorTransferStats(requests.get(), uploaded.get(), downloaded.get(), hits.get())
    fun logicalDevices(): List<LogicalDevice> = synchronized(pools) { pools.values.map { it.copy(workers = it.workers.toSet()) } }
    fun defineDevice(id: String, workers: Set<String>) {
        require(id.matches(Regex("[A-Za-z][A-Za-z0-9_-]{0,63}")) && workers.isNotEmpty() && initial.keys.containsAll(workers))
        synchronized(pools) { check(!closed); require(pools.size < 64 || id in pools); pools[id] = LogicalDevice(id, workers.toSet()) }
    }
    fun forgetJob(id: String) { require(ledger[id]?.state != TensorJobState.RUNNING); ledger.remove(id) }

    suspend fun allocate(values: FloatArray, shape: List<Int> = listOf(values.size), pool: String = "combined"): TensorHandle = transaction("allocate", emptyList()) { job ->
        allocateInternal(values, TensorHandle(UUID.randomUUID().toString(), 1, shape.toList()), pool).also { finish(job, it) }
    }
    /** Compare-and-set latest version; prior versions remain readable until explicitly released. */
    suspend fun replace(previous: TensorHandle, values: FloatArray): TensorHandle = transaction("replace", listOf(previous)) { job ->
        val prior = allocation(previous)
        require(versions[previous.id] == previous.version) { "Stale tensor version; replacement rejected" }
        require(previous.version < 1_000_000_000)
        allocateInternal(values, prior.handle.copy(version = previous.version + 1), "combined").also { finish(job, it) }
    }
    suspend fun gather(handle: TensorHandle): FloatArray = transaction("gather", listOf(handle)) { _ ->
        val record = allocation(handle); admission(record.handle.elements())
        val result = FloatArray(handle.elements())
        record.parts.forEach { part -> currentCoroutineContext().ensureActive(); fetch(part).copyInto(result, part.chunk.slice.start) }
        result
    }
    suspend fun plan(program: HyperLProgram, inputs: Map<String, TensorHandle>, pool: String = "combined"): TensorExecutionPlan = transaction("plan", inputs.values.toList()) { _ ->
        val owned = snapshot(program); val records = validateInputs(owned, inputs); val offers = offers(pool)
        buildPlan(records, offers, pool, TensorHandle(UUID.randomUUID().toString(), 1, records.values.first().handle.shape.toList()))
            .also { validatePlan(owned, it, offers) }
    }
    suspend fun execute(program: HyperLProgram, inputs: Map<String, TensorHandle>, pool: String = "combined"): TensorHandle = transaction("execute", inputs.values.toList()) { job ->
        val owned = snapshot(program); val records = validateInputs(owned, inputs)
        directoryAdmission(records.values.first().parts.size)
        val offers = offers(pool); val output = TensorHandle(UUID.randomUUID().toString(), 1, records.values.first().handle.shape.toList())
        val plan = buildPlan(records, offers, pool, output); validatePlan(owned, plan, offers)
        val parts = coroutineScope {
            plan.placements.groupBy { it.workerId }.map { (worker, placements) -> async(Dispatchers.Default) {
                placements.map { placement -> permits.withPermit {
                    currentCoroutineContext().ensureActive()
                    val index = placement.slice.start / chunkElements
                    val sources = records.mapValues { it.value.parts[index] }
                    val data = sources.mapValues { fetch(it.value) }
                    val missing = sources.values.distinctBy { it.chunk }.filter { worker !in it.owners }
                    missing.chunked(8).forEach { batch -> call(worker, "put", chunks = batch.map { it.chunk }, values = batch.map { data.getValue(sources.entries.first { e -> e.value.chunk == it.chunk }.key) })
                        batch.forEach { it.owners.add(worker) } }
                    val (reply, values) = call(worker, "execute", inputs = sources.mapValues { it.value.chunk }, output = placement.slice, program = owned)
                    val result = values.single()
                    val expected = HyperLCpuBackend(settings.coordinatorBudgetBytes).execute(owned, data)
                    require(result.size == expected.size)
                    result.forEachIndexed { i, value ->
                        if (i % 1024 == 0) currentCoroutineContext().ensureActive()
                        require(if (placement.backend == HyperLTarget.CPU_REFERENCE) value.toRawBits() == expected[i].toRawBits()
                            else abs(value.toDouble() - expected[i]) <= 1e-6 * maxOf(1.0, abs(expected[i].toDouble()))) { "Tensor worker numerical contract mismatch" }
                    }
                    val chunk = reply.chunks.single(); cachePut(chunk, result)
                    Part(chunk, mutableSetOf(worker))
                } }
            } }.awaitAll().flatten().sortedBy { it.chunk.slice.start }
        }
        currentCoroutineContext().ensureActive()
        allocations[output.id to output.version] = Allocation(copyHandle(output), parts); versions[output.id] = output.version
        finish(job, output); output
    }
    suspend fun release(handle: TensorHandle): Unit = transaction("release", listOf(handle)) { _ ->
        val record = allocation(handle)
        drop(record)
        allocations.remove(handle.id to handle.version); record.parts.forEach { cacheRemove(it.chunk) }
    }
    /** Close waits for local cleanup attempts. Remote failures retain only expiring leases, never replayed jobs. */
    suspend fun close() {
        if (closed) return
        running.values.forEach { it.cancel() }
        gate.lock()
        try {
            if (closed) return
            closed = true
            withContext(NonCancellable) { withTimeoutOrNull(settings.jobTimeoutMs) { clients.keys.sorted().forEach { worker -> runCatching { call(worker, "clear") } } } }
            allocations.clear(); synchronized(cache) { cache.clear(); cachedBytes = 0 }
            clients.values.forEach { it.close() }
        } finally { gate.unlock() }
    }
    private suspend fun allocateInternal(values: FloatArray, handle: TensorHandle, pool: String): TensorHandle {
        val n = handle.elements(); require(values.size == n && values.all(Float::isFinite))
        val count = (n + chunkElements - 1) / chunkElements; require(count <= ShardProtocol.MAX_SHARDS); directoryAdmission(count); admission(n)
        val owned = values.copyOf(); val offers = offers(pool).sortedBy { it.id }
        val parts = (0 until n step chunkElements).mapIndexed { index, start ->
            val end = minOf(n, start + chunkElements); val data = owned.copyOfRange(start, end)
            val chunk = TensorChunk(TensorSlice(handle.id, handle.version, start, end), TensorProtocol.digest(data))
            Part(chunk, mutableSetOf(offers[index % offers.size].id))
        }
        coroutineScope { parts.groupBy { it.owners.single() }.map { (worker, group) -> async {
            permits.withPermit { group.chunked(8).forEach { batch ->
                currentCoroutineContext().ensureActive()
                call(worker, "put", chunks = batch.map { it.chunk }, values = batch.map { owned.copyOfRange(it.chunk.slice.start, it.chunk.slice.end) })
            } }
        } }.awaitAll() }
        currentCoroutineContext().ensureActive()
        parts.forEach { cachePut(it.chunk, owned.copyOfRange(it.chunk.slice.start, it.chunk.slice.end)) }
        allocations[handle.id to handle.version] = Allocation(copyHandle(handle), parts); versions[handle.id] = handle.version
        return handle
    }
    private fun copyHandle(handle: TensorHandle) = handle.copy(shape = handle.shape.toList())
    private fun allocation(handle: TensorHandle): Allocation {
        handle.elements(); val record = allocations[handle.id to handle.version] ?: error("Unknown/released tensor handle")
        require(record.handle == handle) { "Tensor shape/version mismatch" }; return record
    }
    private fun snapshot(program: HyperLProgram): HyperLProgram {
        val copy = program.copy(inputs = program.inputs.toSet(), instructions = program.instructions.map { it.copy(inputs = it.inputs.toList()) })
        copy.validate(); require(copy.instructions.all { it.operation in ShardProtocol.operations }) { "Tensor execution supports elementwise operations only; no model or reduction substitution" }
        return copy
    }
    private fun validateInputs(program: HyperLProgram, inputs: Map<String, TensorHandle>): Map<String, Allocation> {
        require(inputs.keys == program.inputs); val records = inputs.mapValues { allocation(it.value) }
        require(records.values.map { it.handle.shape }.distinct().size == 1) { "Tensor shapes must match exactly; no broadcasting" }
        return records
    }
    private suspend fun offers(pool: String): List<WorkerCapabilities> {
        val device = synchronized(pools) { pools[pool] ?: error("Unknown logical device") }
        return device.workers.sorted().map { worker -> fresh(worker) }
    }
    private suspend fun fresh(worker: String): WorkerCapabilities {
        val offer = clients.getValue(worker).capabilities(); val original = initial.getValue(worker)
        require(offer.instanceId == original.instanceId && offer.backend == original.backend && offer.device == original.device && offer.memoryDomains.map { listOf(it.id, it.kind, it.sharedWith, it.maxAllocationBytes, it.recommendedWorkingSetBytes) } == original.memoryDomains.map { listOf(it.id, it.kind, it.sharedWith, it.maxAllocationBytes, it.recommendedWorkingSetBytes) }) {
            "Worker identity/backend/memory evidence changed; reconnect and reallocate explicitly" }
        return offer
    }
    private fun buildPlan(records: Map<String, Allocation>, offers: List<WorkerCapabilities>, pool: String, output: TensorHandle): TensorExecutionPlan {
        val load = offers.associate { it.id to 0L }.toMutableMap()
        val placements = records.values.first().parts.indices.map { index ->
            val sources = records.values.map { it.parts[index] }.distinctBy { it.chunk }
            fun bytes(worker: String) = sources.filter { worker !in it.owners }.sumOf { 4L * (it.chunk.slice.end - it.chunk.slice.start) }
            val worker = offers.minWith(compareBy<WorkerCapabilities> { bytes(it.id) }.thenBy { load.getValue(it.id) }.thenBy { it.id })
            val range = sources.first().chunk.slice; load[worker.id] = load.getValue(worker.id) + range.end - range.start
            TensorPlacement(TensorSlice(output.id, output.version, range.start, range.end), worker.id, worker.backend, bytes(worker.id))
        }
        return TensorExecutionPlan(pool, output, placements, placements.sumOf { it.transferBytes })
    }
    private fun validatePlan(program: HyperLProgram, plan: TensorExecutionPlan, offers: List<WorkerCapabilities>) {
        plan.placements.forEach { placement ->
            val count = placement.slice.end - placement.slice.start; val offer = offers.single { it.id == placement.workerId }
            require(count <= offer.maxShardElements && (program.inputs.size + program.instructions.size).toLong() * count <= HyperLContract.MAX_RETAINED_ELEMENTS)
            require(ShardProtocol.shardBytes(program, count) + 4L * count <= offer.availableAdmissionBytes) { "Worker lacks execution/output headroom" }
            offer.memoryDomains.forEach { domain -> domain.maxAllocationBytes?.let { require(4L * count <= it) } }
        }
        val peak = plan.placements.maxOf { ShardProtocol.shardBytes(program, it.slice.end - it.slice.start) }
        require(cacheBudgetBytes + 1024L * 1024 + minOf(settings.maxParallelism, offers.size) * peak <= settings.coordinatorBudgetBytes) { "Coordinator tensor staging budget exceeded" }
    }
    private suspend fun fetch(part: Part): FloatArray {
        cacheGet(part.chunk)?.let { return it }
        val worker = part.owners.sorted().first() // no failover after an uncertain result
        val (_, values) = call(worker, "get", chunks = listOf(part.chunk))
        return values.single().also { cachePut(part.chunk, it) }
    }
    private suspend fun call(worker: String, action: String, chunks: List<TensorChunk> = emptyList(), values: List<FloatArray> = emptyList(),
        inputs: Map<String, TensorChunk> = emptyMap(), output: TensorSlice? = null, program: HyperLProgram? = null): Pair<TensorReply, List<FloatArray>> {
        currentCoroutineContext().ensureActive(); val cap = fresh(worker)
        val meta = TensorRequest(jobId = UUID.randomUUID().toString(), sessionId = session, workerInstanceId = cap.instanceId,
            action = action, chunks = chunks, inputs = inputs, output = output, program = program, timeoutMs = settings.requestTimeoutMs)
        requests.incrementAndGet(); uploaded.addAndGet(values.sumOf { 4L * it.size })
        return clients.getValue(worker).tensor(meta, values, cap.backend).also { downloaded.addAndGet(it.second.sumOf { data -> 4L * data.size }) }
    }
    private suspend fun drop(record: Allocation) {
        record.parts.flatMap { part -> part.owners.map { it to part.chunk } }.groupBy({ it.first }, { it.second }).forEach { (worker, chunks) ->
            chunks.chunked(8).forEach { call(worker, "drop", chunks = it) }
        }
    }
    private fun admission(elements: Int) { require(cacheBudgetBytes + 1024L * 1024 + 16L * elements + settings.maxParallelism * 2L * 1024 * 1024 <= settings.coordinatorBudgetBytes) { "Coordinator tensor snapshot budget exceeded" } }
    private fun directoryAdmission(parts: Int) { require(allocations.size < 256 && versions.size < 4096 && allocations.values.sumOf { it.parts.size } + parts <= TensorProtocol.MAX_CHUNKS) { "Tensor directory full; release allocations or start a new cell" } }
    private fun cacheGet(chunk: TensorChunk): FloatArray? = synchronized(cache) { cache[chunk]?.also { hits.incrementAndGet() }?.copyOf() }
    private fun cachePut(chunk: TensorChunk, data: FloatArray) = synchronized(cache) {
        val size = 4L * data.size; if (size > cacheBudgetBytes) return@synchronized
        cache.remove(chunk)?.let { cachedBytes -= 4L * it.size }
        while (cachedBytes + size > cacheBudgetBytes) { val entry = cache.entries.iterator(); val oldest = entry.next(); cachedBytes -= 4L * oldest.value.size; entry.remove() }
        cache[chunk] = data.copyOf(); cachedBytes += size
    }
    private fun cacheRemove(chunk: TensorChunk) = synchronized(cache) { cache.remove(chunk)?.let { cachedBytes -= 4L * it.size } }
    private fun finish(id: String, output: TensorHandle) { ledger.computeIfPresent(id) { _, value -> value.copy(output = copyHandle(output)) } }
    private suspend fun <T> transaction(kind: String, dependencies: List<TensorHandle>, action: suspend (String) -> T): T {
        check(gate.tryLock()) { "Tensor coordinator busy; bounded cell accepts one transaction at a time" }
        val id = UUID.randomUUID().toString()
        try {
            check(!closed); require(ledger.size < 256) { "Job ledger full; forget completed jobs" }
            ledger[id] = TensorJob(id, kind, dependencies.map { it.copy(shape = it.shape.toList()) }, TensorJobState.RUNNING)
            return withTimeout(settings.jobTimeoutMs) {
                running[id] = currentCoroutineContext().job
                action(id).also { currentCoroutineContext().ensureActive(); ledger.computeIfPresent(id) { _, value -> value.copy(state = TensorJobState.SUCCEEDED) } }
            }
        } catch (e: Throwable) {
            ledger.computeIfPresent(id) { _, value -> value.copy(state = if (e is CancellationException) TensorJobState.CANCELLED else TensorJobState.FAILED,
                detail = "Transaction aborted; no partial handle published. Remote leases may remain until expiry; no automatic replay.", output = null) }
            throw e
        } finally { running.remove(id); gate.unlock() }
    }
}
