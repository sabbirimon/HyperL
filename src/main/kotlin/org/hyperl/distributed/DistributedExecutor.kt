// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.serialization.Serializable
import java.net.http.*
import java.nio.ByteBuffer
import java.nio.file.Path
import java.time.Duration
import java.util.concurrent.*
import java.util.concurrent.Flow
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

@Serializable data class ShardPlacement(val workerId: String, val start: Int, val end: Int, val estimatedAdmissionBytes: Long)
@Serializable data class DistributedPlan(val format: String = "hyperl-placement/1", val vectorElements: Int,
    val workers: List<WorkerCapabilities>, val shards: List<ShardPlacement>, val maxParallelism: Int,
    val estimatedCoordinatorBytes: Long, val coordinatorBudgetBytes: Long, val orderedReductionAtCoordinator: Boolean,
    val physicalMemoryPooled: Boolean = false, val storageSpillEnabled: Boolean = false)
@Serializable data class ShardMeasurement(val workerId: String, val start: Int, val end: Int, val backend: HyperLTarget,
    val roundTripMs: Double, val workerComputeMs: Double, val sentBytes: Int, val receivedBytes: Int)
@Serializable data class DistributedResult(val format: String = "hyperl-distributed-result/1", val result: FloatArray,
    val plan: DistributedPlan, val shards: List<ShardMeasurement>, val endToEndMs: Double, val cpuVerified: Boolean = true,
    val scope: String = "Bounded elementwise shards; CPU exact or checked GPU tolerance. No model layers, shared address space, automatic spill or speedup claim.")

/** Persistent connections per explicitly approved worker; bounded bodies and cooperative cancellation. */
internal class ShardClient(private val endpoint: WorkerEndpoint, private val timeoutMs: Long): AutoCloseable {
    private val origin = ShardProtocol.origin(endpoint.origin)
    private val token = ShardProtocol.token(Path.of(endpoint.tokenFile))
    private val executor = Executors.newFixedThreadPool(2) { r -> Thread(r, "hyperl-client").apply { isDaemon = true } }
    private val client = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(minOf(timeoutMs, 5000)))
        .followRedirects(HttpClient.Redirect.NEVER).version(HttpClient.Version.HTTP_1_1).executor(executor).build()
    private val pending = ConcurrentHashMap.newKeySet<CompletableFuture<*>>()
    suspend fun capabilities(): WorkerCapabilities {
        val response = send(request("/hyperl/capabilities").GET().build(), 256 * 1024)
        require(response.statusCode() == 200 && type(response) == "application/json") { "Worker capability request rejected" }
        val text = response.body().toString(Charsets.UTF_8); ShardProtocol.checkJson(text, 256 * 1024)
        val cap = ShardProtocol.json.decodeFromString<WorkerCapabilities>(text)
        require(cap.format == ShardProtocol.FORMAT && cap.id == endpoint.id && cap.runtimeRevision == HyperLContract.RUNTIME_REVISION)
        ShardProtocol.uuid(cap.instanceId)
        require(cap.backend in setOf(HyperLTarget.CPU_REFERENCE, HyperLTarget.METAL, HyperLTarget.OPENCL_SPIRV) && cap.operations == ShardProtocol.operations)
        require(cap.numericContract == if (cap.backend == HyperLTarget.CPU_REFERENCE) "ordered-f32-exact/1" else "f32-checked-1e-6/1")
        require(cap.availableAdmissionBytes in 0..cap.configuredBudgetBytes && cap.configuredBudgetBytes in 4L * 1024 * 1024..1024L * 1024 * 1024)
        require(cap.maxShardElements in 1..HyperLContract.MAX_VECTOR_ELEMENTS && cap.maxJobs in 1..8 && cap.activeJobs in 0..8 && cap.memoryDomains.size <= 260)
        return cap
    }
    suspend fun execute(meta: ShardRequest, inputs: Map<String, FloatArray>, target: HyperLTarget): Pair<FloatArray, ShardMeasurement> {
        val bytes = ShardProtocol.encodeRequest(meta, inputs); val digest = ShardProtocol.digest(bytes)
        val call = request("/hyperl/execute").header("Content-Type", ShardProtocol.CONTENT_TYPE).header("X-HyperL-Job", meta.jobId)
            .header("X-HyperL-SHA256", digest).POST(HttpRequest.BodyPublishers.ofByteArray(bytes)).build()
        val started = System.nanoTime()
        val response = send(call, ShardProtocol.MAX_FRAME, meta.jobId)
        require(response.statusCode() == 200 && type(response) == ShardProtocol.CONTENT_TYPE) { "Worker shard rejected (HTTP ${response.statusCode()}); job aborted without retry" }
        require(response.headers().firstValue("X-HyperL-SHA256").orElse("") == ShardProtocol.digest(response.body())) { "Shard response digest mismatch" }
        val (receipt, values) = ShardProtocol.decodeResponse(response.body())
        require(receipt.jobId == meta.jobId && receipt.start == meta.start && receipt.end == meta.end && receipt.requestSha256 == digest && receipt.backend == target)
        require(receipt.computeMs.isFinite() && receipt.computeMs in 0.0..30000.0)
        return values to ShardMeasurement(endpoint.id, meta.start, meta.end, target, (System.nanoTime() - started) / 1e6, receipt.computeMs, bytes.size, response.body().size)
    }
    private fun request(path: String) = HttpRequest.newBuilder(origin.resolve(path)).timeout(Duration.ofMillis(timeoutMs))
        .header("Authorization", "Bearer $token").header("Accept", ShardProtocol.CONTENT_TYPE + ", application/json")
    private fun type(response: HttpResponse<ByteArray>) = response.headers().firstValue("Content-Type").orElse("").substringBefore(';')
    private suspend fun send(request: HttpRequest, bound: Int, jobId: String? = null): HttpResponse<ByteArray> = suspendCancellableCoroutine { continuation ->
        val future = client.sendAsync(request, HttpResponse.BodyHandler { LimitedBody(bound) }); pending.add(future)
        continuation.invokeOnCancellation {
            future.cancel(true)
            if (jobId != null) runCatching {
                // Best effort request, not a termination guarantee. Owner-selected destination only.
                val cancel = client.sendAsync(this.request("/hyperl/jobs/$jobId").timeout(Duration.ofSeconds(2)).DELETE().build(), HttpResponse.BodyHandlers.discarding())
                pending.add(cancel); cancel.whenComplete { _, _ -> pending.remove(cancel) }
            }
        }
        future.whenComplete { value, error -> pending.remove(future)
            if (continuation.isActive) { if (error != null) continuation.resumeWithException(IllegalStateException("Worker transport failed; remote completion may be unknown", error)) else continuation.resume(value) }
        }
    }
    override fun close() { pending.forEach { it.cancel(true) }; executor.shutdownNow() }
    private class LimitedBody(private val limit: Int): HttpResponse.BodySubscriber<ByteArray> {
        private val delegate = HttpResponse.BodySubscribers.ofByteArray()
        private var subscription: Flow.Subscription? = null
        private var count = 0L
        override fun getBody(): CompletionStage<ByteArray> = delegate.body
        override fun onSubscribe(value: Flow.Subscription) { subscription = value; delegate.onSubscribe(value) }
        override fun onNext(items: List<ByteBuffer>) { count += items.sumOf { it.remaining().toLong() }
            if (count > limit) { subscription?.cancel(); delegate.onError(IllegalArgumentException("Worker response size limit")) } else delegate.onNext(items) }
        override fun onError(error: Throwable) = delegate.onError(error)
        override fun onComplete() = delegate.onComplete()
    }
}

/** Split/map/gather with exact range ownership. Only terminal ordered sum stays at the coordinator. */
object DistributedExecutor {
    private data class Shape(val workerProgram: HyperLProgram, val reduce: Boolean, val elements: Int)
    private fun shape(program: HyperLProgram, inputs: Map<String, FloatArray>): Shape {
        program.validate(); require(inputs.keys == program.inputs)
        require(inputs.values.all { it.size in 1..HyperLContract.MAX_VECTOR_ELEMENTS })
        val n = inputs.values.first().size
        require(inputs.values.all { it.size == n }) { "Sharded inputs require equal lengths; no broadcasting" }
        require(inputs.values.sumOf { it.size.toLong() } <= HyperLContract.MAX_RETAINED_ELEMENTS)
        val reductions = program.instructions.filter { it.operation == "sum" }
        if (reductions.isEmpty()) return Shape(program, false, n)
        val last = program.instructions.last()
        require(reductions.size == 1 && last.operation == "sum" && last.output == program.output && program.instructions.size > 1) {
            "Only one terminal ordered sum after elementwise work is distributable; internal/scalar reductions require a different partition contract" }
        return Shape(program.copy(instructions = program.instructions.dropLast(1), output = last.inputs.single()), true, n)
    }
    fun plan(program: HyperLProgram, inputs: Map<String, FloatArray>, settings: DistributedSettings, offers: List<WorkerCapabilities>,
        snapshot: MemorySnapshot = MemoryPlanner.observe()): DistributedPlan {
        settings.validate(); val shape = shape(program, inputs)
        require(offers.size == settings.workers.size && offers.map { it.id }.toSet() == settings.workers.map { it.id }.toSet())
        require(!shape.reduce || offers.all { it.backend == HyperLTarget.CPU_REFERENCE }) { "Ordered reduction requires exact CPU workers; GPU reductions need a separate numerical contract" }
        val base = 1024L * 1024 + 8L * inputs.values.sumOf { it.size.toLong() } + 4L * shape.elements
        val available = minOf(settings.coordinatorBudgetBytes, snapshot.availableBytes())
        val parallelism = minOf(settings.maxParallelism, offers.size)
        val perCall = (available - base).coerceAtLeast(0) / parallelism
        val coordinatorLimit = ShardProtocol.maxElements(shape.workerProgram, perCall)
        require(coordinatorLimit > 0) { "Coordinator lacks staging/verification memory" }
        val sorted = offers.sortedBy { it.id }
        val limits = sorted.associate { cap ->
            require(cap.availableAdmissionBytes in 0..1024L * 1024 * 1024 && cap.maxShardElements in 1..HyperLContract.MAX_VECTOR_ELEMENTS)
            require(cap.operations == ShardProtocol.operations && cap.runtimeRevision == HyperLContract.RUNTIME_REVISION)
            require(cap.format == ShardProtocol.FORMAT && cap.numericContract == if (cap.backend == HyperLTarget.CPU_REFERENCE) "ordered-f32-exact/1" else "f32-checked-1e-6/1")
            require(cap.backend in setOf(HyperLTarget.CPU_REFERENCE, HyperLTarget.METAL, HyperLTarget.OPENCL_SPIRV))
            val allocationElements = cap.memoryDomains.mapNotNull { it.maxAllocationBytes }.minOrNull()?.let { (it / 4).coerceAtMost(HyperLContract.MAX_VECTOR_ELEMENTS.toLong()).toInt() } ?: HyperLContract.MAX_VECTOR_ELEMENTS
            cap.id to minOf(cap.maxShardElements, ShardProtocol.maxElements(shape.workerProgram, cap.availableAdmissionBytes), coordinatorLimit,
                settings.shardElements ?: ((shape.elements + sorted.size - 1) / sorted.size), allocationElements)
        }
        require(limits.values.all { it > 0 }) { "An approved worker lacks shard headroom; remove it or free resources explicitly" }
        val placements = mutableListOf<ShardPlacement>(); var start = 0; var index = 0
        while (start < shape.elements) {
            require(placements.size < ShardProtocol.MAX_SHARDS) { "Shard count limit; increase explicit budgets or shard size" }
            val worker = sorted[index++ % sorted.size]; val end = minOf(shape.elements, start + limits.getValue(worker.id))
            placements += ShardPlacement(worker.id, start, end, ShardProtocol.shardBytes(shape.workerProgram, end - start)); start = end
        }
        val peak = base + placements.sortedByDescending { it.estimatedAdmissionBytes }.take(parallelism).sumOf { it.estimatedAdmissionBytes }
        require(peak <= available) { "Coordinator admission exceeded" }
        return DistributedPlan(vectorElements = shape.elements, workers = sorted, shards = placements, maxParallelism = parallelism,
            estimatedCoordinatorBytes = peak, coordinatorBudgetBytes = settings.coordinatorBudgetBytes, orderedReductionAtCoordinator = shape.reduce)
    }
    suspend fun run(program: HyperLProgram, inputs: Map<String, FloatArray>, settings: DistributedSettings, planOnly: Boolean = false): Pair<DistributedPlan, DistributedResult?> {
        settings.validate()
        val original = program.copy(inputs = program.inputs.toSet(), instructions = program.instructions.map { it.copy(inputs = it.inputs.toList()) })
        val shape = shape(original, inputs)
        // Validate parsing/staging admission before connecting or copying caller buffers.
        val initial = 1024L * 1024 + 8L * inputs.values.sumOf { it.size.toLong() } + 4L * shape.elements + minOf(settings.maxParallelism, settings.workers.size) * 512L * 1024
        require(initial <= minOf(settings.coordinatorBudgetBytes, MemoryPlanner.observe().availableBytes())) { "Coordinator memory budget insufficient" }
        val ownedProgram = shape.workerProgram.copy(inputs = shape.workerProgram.inputs.toSet(), instructions = shape.workerProgram.instructions.map { it.copy(inputs = it.inputs.toList()) })
        val owned = inputs.mapValues { (_, values) -> FloatArray(values.size) { i -> if (i % 1024 == 0) currentCoroutineContext().ensureActive(); values[i].also { require(it.isFinite()) } } }
        val clients = mutableMapOf<String, ShardClient>()
        try { settings.workers.forEach { clients[it.id] = ShardClient(it, settings.requestTimeoutMs) }
            return withTimeout(settings.jobTimeoutMs) { coroutineScope {
            val started = System.nanoTime(); val permits = Semaphore(settings.maxParallelism)
            val observed = settings.workers.map { endpoint -> async { permits.withPermit { clients.getValue(endpoint.id).capabilities() to System.nanoTime() } } }.awaitAll()
            require(observed.all { System.nanoTime() - it.second <= 15_000_000_000L }) { "Stale worker offers; repeat the owner-requested plan" }
            val offers = observed.map { it.first }
            val placement = plan(original, owned, settings, offers)
            if (planOnly) return@coroutineScope placement to null
            val gathered = FloatArray(shape.elements)
            val measurements = placement.shards.groupBy { it.workerId }.map { (workerId, shards) -> async(Dispatchers.Default) {
                val cap = offers.single { it.id == workerId }
                shards.map { shard -> permits.withPermit {
                    currentCoroutineContext().ensureActive()
                    val values = owned.mapValues { (_, source) -> source.copyOfRange(shard.start, shard.end) }
                    val request = ShardRequest(jobId = java.util.UUID.randomUUID().toString(), start = shard.start, end = shard.end, program = ownedProgram, workerInstanceId = cap.instanceId, timeoutMs = settings.requestTimeoutMs)
                    val (result, measured) = clients.getValue(workerId).execute(request, values, cap.backend)
                    val expected = HyperLCpuBackend(settings.coordinatorBudgetBytes).execute(ownedProgram, values)
                    result.forEachIndexed { i, value -> if (i % 1024 == 0) currentCoroutineContext().ensureActive()
                        require(if (cap.backend == HyperLTarget.CPU_REFERENCE) value.toRawBits() == expected[i].toRawBits()
                            else abs(value.toDouble() - expected[i]) <= 1e-6 * maxOf(1.0, abs(expected[i].toDouble()))) { "Worker output differs from admitted numerical contract" } }
                    result.copyInto(gathered, shard.start); measured
                } }
            } }.awaitAll().flatten().sortedBy { it.start }
            currentCoroutineContext().ensureActive()
            val output = if (shape.reduce) { var sum = 0f; gathered.forEachIndexed { i, value -> if (i % 1024 == 0) currentCoroutineContext().ensureActive(); sum += value; require(sum.isFinite()) { "Ordered sum overflow" } }; floatArrayOf(sum) } else gathered
            placement to DistributedResult(result = output, plan = placement, shards = measurements, endToEndMs = (System.nanoTime() - started) / 1e6)
        } } } finally { clients.values.forEach { it.close() } }
    }
}
