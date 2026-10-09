// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.
package org.hyperl.distributed

import org.hyperl.*
import com.sun.net.httpserver.*
import kotlinx.coroutines.*
import kotlinx.coroutines.CancellationException
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.*
import java.net.*
import java.nio.file.*
import java.security.*
import java.util.concurrent.*
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import javax.net.ssl.*

@Serializable data class WorkerSettings(val format: String = "hyperl-worker/1", val id: String, val bind: String = "127.0.0.1",
    val port: Int = 8091, val tokenFile: String, val memoryBudgetBytes: Long = 64L * 1024 * 1024, val maxJobs: Int = 1,
    val maxShardElements: Int = 16384, val backend: HyperLTarget = HyperLTarget.CPU_REFERENCE,
    val bridgeFile: String? = null, val device: String? = null, val scratchDirectory: String? = null,
    val tlsKeystoreFile: String? = null, val tlsPasswordFile: String? = null) {
    fun validate() {
        require(format == "hyperl-worker/1" && id.matches(Regex("[A-Za-z][A-Za-z0-9_-]{0,63}")))
        require(port in 0..65535 && maxJobs in 1..8 && maxShardElements in 1..HyperLContract.MAX_VECTOR_ELEMENTS)
        require(memoryBudgetBytes in 4L * 1024 * 1024..1024L * 1024 * 1024)
        require(bind.length in 1..128 && (bind.matches(Regex("[0-9.]+")) || bind.contains(':')) && '%' !in bind)
        val address = InetAddress.getByName(bind)
        require(tlsKeystoreFile != null || address.isLoopbackAddress && bind in setOf("127.0.0.1", "::1")) { "Non-loopback binding requires TLS" }
        require((tlsKeystoreFile == null) == (tlsPasswordFile == null))
        require(Path.of(tokenFile).isAbsolute)
        require(backend in setOf(HyperLTarget.CPU_REFERENCE, HyperLTarget.OPENCL_SPIRV, HyperLTarget.METAL)) { "Worker backend unavailable; no fallback" }
        if (backend == HyperLTarget.CPU_REFERENCE) require(bridgeFile == null && device == null)
        else require(bridgeFile != null && Path.of(bridgeFile).isAbsolute && device != null && maxJobs == 1) { "GPU workers need an owner-installed bridge, exact device and one job" }
        if (backend == HyperLTarget.OPENCL_SPIRV) require(device!!.toInt() in 0..127)
        if (backend == HyperLTarget.METAL) require(device!!.length in 1..256 && device.none { it.isISOControl() })
    }
}

@Serializable data class WorkerCapabilities(val format: String = ShardProtocol.FORMAT, val id: String, val instanceId: String,
    val backend: HyperLTarget, val runtimeRevision: String = HyperLContract.RUNTIME_REVISION, val operations: Set<String> = ShardProtocol.operations,
    val numericContract: String, val availableAdmissionBytes: Long, val configuredBudgetBytes: Long, val maxShardElements: Int,
    val maxJobs: Int, val activeJobs: Int, val completedShards: Long, val processId: Long, val memoryDomains: List<MemoryDomain>,
    val device: String? = null, val automaticSpill: Boolean = false, val modelLayerPartitioning: Boolean = false)

/** Explicit adapters; no inferred backend, executable plugin download or target substitution. */
interface ShardComputeAdapter {
    val target: HyperLTarget
    val memoryDomains: List<MemoryDomain> get() = emptyList()
    suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray
}

private class ConfiguredAdapter(private val settings: WorkerSettings): ShardComputeAdapter {
    override val target = settings.backend
    private val cpu = if (target == HyperLTarget.CPU_REFERENCE) HyperLCpuBackend(settings.memoryBudgetBytes) else null
    private val openCl = if (target == HyperLTarget.OPENCL_SPIRV) OpenClBridge(Path.of(settings.bridgeFile!!)) else null
    private val metal = if (target == HyperLTarget.METAL) MetalBridge(Path.of(settings.bridgeFile!!)) else null
    override val memoryDomains: List<MemoryDomain> = runBlocking {
        val probe = when (target) {
            HyperLTarget.OPENCL_SPIRV -> ShardProtocol.json.parseToJsonElement(openCl!!.probe()).jsonObject
            HyperLTarget.METAL -> ShardProtocol.json.parseToJsonElement(metal!!.probe()).jsonObject
            else -> null
        }
        if (probe == null) emptyList() else {
            val devices = probe.getValue("devices").jsonArray; require(devices.size <= 128)
            val selected = devices.map { it.jsonObject }.singleOrNull { if (target == HyperLTarget.METAL)
                it["name"]?.jsonPrimitive?.content == settings.device else it["index"]?.jsonPrimitive?.int == settings.device!!.toInt() }
                ?: error("Configured GPU unavailable; no CPU substitution")
            val unified = selected["hasUnifiedMemory"]?.jsonPrimitive?.booleanOrNull
            fun bytes(name: String) = selected[name]?.jsonPrimitive?.longOrNull?.also { require(it >= 0) }
            listOf(MemoryDomain("accelerator", if (unified == true) "UNIFIED_GPU" else "ACCELERATOR_MEMORY",
                capacityBytes = bytes("globalMemoryBytes"), sharedWith = if (unified == true) "system" else null,
                source = "Selected ${target.name} driver probe; no free VRAM or physical reservation; memory technology unknown",
                recommendedWorkingSetBytes = bytes("recommendedMaxWorkingSetSize"),
                maxAllocationBytes = bytes(if (target == HyperLTarget.METAL) "maxBufferLength" else "maxAllocationBytes")))
        }
    }
    override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray = when (target) {
        HyperLTarget.CPU_REFERENCE -> cpu!!.execute(program, inputs)
        HyperLTarget.OPENCL_SPIRV -> openCl!!.executeVerified(settings.device!!.toInt(), program, inputs)
        HyperLTarget.METAL -> metal!!.executeVerified(settings.device!!, program, inputs).result
        else -> error("Worker backend unavailable")
    }
}

/** Application credit leases, not OS/VRAM reservations. One worker cannot overbook its own budget. */
internal class MemoryCredits(private val budget: Long) {
    private var used = 0L
    @Synchronized fun available(): Long = (minOf(budget, MemoryPlanner.observe().availableBytes()) - used).coerceAtLeast(0)
    @Synchronized fun acquire(bytes: Long): Boolean { require(bytes >= 0); if (bytes > available()) return false; used += bytes; return true }
    @Synchronized fun release(bytes: Long) { require(bytes in 0..used); used -= bytes }
}

/** Bounded authenticated foreground service for pure vector shards. No shell, model or arbitrary file paths in requests. */
class ShardWorker(private val settings: WorkerSettings, adapter: ShardComputeAdapter? = null): AutoCloseable {
    private val compute: ShardComputeAdapter
    private val token: ByteArray
    private val server: HttpServer
    private val credits: MemoryCredits
    private val slots: java.util.concurrent.Semaphore
    private val instance = java.util.UUID.randomUUID().toString()
    private val completed = AtomicLong()
    private val jobs = ConcurrentHashMap<String, Job>()
    private val recent = LinkedHashMap<String, Long>()
    private val deadlines = Executors.newSingleThreadScheduledExecutor { r -> Thread(r, "hyperl-deadline").apply { isDaemon = true } }
    private val executor = ThreadPoolExecutor(4 + settings.maxJobs, 4 + settings.maxJobs, 0L, TimeUnit.MILLISECONDS,
        ArrayBlockingQueue(32), { r -> Thread(r, "hyperl-worker").apply { isDaemon = true } }, ThreadPoolExecutor.AbortPolicy())
    @Volatile private var stopped = false
    val origin: String get() = "${if (server is HttpsServer) "https" else "http"}://${if (settings.bind.contains(':')) "[${settings.bind}]" else settings.bind}:${server.address.port}"
    init {
        settings.validate(); token = ShardProtocol.token(Path.of(settings.tokenFile)).toByteArray(Charsets.US_ASCII)
        credits = MemoryCredits(settings.memoryBudgetBytes); slots = java.util.concurrent.Semaphore(settings.maxJobs)
        compute = adapter ?: ConfiguredAdapter(settings); require(compute.target == settings.backend)
        // Provider read limits complement the exchange deadline. Set before initializing the provider.
        System.setProperty("sun.net.httpserver.maxReqTime", "30")
        System.setProperty("sun.net.httpserver.maxRspTime", "30")
        System.setProperty("jdk.httpserver.maxConnections", "32")
        val address = InetSocketAddress(InetAddress.getByName(settings.bind), settings.port)
        server = if (settings.tlsKeystoreFile != null) HttpsServer.create(address, 16).apply {
            httpsConfigurator = HttpsConfigurator(tls(settings.tlsKeystoreFile, settings.tlsPasswordFile!!))
        } else HttpServer.create(address, 16)
        server.executor = executor
        server.createContext("/hyperl") { exchange -> handle(exchange) }
    }
    fun start(): ShardWorker { check(!stopped); server.start(); return this }
    fun capabilities(): WorkerCapabilities = WorkerCapabilities(id = settings.id, instanceId = instance, backend = compute.target,
        numericContract = if (compute.target == HyperLTarget.CPU_REFERENCE) "ordered-f32-exact/1" else "f32-checked-1e-6/1",
        availableAdmissionBytes = credits.available(), configuredBudgetBytes = settings.memoryBudgetBytes, maxShardElements = settings.maxShardElements,
        maxJobs = settings.maxJobs, activeJobs = jobs.size, completedShards = completed.get(), processId = ProcessHandle.current().pid(),
        memoryDomains = MemoryDomains.observe(settings.memoryBudgetBytes, settings.scratchDirectory?.let(Path::of)) + compute.memoryDomains, device = settings.device)

    private fun handle(exchange: HttpExchange) {
        var job: Job? = null
        val activeJob = AtomicReference<Job?>()
        val watchdog = deadlines.schedule({ activeJob.get()?.cancel(); exchange.close() }, 30, TimeUnit.SECONDS)
        try {
            val supplied = exchange.requestHeaders["Authorization"]
            if (supplied?.size != 1 || supplied[0].any { it.code !in 32..126 } || !MessageDigest.isEqual(supplied[0].toByteArray(Charsets.US_ASCII), "Bearer ".toByteArray() + token)) { status(exchange, 401); return }
            if (stopped || exchange.requestURI.rawQuery != null) { status(exchange, 400); return }
            val path = exchange.requestURI.rawPath
            when {
                path == "/hyperl/capabilities" && exchange.requestMethod == "GET" -> {
                    val bytes = ShardProtocol.json.encodeToString(capabilities()).toByteArray()
                    respond(exchange, 200, "application/json", bytes)
                }
                path.startsWith("/hyperl/jobs/") && exchange.requestMethod == "DELETE" -> {
                    val id = path.removePrefix("/hyperl/jobs/"); ShardProtocol.uuid(id)
                    val running = jobs[id]
                    if (running == null) status(exchange, 404) else { running.cancel(); status(exchange, 202) }
                    // 202 acknowledges cancellation requested, never remote termination.
                }
                path == "/hyperl/execute" && exchange.requestMethod == "POST" -> {
                    // The client may receive the previous body's last byte before its handler
                    // releases credits/slot. A bounded admission wait fences that cleanup race
                    // without replaying work or permitting unbounded queued computation.
                    if (!slots.tryAcquire(100, TimeUnit.MILLISECONDS)) { status(exchange, 429); return }
                    var lease = 0L; var id: String? = null
                    try {
                        require(exchange.requestHeaders.getFirst("Content-Type") == ShardProtocol.CONTENT_TYPE)
                        require(exchange.requestHeaders["Content-Length"]?.size == 1)
                        val length = exchange.requestHeaders.getFirst("Content-Length").toInt()
                        require(length in 8..ShardProtocol.MAX_FRAME)
                        id = exchange.requestHeaders.getFirst("X-HyperL-Job"); ShardProtocol.uuid(id)
                        if (!claim(id)) { status(exchange, 409); return }
                        val digest = exchange.requestHeaders.getFirst("X-HyperL-SHA256")
                        require(digest?.matches(Regex("[a-f0-9]{64}")) == true)
                        lease = 2L * length + 512L * 1024
                        if (!credits.acquire(lease)) { lease = 0; status(exchange, 503); return }
                        job = Job(); activeJob.set(job); jobs[id] = job
                        val bytes = exchange.requestBody.readNBytes(length + 1)
                        require(bytes.size == length && ShardProtocol.digest(bytes) == digest)
                        val (request, inputs) = ShardProtocol.decodeRequest(bytes)
                        require(request.jobId == id && request.workerInstanceId == instance && request.end - request.start <= settings.maxShardElements)
                        compute.memoryDomains.forEach { domain -> domain.maxAllocationBytes?.let { require(4L * (request.end - request.start) <= it) } }
                        val estimated = ShardProtocol.shardBytes(request.program, request.end - request.start)
                        if (estimated > lease) {
                            if (!credits.acquire(estimated - lease)) { status(exchange, 503); return }
                            lease = estimated
                        }
                        val started = System.nanoTime()
                        val output = runBlocking(job) { withTimeout(request.timeoutMs) { compute.execute(request.program, inputs) } }
                        val meta = ShardResponse(jobId = id, start = request.start, end = request.end, requestSha256 = digest,
                            backend = compute.target, computeMs = (System.nanoTime() - started) / 1e6)
                        val result = ShardProtocol.encodeResponse(meta, output)
                        exchange.responseHeaders.set("X-HyperL-SHA256", ShardProtocol.digest(result))
                        respond(exchange, 200, ShardProtocol.CONTENT_TYPE, result); completed.incrementAndGet()
                    } finally { if (id != null) jobs.remove(id); job?.cancel(); if (lease > 0) credits.release(lease); slots.release() }
                }
                else -> status(exchange, 404)
            }
        } catch (_: CancellationException) { runCatching { status(exchange, 408) } }
          catch (_: Exception) { runCatching { status(exchange, 422) } }
        finally { watchdog.cancel(false); exchange.close() }
    }
    @Synchronized private fun claim(id: String): Boolean {
        val now = System.currentTimeMillis(); recent.entries.removeIf { now - it.value > 300000 }
        if (id in recent || recent.size >= 1024) return false
        recent[id] = now; return true
    }
    private fun status(exchange: HttpExchange, status: Int) = respond(exchange, status, "application/json", "{\"status\":$status}".toByteArray())
    private fun respond(exchange: HttpExchange, status: Int, type: String, bytes: ByteArray) {
        exchange.responseHeaders.set("Content-Type", type); exchange.responseHeaders.set("Cache-Control", "no-store")
        exchange.sendResponseHeaders(status, bytes.size.toLong()); exchange.responseBody.use { it.write(bytes) }
    }
    override fun close() { stopped = true; jobs.values.forEach { it.cancel() }; server.stop(0); executor.shutdownNow(); deadlines.shutdownNow(); token.fill(0) }
    private fun tls(keyFile: String, passwordFile: String): SSLContext {
        val path = Path.of(keyFile); val passwordPath = Path.of(passwordFile)
        require(path.isAbsolute && Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) && Files.size(path) <= 1024 * 1024)
        require(passwordPath.isAbsolute && Files.isRegularFile(passwordPath, LinkOption.NOFOLLOW_LINKS) && Files.size(passwordPath) in 1..1024)
        val password = Files.readString(passwordPath).trimEnd('\r', '\n').toCharArray()
        try {
            val store = KeyStore.getInstance("PKCS12"); Files.newInputStream(path).use { store.load(it, password) }
            val managers = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm()); managers.init(store, password)
            return SSLContext.getInstance("TLS").apply { init(managers.keyManagers, null, SecureRandom()) }
        } finally { password.fill('\u0000') }
    }
}
