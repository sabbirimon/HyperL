package org.hyperl

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.security.MessageDigest
import kotlinx.serialization.Serializable
import kotlin.math.floor

@Serializable enum class ShardObjective { CAPACITY, LATENCY }
@Serializable enum class ShardDecision { SHARDED_CANDIDATE, SINGLE_DEVICE_PREFERRED, NO_ADMISSIBLE_PLAN, INSUFFICIENT_EVIDENCE }

/** Observed free bytes before the listed reservations. Aliases of physical UMA must use one ID. */
@Serializable data class ShardMemoryDomain(
    val id: String, val availableBytes: Long?, val reservedBytes: Long = 0, val observedAtMs: Long
)

/** A trusted caller supplies observations. These fields are not identity or enrollment proof. */
@Serializable data class ShardWorker(
    val deviceId: String, val memoryDomainId: String, val target: HyperLTarget,
    val runtimeRevision: String, val numericalContract: String,
    val supportedOperations: Set<String>, val programSha256: String,
    val benchmarkElements: Int, val measuredElementsPerSecond: Double,
    val copyBytesPerSecond: Double, val queueMs: Double, val overheadBytes: Long,
    val observedAtMs: Long, val qualified: Boolean = false, val ownerEnabled: Boolean = false
)

@Serializable data class ShardLink(
    val from: String, val to: String, val goodputBytesPerSecond: Double,
    val roundTripMs: Double, val sampleBytes: Long, val observedAtMs: Long,
    val authenticated: Boolean = false
)

/** Aggregate measured source ingress/egress ceilings account for shared network contention. */
@Serializable data class ShardSource(
    val deviceId: String, val memoryDomainId: String, val copyBytesPerSecond: Double,
    val egressBytesPerSecond: Double?, val ingressBytesPerSecond: Double?,
    val observedAtMs: Long
)

@Serializable data class ShardingRequest(
    val format: String = "hyperl-sharding/1", val program: HyperLProgram, val elements: Int,
    val source: ShardSource, val workers: List<ShardWorker>, val memoryDomains: List<ShardMemoryDomain>,
    val links: List<ShardLink>, val objective: ShardObjective = ShardObjective.LATENCY,
    val maxWorkers: Int = 8, val deadlineMs: Double? = null
)

@Serializable data class ShardRange(
    val deviceId: String, val memoryDomainId: String, val startInclusive: Int, val endExclusive: Int,
    val estimatedWorkingBytes: Long, val estimatedBranchMs: Double
)
@Serializable data class ShardDomainUse(val id: String, val requiredBytes: Long, val availableAfterReservations: Long)
@Serializable data class ShardingPlan(
    val format: String, val programSha256: String, val decision: ShardDecision,
    val candidateShards: List<ShardRange>, val memory: List<ShardDomainUse>,
    val estimatedShardedMs: Double?, val estimatedSingleDeviceMs: Double?, val singleDeviceId: String?,
    val capacityRequired: Boolean, val excludedWorkers: Map<String, String>, val reason: String,
    // Required wire fields: serializers must not omit these crucial boundaries as defaults.
    val executionActivated: Boolean, val reservationGuaranteed: Boolean, val timingsAreEstimates: Boolean
)

/** H0: pure bounded placement/admission. No IO, remote dispatch, reservation or trust enrollment.
 * Only independent f32 elementwise graphs qualify. Ordered sum must never be reassociated.
 * At most 8 workers (255 subsets), 262144 elements and 16 physical memory domains.
 */
object HyperLShardingPlanner {
    const val NUMERICAL_CONTRACT = "hyperl-f32-strict/1"
    const val MAX_AGE_MS = 30_000L
    private val id = Regex("[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
    private val hash = Regex("[a-f0-9]{64}")
    private val operations = setOf("add", "multiply", "relu")

    /** Canonical, length-prefixed graph identity, independent of Set insertion order. */
    fun programSha256(program: HyperLProgram): String {
        program.validate()
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { out ->
            out.writeUTF("hyperl-program-digest/1")
            out.writeUTF(program.format)
            out.writeInt(program.inputs.size)
            program.inputs.sorted().forEach(out::writeUTF)
            out.writeInt(program.instructions.size)
            program.instructions.forEach { step ->
                out.writeUTF(step.output); out.writeUTF(step.operation); out.writeInt(step.inputs.size)
                step.inputs.forEach(out::writeUTF)
            }
            out.writeUTF(program.output)
        }
        return MessageDigest.getInstance("SHA-256").digest(bytes.toByteArray()).joinToString("") { "%02x".format(it) }
    }

    fun plan(request: ShardingRequest, nowMs: Long): ShardingPlan {
        request.program.validate()
        require(request.format == "hyperl-sharding/1") { "Unsupported sharding contract" }
        require(request.program.instructions.all { it.operation in operations }) {
            "Only independent add/multiply/relu graphs can be sharded; ordered sum is not reassociated"
        }
        require(nowMs > 0 && request.elements in 2..262144 && request.maxWorkers in 2..8)
        require(request.workers.size in 1..8 && request.memoryDomains.size in 1..16 && request.links.size <= 64)
        require(request.deadlineMs == null || positive(request.deadlineMs))
        require(request.workers.map { it.deviceId }.distinct().size == request.workers.size) { "Duplicate device identity" }
        require(request.memoryDomains.map { it.id }.distinct().size == request.memoryDomains.size) { "Duplicate physical memory domain" }
        require(request.links.map { it.from to it.to }.distinct().size == request.links.size) { "Duplicate directed link" }
        val source = request.source
        require(id.matches(source.deviceId) && id.matches(source.memoryDomainId))
        request.memoryDomains.forEach { domain ->
            require(id.matches(domain.id) && domain.reservedBytes >= 0 && (domain.availableBytes == null || domain.availableBytes >= 0))
        }
        request.workers.forEach { worker ->
            require(id.matches(worker.deviceId) && id.matches(worker.memoryDomainId))
            require(worker.runtimeRevision.length in 1..128 && worker.numericalContract.length in 1..128)
            require(hash.matches(worker.programSha256) && worker.supportedOperations.size <= 64 && worker.supportedOperations.all { it.length in 1..32 })
            require(worker.overheadBytes >= 0)
        }
        request.links.forEach { require(id.matches(it.from) && id.matches(it.to) && it.from != it.to) }
        val digest = programSha256(request.program)
        val excluded = sortedMapOf<String, String>()
        fun result(decision: ShardDecision, reason: String, candidate: Candidate? = null, single: Candidate? = null) = ShardingPlan(
            "hyperl-sharding-plan/1", digest, decision, candidate?.ranges ?: emptyList(), candidate?.memory ?: emptyList(),
            candidate?.timeMs, single?.timeMs, single?.ranges?.single()?.deviceId,
            candidate != null && single == null, excluded.toMap(), reason, false, false, true
        )
        val domains = request.memoryDomains.associateBy { it.id }
        val available = domains.mapNotNull { (name, domain) ->
            domain.availableBytes?.takeIf { fresh(domain.observedAtMs, nowMs) && it >= domain.reservedBytes }
                ?.let { name to (it - domain.reservedBytes) }
        }.toMap()
        if (!fresh(source.observedAtMs, nowMs) || !positive(source.copyBytesPerSecond) || source.memoryDomainId !in available) {
            return result(ShardDecision.INSUFFICIENT_EVIDENCE, "Source memory/copy observations are unknown, stale or invalid")
        }
        val inputCount = request.program.inputs.size
        // Full input + assembled output resident at source. All worker arrays and bounded transfer
        // buffers are counted separately, including when a worker shares source UMA.
        val sourceBytes = Math.addExact(Math.multiplyExact((inputCount + 1L) * 4, request.elements.toLong()), 65536)
        val bytesPerElement = (3L * inputCount + request.program.instructions.size + 2) * 4
        val fixedArrays = 65536L + 32L * (inputCount * 2 + request.program.instructions.size + 1)
        val retainedLimit = 1048576 / (inputCount + request.program.instructions.size)
        val linkMap = request.links.associateBy { it.from to it.to }
        val nodes = request.workers.sortedBy { it.deviceId }.mapNotNull { worker ->
            val outbound = linkMap[source.deviceId to worker.deviceId]
            val inbound = linkMap[worker.deviceId to source.deviceId]
            val remote = worker.deviceId != source.deviceId
            fun validLink(link: ShardLink?) = link != null && link.authenticated && fresh(link.observedAtMs, nowMs) &&
                positive(link.goodputBytesPerSecond) && nonnegative(link.roundTripMs) && link.sampleBytes > 0
            val reason = when {
                !worker.ownerEnabled -> "Owner permission disabled"
                !worker.qualified -> "Backend unqualified"
                !fresh(worker.observedAtMs, nowMs) -> "Worker observation stale or from the future"
                worker.programSha256 != digest || worker.benchmarkElements != request.elements -> "Benchmark workload mismatch"
                worker.numericalContract != NUMERICAL_CONTRACT || !worker.supportedOperations.containsAll(request.program.instructions.map { it.operation }) -> "Incompatible operator or numerical contract"
                !positive(worker.measuredElementsPerSecond) || !positive(worker.copyBytesPerSecond) || !nonnegative(worker.queueMs) -> "Invalid measured compute/copy/queue rate"
                worker.memoryDomainId !in available -> "Unknown or stale physical memory budget"
                worker.overheadBytes > Long.MAX_VALUE - fixedArrays -> "Working memory arithmetic overflow"
                remote && (!validLink(outbound) || !validLink(inbound)) -> "Missing fresh authenticated bidirectional link evidence"
                remote && (source.egressBytesPerSecond?.let(::positive) != true || source.ingressBytesPerSecond?.let(::positive) != true) -> "Missing measured shared source network ceilings"
                else -> null
            }
            if (reason != null) { excluded[worker.deviceId] = reason; null }
            else {
                val base = worker.queueMs + if (remote) outbound!!.roundTripMs + inbound!!.roundTripMs else 0.0
                val perElement = 1000.0 / worker.measuredElementsPerSecond + bytesPerElement * 1000.0 / worker.copyBytesPerSecond +
                    if (remote) (inputCount * 4.0 / outbound!!.goodputBytesPerSecond + 4.0 / inbound!!.goodputBytesPerSecond) * 1000.0 else 0.0
                if (!base.isFinite() || !positive(perElement)) { excluded[worker.deviceId] = "Timing arithmetic overflow"; null }
                else Node(worker, base, perElement, Math.addExact(fixedArrays, worker.overheadBytes), remote)
            }
        }
        if (nodes.isEmpty()) return result(ShardDecision.INSUFFICIENT_EVIDENCE, "No eligible measured workers")
        val candidates = mutableListOf<Candidate>()
        for (mask in 1 until (1 shl nodes.size)) {
            val selected = nodes.filterIndexed { index, _ -> mask and (1 shl index) != 0 }
            if (selected.size > request.maxWorkers || selected.size > request.elements) continue
            candidate(selected, request, available, sourceBytes, bytesPerElement, retainedLimit)?.let(candidates::add)
        }
        val order = compareBy<Candidate> { it.timeMs }.thenBy { it.ranges.size }.thenBy { it.ranges.joinToString(",") { range -> range.deviceId } }
        val single = candidates.filter { it.ranges.size == 1 }.minWithOrNull(order)
        val sharded = candidates.filter { it.ranges.size >= 2 && (request.deadlineMs == null || it.timeMs <= request.deadlineMs) }.minWithOrNull(order)
        if (sharded == null) return result(
            if (single != null && (request.deadlineMs == null || single.timeMs <= request.deadlineMs)) ShardDecision.SINGLE_DEVICE_PREFERRED else ShardDecision.NO_ADMISSIBLE_PLAN,
            "No evaluated multi-device candidate fits the measured memory, deadline and format limits", single = single
        )
        if (request.objective == ShardObjective.LATENCY && single != null && single.timeMs <= sharded.timeMs) {
            return result(ShardDecision.SINGLE_DEVICE_PREFERRED, "Transfer-inclusive estimate favors one device", sharded, single)
        }
        return result(ShardDecision.SHARDED_CANDIDATE,
            if (single == null) "Sharding fits; no single worker fits these bounds. Dispatch and reservation still required"
            else "Sharding candidate fits measured evidence; estimated benefit must be tested on actual hosts", sharded, single)
    }

    private data class Node(val worker: ShardWorker, val baseMs: Double, val msPerElement: Double, val fixedBytes: Long, val remote: Boolean)
    private data class Candidate(val ranges: List<ShardRange>, val memory: List<ShardDomainUse>, val timeMs: Double)

    private fun candidate(nodes: List<Node>, request: ShardingRequest, available: Map<String, Long>, sourceBytes: Long, bytesPerElement: Long, retainedLimit: Int): Candidate? {
        val fixed = mutableMapOf(request.source.memoryDomainId to sourceBytes)
        for (node in nodes) {
            val domain = node.worker.memoryDomainId
            val previous = fixed[domain] ?: 0L
            if (node.fixedBytes > Long.MAX_VALUE - previous) return null
            fixed[domain] = previous + node.fixedBytes
        }
        if (fixed.any { (domain, bytes) -> bytes > available.getValue(domain) }) return null
        val groups = nodes.indices.groupBy { nodes[it].worker.memoryDomainId }
        val capacity = groups.mapValues { (domain, _) ->
            ((available.getValue(domain) - fixed.getValue(domain)) / bytesPerElement).coerceAtMost(request.elements.toLong()).toInt()
        }
        if (groups.any { (domain, members) -> capacity.getValue(domain) < members.size }) return null
        fun limits(time: Double) = nodes.map { node ->
            floor(((time - node.baseMs) / node.msPerElement).coerceIn(0.0, request.elements.toDouble())).toInt().coerceAtMost(retainedLimit)
        }
        fun fits(limits: List<Int>): Boolean = limits.all { it >= 1 } && groups.entries.sumOf { (domain, members) ->
            minOf(capacity.getValue(domain), members.sumOf { limits[it] })
        } >= request.elements
        var high = Math.nextUp(nodes.maxOf { it.baseMs + it.msPerElement * request.elements })
        if (!high.isFinite() || !fits(limits(high))) return null
        var low = 0.0
        repeat(80) {
            val midpoint = low + (high - low) / 2
            if (fits(limits(midpoint))) high = midpoint else low = midpoint
        }
        val limits = limits(high)
        if (!fits(limits)) return null
        val counts = IntArray(nodes.size) { 1 }
        val leftInDomain = groups.mapValues { (domain, members) -> capacity.getValue(domain) - members.size }.toMutableMap()
        var remaining = request.elements - nodes.size
        nodes.indices.sortedWith(compareBy<Int> { nodes[it].msPerElement }.thenBy { nodes[it].worker.deviceId }).forEach { index ->
            val domain = nodes[index].worker.memoryDomainId
            val add = minOf(remaining, limits[index] - 1, leftInDomain.getValue(domain))
            counts[index] += add; remaining -= add; leftInDomain[domain] = leftInDomain.getValue(domain) - add
        }
        if (remaining != 0) return null
        var start = 0
        val ranges = nodes.mapIndexed { index, node ->
            val end = start + counts[index]
            ShardRange(node.worker.deviceId, node.worker.memoryDomainId, start, end,
                Math.addExact(node.fixedBytes, Math.multiplyExact(bytesPerElement, counts[index].toLong())), node.baseMs + counts[index] * node.msPerElement).also { start = end }
        }
        val memory = fixed.map { (domain, bytes) ->
            val elements = groups[domain]?.sumOf { counts[it].toLong() } ?: 0L
            ShardDomainUse(domain, Math.addExact(bytes, Math.multiplyExact(bytesPerElement, elements)), available.getValue(domain))
        }.sortedBy { it.id }
        val remoteElements = nodes.indices.filter { nodes[it].remote }.sumOf { counts[it].toLong() }
        // Conservative shared-bottleneck surcharge, on top of each branch's directed link cost.
        // No overlap or shared-NIC speedup is presumed without a future executor measurement.
        val sharedNetworkMs = if (remoteElements == 0L) 0.0 else remoteElements * 4.0 * 1000.0 *
            (request.program.inputs.size / request.source.egressBytesPerSecond!! + 1.0 / request.source.ingressBytesPerSecond!!)
        val sourceCopyMs = request.elements * (request.program.inputs.size + 1.0) * 4.0 * 1000.0 / request.source.copyBytesPerSecond
        val time = ranges.maxOf { it.estimatedBranchMs } + sharedNetworkMs + sourceCopyMs
        if (!time.isFinite()) return null
        return Candidate(ranges, memory, time)
    }

    private fun positive(value: Double) = value.isFinite() && value > 0.0
    private fun nonnegative(value: Double) = value.isFinite() && value >= 0.0
    private fun fresh(time: Long, now: Long) = time > 0 && time <= now && now - time <= MAX_AGE_MS
}
