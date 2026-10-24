package org.hyperl

import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class HyperLShardingPlannerTest {
    private val now = 50_000L
    private val program = Workspace.program(Workspace.EXAMPLE_PROGRAM)
    private fun fixture(): ShardingRequest {
        val workers = listOf("arm", "x86").map { name -> ShardWorker(
            name, name, HyperLTarget.CPU_REFERENCE, "hyperl-cpu/1", HyperLShardingPlanner.NUMERICAL_CONTRACT,
            setOf("multiply", "relu"), HyperLShardingPlanner.programSha256(program), 1024, 1000.0,
            100_000_000.0, 0.0, 0L, now, qualified = true, ownerEnabled = true
        ) }
        return ShardingRequest(program = program, elements = 1024,
            source = ShardSource("source", "source", 100_000_000.0, 100_000_000.0, 100_000_000.0, now),
            workers = workers,
            memoryDomains = listOf("source", "arm", "x86").map { ShardMemoryDomain(it, 8L * 1024 * 1024, observedAtMs = now) },
            links = workers.flatMap { listOf(
                ShardLink("source", it.deviceId, 100_000_000.0, 1.0, 65536, now, true),
                ShardLink(it.deviceId, "source", 100_000_000.0, 1.0, 65536, now, true)
            ) }
        )
    }

    @Test fun exactPartitionsReassembleTheRealReferenceOutput() = runBlocking {
        val request = fixture()
        val plan = HyperLShardingPlanner.plan(request, now)
        assertEquals(ShardDecision.SHARDED_CANDIDATE, plan.decision)
        assertFalse(plan.executionActivated); assertFalse(plan.reservationGuaranteed); assertTrue(plan.timingsAreEstimates)
        assertEquals(2, plan.candidateShards.size)
        assertTrue(plan.estimatedShardedMs!! < plan.estimatedSingleDeviceMs!!)
        val inputs = mapOf("x" to FloatArray(1024) { (it - 512).toFloat() }, "w" to FloatArray(1024) { (it % 7).toFloat() })
        val actual = FloatArray(1024)
        var offset = 0
        for (shard in plan.candidateShards) {
            assertEquals(offset, shard.startInclusive)
            assertTrue(shard.endExclusive > shard.startInclusive)
            val result = HyperLCpuBackend().execute(program, inputs.mapValues { it.value.copyOfRange(shard.startInclusive, shard.endExclusive) })
            result.copyInto(actual, shard.startInclusive)
            offset = shard.endExclusive
        }
        assertEquals(1024, offset)
        assertArrayEquals(HyperLCpuBackend().execute(program, inputs), actual, 0f)
        // This is one-process correctness evidence, not physical ARM/x86 execution.
        assertTrue(Workspace.json.encodeToString(plan).contains("\"executionActivated\": false"))
    }

    @Test fun orderingIsDeterministicAndProgramDigestBindsDependencies() {
        val request = fixture()
        val original = HyperLShardingPlanner.plan(request, now)
        val reordered = request.copy(workers = request.workers.reversed(), memoryDomains = request.memoryDomains.reversed(), links = request.links.reversed(),
            program = program.copy(inputs = program.inputs.reversed().toSet()))
        assertEquals(original, HyperLShardingPlanner.plan(reordered, now))
        val changed = program.copy(instructions = listOf(program.instructions.first().copy(operation = "add")) + program.instructions.drop(1))
        assertNotEquals(HyperLShardingPlanner.programSha256(program), HyperLShardingPlanner.programSha256(changed))
    }

    @Test fun slowerLinksFavorOneDeviceAndLoadedWorkersReceiveLessWork() {
        val request = fixture()
        val slow = request.copy(source = request.source.copy(deviceId = "arm", memoryDomainId = "arm"),
            links = listOf(ShardLink("arm", "x86", 0.01, 1.0, 65536, now, true), ShardLink("x86", "arm", 0.01, 1.0, 65536, now, true)))
        assertEquals(ShardDecision.SINGLE_DEVICE_PREFERRED, HyperLShardingPlanner.plan(slow, now).decision)
        val unloaded = HyperLShardingPlanner.plan(request, now)
        val loaded = HyperLShardingPlanner.plan(request.copy(objective = ShardObjective.CAPACITY,
            workers = request.workers.map { if (it.deviceId == "x86") it.copy(queueMs = 500.0) else it }), now)
        fun count(plan: ShardingPlan) = plan.candidateShards.single { it.deviceId == "x86" }.let { it.endExclusive - it.startInclusive }
        assertTrue(count(loaded) < count(unloaded))
    }

    @Test fun capacityAdmissionAndSharedMemoryNeverDoubleCountRam() {
        val request = fixture()
        val bounded = request.copy(memoryDomains = request.memoryDomains.map { if (it.id == "source") it else it.copy(availableBytes = 88_000) })
        val plan = HyperLShardingPlanner.plan(bounded, now)
        assertEquals(ShardDecision.SHARDED_CANDIDATE, plan.decision)
        assertTrue(plan.capacityRequired); assertNull(plan.estimatedSingleDeviceMs)
        assertTrue(plan.memory.all { it.requiredBytes <= it.availableAfterReservations })
        val shared = request.copy(workers = request.workers.map { it.copy(memoryDomainId = "shared") },
            memoryDomains = listOf(request.memoryDomains.first(), ShardMemoryDomain("shared", 170_000, observedAtMs = now)))
        val sharedPlan = HyperLShardingPlanner.plan(shared, now)
        assertEquals(ShardDecision.SINGLE_DEVICE_PREFERRED, sharedPlan.decision)
        assertTrue(sharedPlan.candidateShards.isEmpty())
        val sourceShared = bounded.copy(source = request.source.copy(memoryDomainId = "arm"))
        assertEquals(ShardDecision.NO_ADMISSIBLE_PLAN, HyperLShardingPlanner.plan(sourceShared, now).decision)
        val deadlineOnly = HyperLShardingPlanner.plan(request.copy(deadlineMs = 700.0), now)
        assertEquals(ShardDecision.SHARDED_CANDIDATE, deadlineOnly.decision)
        assertFalse(deadlineOnly.capacityRequired)
        assertTrue(deadlineOnly.estimatedSingleDeviceMs!! > 700.0)
    }

    @Test fun staleUnapprovedIncompatibleAndUnknownEvidenceIsExcluded() {
        val request = fixture()
        val invalid = listOf<(ShardWorker) -> ShardWorker>(
            { it.copy(ownerEnabled = false) }, { it.copy(qualified = false) },
            { it.copy(observedAtMs = now - 30_001) }, { it.copy(observedAtMs = now + 1) },
            { it.copy(programSha256 = "a".repeat(64)) }, { it.copy(benchmarkElements = 32) },
            { it.copy(numericalContract = "fast-math") }, { it.copy(supportedOperations = setOf("relu")) },
            { it.copy(measuredElementsPerSecond = Double.NaN) }, { it.copy(copyBytesPerSecond = 0.0) },
            { it.copy(queueMs = Double.POSITIVE_INFINITY) }, { it.copy(overheadBytes = Long.MAX_VALUE) }
        )
        for (change in invalid) {
            val result = HyperLShardingPlanner.plan(request.copy(workers = request.workers.map(change)), now)
            assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, result.decision)
            assertEquals(2, result.excludedWorkers.size)
        }
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(
            memoryDomains = request.memoryDomains.map { it.copy(availableBytes = null) }), now).decision)
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(
            source = request.source.copy(observedAtMs = now - 30_001)), now).decision)
    }

    @Test fun noMissingOneWayUnauthenticatedOrInvalidLinksBecomeZeroCost() {
        val request = fixture()
        val invalid = listOf(emptyList(), request.links.filter { it.from == "source" },
            request.links.map { it.copy(authenticated = false) }, request.links.map { it.copy(observedAtMs = now + 1) },
            request.links.map { it.copy(goodputBytesPerSecond = Double.POSITIVE_INFINITY) },
            request.links.map { it.copy(roundTripMs = -1.0) }, request.links.map { it.copy(sampleBytes = 0) })
        invalid.forEach { assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(links = it), now).decision) }
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(
            source = request.source.copy(ingressBytesPerSecond = null)), now).decision)
    }

    @Test fun boundariesDuplicatesUnknownVersionAndReductionAreRejected() {
        val request = fixture()
        val invalid = listOf(request.copy(elements = 0), request.copy(elements = Int.MAX_VALUE),
            request.copy(format = "hyperl-sharding/999"), request.copy(maxWorkers = 9),
            request.copy(workers = listOf(request.workers[0], request.workers[0])),
            request.copy(memoryDomains = request.memoryDomains + request.memoryDomains[0]),
            request.copy(links = request.links + request.links[0]),
            request.copy(program = HyperLProgram(inputs = setOf("x"), instructions = listOf(HyperLInstruction("sum", "sum", listOf("x"))), output = "sum")))
        invalid.forEach { assertThrows(IllegalArgumentException::class.java) { HyperLShardingPlanner.plan(it, now) } }
        val deadline = HyperLShardingPlanner.plan(request.copy(deadlineMs = 0.1), now)
        assertEquals(ShardDecision.NO_ADMISSIBLE_PLAN, deadline.decision)
    }

    @Test fun futureSourceAndReservedBudgetsCannotAuthorizeWork() {
        val request = fixture()
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(
            source = request.source.copy(observedAtMs = now + 1)), now).decision)
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(request.copy(
            memoryDomains = request.memoryDomains.map { it.copy(reservedBytes = Long.MAX_VALUE) }), now).decision)
        val minRate = request.copy(workers = request.workers.map { it.copy(measuredElementsPerSecond = Double.MIN_VALUE) })
        assertEquals(ShardDecision.INSUFFICIENT_EVIDENCE, HyperLShardingPlanner.plan(minRate, now).decision)
    }
}
