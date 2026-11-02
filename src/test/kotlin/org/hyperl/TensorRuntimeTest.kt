// SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
package org.hyperl

import org.hyperl.distributed.*
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.*
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean

class TensorRuntimeTest {
    private val program = Workspace.program(Workspace.EXAMPLE_PROGRAM)
    private fun fixture(adapter: ShardComputeAdapter? = null, action: suspend (ShardWorker, ShardWorker, DistributedSettings) -> Unit) = runBlocking {
        val dir = Files.createTempDirectory("hyperl-tensor-")
        try {
            val ka = dir.resolve("a"); val kb = dir.resolve("b"); TokenFiles.create(ka); TokenFiles.create(kb)
            ShardWorker(WorkerSettings(id = "a", port = 0, tokenFile = ka.toString()), adapter).start().use { a ->
                ShardWorker(WorkerSettings(id = "b", port = 0, tokenFile = kb.toString()), adapter).start().use { b ->
                    action(a, b, DistributedSettings(workers = listOf(WorkerEndpoint("a", a.origin, ka.toString()), WorkerEndpoint("b", b.origin, kb.toString())), shardElements = 3))
                }
            }
        } finally { Files.walk(dir).use { it.sorted(Comparator.reverseOrder()).forEach(Files::deleteIfExists) } }
    }
    @Test fun residentShardsChainAcrossTwoWorkersWithoutReuploadAndReleaseCredits(): Unit = fixture { a, b, settings ->
        val runtime = TensorRuntime.connect(settings)
        try {
            val x = runtime.allocate(FloatArray(12) { it - 4f }, listOf(3, 4))
            val w = runtime.allocate(FloatArray(12) { 2f }, listOf(3, 4))
            val before = runtime.stats()
            val plan = runtime.plan(program, mapOf("x" to x, "w" to w))
            assertEquals(setOf("a", "b"), plan.placements.map { it.workerId }.toSet()); assertEquals(0L, plan.estimatedTransferBytes)
            val y = runtime.execute(program, mapOf("x" to x, "w" to w))
            val relu = HyperLProgram(inputs = setOf("x"), instructions = listOf(HyperLInstruction("y", "relu", listOf("x"))), output = "y")
            val z = runtime.execute(relu, mapOf("x" to y))
            assertEquals(before.uploadedPayloadBytes, runtime.stats().uploadedPayloadBytes)
            assertArrayEquals(FloatArray(12) { maxOf(0f, (it - 4f) * 2) }, runtime.gather(z), 0f)
            assertTrue(runtime.stats().cacheHits > 0); assertTrue(a.capabilities().completedShards > 0 && b.capabilities().completedShards > 0)
            assertEquals(listOf(3, 4), z.shape)
            runtime.release(x)
            try { runtime.gather(x); fail() } catch (_: IllegalStateException) { }
            assertTrue(runtime.jobs().any { it.state == TensorJobState.FAILED })
        } finally { runtime.close() }
        withTimeout(5000) { while (a.capabilities().activeJobs != 0 || b.capabilities().activeJobs != 0) delay(10) }
        assertEquals(a.capabilities().configuredBudgetBytes, a.capabilities().availableAdmissionBytes)
        assertEquals(b.capabilities().configuredBudgetBytes, b.capabilities().availableAdmissionBytes)
    }
    @Test fun namedPoolsMoveMissingChunksOnceAndThenReuseReplicas(): Unit = fixture { _, _, settings ->
        val runtime = TensorRuntime.connect(settings)
        try {
            runtime.defineDevice("left", setOf("a")); runtime.defineDevice("right", setOf("b"))
            val x = runtime.allocate(FloatArray(9) { it - 2f }, pool = "left")
            val w = runtime.allocate(FloatArray(9) { 2f }, pool = "right")
            val firstPlan = runtime.plan(program, mapOf("x" to x, "w" to w), "right")
            assertEquals(36L, firstPlan.estimatedTransferBytes)
            val before = runtime.stats().uploadedPayloadBytes
            val y = runtime.execute(program, mapOf("x" to x, "w" to w), "right")
            assertEquals(36L, runtime.stats().uploadedPayloadBytes - before)
            assertEquals(0L, runtime.plan(program, mapOf("x" to x, "w" to w), "right").estimatedTransferBytes)
            assertArrayEquals(FloatArray(9) { maxOf(0f, (it - 2f) * 2) }, runtime.gather(y), 0f)
            assertThrows(IllegalArgumentException::class.java) { runtime.defineDevice("unknown", setOf("absent")) }
        } finally { runtime.close() }
    }
    @Test fun immutableVersionsCasShapesAndNoBroadcasting(): Unit = fixture { _, _, settings ->
        val runtime = TensorRuntime.connect(settings)
        try {
            val x = runtime.allocate(floatArrayOf(-0f, 1f, 2f))
            val v2 = runtime.replace(x, floatArrayOf(3f, 4f, 5f))
            assertEquals(x.id, v2.id); assertEquals(2L, v2.version)
            assertEquals((-0f).toRawBits(), runtime.gather(x)[0].toRawBits())
            assertArrayEquals(floatArrayOf(3f, 4f, 5f), runtime.gather(v2), 0f)
            try { runtime.replace(x, floatArrayOf(6f, 7f, 8f)); fail() } catch (_: IllegalArgumentException) { }
            val other = runtime.allocate(floatArrayOf(1f, 2f, 3f), listOf(1, 3))
            try { runtime.execute(program, mapOf("x" to x, "w" to other)); fail() } catch (_: IllegalArgumentException) { }
            try { runtime.gather(x.copy(shape = listOf(1, 3))); fail() } catch (_: IllegalArgumentException) { }
            val sum = program.copy(instructions = program.instructions + HyperLInstruction("total", "sum", listOf(program.output)), output = "total")
            try { runtime.execute(sum, mapOf("x" to x, "w" to v2)); fail() } catch (_: IllegalArgumentException) { }
        } finally { runtime.close() }
    }
    @Test fun callerShapeMutationCannotRewriteTheTensorDirectory(): Unit = fixture { _, _, settings ->
        val runtime = TensorRuntime.connect(settings)
        try {
            val shape = mutableListOf(1, 3)
            val external = runtime.allocate(floatArrayOf(1f, 2f, 3f), shape)
            val stable = external.copy(shape = external.shape.toList())
            shape[0] = 3
            (external.shape as MutableList<Int>)[0] = 2
            assertArrayEquals(floatArrayOf(1f, 2f, 3f), runtime.gather(stable), 0f)
            try { runtime.gather(external); fail() } catch (_: IllegalArgumentException) { }
        } finally { runtime.close() }
    }
    @Test fun noCacheStillReadsAndVerifiesRemoteChunks(): Unit = fixture { _, _, settings ->
        val runtime = TensorRuntime.connect(settings, cacheBudgetBytes = 0)
        try {
            val x = runtime.allocate(floatArrayOf(-1f, 2f, 3f)); val w = runtime.allocate(floatArrayOf(2f, 3f, 4f))
            val y = runtime.execute(program, mapOf("x" to x, "w" to w))
            assertArrayEquals(floatArrayOf(0f, 6f, 12f), runtime.gather(y), 0f)
            assertEquals(0L, runtime.stats().cacheHits)
            assertTrue(runtime.stats().downloadedPayloadBytes >= 48)
        } finally { runtime.close() }
    }
    @Test fun falseWorkerOutputNeverPublishesAHandle(): Unit {
        val adapter = object: ShardComputeAdapter {
            override val target = HyperLTarget.CPU_REFERENCE
            override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>) = FloatArray(inputs.values.first().size) { 999f }
        }
        fixture(adapter) { _, _, settings ->
            val runtime = TensorRuntime.connect(settings)
            try {
                val x = runtime.allocate(floatArrayOf(1f, 2f, 3f)); val w = runtime.allocate(floatArrayOf(2f, 3f, 4f))
                try { runtime.execute(program, mapOf("x" to x, "w" to w)); fail() } catch (_: IllegalArgumentException) { }
                assertNull(runtime.jobs().single { it.kind == "execute" }.output)
                assertEquals(TensorJobState.FAILED, runtime.jobs().single { it.kind == "execute" }.state)
            } finally { runtime.close() }
        }
    }
    @Test fun cancellationStopsRemoteComputeAndRecordsDependencies(): Unit {
        val entered = AtomicBoolean(); val cancelled = AtomicBoolean()
        val adapter = object: ShardComputeAdapter {
            override val target = HyperLTarget.CPU_REFERENCE
            override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray {
                entered.set(true); try { delay(60000); return FloatArray(3) } finally { cancelled.set(true) }
            }
        }
        fixture(adapter) { a, b, settings ->
            val runtime = TensorRuntime.connect(settings)
            try {
                val x = runtime.allocate(floatArrayOf(1f, 2f, 3f)); val w = runtime.allocate(floatArrayOf(2f, 3f, 4f))
                coroutineScope {
                    val task = async { runtime.execute(program, mapOf("x" to x, "w" to w)) }
                    withTimeout(5000) { while (!entered.get()) delay(10) }
                    val job = runtime.jobs().single { it.kind == "execute" }
                    assertEquals(2, job.dependencies.size); assertTrue(runtime.cancel(job.id))
                    try { task.await(); fail() } catch (_: CancellationException) { }
                }
                withTimeout(5000) { while (!cancelled.get() || a.capabilities().activeJobs != 0 || b.capabilities().activeJobs != 0) delay(10) }
                assertEquals(TensorJobState.CANCELLED, runtime.jobs().single { it.kind == "execute" }.state)
            } finally { runtime.close() }
        }
    }
    @Test fun directoryInstanceFenceAndWorkerLossCannotSilentlyReallocate(): Unit = fixture { a, _, settings ->
        try { TensorRuntime.connect(settings, expectedInstances = mapOf("a" to UUID.randomUUID().toString())); fail() } catch (_: IllegalArgumentException) { }
        val runtime = TensorRuntime.connect(settings, cacheBudgetBytes = 0, expectedInstances = mapOf("a" to a.capabilities().instanceId))
        try {
            runtime.defineDevice("left", setOf("a"))
            val x = runtime.allocate(floatArrayOf(1f, 2f, 3f), pool = "left")
            a.close()
            try { runtime.gather(x); fail() } catch (_: Exception) { }
            assertEquals(TensorJobState.FAILED, runtime.jobs().single { it.kind == "gather" }.state)
        } finally { runtime.close() }
    }
    @Test fun workerStoreExpiryIntegritySessionsAndCredits(): Unit {
        val credits = MemoryCredits(4L * 1024 * 1024); var clock = 0L
        val store = TensorStore(credits) { clock }
        val data = floatArrayOf(1f, -0f, 3f)
        val chunk = TensorChunk(TensorSlice(UUID.randomUUID().toString(), 1, 0, 3), TensorProtocol.digest(data))
        val session = UUID.randomUUID().toString()
        val before = credits.available()
        store.put(session, listOf(chunk), listOf(data)); data[0] = 999f
        assertEquals(1f, store.get(session, listOf(chunk)).single()[0], 0f)
        assertThrows(IllegalStateException::class.java) { store.get(UUID.randomUUID().toString(), listOf(chunk)) }
        assertThrows(IllegalArgumentException::class.java) { store.put(session, listOf(chunk.copy(sha256 = "0".repeat(64))), listOf(floatArrayOf(1f, 2f, 3f))) }
        assertTrue(credits.available() < before)
        store.clear(session); assertEquals(before, credits.available())
        store.put(session, listOf(chunk), listOf(floatArrayOf(1f, -0f, 3f)))
        clock += (TensorProtocol.LEASE_MS + 1) * 1_000_000
        store.expire(); assertEquals(before, credits.available())
        assertThrows(IllegalStateException::class.java) { store.get(session, listOf(chunk)) }
    }
    @Test fun protocolRejectsTamperTrailingBytesInvalidShapesAndUnsupportedActions(): Unit {
        val data = floatArrayOf(1f, 2f, 3f)
        val chunk = TensorChunk(TensorSlice(UUID.randomUUID().toString(), 1, 0, 3), TensorProtocol.digest(data))
        val meta = TensorRequest(jobId = UUID.randomUUID().toString(), sessionId = UUID.randomUUID().toString(), workerInstanceId = UUID.randomUUID().toString(), action = "put", chunks = listOf(chunk))
        val bytes = TensorProtocol.request(meta, listOf(data))
        assertArrayEquals(data, TensorProtocol.readRequest(bytes).second.single(), 0f)
        bytes[bytes.lastIndex] = (bytes.last().toInt() xor 1).toByte()
        assertThrows(IllegalArgumentException::class.java) { TensorProtocol.readRequest(bytes) }
        assertThrows(IllegalArgumentException::class.java) { TensorProtocol.readRequest(bytes + byteArrayOf(0)) }
        assertThrows(IllegalStateException::class.java) { TensorProtocol.validate(meta.copy(action = "matmul")) }
        assertThrows(IllegalArgumentException::class.java) { TensorHandle(chunk.slice.tensorId, 1, listOf(Int.MAX_VALUE, Int.MAX_VALUE)).elements() }
    }
}
