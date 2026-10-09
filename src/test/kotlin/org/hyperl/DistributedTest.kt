package org.hyperl

import org.hyperl.distributed.*
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.*
import java.net.URI
import java.net.http.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class DistributedTest {
    private val program = Workspace.program(Workspace.EXAMPLE_PROGRAM)
    private fun fixture(id: String, adapter: ShardComputeAdapter? = null, action: suspend (ShardWorker, Path) -> Unit) = runBlocking {
        val dir = Files.createTempDirectory("hyperl-shard-test-"); val token = dir.resolve("token")
        TokenFiles.create(token)
        try { ShardWorker(WorkerSettings(id = id, port = 0, tokenFile = token.toString(), scratchDirectory = dir.toString()), adapter).start().use { action(it, token) } }
        finally { Files.walk(dir).use { it.sorted(Comparator.reverseOrder()).forEach(Files::deleteIfExists) } }
    }
    private fun settings(vararg nodes: Pair<ShardWorker, Path>, shard: Int? = 3) = DistributedSettings(workers = nodes.map {
        WorkerEndpoint(it.first.capabilities().id, it.first.origin, it.second.toString()) }, maxParallelism = nodes.size, shardElements = shard)
    @Test fun twoRealHttpWorkersComputeParallelShardsAndGatherExactRanges(): Unit = fixture("a") { a, keyA ->
        fixture("b") { b, keyB ->
            val x = FloatArray(31) { it - 12f }; val weights = FloatArray(31) { 2f }
            val input = mapOf("x" to x, "w" to weights)
            val (_, result) = DistributedExecutor.run(program, input, settings(a to keyA, b to keyB))
            assertArrayEquals(HyperLCpuBackend().execute(program, input), result!!.result, 0f)
            assertEquals(setOf("a", "b"), result.shards.map { it.workerId }.toSet())
            assertEquals((0 until 31).toList(), result.shards.flatMap { (it.start until it.end).toList() })
            assertTrue(a.capabilities().completedShards > 0 && b.capabilities().completedShards > 0)
            assertFalse(result.plan.physicalMemoryPooled); assertFalse(result.plan.storageSpillEnabled)
            assertEquals(-12f, x[0], 0f)
            val domain = a.capabilities().memoryDomains
            assertEquals("system", domain.single { it.kind == "JVM_HEAP" }.sharedWith)
            assertNull(domain.single { it.kind == "FILESYSTEM_STORAGE" }.admissionBytes)
        }
    }
    @Test fun parallelDispatchIsRealAndBounded(): Unit {
        val entered = CountDownLatch(2); val active = AtomicInteger(); val peak = AtomicInteger()
        val adapter = object: ShardComputeAdapter {
            override val target = HyperLTarget.CPU_REFERENCE
            override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray {
                val count = active.incrementAndGet(); peak.updateAndGet { maxOf(it, count) }; entered.countDown()
                try { withContext(Dispatchers.IO) { check(entered.await(5, TimeUnit.SECONDS)) }; return HyperLCpuBackend().execute(program, inputs) }
                finally { active.decrementAndGet() }
            }
        }
        fixture("a", adapter) { a, ka -> fixture("b", adapter) { b, kb ->
            DistributedExecutor.run(program, mapOf("x" to FloatArray(8) { it.toFloat() }, "w" to FloatArray(8) { 1f }), settings(a to ka, b to kb))
            assertEquals(2, peak.get()); assertEquals(0, active.get())
        } }
    }
    @Test fun memoryShardsPermitGraphThatExceedsSingleReferenceRetention(): Unit = fixture("a") { a, key ->
        val chain = HyperLProgram(inputs = setOf("x"), instructions = (0 until 8).map { i -> HyperLInstruction("v$i", "relu", listOf(if(i == 0) "x" else "v${i-1}")) }, output = "v7")
        val input = mapOf("x" to FloatArray(262144) { it - 12f })
        try { HyperLCpuBackend().execute(chain, input); fail("Full graph exceeds retained-vector limit") } catch (_: IllegalArgumentException) {}
        val (_, result) = DistributedExecutor.run(chain, input, settings(a to key, shard = 8192))
        assertEquals(32, result!!.shards.size); assertEquals(262144, result.result.size)
        assertEquals(0f, result.result[0], 0f); assertEquals(262131f, result.result.last(), 0f)
        assertTrue(result.plan.shards.all { it.estimatedAdmissionBytes <= a.capabilities().configuredBudgetBytes })
    }
    @Test fun terminalReductionKeepsOrderedF32RatherThanPartialSums(): Unit = fixture("a") { a, key ->
        val p = HyperLProgram(inputs = setOf("x", "w"), instructions = listOf(HyperLInstruction("y", "multiply", listOf("x", "w")), HyperLInstruction("total", "sum", listOf("y"))), output = "total")
        val input = mapOf("x" to floatArrayOf(16777216f, 1f, -16777216f, 1f), "w" to FloatArray(4) { 1f })
        val (_, result) = DistributedExecutor.run(p, input, settings(a to key, shard = 1))
        assertTrue(result!!.plan.orderedReductionAtCoordinator)
        assertArrayEquals(HyperLCpuBackend().execute(p, input), result.result, 0f); assertEquals(1f, result.result.single(), 0f)
    }
    @Test fun invalidShapeAndInternalReductionAreRejectedBeforeNetworking(): Unit = runBlocking {
        val unreachable = DistributedSettings(workers = listOf(WorkerEndpoint("no", "http://127.0.0.1:1", "/missing/key")))
        try { DistributedExecutor.run(program, mapOf("x" to floatArrayOf(1f), "w" to floatArrayOf(1f, 2f)), unreachable); fail() } catch (e: IllegalArgumentException) { assertTrue(e.message!!.contains("equal lengths")) }
        val p = HyperLProgram(inputs = setOf("x"), instructions = listOf(HyperLInstruction("total", "sum", listOf("x")), HyperLInstruction("r", "relu", listOf("total"))), output = "r")
        try { DistributedExecutor.run(p, mapOf("x" to floatArrayOf(1f)), unreachable); fail() } catch (e: IllegalArgumentException) { assertTrue(e.message!!.contains("terminal")) }
        assertThrows(IllegalArgumentException::class.java) { ShardProtocol.origin("http://localhost:8091") }
        assertThrows(IllegalArgumentException::class.java) { ShardProtocol.origin("http://192.168.1.2:8091") }
        assertThrows(IllegalArgumentException::class.java) { WorkerSettings(id="no", bind="0.0.0.0", tokenFile="/key").validate() }
        assertThrows(IllegalArgumentException::class.java) { WorkerSettings(id="no", tokenFile="/key", backend=HyperLTarget.NPU_STABLEHLO).validate() }
    }
    @Test fun orderedReductionCannotSilentlyMixApproximateGpuValues(): Unit = fixture("a") { worker, key ->
        val p = program.copy(instructions=program.instructions + HyperLInstruction("total","sum",listOf(program.output)), output="total")
        val gpu = worker.capabilities().copy(backend=HyperLTarget.METAL,numericContract="f32-checked-1e-6/1")
        assertThrows(IllegalArgumentException::class.java) { DistributedExecutor.plan(p,Workspace.inputs(Workspace.EXAMPLE_INPUTS),settings(worker to key),listOf(gpu)) }
        val plan = DistributedExecutor.plan(program,Workspace.inputs(Workspace.EXAMPLE_INPUTS),settings(worker to key).copy(maxParallelism=16),listOf(worker.capabilities()))
        assertEquals(1,plan.maxParallelism)
    }
    @Test fun incorrectCredentialsCannotObserveOrDispatch(): Unit = fixture("a") { worker, key ->
        val other = key.resolveSibling("other"); TokenFiles.create(other)
        try { DistributedExecutor.run(program, Workspace.inputs(Workspace.EXAMPLE_INPUTS), settings(worker to other)); fail() }
        catch (_: IllegalArgumentException) {}
        assertEquals(0L, worker.capabilities().completedShards)
    }
    @Test fun alteredFramesReplaysAndBusyBudgetsDoNotProduceAcceptedResults(): Unit = fixture("a") { worker, key ->
        val client = HttpClient.newHttpClient()
        fun post(bytes: ByteArray, id: String, digest: String): Int = client.send(HttpRequest.newBuilder(URI(worker.origin+"/hyperl/execute"))
            .header("Authorization", "Bearer "+ShardProtocol.token(key)).header("Content-Type", ShardProtocol.CONTENT_TYPE)
            .header("X-HyperL-Job", id).header("X-HyperL-SHA256", digest).POST(HttpRequest.BodyPublishers.ofByteArray(bytes)).build(), HttpResponse.BodyHandlers.discarding()).statusCode()
        val id = java.util.UUID.randomUUID().toString()
        val frame = ShardProtocol.encodeRequest(ShardRequest(jobId=id, start=0, end=3, program=program, workerInstanceId=worker.capabilities().instanceId), Workspace.inputs(Workspace.EXAMPLE_INPUTS))
        assertEquals(200, post(frame,id,ShardProtocol.digest(frame)))
        assertEquals(409, post(frame,id,ShardProtocol.digest(frame)))
        val id2 = java.util.UUID.randomUUID().toString()
        val changed = ShardProtocol.encodeRequest(ShardRequest(jobId=id2,start=0,end=3,program=program,workerInstanceId=worker.capabilities().instanceId), Workspace.inputs(Workspace.EXAMPLE_INPUTS))
        assertEquals(422, post(changed,id2,"0".repeat(64)))
        val staleId = java.util.UUID.randomUUID().toString()
        val stale = ShardProtocol.encodeRequest(ShardRequest(jobId=staleId,start=0,end=3,program=program,workerInstanceId=java.util.UUID.randomUUID().toString()), Workspace.inputs(Workspace.EXAMPLE_INPUTS))
        assertEquals(422, post(stale,staleId,ShardProtocol.digest(stale)))
        assertEquals(0, worker.capabilities().activeJobs)
    }
    @Test fun incorrectWorkerOutputAbortsGatherWithoutRetry(): Unit {
        val calls = AtomicInteger()
        val adapter = object: ShardComputeAdapter { override val target = HyperLTarget.CPU_REFERENCE
            override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray { calls.incrementAndGet(); return FloatArray(inputs.values.first().size) { 19f } } }
        fixture("a", adapter) { worker, key ->
            try { DistributedExecutor.run(program, Workspace.inputs(Workspace.EXAMPLE_INPUTS), settings(worker to key)); fail() }
            catch (e: IllegalArgumentException) { assertTrue(e.message!!.contains("numerical contract")) }
            assertEquals(1, calls.get())
        }
    }
    @Test fun timedOutJobCancelsWorkerAndReleasesCredits(): Unit {
        val cancelled = AtomicBoolean(); val calls = AtomicInteger()
        val adapter = object: ShardComputeAdapter { override val target = HyperLTarget.CPU_REFERENCE
            override suspend fun execute(program: HyperLProgram, inputs: Map<String, FloatArray>): FloatArray { calls.incrementAndGet(); try { delay(60000); return FloatArray(3) } finally { cancelled.set(true) } } }
        fixture("a", adapter) { worker, key ->
            val config = settings(worker to key).copy(requestTimeoutMs=1000, jobTimeoutMs=3000)
            try { DistributedExecutor.run(program, Workspace.inputs(Workspace.EXAMPLE_INPUTS), config); fail() } catch (_: Exception) {}
            withTimeout(5000) { while (worker.capabilities().activeJobs != 0 || !cancelled.get()) delay(20) }
            assertEquals(1, calls.get()); assertTrue(cancelled.get())
            assertEquals(worker.capabilities().configuredBudgetBytes, worker.capabilities().availableAdmissionBytes)
        }
    }
    @Test fun planningBoundsIndependentDomainsAndLowMemory(): Unit = fixture("a") { worker, key ->
        val cfg = settings(worker to key)
        val cap = worker.capabilities()
        assertThrows(IllegalArgumentException::class.java) { DistributedExecutor.plan(program, Workspace.inputs(Workspace.EXAMPLE_INPUTS), cfg,
            listOf(cap.copy(availableAdmissionBytes=1))) }
        assertThrows(IllegalArgumentException::class.java) { DistributedExecutor.plan(program, Workspace.inputs(Workspace.EXAMPLE_INPUTS), cfg,
            listOf(cap), MemorySnapshot(64L*1024*1024,60L*1024*1024)) }
        assertThrows(FileAlreadyExistsException::class.java) { TokenFiles.create(key) }
        assertThrows(IllegalArgumentException::class.java) { ShardProtocol.decodeRequest(byteArrayOf(1,2,3)) }
    }
}
