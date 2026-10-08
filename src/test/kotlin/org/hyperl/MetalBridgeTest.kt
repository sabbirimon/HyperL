package org.hyperl
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.util.Comparator

class MetalBridgeTest {
    private fun installed():MetalBridge {
        val binary=Path.of("build/native/hyperl-metal").toAbsolutePath()
        assumeTrue("Optional macOS Metal bridge not built",Files.isExecutable(binary));return MetalBridge(binary)
    }
    @Test fun missingBridgeNeverDownloadsOrFallsBack(){assertThrows(IllegalArgumentException::class.java){MetalBridge(Path.of("missing"))}}
    @Test fun unconfirmedSubmissionBlocksFurtherWork()=runBlocking {
        assumeTrue("POSIX subprocess failure fixture",!System.getProperty("os.name").startsWith("Windows"))
        val folder=Files.createTempDirectory("hyperl-metal-failure-test-")
        try {
            val fixture=folder.resolve("failure-host")
            // Fault injection only: no GPU API and no fabricated successful result.
            Files.writeString(fixture,"#!/bin/sh\nprintf pending > submitted.marker\nexit 4\n")
            assertTrue(fixture.toFile().setExecutable(true,true))
            val bridge=MetalBridge(fixture)
            try{bridge.executeVerified("TEST FAILURE ONLY",Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.inputs(Workspace.EXAMPLE_INPUTS));fail("Unconfirmed completion must reject")}
            catch(e:IllegalStateException){assertTrue(e.message.orEmpty().contains("completion unconfirmed"))}
            assertTrue(MetalBridge.cleanupUncertain.get())
            assertThrows(IllegalArgumentException::class.java){MetalBridge(fixture)}
            Unit
        }finally{
            MetalBridge.cleanupUncertain.set(false) // Isolate this fault fixture from unrelated tests.
            Files.walk(folder).use{it.sorted(Comparator.reverseOrder()).forEach{path->Files.deleteIfExists(path)}}
        }
    }
    @Test fun actualProbeAndUnavailableGpuRejection()=runBlocking {
        val bridge=installed();val probe=Workspace.json.parseToJsonElement(bridge.probe()).jsonObject
        assertEquals("hyperl-metal-probe/1",probe.getValue("format").jsonPrimitive.content)
        assertFalse(probe.getValue("inferenceQualified").jsonPrimitive.boolean)
        if(!probe.getValue("available").jsonPrimitive.boolean){
            try{bridge.executeVerified("AMD Radeon Pro 560X",Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.inputs(Workspace.EXAMPLE_INPUTS));fail("Must reject unavailable GPU")}
            catch(e:IllegalArgumentException){assertTrue(e.message.orEmpty().contains("no CPU or other-GPU fallback"))}
        }
    }
    @Test fun overflowAndUnsupportedReductionRejectBeforeDispatch()=runBlocking {
        val bridge=installed();val p=Workspace.program(Workspace.EXAMPLE_PROGRAM)
        try{bridge.executeVerified("AMD Radeon Pro 560X",p,mapOf("x" to floatArrayOf(Float.MAX_VALUE),"w" to floatArrayOf(2f)));fail("Overflow must be visible before ReLU")}
        catch(e:IllegalArgumentException){assertTrue(e.message.orEmpty().contains("nonfinite"))}
        try{bridge.executeVerified("AMD Radeon Pro 560X",p.copy(instructions=p.instructions+HyperLInstruction("total","sum",listOf("positive")),output="total"),Workspace.inputs(Workspace.EXAMPLE_INPUTS));fail("Reduction must reject")}
        catch(e:IllegalArgumentException){assertTrue(e.message.orEmpty().contains("reductions"))}
    }
    @Test fun actualGpuOnlyWhenVisible()=runBlocking {
        val bridge=installed();val probe=Workspace.json.parseToJsonElement(bridge.probe()).jsonObject
        assumeTrue("No Metal GPU visible; actual dispatch unrun",probe.getValue("available").jsonPrimitive.boolean)
        val device=probe.getValue("devices").jsonArray.first().jsonObject.getValue("name").jsonPrimitive.content
        val result=bridge.executeVerified(device,Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.inputs(Workspace.EXAMPLE_INPUTS))
        assertEquals(device,result.device);assertTrue(result.cpuVerified);assertTrue(result.metrics.gpuCompletionConfirmed)
        assertArrayEquals(floatArrayOf(0f,6f,12f),result.result,0.00001f)
    }
}
