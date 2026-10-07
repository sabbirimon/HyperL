package org.hyperl
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.file.Path
import java.nio.file.Files

class OpenClBridgeTest {
    @Test fun missingBridgeDoesNotInstallOrFallback(){assertThrows(IllegalArgumentException::class.java){OpenClBridge(Path.of("missing"))}}
    @Test fun actualInstalledBridgeProbeIsNotInferenceQualification()=runBlocking {
        val binary=Path.of("native/hyperl-opencl").toAbsolutePath()
        assumeTrue("Optional native OpenCL bridge not built",Files.isExecutable(binary))
        val probe=Workspace.json.parseToJsonElement(OpenClBridge(binary).probe()).jsonObject
        assertFalse(probe.getValue("inferenceQualified").jsonPrimitive.boolean)
        if(!probe.getValue("available").jsonPrimitive.boolean){
            try{OpenClBridge(binary).executeVerified(0,Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.inputs(Workspace.EXAMPLE_INPUTS));fail("Unavailable GPU must not fall back")}
            catch(_:IllegalArgumentException){}
        }
    }
    @Test fun actualGpuKernelOnlyWhenGpuIsPresent()=runBlocking {
        val binary=Path.of("native/hyperl-opencl").toAbsolutePath()
        assumeTrue("Optional native bridge not built",Files.isExecutable(binary))
        val bridge=OpenClBridge(binary);val probe=Workspace.json.parseToJsonElement(bridge.probe()).jsonObject
        assumeTrue("No OpenCL GPU visible; no hardware kernel proof",probe.getValue("available").jsonPrimitive.boolean)
        assertArrayEquals(floatArrayOf(0f,6f,12f),bridge.executeVerified(0,Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.inputs(Workspace.EXAMPLE_INPUTS)),0.00001f)
    }
}
