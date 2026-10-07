package org.hyperl
import org.junit.Assert.*
import org.junit.Test

class WorkspaceTest {
    @Test fun jsonAndCpuExampleExecute(){assertEquals("[\n    0.0,\n    6.0,\n    12.0\n]",cpu(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS))}
    @Test fun strictInputsRejectStringsNonfiniteAndWrongKeys(){
        for(text in listOf("{\"x\":[\"1\"]}","{\"x\":[1e100]}"))assertThrows(IllegalArgumentException::class.java){Workspace.inputs(text)}
        assertThrows(IllegalArgumentException::class.java){cpu(Workspace.EXAMPLE_PROGRAM,"{\"x\":[1]}")}
    }
    @Test fun mobileEmissionUsesExplicitBindingsAndNeverClaimsQualification(){
        val p=Workspace.program(Workspace.EXAMPLE_PROGRAM)
        val metal=PortableEmitter.emit(p,HyperLTarget.METAL);assertTrue(metal.source.contains("[[buffer(0)]]"));assertTrue(metal.source.contains("thread_position_in_grid"));assertFalse(metal.deviceQualified)
        val vulkan=PortableEmitter.emit(p,HyperLTarget.VULKAN_SPIRV);assertTrue(vulkan.source.contains("#version 450"));assertTrue(vulkan.source.contains("push_constant"));assertFalse(vulkan.deviceQualified)
        for(t in listOf(HyperLTarget.METAL,HyperLTarget.VULKAN_SPIRV))assertThrows(IllegalArgumentException::class.java){PortableEmitter.emit(p.copy(instructions=p.instructions+HyperLInstruction("total","sum",listOf("positive")),output="total"),t)}
    }
    @Test fun telecomEvidenceDoesNotActivateOrCertifyAnything(){
        val p=TelecomProfile(generation=TelecomGeneration.NR_5G,standardReferences=listOf("3GPP TS 23.501"),ownerApproved=true,isolatedLabReady=true,adapterInstalled=true,conformanceEvidenceSha256="a".repeat(64))
        assertTrue(TelecomPlanner.plan(p).contains("Execution disabled, certified=false"))
        assertThrows(IllegalArgumentException::class.java){TelecomPlanner.plan(p.copy(format="unknown"))}
    }
}
