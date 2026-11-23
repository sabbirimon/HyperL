package org.hyperl
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
class MemoryTest {
    private val program=Workspace.program(Workspace.EXAMPLE_PROGRAM)
    private val inputs=Workspace.inputs(Workspace.EXAMPLE_INPUTS)
    private val roomy=MemorySnapshot(256L*1024*1024,0,512L*1024*1024,256L*1024*1024)
    @Test fun accountsCallerCopiesIntermediatesAndUnknownAcceleratorTiers(){
        val plan=MemoryPlanner.plan(program,inputs,snapshot=roomy)
        assertEquals(24L,plan.inputBytes);assertEquals(48L,plan.retainedVectorBytes)
        assertTrue(plan.admitted);assertTrue(plan.estimatedArrayAndWorkspaceBytes>=84L)
        assertNull(plan.tiers.single{it.type=="HBM"}.available)
        val lowFree=MemoryPlanner.plan(program,inputs,snapshot=roomy.copy(environmentFreeBytes=0))
        assertTrue(lowFree.admitted);assertEquals(true,lowFree.environmentPressureAdvisory)
        assertFalse(plan.automaticSpill);assertFalse(plan.reservationGuaranteed)
    }
    @Test fun rejectsBudgetAndObservedPressureBeforeCpuCopies()=runBlocking {
        assertFalse(MemoryPlanner.plan(program,inputs,1,roomy).admitted)
        assertFalse(MemoryPlanner.plan(program,inputs,snapshot=roomy.copy(heapUsedBytes=roomy.heapLimitBytes-16L*1024*1024)).admitted)
        assertThrows(IllegalArgumentException::class.java){MemoryPlanner.requireAdmission(MemoryPlanner.plan(program,inputs,1,roomy))}
        try{HyperLCpuBackend(1).execute(program,inputs);fail("Budget exceeded")}catch(_:IllegalArgumentException){}
        assertArrayEquals(floatArrayOf(-1f,2f,3f),inputs.getValue("x"),0f)
    }
    @Test fun observesHeapAndRejectsFullGraphMemoryOrShape(){
        val actual=MemoryPlanner.observe();assertTrue(actual.heapLimitBytes>0);assertTrue(actual.heapUsedBytes>=0)
        val big=mapOf("x" to FloatArray(262144),"w" to FloatArray(262144))
        val chain=program.copy(instructions=program.instructions+HyperLInstruction("another","relu",listOf("positive")),output="another")
        assertFalse(MemoryPlanner.plan(chain,big,snapshot=roomy).admitted)
        assertThrows(IllegalArgumentException::class.java){MemoryPlanner.plan(program,inputs+mapOf("w" to FloatArray(2)),snapshot=roomy)}
    }
}
