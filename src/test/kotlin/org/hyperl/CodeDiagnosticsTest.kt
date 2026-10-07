package org.hyperl
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import kotlinx.serialization.json.*
class CodeDiagnosticsTest {
    @Test fun validProgramHasContextCompletionsAndActualSampleCheck()=runBlocking {
        val report=CodeDiagnostics.analyze(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS)
        assertTrue(report.valid);assertTrue(report.sampleCpuChecked);assertTrue(report.memory!!.admitted)
        assertTrue(report.completions.any{it.path=="program.instructions[1].inputs"&&"value" in it.values})
    }
    @Test fun reviewedTypoAndReferenceRepairsAreBoundToExactEditorText()=runBlocking {
        val bad=Workspace.EXAMPLE_PROGRAM.replace("multiply","mulitply").replace("\"inputs\":[\"value\"]","\"inputs\":[\"vlaue\"]")
        val first=CodeDiagnostics.analyze(bad,Workspace.EXAMPLE_INPUTS);assertFalse(first.valid)
        val operation=first.issues.single{it.code=="UNKNOWN_OPERATION"}.repair!!
        val patched=CodeDiagnostics.apply(bad,operation)
        assertThrows(IllegalArgumentException::class.java){CodeDiagnostics.apply(bad+" ",operation)}
        val next=CodeDiagnostics.analyze(patched,Workspace.EXAMPLE_INPUTS)
        val reference=next.issues.single{it.code=="UNDEFINED_REFERENCE"}.repair!!
        val corrected=CodeDiagnostics.apply(patched,reference)
        assertTrue(CodeDiagnostics.analyze(corrected,Workspace.EXAMPLE_INPUTS).valid)
        assertEquals("mulitply",Workspace.json.parseToJsonElement(bad).jsonObject.getValue("instructions").jsonArray[0].jsonObject.getValue("operation").jsonPrimitive.content)
    }
    @Test fun shapesBudgetAndUnqualifiedGpuReductionHaveConcreteSuggestions()=runBlocking {
        assertTrue(CodeDiagnostics.analyze(Workspace.EXAMPLE_PROGRAM,"{\"x\":[1],\"w\":[1,2]}").issues.any{it.code=="INPUT_SHAPE_BINDING"})
        assertTrue(CodeDiagnostics.analyze(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS,1).issues.any{it.code=="MEMORY_ADMISSION"})
        val sum=DeveloperWorkspace.example("Reduction")
        val report=CodeDiagnostics.analyze(sum.program,sum.inputs,backend="OPENCL_GPU")
        assertFalse(report.valid);assertFalse(report.sampleCpuChecked);assertTrue(report.issues.any{it.code=="GPU_REDUCTION"})
    }
    @Test fun actualSampleOverflowCannotBeHiddenByRelu()=runBlocking {
        val report=CodeDiagnostics.analyze(Workspace.EXAMPLE_PROGRAM,"{\"x\":[-3.402823466e38],\"w\":[2]}")
        assertFalse(report.valid);val error=report.issues.single{it.code=="SAMPLE_NUMERICAL_FAILURE"};assertEquals("program.instructions[0]",error.path);assertFalse(report.sampleCpuChecked)
    }
    @Test fun malformedAndOversizedDeclarationFailBoundedly()=runBlocking {
        assertTrue(CodeDiagnostics.analyze("{",Workspace.EXAMPLE_INPUTS).issues.any{it.code=="JSON_SYNTAX"})
        val bad=Workspace.EXAMPLE_PROGRAM.replace("[\"x\",\"w\"]","[\"x\",\"x\"]")
        assertTrue(CodeDiagnostics.analyze(bad,Workspace.EXAMPLE_INPUTS).issues.any{it.code=="INPUT_DECLARATION"})
    }
}
