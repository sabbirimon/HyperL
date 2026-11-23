package org.hyperl
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
class DeveloperWorkspaceTest {
    @Test fun actualWorkspaceRoundTripAndNoOverwrite(){
        val dir=Files.createTempDirectory("hyperl-dev-")
        try{val file=dir.resolve("workspace.hyperl.json");val text=DeveloperWorkspace.encode(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS)
            DeveloperWorkspace.save(file,text);val document=DeveloperWorkspace.parse(Workspace.read(file))
            assertEquals(Workspace.program(Workspace.EXAMPLE_PROGRAM),Workspace.program(document.program))
            assertThrows(IllegalArgumentException::class.java){DeveloperWorkspace.save(file,"replacement")}
            assertEquals(text,Files.readString(file));DeveloperWorkspace.save(file,text,true)
            assertFalse(Files.list(dir).use{it.anyMatch{p->p.fileName.toString().startsWith(".hyperl-editor-")}})
        }finally{Files.walk(dir).use{it.sorted(Comparator.reverseOrder()).forEach(Files::deleteIfExists)}}
    }
    @Test fun examplesValidateAndDocumentsCannotCarryBuildHooks(){
        for(name in listOf("Elementwise","Reduction")){val example=DeveloperWorkspace.example(name);DeveloperWorkspace.parse(DeveloperWorkspace.encode(example.program,example.inputs))}
        val text=DeveloperWorkspace.encode(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS)
        assertThrows(IllegalArgumentException::class.java){DeveloperWorkspace.parse(text.replace("\"format\":", "\"buildHook\": \"not executed\", \"format\":"))}
        assertThrows(IllegalArgumentException::class.java){DeveloperWorkspace.encode(Workspace.EXAMPLE_PROGRAM,"{\"x\":[1],\"w\":[1,2]}")}
    }
}
