package org.hyperl

import kotlinx.serialization.json.*
import java.nio.file.*
import java.nio.file.attribute.PosixFilePermissions

/** Owner-opened local project document, never a build script or executable plugin. */
data class DeveloperDocument(val program:String,val inputs:String)
object DeveloperWorkspace {
    fun parse(text:String):DeveloperDocument {
        require(text.length<=Workspace.MAX_JSON_BYTES)
        val root=Workspace.json.parseToJsonElement(text).jsonObject
        require(root.keys==setOf("format","program","inputs") && root.getValue("format").jsonPrimitive.content=="hyperl-workspace/1"){"Unsupported workspace document"}
        val p=Workspace.json.encodeToString(root.getValue("program"));val i=Workspace.json.encodeToString(root.getValue("inputs"))
        val program=Workspace.program(p);val inputs=Workspace.inputs(i)
        require(inputs.keys==program.inputs);MemoryPlanner.plan(program,inputs)
        return DeveloperDocument(p,i)
    }
    fun encode(program:String,inputs:String):String {
        val text=Workspace.json.encodeToString(buildJsonObject {put("format","hyperl-workspace/1");put("program",Workspace.json.parseToJsonElement(program));put("inputs",Workspace.json.parseToJsonElement(inputs))})
        parse(text);return text
    }
    fun save(file:Path,text:String,replace:Boolean=false){
        val bytes=text.toByteArray(Charsets.UTF_8);require(bytes.size<=Workspace.MAX_JSON_BYTES)
        require(!Files.isSymbolicLink(file)){"Symlink output rejected"}
        if(Files.exists(file,LinkOption.NOFOLLOW_LINKS)){require(replace && Files.isRegularFile(file,LinkOption.NOFOLLOW_LINKS)){"Output exists"}}
        val parent=file.toAbsolutePath().parent
        val attrs=if(Files.getFileStore(parent).supportsFileAttributeView("posix"))arrayOf(PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rw-------")))else emptyArray()
        val temporary=Files.createTempFile(parent,".hyperl-editor-",".tmp",*attrs)
        try{Files.write(temporary,bytes);if(replace)Files.move(temporary,file,StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING)else Files.move(temporary,file,StandardCopyOption.ATOMIC_MOVE)}finally{Files.deleteIfExists(temporary)}
    }
    fun example(name:String):DeveloperDocument=when(name){
        "Elementwise"->DeveloperDocument(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS)
        "Reduction"->DeveloperDocument("""{"format":"hyperl/1","inputs":["x"],"instructions":[{"output":"total","operation":"sum","inputs":["x"]}],"output":"total"}""","""{"x":[1,2,3,4]}""")
        else->error("Unknown example")
    }
}
