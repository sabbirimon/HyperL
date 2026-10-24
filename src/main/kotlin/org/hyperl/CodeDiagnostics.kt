package org.hyperl

import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.*
import java.security.MessageDigest

@Serializable data class SuggestedRepair(val programSha256:String,val instruction:Int,val input:Int?=null,val before:String,val after:String)
@Serializable data class CodeIssue(val severity:String,val code:String,val path:String,val message:String,val suggestion:String,val repair:SuggestedRepair?=null)
@Serializable data class CodeCompletion(val path:String,val values:List<String>,val explanation:String)
@Serializable data class DiagnosticReport(val valid:Boolean,val issues:List<CodeIssue>,val completions:List<CodeCompletion>,val sampleCpuChecked:Boolean,val sampleScope:String="Provided inputs only; no proof for all values or accelerator execution",val memory:MemoryPlan?=null)
/** Bounded local semantic analysis plus real CPU sample checking; no provider or shell. */
object CodeDiagnostics {
    private val operations=listOf("add","multiply","relu","sum")
    private val names=Regex(HyperLContract.IDENTIFIER_PATTERN)
    private fun hash(text:String)=MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8)).joinToString(""){"%02x".format(it)}
    private fun nearest(value:String,candidates:Collection<String>):String?=candidates.filter{it!=value}.minByOrNull{distance(value,it)}?.takeIf{distance(value,it)<=3}
    private fun distance(a:String,b:String):Int {
        if(a.length>32 || b.length>32)return Int.MAX_VALUE
        var previous=IntArray(b.length+1){it}
        for(i in a.indices){val row=IntArray(b.length+1);row[0]=i+1
            for(j in b.indices)row[j+1]=minOf(row[j]+1,previous[j+1]+1,previous[j]+if(a[i]==b[j])0 else 1)
            previous=row
        };return previous[b.length]
    }
    suspend fun analyze(programText:String,inputText:String,budget:Long=MemoryPlanner.DEFAULT_BUDGET,backend:String="CPU_REFERENCE"):DiagnosticReport {
        val issues=mutableListOf<CodeIssue>();val completions=mutableListOf<CodeCompletion>()
        fun issue(code:String,path:String,message:String,suggestion:String,repair:SuggestedRepair?=null,severity:String="ERROR"){if(issues.size<256)issues.add(CodeIssue(severity,code,path.take(256),message.take(1024),suggestion,repair))}
        require(programText.length<=Workspace.MAX_JSON_BYTES && inputText.length<=Workspace.MAX_JSON_BYTES)
        val raw=try{Workspace.json.parseToJsonElement(programText).jsonObject}catch(e:Exception){issue("JSON_SYNTAX","program",e.message?:"Invalid JSON","Check quoting, commas and matching object/array delimiters.");return DiagnosticReport(false,issues,completions,false)}
        if(raw.size>16){issue("PROGRAM_FIELD_LIMIT","program","Too many program fields","Keep only the four supported fields; diagnostic output is bounded.");return DiagnosticReport(false,issues,completions,false)}
        val format=raw["format"]?.let{(it as? JsonPrimitive)?.content}
        if(format!=null&&format!="hyperl/1")issue("VERSION","program.format","Unsupported language version","Use hyperl/1 only if you intend its f32 semantics.")
        for(key in raw.keys-setOf("format","inputs","instructions","output"))issue("UNKNOWN_FIELD","program.$key","Unknown program field","Remove unsupported metadata/build hooks; workspaces never execute hooks.")
        val declarations=raw["inputs"] as? JsonArray
        if(declarations==null || declarations.size !in 1..8){issue("INPUT_DECLARATION","program.inputs","Expected 1–8 declared inputs","Supply a bounded list of distinct identifiers.");return DiagnosticReport(false,issues,completions,false)}
        val declared=declarations.mapNotNull{(it as? JsonPrimitive)?.takeIf{p->p.isString}?.content}.orEmpty()
        if(declared.size !in 1..8 || declared.any{!it.matches(names)} || declared.toSet().size!=declared.size){issue("INPUT_DECLARATION","program.inputs","Expected 1–8 unique legal input names","Use distinct ASCII identifiers, max 32 characters.");return DiagnosticReport(false,issues,completions,false)}
        val steps=raw["instructions"] as? JsonArray
        if(steps==null || steps.size !in 1..64){issue("INSTRUCTION_COUNT","program.instructions","Expected 1–64 instructions","Supply a bounded ordered DAG.");return DiagnosticReport(false,issues,completions,false)}
        val defined=declared.toMutableSet();val signature=hash(programText)
        for((index,value) in steps.withIndex()){
            currentCoroutineContext().ensureActive()
            val step=value as? JsonObject
            if(step==null){issue("STEP_OBJECT","program.instructions[$index]","Instruction must be an object","Use output, operation and inputs fields.");continue}
            val op=(step["operation"] as? JsonPrimitive)?.content.orEmpty()
            val out=(step["output"] as? JsonPrimitive)?.content.orEmpty()
            val arguments=step["inputs"] as? JsonArray
            if(arguments==null || arguments.size !in 1..2){issue("ARITY","program.instructions[$index].inputs","Expected 1–2 references","Use the operator's bounded arity.");continue}
            val args=arguments.mapNotNull{(it as? JsonPrimitive)?.takeIf{p->p.isString}?.content}.orEmpty()
            completions.add(CodeCompletion("program.instructions[$index].operation",operations,"Supported f32 operations."))
            completions.add(CodeCompletion("program.instructions[$index].inputs",defined.sorted(),"Only declared inputs or earlier outputs are legal references."))
            if(op !in operations){val closest=nearest(op,operations);issue("UNKNOWN_OPERATION","program.instructions[$index].operation","Unsupported operation '$op'",closest?.let{"Review replacing '$op' with '$it'."}?:"Choose add, multiply, relu or sum.",closest?.let{SuggestedRepair(signature,index,before=op,after=it)})}
            else if(args.size!=if(op=="add"||op=="multiply")2 else 1)issue("ARITY","program.instructions[$index].inputs","Wrong number of operands for $op","${if(op=="add"||op=="multiply")"Two" else "One"} earlier references required; no implicit arguments.")
            for((input,name) in args.withIndex())if(name !in defined){val closest=nearest(name,defined);issue("UNDEFINED_REFERENCE","program.instructions[$index].inputs[$input]","'$name' is not defined before this step",closest?.let{"Review reference '$it'; a similar name does not prove your intended dependency."}?:"Declare the input or use an earlier output.",closest?.let{SuggestedRepair(signature,index,input,name,it)})}
            if(!out.matches(names)||out in defined)issue("OUTPUT_NAME","program.instructions[$index].output","Output name is invalid or already defined","Use a new identifier and explicitly update intended consumers.")
            else defined.add(out)
        }
        val output=(raw["output"] as? JsonPrimitive)?.content.orEmpty()
        completions.add(CodeCompletion("program.output",defined.sorted(),"Select an existing input or step output."))
        if(output !in defined)issue("FINAL_OUTPUT","program.output","Final output is undefined","Choose a declared input or instruction output.")
        var p:HyperLProgram?=null;var values:Map<String,FloatArray>?=null;var memory:MemoryPlan?=null
        try{p=Workspace.program(programText)}catch(e:Exception){if(issues.none{it.severity=="ERROR"})issue("PROGRAM_INVALID","program",e.message?:"Invalid program","Correct required fields and instruction dependencies.")}
        try{values=Workspace.inputs(inputText)}catch(e:Exception){issue("INPUT_VALUES","inputs",e.message?:"Invalid finite f32 vectors","Use 1–8 arrays of finite f32 numbers within vector limits.")}
        if(p!=null&&values!=null){
            try{memory=MemoryPlanner.plan(p,values,budget);if(!memory.admitted)issue("MEMORY_ADMISSION","memory",memory.reason,"Partition the work or adjust your explicit budget within observed JVM headroom; no automatic spill.")}
            catch(e:Exception){issue("INPUT_SHAPE_BINDING","inputs",e.message?:"Input shape/binding mismatch","Use exactly the declared inputs and equal vector lengths for add/multiply; broadcasting is unavailable.")}
            if(backend in setOf("OPENCL_GPU","METAL_GPU")&&p.instructions.any{it.operation=="sum"})issue("GPU_REDUCTION","program.instructions","The selected GPU bridge has no reduction dispatch","Explicitly choose CPU_REFERENCE for this program or wait for a qualified reduction adapter.")
            if(backend !in setOf("CPU_REFERENCE","OPENCL_GPU","METAL_GPU"))issue("BACKEND_UNAVAILABLE","backend","Backend is not installed in this workbench","Use an explicitly available backend; source emission alone does not install one.")
        }
        var checked=false
        if(p!=null&&values!=null&&issues.none{it.severity=="ERROR"}){
            try{HyperLCpuBackend(budget).execute(p,values);checked=true}
            catch(e:IllegalArgumentException){val outputName=e.message?.substringAfter("nonfinite result at ","")?.takeIf{it.isNotEmpty()};val index=p.instructions.indexOfFirst{it.output==outputName};issue("SAMPLE_NUMERICAL_FAILURE",if(index>=0)"program.instructions[$index]" else "program",e.message?:"CPU sample rejected","Inspect input scale and operation range; f32 overflow cannot be hidden by a later ReLU. Higher precision requires future explicit semantics.")}
        }
        return DiagnosticReport(issues.none{it.severity=="ERROR"},issues,completions,checked,memory=memory)
    }
    /** Applies only one reviewed operation/reference correction to the exact analyzed text. */
    fun apply(programText:String,repair:SuggestedRepair):String {
        require(hash(programText)==repair.programSha256){"Editor changed; analyze again before applying this suggestion"}
        val root=Workspace.json.parseToJsonElement(programText).jsonObject
        val steps=root.getValue("instructions").jsonArray.toMutableList();require(repair.instruction in steps.indices && repair.instruction in 0..63)
        val step=steps[repair.instruction].jsonObject.toMutableMap()
        if(repair.input==null){require(repair.after in operations && step.getValue("operation").jsonPrimitive.content==repair.before);step["operation"]=JsonPrimitive(repair.after)}
        else{
            val defined=root.getValue("inputs").jsonArray.map{it.jsonPrimitive.content}.toMutableSet()
            steps.take(repair.instruction).forEach{defined.add(it.jsonObject.getValue("output").jsonPrimitive.content)}
            require(repair.after in defined);val args=step.getValue("inputs").jsonArray.toMutableList();require(repair.input in args.indices && repair.input in 0..1 && args[repair.input].jsonPrimitive.content==repair.before);args[repair.input]=JsonPrimitive(repair.after);step["inputs"]=JsonArray(args)
        }
        steps[repair.instruction]=JsonObject(step);return Workspace.json.encodeToString(JsonObject(root+mapOf("instructions" to JsonArray(steps))))
    }
}
