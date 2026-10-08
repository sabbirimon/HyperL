package org.hyperl

import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import javax.swing.SwingUtilities
import kotlin.system.exitProcess

fun main(args:Array<String>) {
    try {
        when(args.firstOrNull() ?: "help") {
            "help","--help","-h"->println("""HyperL ${Workspace.VERSION} — experimental portable AI kernels
Usage:
  hyperl gui
  hyperl capabilities
  hyperl library
  hyperl recipe RECIPE_ID
  hyperl recipe-run RECIPE_ID INPUTS.json [MEMORY_BUDGET_BYTES]
  hyperl diagnose PROGRAM.json INPUTS.json [MEMORY_BUDGET_BYTES]
  hyperl validate PROGRAM.json
  hyperl workspace-new FILE.hyperl.json
  hyperl workspace-validate FILE.hyperl.json
  hyperl run PROGRAM.json INPUTS.json [MEMORY_BUDGET_BYTES]
  hyperl sum-precise INPUTS.json [MEMORY_BUDGET_BYTES]
  hyperl memory-plan PROGRAM.json INPUTS.json [MEMORY_BUDGET_BYTES]
  hyperl emit TARGET PROGRAM.json
  hyperl gpu-probe /absolute/path/to/hyperl-opencl
  hyperl gpu-run /absolute/path/to/hyperl-opencl DEVICE_INDEX PROGRAM.json INPUTS.json
  hyperl metal-probe /absolute/path/to/hyperl-metal
  hyperl metal-run /absolute/path/to/hyperl-metal "EXACT METAL DEVICE NAME" PROGRAM.json INPUTS.json
  hyperl telecom-plan PROFILE.json
  hyperl cluster-plan PROFILE.json
  hyperl node-probe ORIGIN TOKEN_FILE
  hyperl keygen KEYFILE
  hyperl data-import SOURCE DATASET_DIRECTORY KEYFILE MAX_BYTES
  hyperl data-export DATASET_DIRECTORY DESTINATION KEYFILE MAX_BYTES
  hyperl data-rekey DATASET_DIRECTORY NEW_DATASET_DIRECTORY OLD_KEYFILE NEW_KEYFILE MAX_BYTES
Targets: LLVM_CPU, CUDA, ROCM_HIP, OPENCL_SPIRV, METAL, VULKAN_SPIRV.
CPU_REFERENCE and explicitly supplied OpenCL/Metal bridges can execute; actual GPU qualification is separate. Emission is source only.
Network access only through explicit authenticated node-probe; no shell, auto installs, radio or privileges.""")
            "gui"->{require(args.size==1);SwingUtilities.invokeLater{HyperLWindow.show()}}
            "capabilities"->{require(args.size==1);println(Workspace.capabilities())}
            "library"->{require(args.size==1);HyperLLibrary.ids.forEach{id->val recipe=HyperLLibrary.recipe(id);println("$id · ${recipe.title} · ${recipe.purpose} · ${if(recipe.elementwiseOnly)"elementwise source eligible; hardware qualification separate" else "CPU reduction"}")}}
            "recipe"->{require(args.size==2);val recipe=HyperLLibrary.recipe(args[1]);println(DeveloperWorkspace.encode(HyperLCodec.json.encodeToString(recipe.program),HyperLCodec.json.encodeToString(recipe.example)))}
            "recipe-run"->{require(args.size in 3..4);val recipe=HyperLLibrary.recipe(args[1]);println(cpu(Workspace.json.encodeToString(recipe.program),Workspace.read(Path.of(args[2])),args.getOrNull(3)?.toLong()?:MemoryPlanner.DEFAULT_BUDGET))}
            "workspace-new"->{require(args.size==2);DeveloperWorkspace.save(Path.of(args[1]),DeveloperWorkspace.encode(Workspace.EXAMPLE_PROGRAM,Workspace.EXAMPLE_INPUTS));println("Workspace created; nothing executed")}
            "workspace-validate"->{require(args.size==2);DeveloperWorkspace.parse(Workspace.read(Path.of(args[1])));println("Valid hyperl-workspace/1") }
            "diagnose"->{require(args.size in 3..4);val report=runBlocking{CodeDiagnostics.analyze(Workspace.read(Path.of(args[1])),Workspace.read(Path.of(args[2])),args.getOrNull(3)?.toLong()?:MemoryPlanner.DEFAULT_BUDGET)};println(Workspace.json.encodeToString(report));if(!report.valid)exitProcess(2)}
            "validate"->{require(args.size==2);Workspace.program(Workspace.read(Path.of(args[1])));println("Valid hyperl/1 program")}
            "run"->{require(args.size in 3..4);println(cpu(Workspace.read(Path.of(args[1])),Workspace.read(Path.of(args[2])),args.getOrNull(3)?.toLong()?:MemoryPlanner.DEFAULT_BUDGET))}
            "sum-precise"->{require(args.size in 2..3);val inputs=Workspace.inputs(Workspace.read(Path.of(args[1])));require(inputs.keys==setOf("x"));println(runBlocking{Workspace.result(floatArrayOf(PreciseReduction.sum(inputs.getValue("x"),args.getOrNull(2)?.toLong()?:MemoryPlanner.DEFAULT_BUDGET)))})}
            "memory-plan"->{require(args.size in 3..4);println(Workspace.json.encodeToString(MemoryPlanner.plan(Workspace.program(Workspace.read(Path.of(args[1]))),Workspace.inputs(Workspace.read(Path.of(args[2]))),args.getOrNull(3)?.toLong()?:MemoryPlanner.DEFAULT_BUDGET)))}
            "emit"->{require(args.size==3);println(PortableEmitter.emit(Workspace.program(Workspace.read(Path.of(args[2]))),HyperLTarget.valueOf(args[1])).source)}
            "gpu-probe"->{require(args.size==2);println(runBlocking { OpenClBridge(Path.of(args[1])).probe() })}
            "metal-probe"->{require(args.size==2);println(runBlocking { MetalBridge(Path.of(args[1])).probe() })}
            "metal-run"->{require(args.size==5);println(runBlocking {Workspace.json.encodeToString(MetalBridge(Path.of(args[1])).executeVerified(args[2],Workspace.program(Workspace.read(Path.of(args[3]))),Workspace.inputs(Workspace.read(Path.of(args[4])))))})}
            "gpu-run"->{require(args.size==5);val bridge=OpenClBridge(Path.of(args[1]));println(runBlocking {
                Workspace.result(bridge.executeVerified(args[2].toInt(),Workspace.program(Workspace.read(Path.of(args[3]))),Workspace.inputs(Workspace.read(Path.of(args[4])))))
            })}
            "telecom-plan"->{require(args.size==2);println(TelecomPlanner.plan(Workspace.json.decodeFromString<TelecomProfile>(Workspace.read(Path.of(args[1])))))}
            "cluster-plan"->{require(args.size==2);println(ClusterPlanner.plan(Workspace.json.decodeFromString<ClusterProfile>(Workspace.read(Path.of(args[1])))))}
            "node-probe"->{require(args.size==3);println(NodeProbe.observe(args[1],Path.of(args[2])))}
            "keygen"->{require(args.size==2);LargeData.keygen(Path.of(args[1]));println("Key created; keep it private and separate from datasets")}
            "data-import"->{require(args.size==5);println(runBlocking{LargeData.import(Path.of(args[1]),Path.of(args[2]),Path.of(args[3]),args[4].toLong())})}
            "data-export"->{require(args.size==5);println(runBlocking{LargeData.export(Path.of(args[1]),Path.of(args[2]),Path.of(args[3]),args[4].toLong())})}
            "data-rekey"->{require(args.size==6);println(runBlocking{LargeData.rekey(Path.of(args[1]),Path.of(args[2]),Path.of(args[3]),Path.of(args[4]),args[5].toLong())})}
            else->error("Unknown command; run hyperl help")
        }
    } catch(e:Exception) {System.err.println("HyperL failed: ${e.message?.take(1024)}");exitProcess(2)}
}
