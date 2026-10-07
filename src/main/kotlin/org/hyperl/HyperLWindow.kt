package org.hyperl

import kotlinx.coroutines.*
import java.awt.*
import java.nio.file.Path
import javax.swing.*

/** Desktop GUI uses the same validated program/runtime API as the CLI. */
class HyperLPanel: JPanel(BorderLayout()) {
    val program=JTextArea(Workspace.json.encodeToString(Workspace.json.parseToJsonElement(Workspace.EXAMPLE_PROGRAM)),18,40)
    val inputs=JTextArea(Workspace.json.encodeToString(Workspace.json.parseToJsonElement(Workspace.EXAMPLE_INPUTS)),8,40)
    val output=JTextArea("Ready. CPU reference is available. GPU execution requires an explicitly installed bridge.",16,50)
    val backend=JComboBox(arrayOf("CPU_REFERENCE","OPENCL_GPU"))
    val sourceTarget=JComboBox(arrayOf("LLVM_CPU","CUDA","ROCM_HIP","OPENCL_SPIRV","METAL","VULKAN_SPIRV"))
    val bridge=JTextField("",28)
    val device=JSpinner(SpinnerNumberModel(0,0,127,1))
    val run=JButton("Run");val emit=JButton("Emit source");val probe=JButton("Probe GPU");val stop=JButton("Stop");val memory=JButton("Memory plan")
    private val scope=CoroutineScope(SupervisorJob()+Dispatchers.Default)
    private var job:Job?=null
    init {
        output.isEditable=false
        for(area in listOf(program,inputs,output)){area.font=Font(Font.MONOSPACED,Font.PLAIN,13);area.lineWrap=true;area.wrapStyleWord=true}
        val editors=JPanel(GridLayout(1,2,12,0))
        val inputPanel=JPanel(BorderLayout());inputPanel.add(JLabel("Input vectors (JSON)"),BorderLayout.NORTH);inputPanel.add(JScrollPane(inputs))
        val left=JPanel(BorderLayout());left.add(JLabel("hyperl/1 program (JSON)"),BorderLayout.NORTH);left.add(JScrollPane(program));left.add(inputPanel,BorderLayout.SOUTH)
        editors.add(left);editors.add(JScrollPane(output));add(editors)
        val controls=JPanel();controls.add(backend);controls.add(run);controls.add(stop);controls.add(sourceTarget);controls.add(emit);controls.add(probe);controls.add(memory)
        val configuration=JPanel();configuration.add(JLabel("OpenCL executable (absolute path):"));configuration.add(bridge);configuration.add(JLabel("GPU index:"));configuration.add(device)
        val top=JPanel(GridLayout(2,1));top.add(controls);top.add(configuration);add(top,BorderLayout.NORTH)
        add(JLabel("Local experimental kernels • no network/radio access • source emission does not qualify devices"),BorderLayout.SOUTH)
        run.addActionListener {
            val text=program.text;val inputText=inputs.text;val selected=backend.selectedItem as String;val binary=bridge.text;val index=device.value as Int
            start { val p=Workspace.program(text);val values=Workspace.inputs(inputText)
                val result=if(selected=="CPU_REFERENCE")HyperLCpuBackend().execute(p,values) else OpenClBridge(Path.of(binary)).executeVerified(index,p,values)
                "Backend: $selected\nResult: ${Workspace.result(result)}\nOpenCL mode verifies this invocation against CPU; no speed claim."
            }
        }
        emit.addActionListener {val text=program.text;val target=HyperLTarget.valueOf(sourceTarget.selectedItem as String);start{PortableEmitter.emit(Workspace.program(text),target).source}}
        memory.addActionListener {val text=program.text;val values=inputs.text;start{Workspace.json.encodeToString(MemoryPlanner.plan(Workspace.program(text),Workspace.inputs(values)))}}
        probe.addActionListener {val binary=bridge.text;start{OpenClBridge(Path.of(binary)).probe()}}
        stop.addActionListener{job?.cancel();output.text="Stop requested; awaiting local task/native cleanup. Remote termination is not implied."}
    }
    private fun start(action:suspend ()->String) {
        if(job?.isActive==true)return
        run.isEnabled=false;emit.isEnabled=false;probe.isEnabled=false;memory.isEnabled=false
        job=scope.launch {
            var message="Stopped after local cleanup."
            try{message=action()}catch(_:CancellationException){}catch(e:Exception){message="Failed: ${e.message?.take(1024)}"}
            finally{val finalMessage=message;SwingUtilities.invokeLater {output.text=finalMessage;run.isEnabled=true;emit.isEnabled=true;probe.isEnabled=true;memory.isEnabled=true}}
        }
    }
    fun close(){scope.cancel()}
}
object HyperLWindow {
    fun show(){
        check(!GraphicsEnvironment.isHeadless()){"Desktop display unavailable; use CLI commands"}
        val panel=HyperLPanel();val window=JFrame("HyperL — portable AI kernel workbench")
        window.defaultCloseOperation=WindowConstants.DISPOSE_ON_CLOSE;window.contentPane=panel;window.setSize(1150,700)
        window.addWindowListener(object:java.awt.event.WindowAdapter(){override fun windowClosed(e:java.awt.event.WindowEvent){panel.close()}})
        window.setLocationRelativeTo(null);window.isVisible=true
    }
}
