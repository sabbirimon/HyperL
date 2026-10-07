package org.hyperl

import kotlinx.coroutines.*
import org.fife.ui.rsyntaxtextarea.SyntaxConstants
import java.awt.*
import java.awt.event.*
import java.nio.file.Files
import java.nio.file.Path
import javax.swing.*
import javax.swing.border.EmptyBorder
import javax.swing.filechooser.FileNameExtensionFilter

/** Real local developer tools share the CLI's validation and execution contracts. */
class HyperLPanel: JPanel(BorderLayout(0,18)) {
    private val initializedTheme=DesktopTheme.install()
    val program=DesktopTheme.editor("""{
  "format": "hyperl/1",
  "inputs": ["x", "w"],
  "instructions": [
    {"output": "value", "operation": "multiply",
     "inputs": ["x", "w"]},
    {"output": "positive", "operation": "relu",
     "inputs": ["value"]}
  ],
  "output": "positive"
}""")
    val inputs=DesktopTheme.editor("""{
  "x": [-1, 2, 3],
  "w": [2, 3, 4]
}""")
    val output=DesktopTheme.editor("Ready to build.\n\nRun the CPU example or inspect its memory plan.\nSource emission does not qualify an accelerator.\n\nOpenCL execution requires an explicitly installed bridge.",false)
    val backend=JComboBox(arrayOf("CPU_REFERENCE","OPENCL_GPU"))
    val sourceTarget=JComboBox(arrayOf("LLVM_CPU","CUDA","ROCM_HIP","OPENCL_SPIRV","METAL","VULKAN_SPIRV"))
    val bridge=JTextField("",24);val device=JSpinner(SpinnerNumberModel(0,0,127,1))
    val memoryBudget=JTextField(MemoryPlanner.DEFAULT_BUDGET.toString(),10)
    val run=JButton("Run kernel");val emit=JButton("Emit source");val probe=JButton("Probe GPU");val stop=JButton("Stop")
    val memory=JButton("Memory plan");val validate=JButton("Validate");val format=JButton("Format")
    val open=JButton("Open");val save=JButton("Save");val export=JButton("Export output")
    val analyze=JButton("Analyze");val review=JButton("Review fix")
    private var diagnosticReport:DiagnosticReport?=null
    val findText=JTextField("",16);val find=JButton("Find next")
    val examples=JComboBox(arrayOf("Elementwise","Reduction"));val loadExample=JButton("Load example")
    val state=DesktopTheme.label("READY",12,DesktopTheme.cyan)
    private val scope=CoroutineScope(SupervisorJob()+Dispatchers.Default)
    private var job:Job?=null
    private var searchEditor=program
    private var dirty=false
    private val configurationControls=listOf<JComponent>(backend,sourceTarget,bridge,device,memoryBudget,findText,examples)
    private val actions=listOf(run,emit,probe,memory,validate,format,open,save,export,find,loadExample,analyze,review)
    init {
        background=DesktopTheme.background;border=EmptyBorder(22,24,16,24)
        output.syntaxEditingStyle=SyntaxConstants.SYNTAX_STYLE_NONE
        actions.forEach{DesktopTheme.button(it,it===run)};DesktopTheme.button(stop);stop.foreground=Color(0xFFA5A5);stop.isEnabled=false
        backend.toolTipText="Only CPU reference and an explicitly installed OpenCL bridge execute."
        memoryBudget.toolTipText="CPU byte budget; GPU VRAM is not measured by this policy."
        bridge.toolTipText="An absolute path to your reviewed, installed OpenCL bridge; never downloaded automatically."
        val header=JPanel(BorderLayout(20,0));header.isOpaque=false
        val brand=JPanel(BorderLayout(0,7));brand.isOpaque=false
        val title=JPanel(FlowLayout(FlowLayout.LEFT,12,0));title.isOpaque=false
        val logo=DesktopTheme.label("HL",22,DesktopTheme.background);logo.isOpaque=true;logo.background=DesktopTheme.cyan;logo.border=EmptyBorder(8,11,8,11)
        title.add(logo);title.add(DesktopTheme.label("HyperL",30));title.add(DesktopTheme.label("DEVELOPER WORKBENCH",11,DesktopTheme.muted))
        brand.add(title,BorderLayout.NORTH);brand.add(DesktopTheme.label("Write a kernel. Inspect memory. Build with evidence.",14,DesktopTheme.muted),BorderLayout.SOUTH)
        header.add(brand);val edition=JPanel(GridLayout(2,1,0,7));edition.isOpaque=false;edition.add(DesktopTheme.label("LOCAL  /  ALPHA.2",11,DesktopTheme.muted));edition.add(state);header.add(edition,BorderLayout.EAST)
        val top=JPanel(BorderLayout(0,16));top.isOpaque=false;top.add(header,BorderLayout.NORTH)
        val snapshot=MemoryPlanner.observe()
        val environment=DesktopTheme.panel(FlowLayout(FlowLayout.LEFT,20,0))
        environment.add(DesktopTheme.label("ENVIRONMENT",10,DesktopTheme.muted));environment.add(DesktopTheme.label("${System.getProperty("os.name")}  ·  ${System.getProperty("os.arch")}",12))
        environment.add(DesktopTheme.label("JVM heap limit: ${snapshot.heapLimitBytes/(1024*1024)} MiB",12))
        environment.add(DesktopTheme.label("CPU reference available",12,DesktopTheme.accent));environment.add(DesktopTheme.label("GPU requires probe",12,DesktopTheme.muted));top.add(environment,BorderLayout.SOUTH);add(top,BorderLayout.NORTH)

        val tools=JPanel(GridLayout(2,1,0,7));tools.isOpaque=false
        val fileTools=JPanel(FlowLayout(FlowLayout.LEFT,8,0));fileTools.isOpaque=false
        listOf(open,save,validate,format).forEach(fileTools::add)
        val exampleTools=JPanel(FlowLayout(FlowLayout.LEFT,8,0));exampleTools.isOpaque=false
        exampleTools.add(DesktopTheme.label("START FROM",10,DesktopTheme.muted));exampleTools.add(examples);exampleTools.add(loadExample);exampleTools.add(analyze);exampleTools.add(review)
        tools.add(fileTools);tools.add(exampleTools)
        val editors=JTabbedPane();editors.addTab("Program",editorCard("01  PROGRAM","hyperl/1 · JSON",program));editors.addTab("Inputs",editorCard("02  INPUT VECTORS","finite f32 · JSON",inputs))
        editors.minimumSize=Dimension(330,260)
        val left=JPanel(BorderLayout(0,12));left.isOpaque=false;left.add(tools,BorderLayout.NORTH);left.add(editors)
        val search=JPanel(FlowLayout(FlowLayout.LEFT,8,0));search.isOpaque=false;search.add(DesktopTheme.label("Literal search",11,DesktopTheme.muted));search.add(findText);search.add(find);left.add(search,BorderLayout.SOUTH)
        val right=DesktopTheme.panel(BorderLayout(0,12));right.minimumSize=Dimension(320,260)
        val resultHeader=JPanel(BorderLayout());resultHeader.isOpaque=false;resultHeader.add(DesktopTheme.label("03  OUTPUT / DIAGNOSTICS",11,DesktopTheme.muted));resultHeader.add(export,BorderLayout.EAST);right.add(resultHeader,BorderLayout.NORTH);right.add(DesktopTheme.scroll(output))
        val split=JSplitPane(JSplitPane.HORIZONTAL_SPLIT,left,right);split.resizeWeight=0.53;split.dividerSize=14;split.border=null;split.background=DesktopTheme.background
        add(split)

        val bottom=JPanel(BorderLayout(0,12));bottom.isOpaque=false
        val execution=DesktopTheme.panel(BorderLayout(0,10));val commands=JPanel(FlowLayout(FlowLayout.LEFT,8,0));commands.isOpaque=false
        commands.add(backend);commands.add(run);commands.add(stop);commands.add(memory);commands.add(DesktopTheme.label("CPU budget (bytes)",11,DesktopTheme.muted));commands.add(memoryBudget)
        val sourceCommands=JPanel(FlowLayout(FlowLayout.LEFT,8,0));sourceCommands.isOpaque=false
        sourceCommands.add(sourceTarget);sourceCommands.add(emit);sourceCommands.add(DesktopTheme.label("Generate source · compilation / device qualification remains separate",11,DesktopTheme.muted))
        val commandRows=JPanel(GridLayout(2,1,0,9));commandRows.isOpaque=false;commandRows.add(commands);commandRows.add(sourceCommands);execution.add(commandRows,BorderLayout.NORTH)
        val gpu=JPanel(FlowLayout(FlowLayout.LEFT,8,0));gpu.isOpaque=false;gpu.add(DesktopTheme.label("OpenCL bridge",11,DesktopTheme.muted));gpu.add(bridge);gpu.add(DesktopTheme.label("Device",11,DesktopTheme.muted));gpu.add(device);gpu.add(probe);execution.add(gpu,BorderLayout.SOUTH)
        bottom.add(execution);bottom.add(DesktopTheme.label("Ctrl / Cmd + Enter to run  ·  Local files only  ·  Full SDK, model engines and device profiler are later milestones",11,DesktopTheme.muted),BorderLayout.SOUTH);add(bottom,BorderLayout.SOUTH)
        for(editor in listOf(program,inputs)){
            editor.addFocusListener(object:FocusAdapter(){override fun focusGained(e:FocusEvent){searchEditor=editor}})
            editor.document.addDocumentListener(object:javax.swing.event.DocumentListener{
                override fun insertUpdate(e:javax.swing.event.DocumentEvent){dirty=true};override fun removeUpdate(e:javax.swing.event.DocumentEvent){dirty=true};override fun changedUpdate(e:javax.swing.event.DocumentEvent){dirty=true}
            })
        }
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(KeyStroke.getKeyStroke("control ENTER"),"run")
        getInputMap(WHEN_ANCESTOR_OF_FOCUSED_COMPONENT).put(KeyStroke.getKeyStroke("meta ENTER"),"run")
        actionMap.put("run",object:AbstractAction(){override fun actionPerformed(e:ActionEvent){if(run.isEnabled)run.doClick()}})
        run.addActionListener {
            val text=program.text;val inputText=inputs.text;val selected=backend.selectedItem as String;val binary=bridge.text;val index=device.value as Int;val budget=memoryBudget.text
            start {val p=Workspace.program(text);val values=Workspace.inputs(inputText);val bytes=budget.toLong()
                val result=if(selected=="CPU_REFERENCE")HyperLCpuBackend(bytes).execute(p,values) else {MemoryPlanner.requireAdmission(MemoryPlanner.plan(p,values,bytes));OpenClBridge(Path.of(binary)).executeVerified(index,p,values)}
                "Backend: $selected\nResult: ${Workspace.result(result)}\n\nOpenCL verifies each request against CPU; no speed claim."
            }
        }
        emit.addActionListener {val text=program.text;val target=HyperLTarget.valueOf(sourceTarget.selectedItem as String);start(SyntaxConstants.SYNTAX_STYLE_C){PortableEmitter.emit(Workspace.program(text),target).source}}
        memory.addActionListener {val text=program.text;val values=inputs.text;val budget=memoryBudget.text;start{Workspace.json.encodeToString(MemoryPlanner.plan(Workspace.program(text),Workspace.inputs(values),budget.toLong()))}}
        probe.addActionListener {val binary=bridge.text;start{OpenClBridge(Path.of(binary)).probe()}}
        validate.addActionListener {val p=program.text;val i=inputs.text;start{DeveloperWorkspace.encode(p,i);"Valid hyperl/1 workspace.\nDependencies, finite inputs and shapes passed.\nUse Memory plan for admission and Run for actual execution."}}
        format.addActionListener {val p=program.text;val i=inputs.text;start{val doc=DeveloperWorkspace.parse(DeveloperWorkspace.encode(p,i));SwingUtilities.invokeAndWait{program.text=doc.program;inputs.text=doc.inputs};"Workspace validated and JSON formatted."}}
        analyze.addActionListener {val p=program.text;val i=inputs.text;val budget=memoryBudget.text;val selected=backend.selectedItem as String;start{val report=CodeDiagnostics.analyze(p,i,budget.toLong(),selected);SwingUtilities.invokeAndWait{diagnosticReport=report};Workspace.json.encodeToString(report)}}
        review.addActionListener {
            val repairs=diagnosticReport?.issues?.filter{it.repair!=null}.orEmpty()
            if(repairs.isEmpty()){JOptionPane.showMessageDialog(this,"Analyze the current program first. No reviewable typo/reference repair is available.");return@addActionListener}
            val labels=repairs.map{"${it.path}: ${it.repair!!.before} → ${it.repair.after}"}.toTypedArray()
            val selected=JOptionPane.showInputDialog(this,"Choose a suggested edit to review. Its correctness still needs validation and tests.","HyperL suggested fix",JOptionPane.QUESTION_MESSAGE,null,labels,labels.first()) as? String?:return@addActionListener
            val repair=repairs[labels.indexOf(selected)].repair!!
            if(JOptionPane.showConfirmDialog(this,"Replace ${repair.before} with ${repair.after}?\nOnly this field changes; Analyze again afterward.","Apply reviewed suggestion",JOptionPane.OK_CANCEL_OPTION)!=JOptionPane.OK_OPTION)return@addActionListener
            val text=program.text;start{val patched=CodeDiagnostics.apply(text,repair);SwingUtilities.invokeAndWait{program.text=patched;diagnosticReport=null};"Applied the reviewed suggestion. Analyze and test again before execution."}
        }
        find.addActionListener {val term=findText.text;if(term.isEmpty()||term.length>256)return@addActionListener;val text=searchEditor.text;val start=searchEditor.selectionEnd;var index=text.indexOf(term,start);if(index<0)index=text.indexOf(term);if(index>=0){searchEditor.requestFocusInWindow();searchEditor.select(index,index+term.length);state.text="MATCH FOUND"}else state.text="NO MATCH"}
        loadExample.addActionListener {if(!allowDiscard())return@addActionListener;val example=DeveloperWorkspace.example(examples.selectedItem as String);program.text=pretty(example.program);inputs.text=pretty(example.inputs);dirty=false;state.text="EXAMPLE LOADED"}
        open.addActionListener {if(!allowDiscard())return@addActionListener;val file=choose(false,"Open HyperL workspace")?:return@addActionListener;start{val doc=DeveloperWorkspace.parse(Workspace.read(file));SwingUtilities.invokeAndWait{program.text=doc.program;inputs.text=doc.inputs;dirty=false};"Opened ${file.fileName}.\nNothing was executed."}}
        save.addActionListener {val file=choose(true,"Save HyperL workspace")?:return@addActionListener;val replace=Files.exists(file);if(replace&&!confirmReplace(file))return@addActionListener;val p=program.text;val i=inputs.text;start{DeveloperWorkspace.save(file,DeveloperWorkspace.encode(p,i),replace);SwingUtilities.invokeAndWait{dirty=false};"Saved ${file.fileName}."}}
        export.addActionListener {val file=choose(true,"Export displayed output")?:return@addActionListener;val replace=Files.exists(file);if(replace&&!confirmReplace(file))return@addActionListener;val text=output.text;start{DeveloperWorkspace.save(file,text,replace);"Exported displayed output to ${file.fileName}.\nNo source was compiled or executed."}}
        stop.addActionListener{job?.cancel();state.text="STOPPING";output.text="Stop requested; awaiting local task/native cleanup. Remote termination is not implied."}
    }
    private fun editorCard(title:String,detail:String,editor:org.fife.ui.rsyntaxtextarea.RSyntaxTextArea)=DesktopTheme.panel(BorderLayout(0,10)).also{
        val heading=JPanel(BorderLayout());heading.isOpaque=false;heading.add(DesktopTheme.label(title,11,DesktopTheme.muted));heading.add(DesktopTheme.label(detail,11,DesktopTheme.cyan),BorderLayout.EAST);it.add(heading,BorderLayout.NORTH);it.add(DesktopTheme.scroll(editor))
    }
    private fun allowDiscard()=!dirty||JOptionPane.showConfirmDialog(this,"Replace unsaved editor changes?","HyperL workspace",JOptionPane.OK_CANCEL_OPTION)==JOptionPane.OK_OPTION
    fun requestClose():Boolean=allowDiscard()
    private fun confirmReplace(file:Path)=JOptionPane.showConfirmDialog(this,"Replace ${file.fileName}?","Save selected file",JOptionPane.OK_CANCEL_OPTION)==JOptionPane.OK_OPTION
    private fun choose(save:Boolean,title:String):Path? {
        val chooser=JFileChooser();chooser.dialogTitle=title;chooser.fileFilter=FileNameExtensionFilter("HyperL workspace / JSON / source", "json","c","cl","metal","glsl","txt")
        if(save)chooser.selectedFile=java.io.File("workspace.hyperl.json")
        val result=if(save)chooser.showSaveDialog(this)else chooser.showOpenDialog(this)
        return if(result==JFileChooser.APPROVE_OPTION)chooser.selectedFile.toPath() else null
    }
    private fun start(syntax:String=SyntaxConstants.SYNTAX_STYLE_NONE,action:suspend ()->String) {
        if(job?.isActive==true)return
        actions.forEach{it.isEnabled=false};configurationControls.forEach{it.isEnabled=false};program.isEditable=false;inputs.isEditable=false;stop.isEnabled=true;state.text="RUNNING"
        job=scope.launch {
            var message="Stopped after local cleanup.";var status="STOPPED"
            try{message=action();status="COMPLETED"}catch(_:CancellationException){}catch(e:Exception){message="Failed: ${e.message?.take(1024)}";status="FAILED"}
            finally{val finalMessage=message;val finalStatus=status;SwingUtilities.invokeLater{output.syntaxEditingStyle=if(finalMessage.trimStart().startsWith("{"))SyntaxConstants.SYNTAX_STYLE_JSON else syntax;output.text=finalMessage;output.caretPosition=0;state.text=finalStatus;actions.forEach{it.isEnabled=true};configurationControls.forEach{it.isEnabled=true};program.isEditable=true;inputs.isEditable=true;stop.isEnabled=false}}
        }
    }
    fun close(){scope.cancel()}
    companion object {private fun pretty(text:String)=Workspace.json.encodeToString(Workspace.json.parseToJsonElement(text))}
}
object HyperLWindow {
    fun show(){
        check(!GraphicsEnvironment.isHeadless()){"Desktop display unavailable; use CLI commands"}
        val panel=HyperLPanel();val window=JFrame("HyperL — Developer Workbench")
        window.defaultCloseOperation=WindowConstants.DO_NOTHING_ON_CLOSE;window.contentPane=panel;window.setSize(1280,820);window.minimumSize=Dimension(1000,700)
        window.addWindowListener(object:WindowAdapter(){override fun windowClosing(e:WindowEvent){if(panel.requestClose())window.dispose()};override fun windowClosed(e:WindowEvent){panel.close()}})
        window.setLocationRelativeTo(null);window.isVisible=true
    }
}
