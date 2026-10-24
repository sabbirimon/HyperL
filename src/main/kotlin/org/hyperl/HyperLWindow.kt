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
class HyperLPanel: JPanel(BorderLayout(0,10)) {
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
    val output=DesktopTheme.editor("Ready to build.\n\nRun the CPU example or inspect its memory plan.\nSource emission does not qualify an accelerator.\n\nGPU execution requires an explicitly installed OpenCL/Metal bridge.",false)
    val backend=JComboBox(arrayOf("CPU_REFERENCE","OPENCL_GPU","METAL_GPU"))
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
    val state=DesktopTheme.badge("READY")
    val executionTabs=JTabbedPane()
    val appearanceSelector=JComboBox(DesktopAppearance.entries.toTypedArray())
    val focusEditor=JToggleButton("Focus")
    private val scope=CoroutineScope(SupervisorJob()+Dispatchers.Default)
    private var job:Job?=null
    private var searchEditor=program
    private var dirty=false
    private val configurationControls=listOf<JComponent>(backend,sourceTarget,bridge,device,memoryBudget,findText,examples)
    private val actions=listOf(run,emit,probe,memory,validate,format,open,save,export,find,loadExample,analyze,review)
    init {
        background=DesktopTheme.background;border=EmptyBorder(8,12,8,12)
        output.syntaxEditingStyle=SyntaxConstants.SYNTAX_STYLE_NONE
        output.font=DesktopTheme.codeFont(13)
        actions.forEach{DesktopTheme.button(it,it===run)};DesktopTheme.button(stop);stop.foreground=DesktopTheme.error;stop.isEnabled=false
        backend.toolTipText="CPU reference, or explicitly installed reviewed OpenCL/Metal bridges. GPU hardware qualification is separate."
        memoryBudget.toolTipText="CPU byte budget; GPU VRAM is not measured by this policy."
        bridge.toolTipText="An absolute path to the reviewed bridge for your selected GPU backend; never downloaded automatically."
        findText.accessibleContext.accessibleName="Literal search in focused editor"
        bridge.accessibleContext.accessibleName="Reviewed GPU bridge executable path"
        memoryBudget.accessibleContext.accessibleName="CPU memory budget in bytes"
        val header=DesktopTheme.panel(BorderLayout(12,0),hero=true)
        header.border=EmptyBorder(4,8,4,8)
        val brand=JPanel(FlowLayout(FlowLayout.LEFT,8,0));brand.isOpaque=false
        val logo=DesktopTheme.badge("HL");logo.font=DesktopTheme.uiFont(14,true);logo.border=EmptyBorder(3,6,3,6)
        brand.add(logo);brand.add(DesktopTheme.label("HyperL",18,bold=true));brand.add(DesktopTheme.label("WORKBENCH",9,DesktopTheme.muted,true))
        header.add(brand,BorderLayout.WEST)
        val appearance=JPanel(FlowLayout(FlowLayout.RIGHT,8,0));appearance.isOpaque=false
        val glass=JCheckBox("Glass",DesktopTheme.glassEnabled);glass.isOpaque=false;glass.font=DesktopTheme.uiFont(11);glass.foreground=DesktopTheme.muted
        glass.toolTipText="Subtle static surface tint and highlights; turn off for solid panels."
        glass.addActionListener{DesktopTheme.glassEnabled=glass.isSelected;try{DesktopTheme.select(DesktopTheme.appearance)}catch(_:Exception){state.toolTipText="Glass preference could not be saved"};repaint()}
        state.border=BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(DesktopTheme.line),EmptyBorder(3,8,3,8))
        appearanceSelector.selectedItem=DesktopTheme.appearance;appearanceSelector.font=DesktopTheme.uiFont(11);appearanceSelector.toolTipText="Workbench appearance; saved for the next launch"
        appearanceSelector.addActionListener{try{DesktopTheme.select(appearanceSelector.selectedItem as DesktopAppearance);DesktopTheme.refresh(this);DesktopTheme.status(state,state.text)}catch(_:Exception){DesktopTheme.status(state,"FAILED");state.toolTipText="Appearance could not be saved"}}
        focusEditor.font=DesktopTheme.uiFont(11);focusEditor.toolTipText="Give the editor the output pane’s space; preserve results and runtime controls"
        appearance.add(appearanceSelector);appearance.add(focusEditor);appearance.add(glass);appearance.add(state);header.add(appearance,BorderLayout.EAST)
        val top=JPanel();top.layout=BoxLayout(top,BoxLayout.Y_AXIS);top.isOpaque=false;top.add(header);top.add(Box.createVerticalStrut(6))
        val snapshot=MemoryPlanner.observe()
        val environment=JPanel(GridLayout(1,4,8,0));environment.isOpaque=false
        environment.add(environmentCard("HOST", "${System.getProperty("os.name")} · ${System.getProperty("os.arch")}",DesktopTheme.foreground))
        environment.add(environmentCard("JVM HEAP LIMIT", "${snapshot.heapLimitBytes/(1024*1024)} MiB",DesktopTheme.foreground))
        environment.add(environmentCard("CPU RUNTIME", "Reference ready",DesktopTheme.accent))
        environment.add(environmentCard("GPU RUNTIME", "Probe required",DesktopTheme.violet))
        top.add(environment);add(top,BorderLayout.NORTH)

        val tools=JPanel(GridLayout(2,1,0,5));tools.isOpaque=false
        val fileTools=JPanel(FlowLayout(FlowLayout.LEFT,7,0));fileTools.isOpaque=false
        listOf(open,save,validate,format).forEach(fileTools::add)
        val exampleTools=JPanel(FlowLayout(FlowLayout.LEFT,7,0));exampleTools.isOpaque=false
        exampleTools.add(DesktopTheme.label("EXAMPLE",10,DesktopTheme.muted,true));exampleTools.add(examples);exampleTools.add(loadExample);exampleTools.add(analyze);exampleTools.add(review)
        tools.add(fileTools);tools.add(exampleTools)
        val editors=JTabbedPane();editors.addTab("Program",editorCard("KERNEL PROGRAM","hyperl/1 · JSON",program));editors.addTab("Inputs",editorCard("INPUT VECTORS","finite f32 · JSON",inputs))
        editors.minimumSize=Dimension(330,240)
        val left=JPanel(BorderLayout(0,6));left.isOpaque=false;left.minimumSize=Dimension(330,240);left.add(tools,BorderLayout.NORTH);left.add(editors)
        val search=JPanel(FlowLayout(FlowLayout.LEFT,7,0));search.isOpaque=false;search.add(DesktopTheme.label("Search",11,DesktopTheme.muted));search.add(findText);search.add(find);left.add(search,BorderLayout.SOUTH)
        val right=DesktopTheme.panel(BorderLayout(0,8));right.minimumSize=Dimension(320,240)
        val resultHeader=JPanel(BorderLayout());resultHeader.isOpaque=false
        val resultTitle=JPanel(BorderLayout(0,5));resultTitle.isOpaque=false
        resultTitle.add(DesktopTheme.label("Output & diagnostics",15,bold=true),BorderLayout.NORTH)
        resultTitle.add(DesktopTheme.label("RESULTS · SOURCE · MEMORY · ANALYSIS",9,DesktopTheme.muted),BorderLayout.SOUTH)
        resultHeader.add(resultTitle);resultHeader.add(export,BorderLayout.EAST);right.add(resultHeader,BorderLayout.NORTH);right.add(DesktopTheme.scroll(output))
        val split=object:JSplitPane(HORIZONTAL_SPLIT,left,right){
            private var initialLayout=true
            override fun doLayout(){
                if(initialLayout && width>=left.minimumSize.width+right.minimumSize.width+dividerSize){
                    dividerLocation=(width*0.70).toInt().coerceIn(left.minimumSize.width,width-right.minimumSize.width-dividerSize)
                    initialLayout=false
                }
                super.doLayout()
            }
        };var previousDivider=0
        focusEditor.addActionListener{if(focusEditor.isSelected){previousDivider=split.dividerLocation;right.isVisible=false;environment.isVisible=false;split.dividerSize=0;split.dividerLocation=split.width}else{right.isVisible=true;environment.isVisible=true;split.dividerSize=8;split.dividerLocation=previousDivider};revalidate();repaint()}
        split.resizeWeight=0.70;split.dividerSize=8;split.border=null;split.background=DesktopTheme.background;split.isOpaque=false
        add(split)

        val bottom=JPanel(BorderLayout(0,6));bottom.isOpaque=false
        val execution=DesktopTheme.panel(BorderLayout(0,7))
        val commands=JPanel(FlowLayout(FlowLayout.LEFT,8,0));commands.isOpaque=false
        commands.add(DesktopTheme.label("BACKEND",10,DesktopTheme.muted,true));commands.add(backend);commands.add(run);commands.add(stop)
        commands.add(DesktopTheme.label("Budget (bytes)",11,DesktopTheme.muted));commands.add(memoryBudget);commands.add(memory)
        execution.add(commands,BorderLayout.NORTH)
        val sources=DesktopTheme.panel(BorderLayout(0,7))
        val sourceCommands=JPanel(FlowLayout(FlowLayout.LEFT,8,0));sourceCommands.isOpaque=false
        sourceCommands.add(DesktopTheme.label("TARGET",10,DesktopTheme.muted,true));sourceCommands.add(sourceTarget);sourceCommands.add(emit)
        sources.add(sourceCommands,BorderLayout.NORTH)
        val gpuSetup=DesktopTheme.panel(BorderLayout(0,7))
        val gpu=JPanel(FlowLayout(FlowLayout.LEFT,8,0));gpu.isOpaque=false
        gpu.add(DesktopTheme.label("Bridge path",11,DesktopTheme.muted));gpu.add(bridge);gpu.add(DesktopTheme.label("Device",11,DesktopTheme.muted));gpu.add(device);gpu.add(probe)
        gpuSetup.add(gpu,BorderLayout.NORTH)
        executionTabs.addTab("Execution",execution);executionTabs.addTab("Source emission",sources);executionTabs.addTab("GPU setup",gpuSetup)
        bottom.add(executionTabs)
        val footer=JPanel(BorderLayout());footer.isOpaque=false
        val footerHint=DesktopTheme.label("Ctrl / Cmd + Enter to run · hyperl/1 · finite f32",10,DesktopTheme.muted)
        footer.add(footerHint)
        executionTabs.addChangeListener {footerHint.text=when(executionTabs.selectedIndex){
            1->"Source only · Compilation and device qualification remain separate"
            2->"Reviewed installed bridge · Probe establishes GPU availability"
            else->"Ctrl / Cmd + Enter to run · hyperl/1 · finite f32"
        }}
        footer.add(DesktopTheme.label("Full SDK & native profiler planned",10,DesktopTheme.muted),BorderLayout.EAST)
        bottom.add(footer,BorderLayout.SOUTH);add(bottom,BorderLayout.SOUTH)
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
                if(selected=="METAL_GPU"){
                    MemoryPlanner.requireAdmission(MemoryPlanner.plan(p,values,bytes));Workspace.json.encodeToString(MetalBridge(Path.of(binary)).executeVerified(index,p,values))
                }else{
                    val result=if(selected=="CPU_REFERENCE")HyperLCpuBackend(bytes).execute(p,values) else {MemoryPlanner.requireAdmission(MemoryPlanner.plan(p,values,bytes));OpenClBridge(Path.of(binary)).executeVerified(index,p,values)}
                    "Backend: $selected\nResult: ${Workspace.result(result)}\n\nGPU requests require CPU verification; no speed claim."
                }
            }
        }
        emit.addActionListener {val text=program.text;val target=HyperLTarget.valueOf(sourceTarget.selectedItem as String);start(SyntaxConstants.SYNTAX_STYLE_C){PortableEmitter.emit(Workspace.program(text),target).source}}
        memory.addActionListener {val text=program.text;val values=inputs.text;val budget=memoryBudget.text;start{Workspace.json.encodeToString(MemoryPlanner.plan(Workspace.program(text),Workspace.inputs(values),budget.toLong()))}}
        probe.addActionListener {val binary=bridge.text;val selected=backend.selectedItem as String;start{if(selected=="METAL_GPU")MetalBridge(Path.of(binary)).probe() else OpenClBridge(Path.of(binary)).probe()}}
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
        find.addActionListener {val term=findText.text;if(term.isEmpty()||term.length>256)return@addActionListener;val text=searchEditor.text;val start=searchEditor.selectionEnd;var index=text.indexOf(term,start);if(index<0)index=text.indexOf(term);if(index>=0){searchEditor.requestFocusInWindow();searchEditor.select(index,index+term.length);DesktopTheme.status(state,"MATCH FOUND")}else DesktopTheme.status(state,"NO MATCH")}
        loadExample.addActionListener {if(!allowDiscard())return@addActionListener;val example=DeveloperWorkspace.example(examples.selectedItem as String);program.text=pretty(example.program);inputs.text=pretty(example.inputs);dirty=false;state.text="EXAMPLE LOADED"}
        open.addActionListener {if(!allowDiscard())return@addActionListener;val file=choose(false,"Open HyperL workspace")?:return@addActionListener;start{val doc=DeveloperWorkspace.parse(Workspace.read(file));SwingUtilities.invokeAndWait{program.text=doc.program;inputs.text=doc.inputs;dirty=false};"Opened ${file.fileName}.\nNothing was executed."}}
        save.addActionListener {val file=choose(true,"Save HyperL workspace")?:return@addActionListener;val replace=Files.exists(file);if(replace&&!confirmReplace(file))return@addActionListener;val p=program.text;val i=inputs.text;start{DeveloperWorkspace.save(file,DeveloperWorkspace.encode(p,i),replace);SwingUtilities.invokeAndWait{dirty=false};"Saved ${file.fileName}."}}
        export.addActionListener {val file=choose(true,"Export displayed output")?:return@addActionListener;val replace=Files.exists(file);if(replace&&!confirmReplace(file))return@addActionListener;val text=output.text;start{DeveloperWorkspace.save(file,text,replace);"Exported displayed output to ${file.fileName}.\nNo source was compiled or executed."}}
        stop.addActionListener{job?.cancel();DesktopTheme.status(state,"STOPPING");output.text="Stop requested; awaiting local task/native cleanup. Remote termination is not implied."}
    }
    override fun paintComponent(g:Graphics){DesktopTheme.paintBackdrop(this,g)}
    private fun environmentCard(title:String,value:String,color:Color)=DesktopTheme.panel(BorderLayout(6,0)).also{
        it.border=EmptyBorder(4,8,4,8)
        it.add(DesktopTheme.label(title,9,DesktopTheme.muted,true),BorderLayout.WEST)
        it.add(DesktopTheme.label(value,11,color,true),BorderLayout.CENTER)
    }
    private fun editorCard(title:String,detail:String,editor:org.fife.ui.rsyntaxtextarea.RSyntaxTextArea)=DesktopTheme.panel(BorderLayout(0,8)).also{
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
        actions.forEach{it.isEnabled=false};configurationControls.forEach{it.isEnabled=false};program.isEditable=false;inputs.isEditable=false;stop.isEnabled=true;DesktopTheme.status(state,"RUNNING")
        job=scope.launch {
            var message="Stopped after local cleanup.";var status="STOPPED"
            try{message=action();status="COMPLETED"}catch(_:CancellationException){}catch(e:Exception){message="Failed: ${e.message?.take(1024)}";status="FAILED"}
            finally{val finalMessage=message;val finalStatus=status;SwingUtilities.invokeLater{output.syntaxEditingStyle=if(finalMessage.trimStart().startsWith("{"))SyntaxConstants.SYNTAX_STYLE_JSON else syntax;output.text=finalMessage;output.caretPosition=0;DesktopTheme.status(state,finalStatus);actions.forEach{it.isEnabled=true};configurationControls.forEach{it.isEnabled=true};program.isEditable=true;inputs.isEditable=true;stop.isEnabled=false}}
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
