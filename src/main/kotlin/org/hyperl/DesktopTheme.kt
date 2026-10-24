package org.hyperl

import com.formdev.flatlaf.FlatDarkLaf
import com.formdev.flatlaf.FlatLightLaf
import java.nio.file.Files
import java.nio.file.Path
import java.util.Properties
import org.fife.ui.rsyntaxtextarea.*
import org.fife.ui.rtextarea.RTextScrollPane
import java.awt.*
import java.awt.geom.RoundRectangle2D
import javax.swing.*
import javax.swing.border.EmptyBorder

enum class DesktopAppearance(val label:String) { GRAPHITE("Graphite"),AURORA("Aurora"),PAPER("Paper"); override fun toString()=label }

/** Static vector surfaces and bundled fonts. No animation, network or kernel work. */
object DesktopTheme {
    var appearance=DesktopAppearance.AURORA;private set
    val light get()=appearance==DesktopAppearance.PAPER
    val background get()=Color(if(light)0xF4F6F9 else if(appearance==DesktopAppearance.GRAPHITE)0x08090B else 0x070B12)
    val surface get()=Color(if(light)0xFFFFFF else if(appearance==DesktopAppearance.GRAPHITE)0x15171B else 0x101824)
    val line get()=Color(if(light)0xCCD4E0 else if(appearance==DesktopAppearance.GRAPHITE)0x35383F else 0x253346)
    val foreground get()=Color(if(light)0x17202C else 0xE9EFF7)
    val muted get()=Color(if(light)0x536176 else 0xA5B4CA)
    val accent get()=Color(if(light)0x197144 else if(appearance==DesktopAppearance.GRAPHITE)0x83E5C8 else 0xC2EF87)
    val cyan get()=Color(if(light)0x087D78 else 0x72E0CE)
    val violet get()=Color(if(light)0x7445A5 else 0xBBA7F8)
    val warning get()=Color(if(light)0x924500 else 0xEAC388)
    val error get()=Color(if(light)0xAD2639 else 0xFFACAF)
    var glassEnabled=System.getProperty("hyperl.ui.glass","true").toBoolean()
    private fun settingsFile()=Path.of(System.getProperty("hyperl.ui.settings",Path.of(System.getProperty("user.home"),".hyperl","ui.properties").toString()))
    private var loaded=false
    fun select(value:DesktopAppearance,persist:Boolean=true){
        appearance=value;loaded=true;install()
        if(persist){val file=settingsFile();Files.createDirectories(file.parent);val props=Properties();props.setProperty("appearance",value.name);props.setProperty("glass",glassEnabled.toString());Files.newOutputStream(file).use{props.store(it,"HyperL visual preferences")}}
    }
    private val regular=loadFont("Inter-Regular.otf",Font.SANS_SERIF)
    private val semibold=loadFont("Inter-SemiBold.otf",Font.SANS_SERIF)
    private val mono=loadFont("JetBrainsMono-Regular.ttf",Font.MONOSPACED)

    private fun loadFont(name:String,fallback:String):Font {
        return try {
            DesktopTheme::class.java.getResourceAsStream("/org/hyperl/fonts/$name")?.use {
                Font.createFont(Font.TRUETYPE_FONT,it).also { font -> GraphicsEnvironment.getLocalGraphicsEnvironment().registerFont(font) }
            } ?: Font(fallback,Font.PLAIN,13)
        } catch(_:java.io.IOException){Font(fallback,Font.PLAIN,13)}
          catch(_:FontFormatException){Font(fallback,Font.PLAIN,13)}
    }
    fun uiFont(size:Int=13,bold:Boolean=false)=(if(bold)semibold else regular).deriveFont(size.toFloat())
    fun codeFont(size:Int=14)=mono.deriveFont(size.toFloat())
    fun install(){
        if(!loaded){loaded=true;runCatching {val file=settingsFile();if(Files.exists(file) && Files.size(file)<=4096){val props=Properties();Files.newInputStream(file).use{props.load(it)};appearance=DesktopAppearance.entries.firstOrNull{it.name==props.getProperty("appearance")} ?: DesktopAppearance.AURORA;glassEnabled=props.getProperty("glass",glassEnabled.toString()).toBoolean()}}}
        if(light){if(UIManager.getLookAndFeel() !is FlatLightLaf)check(FlatLightLaf.setup()){ "Desktop theme unavailable"}}
        else if(UIManager.getLookAndFeel() !is FlatDarkLaf){check(FlatDarkLaf.setup()){ "Desktop theme unavailable"}}
        UIManager.put("defaultFont",uiFont())
        UIManager.put("Panel.background",background);UIManager.put("SplitPane.background",background)
        UIManager.put("TextField.background",surface);UIManager.put("TextField.foreground",foreground)
        UIManager.put("ComboBox.background",surface);UIManager.put("Label.foreground",foreground)
        UIManager.put("Component.borderColor",line);UIManager.put("Component.focusColor",cyan)
        UIManager.put("Component.focusWidth",1);UIManager.put("Component.arc",9)
        UIManager.put("Button.arc",9);UIManager.put("Button.background",surface)
        UIManager.put("Button.hoverBackground",line);UIManager.put("Button.pressedBackground",line)
        UIManager.put("TabbedPane.selectedBackground",surface);UIManager.put("TabbedPane.background",background)
        UIManager.put("TabbedPane.underlineColor",cyan);UIManager.put("TabbedPane.tabHeight",32)
        UIManager.put("ScrollBar.width",10);UIManager.put("ScrollBar.thumbArc",10)
    }
    fun label(text:String,size:Int=13,color:Color=foreground,bold:Boolean=false)=JLabel(text).also{
        it.font=uiFont(size,bold);it.foreground=color
        it.putClientProperty("hyperl.tint",when(color){muted->"muted";accent->"accent";cyan->"cyan";violet->"violet";warning->"warning";error->"error";else->"foreground"})
    }
    fun panel(layout:LayoutManager=BorderLayout(),hero:Boolean=false)=object:JPanel(layout){
        override fun paintComponent(g:Graphics){
            val p=g.create() as Graphics2D
            try {
                p.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON)
                val shape=RoundRectangle2D.Float(0.5f,0.5f,width-1f,height-1f,20f,20f)
                val top=if(light)Color(0xFFFFFF) else if(appearance==DesktopAppearance.GRAPHITE)Color(0x1D2026) else if(hero)Color(0x112624)else Color(0x121C29)
                val bottom=if(light)Color(0xEEF2F8) else if(appearance==DesktopAppearance.GRAPHITE)Color(0x101216) else if(hero)Color(0x151629)else Color(0x0B121C)
                p.paint=GradientPaint(0f,0f,if(glassEnabled)Color(top.red,top.green,top.blue,225)else top,width.toFloat(),height.toFloat(),if(glassEnabled)Color(bottom.red,bottom.green,bottom.blue,244)else bottom)
                p.fill(shape)
                if(glassEnabled){
                    // A static sheen and translucent tint; no backdrop capture, blur or repaint timer.
                    val clipped=p.create() as Graphics2D
                    try {
                        clipped.clip(shape)
                        clipped.paint=GradientPaint(0f,0f,Color(215,238,255,if(hero)19 else 12),0f,65f,Color(215,238,255,0))
                        clipped.fillRect(0,0,width,65)
                        clipped.paint=GradientPaint(0f,0f,Color(205,239,244,65),width.toFloat(),height.toFloat(),Color(160,174,211,14))
                        clipped.draw(shape)
                    }finally{clipped.dispose()}
                }else{p.color=if(hero)Color(0x2A4140)else line;p.draw(shape)}
                if(hero){p.color=Color(0x72,0xE0,0xCE,105);p.drawLine(24,1,width/3,1)}
            }finally{p.dispose()}
        }
    }.also{it.isOpaque=false;it.background=surface;it.border=EmptyBorder(10,10,10,10)}

    fun paintBackdrop(component:JComponent,g:Graphics){
        val p=g.create() as Graphics2D
        try {
            val w=component.width.toFloat().coerceAtLeast(1f);val h=component.height.toFloat().coerceAtLeast(1f)
            p.paint=GradientPaint(0f,0f,background,w,h,if(light) surface else background.darker());p.fillRect(0,0,component.width,component.height)
            p.paint=RadialGradientPaint(0f,0f,w*0.70f,floatArrayOf(0f,1f),arrayOf(Color(0x28,0x82,0x71,38),Color(0x28,0x82,0x71,0)))
            p.fillRect(0,0,component.width,component.height)
            p.paint=RadialGradientPaint(w,h*0.2f,w*0.65f,floatArrayOf(0f,1f),arrayOf(Color(0x61,0x45,0xA0,35),Color(0x61,0x45,0xA0,0)))
            p.fillRect(0,0,component.width,component.height)
        }finally{p.dispose()}
    }
    fun badge(text:String,color:Color=cyan)=label(text,11,color,true).also{
        it.border=BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(Color(color.red,color.green,color.blue,65),1,true),EmptyBorder(6,10,6,10))
        it.isOpaque=true;it.background=surface
    }
    fun status(label:JLabel,text:String){
        label.text=text
        label.foreground=when(text){"FAILED"->error;"STOPPING","STOPPED","NO MATCH"->warning;"COMPLETED","MATCH FOUND"->accent;else->cyan}
    }
    fun editor(text:String,editable:Boolean=true)=RSyntaxTextArea().also{
        it.text=text;it.isEditable=editable;it.syntaxEditingStyle=SyntaxConstants.SYNTAX_STYLE_JSON
        it.isCodeFoldingEnabled=true;it.font=codeFont()
        it.background=background;it.foreground=foreground;it.caretColor=cyan
        it.currentLineHighlightColor=if(light)Color(0xE9EEF5) else surface;it.selectionColor=if(light)Color(0xD4E6F0) else Color(0x315064)
        it.margin=Insets(6,8,6,8);it.setAnimateBracketMatching(false)
        val scheme=it.syntaxScheme
        for(index in 0 until scheme.styleCount){scheme.getStyle(index)?.foreground=foreground}
        scheme.getStyle(Token.LITERAL_STRING_DOUBLE_QUOTE).foreground=cyan
        scheme.getStyle(Token.LITERAL_NUMBER_DECIMAL_INT).foreground=warning
        scheme.getStyle(Token.LITERAL_NUMBER_FLOAT).foreground=warning
        scheme.getStyle(Token.RESERVED_WORD).foreground=violet
        scheme.getStyle(Token.COMMENT_EOL).foreground=muted
    }
    fun scroll(editor:RSyntaxTextArea)=RTextScrollPane(editor).also{
        it.border=BorderFactory.createLineBorder(line);it.setLineNumbersEnabled(true)
        it.gutter.background=background;it.gutter.lineNumberColor=muted
        it.gutter.lineNumberFont=codeFont(12);it.gutter.borderColor=line
    }
    fun button(button:JButton,primary:Boolean=false){
        button.putClientProperty("hyperl.primary",primary)
        button.font=uiFont(12,true);button.margin=Insets(5,10,5,10)
        button.background=if(primary)accent else surface;button.foreground=if(primary)background else foreground
        button.putClientProperty("JButton.buttonType","roundRect")
        button.accessibleContext.accessibleDescription=button.text
        if(primary)button.putClientProperty("FlatLaf.style","hoverBackground: #${Integer.toHexString(accent.brighter().rgb).takeLast(6)}; pressedBackground: #${Integer.toHexString(accent.darker().rgb).takeLast(6)}")
    }
    fun refresh(root:JComponent){
        SwingUtilities.updateComponentTreeUI(root)
        fun visit(component:Component){
            when(component){
                is RSyntaxTextArea->{component.background=background;component.foreground=foreground;component.caretColor=cyan;component.currentLineHighlightColor=if(light)Color(0xE9EEF5) else surface;component.selectionColor=if(light)Color(0xD4E6F0) else Color(0x315064)
                    val scheme=component.syntaxScheme;for(index in 0 until scheme.styleCount)scheme.getStyle(index)?.foreground=foreground
                    scheme.getStyle(Token.LITERAL_STRING_DOUBLE_QUOTE).foreground=cyan;scheme.getStyle(Token.RESERVED_WORD).foreground=violet;scheme.getStyle(Token.LITERAL_NUMBER_DECIMAL_INT).foreground=warning;scheme.getStyle(Token.LITERAL_NUMBER_FLOAT).foreground=warning;scheme.getStyle(Token.COMMENT_EOL).foreground=muted}
                is RTextScrollPane->{component.border=BorderFactory.createLineBorder(line);component.gutter.background=background;component.gutter.lineNumberColor=muted;component.gutter.borderColor=line}
                is JLabel->component.foreground=when(component.getClientProperty("hyperl.tint")){"muted"->muted;"accent"->accent;"cyan"->cyan;"violet"->violet;"warning"->warning;"error"->error;else->foreground}
                is JButton->button(component,component.getClientProperty("hyperl.primary")==true)
                is JTextField->{component.background=surface;component.foreground=foreground;component.caretColor=cyan}
                is JPanel->component.background=background
            }
            if(component is Container)component.components.forEach(::visit)
        }
        visit(root);root.revalidate();root.repaint()
    }

}
