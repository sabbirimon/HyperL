package org.hyperl

import com.formdev.flatlaf.FlatDarkLaf
import org.fife.ui.rsyntaxtextarea.*
import org.fife.ui.rtextarea.RTextScrollPane
import java.awt.*
import java.awt.geom.RoundRectangle2D
import javax.swing.*
import javax.swing.border.EmptyBorder

/** Static vector surfaces and bundled fonts. No animation, network or kernel work. */
object DesktopTheme {
    val background=Color(0x090E17);val surface=Color(0x131C29);val line=Color(0x2A394D)
    val foreground=Color(0xE9EFF7);val muted=Color(0xA5B4CA)
    val accent=Color(0xC2EF87);val cyan=Color(0x72E0CE);val violet=Color(0xBBA7F8)
    val warning=Color(0xEAC388);val error=Color(0xFFACAF)
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
        if(UIManager.getLookAndFeel() !is FlatDarkLaf){check(FlatDarkLaf.setup()){ "Desktop theme unavailable"}}
        UIManager.put("defaultFont",uiFont())
        UIManager.put("Panel.background",background);UIManager.put("SplitPane.background",background)
        UIManager.put("TextField.background",Color(0x0E1622));UIManager.put("TextField.foreground",foreground)
        UIManager.put("ComboBox.background",Color(0x192536));UIManager.put("Label.foreground",foreground)
        UIManager.put("Component.borderColor",line);UIManager.put("Component.focusColor",cyan)
        UIManager.put("Component.focusWidth",1);UIManager.put("Component.arc",9)
        UIManager.put("Button.arc",9);UIManager.put("Button.background",Color(0x1C2B3D))
        UIManager.put("Button.hoverBackground",Color(0x2B4056));UIManager.put("Button.pressedBackground",Color(0x314B60))
        UIManager.put("TabbedPane.selectedBackground",surface);UIManager.put("TabbedPane.background",background)
        UIManager.put("TabbedPane.underlineColor",cyan);UIManager.put("TabbedPane.tabHeight",32)
        UIManager.put("ScrollBar.width",10);UIManager.put("ScrollBar.thumbArc",10)
    }
    fun label(text:String,size:Int=13,color:Color=foreground,bold:Boolean=false)=JLabel(text).also{
        it.font=uiFont(size,bold);it.foreground=color
    }
    fun panel(layout:LayoutManager=BorderLayout(),hero:Boolean=false)=object:JPanel(layout){
        override fun paintComponent(g:Graphics){
            val p=g.create() as Graphics2D
            try {
                p.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON)
                val shape=RoundRectangle2D.Float(0.5f,0.5f,width-1f,height-1f,20f,20f)
                p.paint=GradientPaint(0f,0f,if(hero)Color(0x183332)else Color(0x172333),width.toFloat(),height.toFloat(),if(hero)Color(0x1A1C32)else Color(0x101925))
                p.fill(shape);p.color=if(hero)Color(0x35504F)else line;p.draw(shape)
                if(hero){p.color=Color(0x72,0xE0,0xCE,105);p.drawLine(24,1,width/3,1)}
            }finally{p.dispose()}
        }
    }.also{it.isOpaque=false;it.background=surface;it.border=EmptyBorder(14,16,14,16)}

    fun paintBackdrop(component:JComponent,g:Graphics){
        val p=g.create() as Graphics2D
        try {
            val w=component.width.toFloat().coerceAtLeast(1f);val h=component.height.toFloat().coerceAtLeast(1f)
            p.paint=GradientPaint(0f,0f,Color(0x0C1820),w,h,Color(0x080C14));p.fillRect(0,0,component.width,component.height)
            p.paint=RadialGradientPaint(0f,0f,w*0.70f,floatArrayOf(0f,1f),arrayOf(Color(0x28,0x82,0x71,62),Color(0x28,0x82,0x71,0)))
            p.fillRect(0,0,component.width,component.height)
            p.paint=RadialGradientPaint(w,h*0.2f,w*0.65f,floatArrayOf(0f,1f),arrayOf(Color(0x61,0x45,0xA0,52),Color(0x61,0x45,0xA0,0)))
            p.fillRect(0,0,component.width,component.height)
        }finally{p.dispose()}
    }
    fun badge(text:String,color:Color=cyan)=label(text,11,color,true).also{
        it.border=BorderFactory.createCompoundBorder(BorderFactory.createLineBorder(Color(color.red,color.green,color.blue,65),1,true),EmptyBorder(6,10,6,10))
        it.isOpaque=true;it.background=Color(0x13222C)
    }
    fun status(label:JLabel,text:String){
        label.text=text
        label.foreground=when(text){"FAILED"->error;"STOPPING","STOPPED","NO MATCH"->warning;"COMPLETED","MATCH FOUND"->accent;else->cyan}
    }
    fun editor(text:String,editable:Boolean=true)=RSyntaxTextArea().also{
        it.text=text;it.isEditable=editable;it.syntaxEditingStyle=SyntaxConstants.SYNTAX_STYLE_JSON
        it.isCodeFoldingEnabled=true;it.font=codeFont()
        it.background=Color(0x0C131F);it.foreground=foreground;it.caretColor=cyan
        it.currentLineHighlightColor=Color(0x192A3A);it.selectionColor=Color(0x315064)
        it.margin=Insets(10,12,10,12);it.setAnimateBracketMatching(false)
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
        it.gutter.background=Color(0x0C131F);it.gutter.lineNumberColor=muted
        it.gutter.lineNumberFont=codeFont(12);it.gutter.borderColor=line
    }
    fun button(button:JButton,primary:Boolean=false){
        button.font=uiFont(12,true);button.margin=Insets(7,12,7,12)
        button.background=if(primary)accent else Color(0x1C2B3D);button.foreground=if(primary)background else foreground
        button.putClientProperty("JButton.buttonType","roundRect")
        button.accessibleContext.accessibleDescription=button.text
        if(primary)button.putClientProperty("FlatLaf.style","hoverBackground: #D1FAA4; pressedBackground: #A5D671")
    }
}
