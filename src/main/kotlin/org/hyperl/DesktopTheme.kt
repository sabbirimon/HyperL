package org.hyperl
import com.formdev.flatlaf.FlatDarkLaf
import org.fife.ui.rsyntaxtextarea.*
import org.fife.ui.rtextarea.RTextScrollPane
import java.awt.*
import javax.swing.*
import javax.swing.border.EmptyBorder

object DesktopTheme {
    val background=Color(0x0B1119);val surface=Color(0x131D29);val line=Color(0x283849)
    val foreground=Color(0xE6EDF5);val muted=Color(0xA2B2C4);val accent=Color(0xB9F36B);val cyan=Color(0x68DADB)
    fun install(){
        if(UIManager.getLookAndFeel() !is FlatDarkLaf){check(FlatDarkLaf.setup()){ "Desktop theme unavailable"}}
        UIManager.put("Panel.background",background);UIManager.put("SplitPane.background",background)
        UIManager.put("TextField.background",surface);UIManager.put("ComboBox.background",surface);UIManager.put("Label.foreground",foreground)
        UIManager.put("Component.focusColor",cyan);UIManager.put("Button.arc",10)
        UIManager.put("Component.arc",10);UIManager.put("TabbedPane.selectedBackground",surface);UIManager.put("TabbedPane.background",background)
        UIManager.put("defaultFont",Font(Font.SANS_SERIF,Font.PLAIN,13))
    }
    fun label(text:String,size:Int=13,color:Color=foreground)=JLabel(text).also{it.font=Font(Font.SANS_SERIF,Font.PLAIN,size);it.foreground=color}
    fun panel(layout:LayoutManager=BorderLayout())=JPanel(layout).also{it.isOpaque=true;it.background=surface;it.border=EmptyBorder(16,18,16,18)}
    fun editor(text:String,editable:Boolean=true)=RSyntaxTextArea().also{
        it.text=text;it.isEditable=editable;it.syntaxEditingStyle=SyntaxConstants.SYNTAX_STYLE_JSON
        it.isCodeFoldingEnabled=true;it.font=Font(Font.MONOSPACED,Font.PLAIN,13)
        it.background=Color(0x0E1722);it.foreground=foreground;it.caretColor=cyan
        it.currentLineHighlightColor=Color(0x1C2A38);it.selectionColor=Color(0x2B5363)
        it.margin=Insets(10,12,10,12);it.setAnimateBracketMatching(false)
        val scheme=it.syntaxScheme
        for(index in 0 until scheme.styleCount){scheme.getStyle(index)?.foreground=foreground}
        scheme.getStyle(Token.LITERAL_STRING_DOUBLE_QUOTE).foreground=cyan
        scheme.getStyle(Token.LITERAL_NUMBER_DECIMAL_INT).foreground=accent
        scheme.getStyle(Token.LITERAL_NUMBER_FLOAT).foreground=accent
        scheme.getStyle(Token.RESERVED_WORD).foreground=Color(0xC7A6FF)
    }
    fun scroll(editor:RSyntaxTextArea)=RTextScrollPane(editor).also{
        it.border=BorderFactory.createLineBorder(line);it.setLineNumbersEnabled(true)
        it.gutter.background=Color(0x0E1722);it.gutter.lineNumberColor=muted
        it.gutter.borderColor=line
    }
    fun button(button:JButton,primary:Boolean=false){
        button.font=Font(Font.SANS_SERIF,Font.BOLD,12);button.margin=Insets(9,14,9,14)
        button.background=if(primary)accent else Color(0x203142);button.foreground=if(primary)background else foreground
        button.putClientProperty("JButton.buttonType","roundRect")
        button.accessibleContext.accessibleDescription=button.text
    }
}
