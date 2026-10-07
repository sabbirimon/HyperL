package org.hyperl
import org.junit.Assert.*
import org.junit.Test
import javax.swing.SwingUtilities

class GuiTest {
    @Test fun realPanelButtonsEmitMetalAndExecuteCpu() {
        lateinit var panel:HyperLPanel
        SwingUtilities.invokeAndWait{panel=HyperLPanel();panel.sourceTarget.selectedItem="METAL";panel.emit.doClick();assertFalse(panel.backend.isEnabled)}
        try {
            await {panel.output.text.contains("kernel void hyperl_kernel") && panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.run.doClick()}
            await{panel.output.text.contains("Backend: CPU_REFERENCE") && panel.run.isEnabled}
            SwingUtilities.invokeAndWait{assertTrue(panel.output.text.contains("12.0"));assertFalse(panel.output.isEditable)}
        }finally{SwingUtilities.invokeAndWait{panel.close()}}
    }
    @Test fun realDeveloperToolsValidateFormatMemoryAndFind(){
        lateinit var panel:HyperLPanel
        SwingUtilities.invokeAndWait{panel=HyperLPanel();panel.validate.doClick()}
        try{await{panel.output.text.contains("Valid hyperl/1 workspace")&&panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.format.doClick()};await{panel.output.text.contains("JSON formatted")&&panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.memory.doClick()};await{panel.output.text.contains("estimatedArrayAndWorkspaceBytes")&&panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.analyze.doClick()};await{panel.output.text.contains("completions")&&panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.findText.text="multiply";panel.find.doClick();assertEquals("multiply",panel.program.selectedText);assertFalse(panel.stop.isEnabled)}
        }finally{SwingUtilities.invokeAndWait{panel.close()}}
    }
    private fun await(condition:()->Boolean){
        val deadline=System.nanoTime()+5_000_000_000L
        while(System.nanoTime()<deadline){var ready=false;SwingUtilities.invokeAndWait{ready=condition()};if(ready)return;Thread.sleep(20)}
        fail("GUI task did not finish within five seconds")
    }
}
