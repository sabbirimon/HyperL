package org.hyperl
import org.junit.Assert.*
import org.junit.Test
import javax.swing.SwingUtilities

class GuiTest {
    @Test fun realPanelButtonsEmitMetalAndExecuteCpu() {
        lateinit var panel:HyperLPanel
        SwingUtilities.invokeAndWait{panel=HyperLPanel();panel.sourceTarget.selectedItem="METAL";panel.emit.doClick()}
        try {
            await {panel.output.text.contains("kernel void hyperl_kernel") && panel.run.isEnabled}
            SwingUtilities.invokeAndWait{panel.run.doClick()}
            await{panel.output.text.contains("Backend: CPU_REFERENCE") && panel.run.isEnabled}
            SwingUtilities.invokeAndWait{assertTrue(panel.output.text.contains("12.0"));assertFalse(panel.output.isEditable)}
        }finally{SwingUtilities.invokeAndWait{panel.close()}}
    }
    private fun await(condition:()->Boolean){
        val deadline=System.nanoTime()+5_000_000_000L
        while(System.nanoTime()<deadline){var ready=false;SwingUtilities.invokeAndWait{ready=condition()};if(ready)return;Thread.sleep(20)}
        fail("GUI task did not finish within five seconds")
    }
}
