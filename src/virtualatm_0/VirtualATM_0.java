/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package virtualatm_0;

import java.awt.Dimension;
import java.awt.Toolkit;

/**
 *
 * @author welcome
 */
public class VirtualATM_0 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        LoginFrame lf=new LoginFrame();
        Dimension d=Toolkit.getDefaultToolkit().getScreenSize();
        lf.setVisible(true);
        lf.setSize(d);
    }
    
}
