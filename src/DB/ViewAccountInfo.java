/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DB;

import virtualatm_0.ViewAccountDetailsFrame;
import com.mysql.jdbc.Connection;
import com.mysql.jdbc.Statement;
import java.sql.DriverManager;
import java.sql.ResultSet;

/**
 *
 * @author welcome
 */
public class ViewAccountInfo
{
    public void getDataView()
    {
        try
        {
             Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection conn=(Connection) DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st1=(Statement) conn.createStatement();
            Statement st2=(Statement) conn.createStatement();

            String query="Select * from account_info ";

            ResultSet rs1=st1.executeQuery(query);
            ResultSet rs2=st2.executeQuery(query);
            
            int row=0;
            while(rs1.next())
            {
                row++;
            }
            int i=0;
            String data[][]=new String[row][4];
            while(rs2.next())
            {
                String accountno=rs2.getString(1);
                data[i][0]=accountno;
               
                String customername=rs2.getString(3);
                data[i][1]=customername;
                
                String mobile=rs2.getString(5);
                data[i][2]=mobile;
                
                String adharno=rs2.getString(7);
                data[i][3]=adharno;
                
                
                i++;
            }
            ViewAccountDetailsFrame.data1=data;
            conn.close();
            st1.close();
            st2.close();
        }
        catch(Exception ex)
        {
            System.out.println(ex);
        }
                
    }
}
