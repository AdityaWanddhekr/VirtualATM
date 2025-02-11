/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DB;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;

/**
 *
 * @author welcome
 */
public class AccountNumberFetcher 
{
    public ArrayList getData()
    {
        ArrayList a1=new ArrayList();
       
        try
        {
            
            Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection con=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=con.createStatement();
            
            String query="select * from account_info" ;
            ResultSet rs=st.executeQuery(query);
            
           while(rs.next())
           {
                a1.add(rs.getString(1));
           }
          //  System.out.println("a1 is: "+a1);
           con.close();
        }
        catch(Exception e)
        {
            System.out.println(e);
           
        }
      
    return a1;
    }
}
