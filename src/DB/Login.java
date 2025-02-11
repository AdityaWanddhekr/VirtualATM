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

/**
 *
 * @author welcome
 */
public class Login 
{
     public boolean doLogin(String username,String password)
    {
        boolean flag=true;
        try
        {
            Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=conn.createStatement();

            String query="Select * from registration_info  where user_name='"+username+"'and password='"+password+"'";
            ResultSet rs=st.executeQuery(query);
       
            if(rs.next())
            {
                flag=true;
            }
            else
            {
                flag=false;
            }
            
            conn.close();
        }
        catch(Exception e)
        {
            System.out.println(e);
           
        }

        return flag;

    }
}
