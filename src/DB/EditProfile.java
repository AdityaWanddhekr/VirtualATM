/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DB;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

/**
 *
 * @author welcome
 */
public class EditProfile 
{
    public boolean doUpdate(String empid,String name,String email,String mobile,String uname,String pass)
    {

        boolean flag=true;
        try
        {
           Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=conn.createStatement();
            String query = "update  registration_info set emp_id='"+empid+"',emp_name='"+name+"',  email_id='"+email+"',mobile_number='"+mobile+"',password='"+pass+"'where user_name='"+uname+"'";
            int x=st.executeUpdate(query);
            if(x>0)
                flag=true;
            else
                flag=false;

            conn.close();

        }
        catch(Exception ex)
        {
            System.out.println(ex);
            flag=false;
        }

        return flag;
    }
}
