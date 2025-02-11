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
public class NewAccountCreation 
{
    public boolean doCreateAccount(String acc_no, String acc_type ,String name,String address,String mobile,String email,String adhar_no,String pan_no,String initial_amt )
    {

        boolean flag=true;
     
        
        try
        {
            Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=conn.createStatement();
            String query = "insert into account_info values ('"+acc_no+"','"+acc_type+"','"+name+"','"+address+"','"+mobile+"','"+email+"','"+adhar_no+"','"+pan_no+"','"+initial_amt+"')";
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
            
        }

        return flag;
    }
}
