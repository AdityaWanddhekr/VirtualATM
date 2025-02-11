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
public class UpdateAccount
{
    public boolean doUpdate(String acc_no,String acc_type,String name,String address,String mobile,String email,String adhar_no,String pan_no)
    {

        boolean flag=true;
        try
        {
           Class.forName("com.mysql.jdbc.Driver").newInstance();
            Connection conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=conn.createStatement();
            String query = "update  account_info set account_type='"+acc_type+"',customer_name='"+name+"',  address='"+address+"',mobile_number='"+mobile+"',email_id='"+email+"',adhar_no='"+adhar_no+"',pan_no='"+pan_no+"' where account_no='"+acc_no+"'";
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
