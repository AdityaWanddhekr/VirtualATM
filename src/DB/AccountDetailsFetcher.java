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
public class AccountDetailsFetcher
{
    public ArrayList getDetails(String acc_no)
    {

     
        ArrayList ar= new ArrayList();
        try
        {
            Class.forName("com.mysql.jdbc.Driver");
            Connection conn=DriverManager.getConnection("jdbc:mysql://localhost:3306/virtualatm","root","root");
            Statement st=conn.createStatement();

            String query="Select * from account_info where account_no='"+acc_no+"'";
            ResultSet rs=st.executeQuery(query);
            
            if(rs.next())
            {

                ar.add(rs.getString(1));
                ar.add(rs.getString(2));
                ar.add(rs.getString(3));
                ar.add(rs.getString(4));
                ar.add(rs.getString(5));
                ar.add(rs.getString(6));
                ar.add(rs.getString(7));
                ar.add(rs.getString(8));
            }
            conn.close();
            System.out.println("ar is  "+ar);
    
        }
         
            
        catch(Exception e)
        {
            System.out.println(e);
            
        }
       
      return ar;
    }
}
