/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package virtualatm_0;

import DB.AccountNumberFetcher;
import java.util.ArrayList;

/**
 *
 * @author welcome
 */
public class AccountNumber
{
    public int getAccountNumber()
    {
        int assgno=0;
        ArrayList accountNum=new AccountNumberFetcher().getData();
       // System.out.println(SerialNum);
        
        
        if(accountNum.isEmpty())
        {
            assgno=101;

        }
        else
        {
            String big=(String)accountNum.get(0);
                 int bg=Integer.parseInt(big);
            for (int i = 0; i < accountNum.size(); i++) 
            {
                
                 
                 String sr=(String)accountNum.get(i);
                 int s=Integer.parseInt(sr);
                 if(s>bg)
                 {
                     bg=s;
                 }
                assgno=bg+1;
            }

        }
        accountNum.add(assgno);
      
        return assgno;
    }
}
