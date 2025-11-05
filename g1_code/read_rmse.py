import io
import os
import re
import shutil
import numpy as np
import pandas as pd

def read_rmse(path='/work/mse-minzw/lrz/g1/'):
    df_e = pd.DataFrame(index=np.arange(50),columns=np.arange(50))
    df_f = pd.DataFrame(index=np.arange(50),columns=np.arange(50))
    eta_list = [10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
    drs_list = [0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,
       0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,
       0.625,0.650,0.675,0.700,0.725,0.75]
    ii = 0
    for i in eta_list:
        jj = 0
        for j in drs_list:
            path_pl = path + '{}-{:0.3f}'.format(i, j) + '/pyamff.log'
            file = open(path_pl)
            out = file.readlines()
            rmse = out[-1]
            num_list = rmse.split()
            EnergyRMSE = float(num_list[-2])
            ForceRMSE = float(num_list[-1])
            df_e.iloc[ii, jj] = EnergyRMSE
            df_f.iloc[ii, jj] = ForceRMSE
            jj += 1
            print('t',i,'-',j)
        ii += 1
        
    with pd.ExcelWriter('./train_rmse1.xlsx') as writer1:
        df_e.to_excel(writer1, sheet_name='t_e')
        df_f.to_excel(writer1, sheet_name='t_f') 
                
    
    p_df_e = pd.DataFrame(index=np.arange(50),columns=np.arange(50))
    p_df_f = pd.DataFrame(index=np.arange(50),columns=np.arange(50)) 
    pii = 0
    for i in eta_list:
        pjj = 0
        for j in drs_list:            
            path_prl = path + '{}-{:0.3f}'.format(i, j) + '/pred.log'
            pfile = open(path_prl)
            out = pfile.readlines()
            rmse = out[-36]
            pattern = r'[\d.]+'
            number = re.findall(pattern, rmse)
            EnergyRMSE_str = number[0]
            ForceRMSE_str = number[1]
            EnergyRMSE = float(EnergyRMSE_str)
            ForceRMSE = float(ForceRMSE_str)
            p_df_e.iloc[pii, pjj] = EnergyRMSE
            p_df_f.iloc[pii, pjj] = ForceRMSE 
            pjj += 1
            print('p',i,'-',j)
        pii += 1    
        
    with pd.ExcelWriter('./pred_rmse1.xlsx') as writer2:
        p_df_e.to_excel(writer2, sheet_name='p_e')
        p_df_f.to_excel(writer2, sheet_name='p_f')        
        
read_rmse()

