#合并一个eta-drs里的所有fps
import os
import numpy as np
import pandas as pd

print("Program Initiated.")
code_time_s = time.perf_counter()

def com_fps(path_csv,img_num=1225):
    os.chdir(path_csv)
    df_all_ori = pd.DataFrame()
    df_all_nor = pd.DataFrame()
    for i in range(img_num):
        img_i_ori = path_csv + '/Ge_fps_{}_ori.csv'.format(i)
        fps_i_ori = pd.read_csv(img_i_ori)
        fps_i_ori = fps_i_ori.iloc[:,:-15]
        df_all_ori = pd.concat([df_all_ori,fps_i_ori],axis=0)
        
        img_i_nor = path_csv + '/Ge_fps_{}_nor.csv'.format(i)
        fps_i_nor = pd.read_csv(img_i_nor)
        fps_i_nor = fps_i_nor.iloc[:,:-15]
        df_all_nor = pd.concat([df_all_nor,fps_i_nor],axis=0)
    df_all_nor.to_csv('./fps_all_nor.csv')
    df_all_ori.to_csv('./fps_all_ori.csv')
    return None


eta = [10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
drs = [0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,0.375,0.400,
       0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,0.625,0.650,0.675,0.700,0.725,0.750]

for e in eta:
    for d in drs:
        pa_ = 'cd /work/mse-minzw/lrz/g1/{}-{:0.3f}/fps/fps_csv'.format(e,d)
        com_fps(path_csv=pa_,img_num=1225)
        print('Done   {}-{:0.3f}'.format(e,d))
code_time_e = time.perf_counter()
print('Program execution completed successfully.Code-Time is {} Sec.'.format(np.round(code_time_e-code_time_s, 2)))