#3.合并一个eta,所有drs里的所有fps
import os
import time
import numpy as np
import pandas as pd

print("Program Initiated.")
code_time_s = time.perf_counter()

eta = [10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
drs = [0.075,0.100,0.125,0.150,0.175,0.200,0.225,0.250,0.275,0.300,0.325,0.350,
       0.375,0.400,0.425,0.450,0.475,0.500,0.525,0.550,0.575,0.600,0.625,0.650,
       0.675,0.700,0.725,0.75]

for e in eta:
    print('------------------------eta-{}------------------------'.format(e))
    name_all = []
    df_all = pd.DataFrame()
    for d in drs:
        pa_ = '/work/mse-minzw/lrz/g1/{}-{:0.3f}/fps/fps_csv/fps_all_nor.csv'.format(e,d)
        fps_e_di = pd.read_csv(pa_)
        fps_e_di_ = fps_e_di.iloc[:,1:]
        name_num = fps_e_di_.shape[1]
        df_all = pd.concat([df_all,fps_e_di_],axis=1)
        print('Done-{}-{:0.3f}'.format(e,d))
    df_all.to_csv('./eta-{}-all_fps.csv'.format(e))
    print('Done---------eta = {}'.format(e))
        

code_time_e = time.perf_counter()
print(
    'Program execution completed successfully.Code-Time is {} Sec.'.format(np.round(code_time_e-code_time_s, 2)))



