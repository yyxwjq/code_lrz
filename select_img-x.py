# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------
[File-Information:]
    -. Created on 2023/10/27
    -. Updated on 2024/03/15
    -. Developer: Renzhe Li
    -. Email: 1135116956@qq.com
    -. Development Team: SUSTech Materials Department, Renzhe Li, Chuan Zhou, Lei Li*
-------------------------------------------------------------------------------------
[Description:]
1.The Fingerprints file calculated using PyAMFF is too large and converted to .csv. Taking the PdH system as an example,
 the i-th structure will convert fps_i.pckl into Pd_fps_i.csv and H_fps_i.csv.
2.Currently the code can only solve single-element, two-element and three-element systems.
3.For a system, the types of elements in each structure must be the same.
  For example, structure I contains two elements（A and B）, and structure J also contains both elements（A and B）.
4.The selected structure is saved as train.traj, and the remaining structures are saved as test.traj.
  That is, the original-data-set = train.traj + test.traj
5. The user must enter parameters in the # Parameter setting section. T_i controls the filtering rate, and fps_path_ and
   img_path_ are the paths of fingerprints and the original-data-set.f_cp_ is the path to generate the csv file. 
   When the program runs for the first time, the three parameters fps_path_, img_path_ and f_cp_ do not need to be changed.
6.The fingerprint of the selected local-chemical-environment will be stored as a .csv file. Named bank-{symbol}-fps.csv
7.t_i can be tried starting from 0.1 (or 0.2, 0.3,...,1.0,...).
  If you want to reduce the screening rate, increase ti, otherwise decrease it.
-------------------------------------------------------------------------------------
"""


# 0.Parameter setting
t_1 = 0.9
t_2 = 0.6
t_3 = 0.5
_path_all_ = '/scratch/2025-08-06/mse-yulk/J/opt-xifu'

traj_symbol_ = None
f_cp_ = _path_all_ + '/fps_csv'
img_path_ = _path_all_ + '/train.traj'
fps_path_ = _path_all_ + '/fingerprints'


# 1.Define functions
print("Program-Initiated.")
import os
import ase
import csv
import time
import torch
import pickle
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory


# 2.Get the elements of a structure.
code_time_s = time.perf_counter()
def get_symbols(image_path):
    s = []
    img = Trajectory(image_path)
    symbols_all = img[0].get_chemical_symbols()
    symbols_all_set = list(set(symbols_all))
    symbols_all_set.sort()
    if len(symbols_all_set) == 1:
        s.append(symbols_all_set[0])
    elif len(symbols_all_set) == 2:
        s.append(symbols_all_set[0])
        s.append(symbols_all_set[1])
    elif len(symbols_all_set) == 3:
        s.append(symbols_all_set[0])
        s.append(symbols_all_set[1])
        s.append(symbols_all_set[2])
    return s


# 3.xxx.pckl --> xxx.csv
def fps_nor1_to_csv(fps_path, img_path, symbol, fps_csvpath):
    img_all = Trajectory(img_path)
    # fprange.pckl --> .csv
    print('Start converting | fprange.pckl --> .csv')
    os.makedirs(fps_csvpath)
    fprange_path = os.path.join(fps_path, 'fprange.pckl')
    with open(fprange_path, 'rb') as f:
        fp_all = pickle.load(f)
        for smb in symbol:
            fp_ = fp_all[smb]
            min_ = fp_[0]
            max_min_ = fp_[2]
            min_np = min_.numpy()
            max_min_np = max_min_.numpy()
            npi = np.concatenate((min_np, max_min_np))
            npi_ = npi.reshape(2, -1)
            dfi = pd.DataFrame(npi_)
            dfi.to_csv(fps_csvpath + '/fpsrange_{}.csv'.format(smb), index=False)

    # fps_i.pckl --> .csv
    print('Start converting | fps_i.pckl --> .csv')
    data_num = len(img_all)
    for smb in symbol:
        fprange_path1 = os.path.join(fps_csvpath, 'fpsrange_{}.csv'.format(smb))
        fprange_i = pd.read_csv(fprange_path1)
        min_i_np = fprange_i.iloc[0, :]
        min_i = torch.tensor(min_i_np)
        max_min_i_np = fprange_i.iloc[1, :]
        max_min_i = torch.tensor(max_min_i_np)
        for num in range(data_num):
            fp_path = os.path.join(fps_path, 'fps_{}.pckl'.format(num))
            with open(fp_path, 'rb') as f:
                fp_all = pickle.load(f)
                fp_i = fp_all.allElement_fps[smb]
                fp_i = (fp_i - min_i) / max_min_i
                fp_i_np = fp_i.numpy()
                dfi = pd.DataFrame(fp_i_np)
                pathfpsi = os.path.join(fps_csvpath, '{}_fps_{}.csv'.format(smb, num))
                dfi.to_csv(pathfpsi, index=False)
                print('Completed {}-{}'.format(smb, num))
    return None

                
# 4.Screen for representative structures.
def screen_img(tsd_1,
               tsd_2,
               tsd_3,
               img_path,
               csv_path_,
               symbol_all,
               initial_num1=0,
               initial_num2=0,
               initial_num11=0,
               initial_num22=0):
    # initialize
    print('Start-Screening-Structure-Job.')
    print('Progress        Bank-Local       Bank-Image        Rate        Rate-all')
    image_all = Trajectory(img_path)
    element_num = len(symbol_all)
    img_num = len(image_all)
    bank_sym0 = []
    bank_sym1 = []
    bank_sym2 = []
    if element_num == 1:
        path_sym0 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[0])
        if os.path.exists(path_sym0):
            data_sym0 = pd.read_csv(path_sym0)
            bank_sym0 = [torch.tensor(data_sym0.iloc[do,:]) for do in range(len(data_sym0))]
    elif element_num == 2:
        path_sym0 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[0])
        path_sym1 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[1])
        if os.path.exists(path_sym0):
            data_sym0 = pd.read_csv(path_sym0)
            bank_sym0 = [torch.tensor(data_sym0.iloc[do,:]) for do in range(len(data_sym0))]            
        if os.path.exists(path_sym1):
            data_sym1 = pd.read_csv(path_sym1)
            bank_sym1 = [torch.tensor(data_sym1.iloc[do,:]) for do in range(len(data_sym1))]
    elif element_num == 3:
        path_sym0 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[0])
        path_sym1 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[1])
        path_sym2 = _path_all_ + '/bank-{}-fps.csv'.format(symbol_all[2])
        if os.path.exists(path_sym0):
            data_sym0 = pd.read_csv(path_sym0)
            bank_sym0 = [torch.tensor(data_sym0.iloc[do,:]) for do in range(len(data_sym0))]
        if os.path.exists(path_sym1):
            data_sym1 = pd.read_csv(path_sym1)
            bank_sym1 = [torch.tensor(data_sym1.iloc[do,:]) for do in range(len(data_sym1))]
        if os.path.exists(path_sym2):
            data_sym2 = pd.read_csv(path_sym2)
            bank_sym2 = [torch.tensor(data_sym2.iloc[do,:]) for do in range(len(data_sym2))]
    bank_img0 = [lc0 for lc0 in bank_sym0]
    bank_img1 = [lc1 for lc1 in bank_sym1]
    bank_img2 = [lc2 for lc2 in bank_sym2]
    bank = [bank_img0, bank_img1, bank_img2]
    atom_all0 = {}
    atom_all1 = {}
    atom_all2 = {}
    atom_all = [atom_all0, atom_all1, atom_all2]
    index_ = []
    indexc = []
    threshold_all = []
    len_bank_sym0_s = len(bank[0])
    len_bank_sym1_s = len(bank[1])
    len_bank_sym2_s = len(bank[2])
    if element_num == 1:     
        if len(bank_img0) == 0:
            index_.append(initial_num1)
            initial_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[0], initial_num1))
            initial_all = pd.read_csv(initial_path)
            initial_ = initial_all.iloc[initial_num2, :]
            initial_t = torch.tensor(initial_)
            bank_img0.append(initial_t)
        bank[0] = bank_img0
        threshold_1 = tsd_1
        threshold_all.append(threshold_1)
    elif element_num == 2:
        if (len(bank_img0) == 0) and (len(bank_img1) == 0):
            index_.append(initial_num11)
            initial_0_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[0], initial_num11))
            initial_1_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[1], initial_num11))
            initial_0_all = pd.read_csv(initial_0_path)
            initial_1_all = pd.read_csv(initial_1_path)
            initial_0 = initial_0_all.iloc[initial_num22, :]
            initial_1 = initial_1_all.iloc[initial_num22, :]
            initial_0_t = torch.tensor(initial_0)
            initial_1_t = torch.tensor(initial_1)
            bank_img0.append(initial_0_t)
            bank_img1.append(initial_1_t)
        bank[0] = bank_img0
        bank[1] = bank_img1
        threshold_1 = tsd_1
        threshold_2 = tsd_2
        threshold_all.append(threshold_1)
        threshold_all.append(threshold_2)
    elif element_num == 3:
        if (len(bank_img0) == 0) and (len(bank_img1) == 0) and (len(bank_img2) == 0):
            index_.append(initial_num11)
            initial_0_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[0], initial_num11))
            initial_1_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[1], initial_num11))
            initial_2_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[2], initial_num11))
            initial_0_all = pd.read_csv(initial_0_path)
            initial_1_all = pd.read_csv(initial_1_path)
            initial_2_all = pd.read_csv(initial_2_path)
            initial_0 = initial_0_all.iloc[initial_num22, :]
            initial_1 = initial_1_all.iloc[initial_num22, :]
            initial_2 = initial_2_all.iloc[initial_num22, :]
            initial_0_t = torch.tensor(initial_0)
            initial_1_t = torch.tensor(initial_1)
            initial_2_t = torch.tensor(initial_2)
            bank_img0.append(initial_0_t)
            bank_img1.append(initial_1_t)
            bank_img2.append(initial_2_t)
        bank[0] = bank_img0
        bank[1] = bank_img1
        bank[2] = bank_img2
        threshold_1 = tsd_1
        threshold_2 = tsd_2
        threshold_3 = tsd_3
        threshold_all.append(threshold_1)
        threshold_all.append(threshold_2)
        threshold_all.append(threshold_3)
    # Iterate through all img.
    for i in range(img_num):
        len_0 = len(bank[0])
        len_1 = len(bank[1])
        len_2 = len(bank[2])
        lc_atoms = [[], [], []]
        for smb_num in range(len(symbol_all)):  # Traverse elements in image.
            # Read fingerprint into memory.
            fp_1_all_path = os.path.join(csv_path_, '{}_fps_{}.csv'.format(symbol_all[smb_num], i))
            fp_1_all_ = pd.read_csv(fp_1_all_path)
            # The data type is tensor.
            bk_i = bank[smb_num]
            fp_1_all = np.array(fp_1_all_)
            fp_1_all_t = torch.tensor(fp_1_all)
            bank_img_array = np.array([ten_sor.numpy() for ten_sor in bk_i])
            bank_img_t = torch.tensor(bank_img_array)
            # Calculate the distance matrix between this img and the local environment in the bank.
            dis_all = torch.cdist(fp_1_all_t, bank_img_t, p=2)
            # Compare with img itself. The 0th atom is calculated separately, and the rest are looped.
            img_compare_itself = []
            if torch.min(dis_all[0]) >= threshold_all[smb_num]:
                bk_i.append(torch.tensor(fp_1_all[0]))
                lc_atoms[smb_num].append(0)
                img_compare_itself.append(torch.tensor(fp_1_all[0]))
            for d_i in range(1, len(dis_all)):
                go_on_cal = (torch.min(dis_all[d_i]) >= threshold_all[smb_num] and len(img_compare_itself) > 0)
                go_on_no_cal = (torch.min(dis_all[d_i]) >= threshold_all[smb_num] and len(img_compare_itself) == 0)
                if go_on_no_cal:
                    bk_i.append(torch.tensor(fp_1_all[d_i]))
                    lc_atoms[smb_num].append(d_i)
                    img_compare_itself.append(torch.tensor(fp_1_all[d_i]))
                if go_on_cal:
                    i_c_i_a = np.array([t_s.numpy() for t_s in img_compare_itself])
                    i_c_i_t = torch.tensor(i_c_i_a)
                    d_i_t = torch.tensor(fp_1_all[d_i]).unsqueeze(0)
                    d_a = torch.cdist(d_i_t, i_c_i_t, p=2)
                    if torch.min(d_a) >= threshold_all[smb_num]:
                        bk_i.append(torch.tensor(fp_1_all[d_i]))
                        img_compare_itself.append(torch.tensor(fp_1_all[d_i]))
                        lc_atoms[smb_num].append(d_i)
        # Determine whether this img contains a representative local_environment.
        atom_all[0][i] = lc_atoms[0]
        atom_all[1][i] = lc_atoms[1]
        atom_all[2][i] = lc_atoms[2]
        accept_ = (len(bank[0]) > len_0 or len(bank[1]) > len_1 or len(bank[2]) > len_2)
        if accept_:
            index_.append(i)
        else:
            indexc.append(i)
        # Print the steps.
        numnow_rate = round((len(index_)-1)/(i+1)*100, 2)
        accept_rate = round((len(index_)-1)/img_num*100, 2)
        a_0 = len(bank[0])
        a_1 = len(bank[1])
        a_2 = len(bank[2])
        print(
            '{}/{}            {}|{}|{}            {}            {}%            {}%'.
            format(i+1, 
                   img_num, 
                   a_0, 
                   a_1, 
                   a_2,
                   len(index_)-1, 
                   numnow_rate,
                   accept_rate))
    # Write the filtered img to the train.traj file.
    index_s = set(index_)
    index_l = list(index_s)
    new_set = []
    new_setc = []
    len_bank_sym0_e = len(bank[0])
    len_bank_sym1_e = len(bank[1])
    len_bank_sym2_e = len(bank[2])
    for i in index_l:
        new_set.append(image_all[i])
    ase.io.write(filename='./train.traj', images=new_set)
    for ii in indexc:
        new_setc.append(image_all[ii])
    ase.io.write(filename='./test.traj', images=new_setc)
    print('  ')
    print(
'''
  _   _          _      _
 | | | |_ __  __| |__ _| |_ ___
 | |_| | '_ \/ _` / _` |  _/ -_)
  \___/| .__/\__,_\__,_|\__\___|
       |_|

''')
    print('----------------------------------------------------------------------')
    print('[Screening-Structure-Job-Information:]')
    print('    -. The elements in the system are: {}'.format(symbol_all))
    print('    -. Number of representative configurations in the original Atom_Bank: {}|{}|{}'.
                  format(len_bank_sym0_s, 
                         len_bank_sym1_s, 
                         len_bank_sym2_s))
    print('    -. Number of representative configurations in the after screening Atom_Bank: {}|{}|{}'.
                  format(len_bank_sym0_e, 
                         len_bank_sym1_e, 
                         len_bank_sym2_e))
    print('    -. Number of representative configurations accepted in Atom_Bank: {}|{}|{}'.
                  format(len_bank_sym0_e-len_bank_sym0_s, 
                         len_bank_sym1_e-len_bank_sym1_s, 
                         len_bank_sym2_e-len_bank_sym2_s))
    print('Finish-Screening-Structure-Job.')
    return bank, atom_all
                

# 5.Write fingerprints of all representative local-chemical-environments in the bank into csv file.
def write_bank_fps_to_csv(bank, symbol):
    if len(bank[0]) != 0:
        bank_0_a = []
        for h in bank[0]:
            nph = np.array(h)
            bank_0_a.append(nph)
        df_norm = pd.DataFrame(np.array(bank_0_a))
        df_norm.to_csv('bank-{}-fps.csv'.format(symbol[0]), encoding='utf-8', index=None)
    if len(bank[1]) != 0:
        bank_1_a = []
        for h in bank[1]:
            nph = np.array(h)
            bank_1_a.append(nph)
        df_norm = pd.DataFrame(np.array(bank_1_a))
        df_norm.to_csv('bank-{}-fps.csv'.format(symbol[1]), encoding='utf-8', index=None)
    if len(bank[2]) != 0:
        bank_2_a = []
        for h in bank[2]:
            nph = np.array(h)
            bank_2_a.append(nph)
        df_norm = pd.DataFrame(np.array(bank_2_a))
        df_norm.to_csv('bank-{}-fps.csv'.format(symbol[2]), encoding='utf-8', index=None)
    print('    -. Write local-configuration-fps to .csv file.')
    return None

    
# 6.lc_to_csv.
def write_lc_to_csv(l_c_all, symbol):
    csv_file = 'local_config.csv'
    keys0 = l_c_all[0].keys()
    if len(symbol) == 1:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 
                             '{}'.format(symbol[0])]) 
            for key in keys0:
                writer.writerow([key, 
                                 l_c_all[0][key]])            
    elif len(symbol) == 2:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 
                             '{}'.format(symbol[0]), 
                             '{}'.format(symbol[1])])
            for key in keys0:
                writer.writerow([key, 
                                 l_c_all[0][key], 
                                 l_c_all[1][key]])                
    elif len(symbol) == 3:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 
                             '{}'.format(symbol[0]), 
                             '{}'.format(symbol[1]), 
                             '{}'.format(symbol[2])])
            for key in keys0:
                writer.writerow([key, 
                                 l_c_all[0][key], 
                                 l_c_all[1][key],
                                 l_c_all[2][key]])
    print('    -. Write local configuration to csv file.')
    return None


# 7.Encapsulate all the above functions as main functions and run.
def run_select_img(fps_csvp_, 
                   traj_symbol=traj_symbol_):
    if traj_symbol == None:
        traj_symbol = get_symbols(image_path=img_path_)
    if not os.path.exists(fps_csvp_):
        fps_nor1_to_csv(fps_path=fps_path_, 
                        img_path=img_path_, 
                        symbol=traj_symbol, 
                        fps_csvpath=fps_csvp_)
    bank12, atom_all_ = screen_img(tsd_1=t_1, 
                                   tsd_2=t_2,
                                   tsd_3=t_3,
                                   initial_num1=0,
                                   initial_num2=0,
                                   initial_num11=0,
                                   initial_num22=0,
                                   img_path=img_path_,
                                   csv_path_=fps_csvp_,
                                   symbol_all=traj_symbol,)
    write_bank_fps_to_csv(bank=bank12, 
                          symbol=traj_symbol)
    write_lc_to_csv(l_c_all=atom_all_, 
                    symbol=traj_symbol)
    code_time_e = time.perf_counter()
    print('----------------------------------------------------------------------')
    print('[Pying-Information:]')
    print('    -. Program execution completed successfully.')
    print('    -. Code-Time is {} Sec.'.format(np.round(code_time_e-code_time_s, 2)))
    print('----------------------------------------------------------------------')
    return None


# 8.Run-Code.
run_select_img(fps_csvp_=f_cp_)
