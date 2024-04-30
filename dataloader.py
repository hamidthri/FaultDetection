import numpy as np
import scipy.io
import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn
import sklearn.model_selection
from torch import optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data_path = 'G:\\Electrical_ENG\\Fault\\project\\data'

loads = ['1', '2', '3']
diams = ['d_0.18', 'd_0.36', 'd_0.54']


test = os.path.join(data_path, diams[0], loads[0])
inner_Race_1_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\1\\inner_race\\110.mat')
data_IF_1_18 = inner_Race_1_18['X110_DE_time']
inner_Race_2_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\2\\inner_race\\111.mat')
data_IF_2_18 = inner_Race_2_18['X111_DE_time']
inner_Race_3_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\3\\inner_race\\112.mat')
data_IF_3_18 = inner_Race_3_18['X112_DE_time']
outer_Race_1_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\1\\outer_race\\136.mat')
data_OF_1_18 = outer_Race_1_18['X136_DE_time']
outer_Race_2_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\2\\outer_race\\137.mat')
data_OF_2_18 = outer_Race_2_18['X137_DE_time']
outer_Race_3_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\3\\outer_race\\138.mat')
data_OF_3_18 = outer_Race_3_18['X138_DE_time']
roller_Race_1_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\1\\roller_falt\\123.mat')
data_RF_1_18 = roller_Race_1_18['X123_DE_time']
roller_Race_2_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\2\\roller_falt\\124.mat')
data_RF_2_18 = roller_Race_2_18['X124_DE_time']
roller_Race_3_18 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.18\\3\\roller_falt\\125.mat')
data_RF_3_18 = roller_Race_3_18['X125_DE_time']
#########################   36
inner_Race_1_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\1\\inner_race\\175.mat')
data_IF_1_36 = inner_Race_1_36['X175_DE_time']
inner_Race_2_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\2\\inner_race\\176.mat')
data_IF_2_36 = inner_Race_2_36['X176_DE_time']
inner_Race_3_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\3\\inner_race\\177.mat')
data_IF_3_36 = inner_Race_3_36['X177_DE_time']
outer_Race_1_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\1\\outer_race\\202.mat')
data_OF_1_36 = outer_Race_1_36['X202_DE_time']
outer_Race_2_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\2\\outer_race\\203.mat')
data_OF_2_36 = outer_Race_2_36['X203_DE_time']
outer_Race_3_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\3\\outer_race\\204.mat')
data_OF_3_36 = outer_Race_3_36['X204_DE_time']
roller_Race_1_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\1\\roller_falt\\190.mat')
data_RF_1_36 = roller_Race_1_36['X190_DE_time']
roller_Race_2_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\2\\roller_falt\\191.mat')
data_RF_2_36 = roller_Race_2_36['X191_DE_time']
roller_Race_3_36 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.36\\3\\roller_falt\\192.mat')
data_RF_3_36 = roller_Race_3_36['X192_DE_time']
###################  54
inner_Race_1_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\1\\inner_race\\214.mat')
data_IF_1_54 = inner_Race_1_54['X214_DE_time']
inner_Race_2_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\2\\inner_race\\215.mat')
data_IF_2_54 = inner_Race_2_54['X215_DE_time']
inner_Race_3_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\3\\inner_race\\217.mat')
data_IF_3_54 = inner_Race_3_54['X217_DE_time']
outer_Race_1_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\1\\outer_race\\239.mat')
data_OF_1_54 = outer_Race_1_54['X239_DE_time']
outer_Race_2_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\2\\outer_race\\240.mat')
data_OF_2_54 = outer_Race_2_54['X240_DE_time']
outer_Race_3_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\3\\outer_race\\241.mat')
data_OF_3_54 = outer_Race_3_54['X241_DE_time']
roller_Race_1_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\1\\roller_falt\\227.mat')
data_RF_1_54 = roller_Race_1_54['X227_DE_time']
roller_Race_2_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\2\\roller_falt\\228.mat')
data_RF_2_54 = roller_Race_2_54['X228_DE_time']
roller_Race_3_54 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\d_0.54\\3\\roller_falt\\229.mat')
data_RF_3_54 = roller_Race_3_54['X229_DE_time']

############# normal

normal_1 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\\normal\\1\\98.mat')
data_normal_1 = normal_1['X098_DE_time']
normal_2 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\\normal\\2\\99.mat')
data_normal_2 = normal_2['X099_DE_time']
normal_3 = scipy.io.loadmat('G:\Electrical_ENG\Fault\project\data\\normal\\3\\100.mat')
data_normal_3 = normal_3['X100_DE_time']

################data

data_all = {
    'normal': {
        '1': data_normal_1,
        '2': data_normal_2,
        '3': data_normal_3
    },
    'roller_Race_18': {
        '1': data_RF_1_18,
        '2': data_RF_2_18,
        '3': data_RF_3_18
    },
    'roller_Race_36': {
        '1': data_RF_1_36,
        '2': data_RF_2_36,
        '3': data_RF_3_36
    },
    'roller_Race_54': {
        '1': data_RF_1_54,
        '2': data_RF_2_54,
        '3': data_RF_3_54
    },
    'inner_Race_18': {
        '1': data_IF_1_18,
        '2': data_IF_2_18,
        '3': data_IF_3_18
    },
    'inner_Race_36': {
        '1': data_IF_1_36,
        '2': data_IF_2_36,
        '3': data_IF_3_36

    },
    'inner_Race_54': {
        '1': data_IF_1_54,
        '2': data_IF_2_54,
        '3': data_IF_3_54
    },
    'outer_Race_18': {
        '1': data_OF_1_18,
        '2': data_OF_2_18,
        '3': data_OF_3_18
    },
    'outer_Race_36': {
        '1': data_OF_1_36,
        '2': data_OF_2_36,
        '3': data_OF_3_36
    },
    'outer_Race_54': {
        '1': data_OF_1_54,
        '2': data_OF_2_54,
        '3': data_OF_3_54
    },
}
# np.savez_compressed('data_all.npz', **data_all)
np.save('data_all.npy', data_all, allow_pickle=True)
