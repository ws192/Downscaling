import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset

'''
    显存要求过大请在服务器完成数据预处理
'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机
setup_seed(888888)

data1 = np.load(r'../data/merged_td.npz')
td = data1['td'][:, :96, 200:296]
print(td.shape)  # (1500, 96, 96)

data2 = np.load(r'../data/merged_wd.npz')
wd = data2['wd'][:, :96, 200:296]

data3 = np.load(r'../data/merged_um.npz')
um = data3['um'][:, :96, 200:296]
print(um.shape)  # (1500, 96, 96)

data4 = np.load(r'../data/merged_vm.npz')
vm = data4['vm'][:, :96, 200:296]

data5 = np.load(r'../data/merged_sst.npz')
tm = data5['sst'][:, :96, 200:296]

data6 = np.load(r'../data/merged_Q.npz')
Q = data6['Q'][:, :96, 200:296]

data7 = np.load(r'../data/merged_mld.npz')
mld = data7['mld'][:, :96, 200:296]

arrays = {
    'Q': Q,  # 替换为真实数据
    'tm': tm,
    'td': td,
    'um': um,
    'vm': vm,
    'wd': wd,
    'mld': mld
}

np.savez_compressed('DWQ_pinn.npz', **arrays)
