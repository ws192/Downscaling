import h5py
import hdf5storage
import numpy as np

# data=np.load(r'C:\Users\31860\Desktop\tmplow.npy').reshape(5479*12,32,32)
# print(data.shape)

# matlab处理完转回python npy格式文件
data = h5py.File(r'D:\SST\tmplow.mat')
res_sst = data['tmp'][:]
res_sst = np.transpose(res_sst, [2, 1, 0])
res_sst = np.nan_to_num(res_sst)
print(res_sst.shape)
np.save(r'D:\SST\tmplow.npy', res_sst.astype(np.float16).reshape((-1, 12, 128, 128)))

import scipy.io as sio
import hdf5storage

# a=np.load(r'C:\Users\31860\Desktop\tmplow.npy').reshape(5479*12,32,32).astype(np.float32)
# hdf5storage.savemat(r'K:\Data\tmplow.mat', {'a': a})
