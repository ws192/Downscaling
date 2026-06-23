import numpy as np
from netCDF4 import Dataset
import torch
nc_file_msl = Dataset(r"C:\Users\31860\PycharmProjects\DeepR\my_SR\data\msl_up.nc")
nc_file_sst = Dataset(r"C:\Users\31860\PycharmProjects\DeepR\my_SR\data\sst_up.nc")
nc_file_slhf = Dataset(r"C:\Users\31860\PycharmProjects\DeepR\my_SR\data\slhf_up.nc")
# print(nc_file.variables['latitude'][:])
data_msl=nc_file_msl.variables['msl'][:]
data_sst=nc_file_sst.variables['sst'][:]
data_slhf=nc_file_slhf.variables['slhf'][:]
arr=np.zeros((1440,721),dtype=float)
for i in range(1440):
    arr[i]=nc_file_msl.variables['latitude'][:]
arr_latitude=arr.T
# print(arr_latitude)
# print(arr_latitude.shape)

arr=np.zeros((721,1440),dtype=float)
for i in range(721):
    arr[i]=nc_file_msl.variables['longitude'][:]
arr_longitude=arr
# print(arr_longitude)
# print(arr_longitude.shape)
dataset=[]

# print(lon_lat_sst_slhf_msl_concat)
# print(lon_lat_sst_slhf_msl_concat.shape)
#输入条件：经纬度+海温
for i in range(36):
    lon_lat_sst_slhf_msl_concat = np.zeros((4, 721, 1440), dtype=float)
    lon_lat_sst_slhf_msl_concat[0, :] = arr_latitude
    lon_lat_sst_slhf_msl_concat[1, :] = arr_longitude
    lon_lat_sst_slhf_msl_concat[2,:]=data_sst[i]
    # lon_lat_sst_slhf_msl_concat[3,:]=data_slhf[i]
    lon_lat_sst_slhf_msl_concat[3,:]=data_msl[i]/1000
    dataset.append(lon_lat_sst_slhf_msl_concat)

dataset=[torch.Tensor(matrix) for matrix in dataset]
print(dataset[0])
print(dataset[0].shape)


