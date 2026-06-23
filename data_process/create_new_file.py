import netCDF4 as nc

# 读取现有的netCDF文件
import numpy.ma
import numpy as np

dataset = nc.Dataset(r"D:\2mtmp.nc")

# 获取对应的变量
latitude = dataset.variables['latitude'][:]
longitude = dataset.variables['longitude'][:]
print(latitude)
print(latitude.shape)
# sst = dataset.variables['t2m'][:,0,:,:][:1002,:,:]
# print(sst[0][-1][0])
# sst=numpy.ma.filled(sst,fill_value=np.nan)
# 获取感兴趣区域的索引范围
lat_start = (latitude < 49.75).tolist().index(True)
print(lat_start)
lat_end = (latitude <10).tolist().index(True)
print(lat_end)
lon_end = (longitude > 159.75).tolist().index(True)
lon_start = (longitude > 120).tolist().index(True)
# #
stride=8



# # # 提取感兴趣区域的数据
# sst = sst[:, lat_start:lat_end+1:stride, lon_start:lon_end+1:stride]
# # print(sst[100])
# # sst=numpy.ma.filled(sst,fill_value=0)
# latitude = latitude[lat_start:lat_end+1:stride]
# longitude = longitude[lon_start:lon_end+1:stride]
# print(latitude)
# print(longitude)
# print(latitude.shape)
# print(longitude.shape)
# ## 创建新的netCDF文件
# new_dataset = nc.Dataset(r"D:\2mtmp_forbi.nc", 'w', format='NETCDF4')
#
# # 定义新的维度和变量
# new_dataset.createDimension('latitude', len(latitude))
# new_dataset.createDimension('longitude', len(longitude))
# new_dataset.createDimension('time', None) # 在新的文件中使用无限的时间维度
#
# new_latitude = new_dataset.createVariable('latitude', 'd', ('latitude',))
# new_longitude = new_dataset.createVariable('longitude', 'd', ('longitude',))
# new_sst = new_dataset.createVariable('t2m', 'd', ('time', 'latitude', 'longitude'))
#
# # 将数据写入新的netCDF文件
# new_latitude[:] = latitude
# new_longitude[:] = longitude
# new_sst[:] = sst
#
# # 关闭文件
# new_dataset.close()
# dataset.close()