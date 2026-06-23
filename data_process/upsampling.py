# import netCDF4 as nc
# import numpy as np
# from scipy.interpolate import interp2d
#
# # 读取原始netCDF文件
# nc_file = nc.Dataset(r'C:\Users\31860\PycharmProjects\DeepR\my_SR\data\msl_down.nc')
#
# # 获取原始数据
# longitude = nc_file.variables['longitude'][:]
# latitude = nc_file.variables['latitude'][:]
# msl_data = nc_file.variables['msl'][:]
#
# # 定义新的经纬度网格
# new_longitude = np.linspace(-179.875, 179.875, 1440)
# new_latitude = np.linspace(-89.875, 89.875, 721)
#
# # 创建上采样后的经纬度网格
# new_longitude_grid, new_latitude_grid = np.meshgrid(new_longitude, new_latitude)
#
# # 创建上采样后的数据数组
# upsampled_msl_data = np.zeros((msl_data.shape[0], new_latitude.shape[0], new_longitude.shape[0]))
#
# # 插值操作
# for i in range(msl_data.shape[0]):
#     f = interp2d(longitude, latitude, msl_data[i, :, :], kind='cubic')
#     upsampled_msl_data[i, :, :] = f(new_longitude, new_latitude)
#
# # 创建新的netCDF文件
# output_file = nc.Dataset(r'C:\Users\31860\PycharmProjects\DeepR\my_SR\data\msl_up.nc', 'w', format='NETCDF4')
#
# # 创建维度
# longitude_dim = output_file.createDimension('longitude', new_longitude.shape[0])
# latitude_dim = output_file.createDimension('latitude', new_latitude.shape[0])
# time_dim = output_file.createDimension('time', msl_data.shape[0])
#
# # 创建变量
# longitude_var = output_file.createVariable('longitude', 'f4', ('longitude',))
# latitude_var = output_file.createVariable('latitude', 'f4', ('latitude',))
# time_var = output_file.createVariable('time', 'i4', ('time',))
# msl_var = output_file.createVariable('msl', 'f4', ('time', 'latitude', 'longitude'))
#
# # 设置变量的属性
# longitude_var.units = 'degrees_east'
# latitude_var.units = 'degrees_north'
# time_var.units = 'hours since 1970-01-01 00:00:00'
# msl_var.units = 'hPa'
# msl_var.coordinates = 'longitude latitude'
#
# # 将数据写入变量
# longitude_var[:] = new_longitude
# latitude_var[:] = new_latitude
# time_var[:] = range(msl_data.shape[0])
# msl_var[:] = upsampled_msl_data
#
# # 关闭netCDF文件
# output_file.close()
# nc_file.close()
# #
import numpy as np
import xarray as xr
from scipy.interpolate import interp2d
# 读取原始netCDF文件
ds = xr.open_dataset(r'D:\s_low.nc')

# 上采样
upsampled_ds = ds.interp(latitude=np.linspace(49.5, 9.75, 160), longitude=np.linspace(120.25, 160, 160),method='cubic')
# print(np.linspace(49.5, -30.25, 320))
# print(np.linspace(90.25, 170, 320))
upsampled_ds.to_netcdf(r'D:\s_low_up.nc')
ds.close()
#
# import numpy as np
# import xarray as xr
# from scipy.interpolate import griddata
#
# # 读取原始netCDF文件
# ds = xr.open_dataset(r'D:\2mtmp_low.nc')
#
# # 指定插值网格的经纬度范围和分辨率
# lat_res = 160
# lon_res = 160
# lat_values = np.linspace(49.5,9.75,  lat_res)
# lon_values = np.linspace(120.25, 160, lon_res)
#
# # 创建目标网格的经纬度坐标
# lon_target, lat_target = np.meshgrid(lon_values, lat_values)
#
# # 获取原始数据的经纬度和值
# original_lat = ds['latitude'].values
# original_lon = ds['longitude'].values
# original_data = ds['t2m'].values  # 替换为你的变量名
# print(original_data[0])
#
# interpolated_data=np.zeros((1003,160,160))
# for i in range(1002):
#
#     # 使用Scipy的griddata进行插值，但在计算过程中跳过NaN点
#     interpolated_data[i][:] = griddata((original_lat,original_lon ), original_data[i].ravel(),
#                                  (lat_target,lon_target), method='cubic')
#
#     # print(interpolated_data[i])
#
#
#
# # 创建一个新的xarray数据集
# upsampled_ds = xr.Dataset({
#     't2m': (['time','latitude', 'longitude'], interpolated_data)
# }, coords={'latitude': lat_values, 'longitude': lon_values})
#
# # 保存插值结果到netCDF文件
# upsampled_ds.to_netcdf(r'D:\2mtmp_up.nc')
#
# # 关闭原始数据集
# ds.close()

# import numpy as np
# import xarray as xr
# from scipy.interpolate import griddata
#
# # 读取原始netCDF文件
# ds = xr.open_dataset(r'D:\data_low_for_bicubic.nc')
#
# # 指定插值网格的经纬度范围和分辨率
# lat_res = 160
# lon_res = 160
# lat_values = np.linspace(49.5, 9.75, lat_res)
# lon_values = np.linspace(120.25, 160, lon_res)
#
# # 创建目标网格的经纬度坐标
# lon_target, lat_target = np.meshgrid(lon_values, lat_values)
#
# # 获取原始数据的经纬度和值
# original_lat = ds['latitude'].values
# original_lon = ds['longitude'].values
# original_data = ds['sst'].values  # 替换为你的变量名
# interpolated_data=np.zeros((1003,160,160))
# for i in range(1003):
#     # 创建一个mask来标识NaN点
#     mask = np.isnan(original_data[i])  # 使用一个时间步的数据来创建mask
#
#     # 使用np.meshgrid将经纬度转换为二维数组，使其与mask具有相同维度
#     original_lat, original_lon = np.meshgrid(original_lat, original_lon)
#
#     # 获取非NaN点的坐标和对应的值
#     non_nan_lat = original_lat[~mask]
#     non_nan_lon = original_lon[~mask]
#     non_nan_data = original_data[i][~mask]  # 使用一个时间步的数据来获取非NaN点的值
#
#     # 使用Scipy的griddata进行插值，但在计算过程中跳过NaN点
#     interpolated_data[i][:] = griddata((non_nan_lon.ravel(), non_nan_lat.ravel()), non_nan_data.ravel(),
#                                  (lon_target, lat_target), method='cubic')
#
# # 创建一个新的xarray数据集
# upsampled_ds = xr.Dataset({
#     'sst': (['latitude', 'longitude'], interpolated_data)
# }, coords={'latitude': lat_values, 'longitude': lon_values})
#
# # 保存插值结果到netCDF文件
# upsampled_ds.to_netcdf(r'D:\data_upbycubic.nc')
#
# # 关闭原始数据集
# ds.close()
