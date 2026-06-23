import xarray as xr

# 读取数据
ds = xr.open_dataset(r'C:\Users\31860\PycharmProjects\DeepR\my_SR\data\slhf.nc')

# 降采样
ds = ds.coarsen(latitude=2, longitude=2,boundary='pad').mean()

# 存储到新的netcdf文件中
ds.to_netcdf(r'C:\Users\31860\PycharmProjects\DeepR\my_SR\data\slhf_down.nc')
