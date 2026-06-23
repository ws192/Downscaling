import netCDF4 as nc

# 打开netCDF文件以进行读写操作
ncfile = nc.Dataset(r'D:\2mtmp_low.nc', 'r+')

# 获取t2m变量的维度
time_dim = ncfile.dimensions['time']

# 更新t2m变量的数据，仅保留前1002个时间步骤
ncfile.variables['t2m'][:] = ncfile.variables['t2m'][:1002, :, :]

# 更新时间维度的大小为1002
time_dim.setlen(1002)

# 关闭并保存文件
ncfile.close()
