import torch
import torch.nn.functional as F
import numpy as np
from netCDF4 import Dataset
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(-1,1))
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
np.set_printoptions(threshold=np.inf)
from netCDF4 import Dataset
nc_file = Dataset(r'D:\2mtmp_forbi.nc','a')
nc_file_normal = Dataset(r'D:\2mtmp_normal.nc','a')

tmp=np.zeros((1,1,20,20))
tmp[0][0][:]=nc_file.variables['t2m'][1001]
t=torch.Tensor(tmp)
t_up=F.interpolate(t,size=(160,240),mode='bicubic')
t_up=t_up[0][0][:].numpy()
# sst=t_up[0][0][:]
# scaler.fit(sst)
# sst=scaler.transform(sst)
# t_up[0][0][:]=torch.Tensor(sst)
#
# file_sst_gt1 = Dataset(r"D:\2mtmp_normal.nc").variables['t2m'][:]
# gt=file_sst_gt1[1001]
# scaler.fit(gt)
# gt=scaler.transform(gt)
# a=np.zeros((1,1,160,160))
# a[0][0][:]=gt
# a=torch.Tensor(a)
# print(torch.sqrt(F.mse_loss(a,t_up)))



nc_file_lsm=Dataset(r'D:\lsm.nc')
# 读取平均表面压力数据和land sea mask数据
lsm_data = nc_file_lsm.variables['lsm'][:]

t_up = np.ma.masked_where(lsm_data[10] != 0, t_up)

lat = nc_file_normal.variables["latitude"][:]
print(lat)
lon = nc_file_normal.variables['longitude'][:]
print(lon)
# 创建绘图窗口
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置绘图的范围
# ax.set_extent([120, 160, 10, 50], crs=ccrs.PlateCarree())
# ax.set_extent([-170, -154, -46, -30], crs=ccrs.PlateCarree())
# 绘制数据
# levels = np.arange(10, sosstsst.max(), 0.1)  # 设置等值线间隔
levels = np.linspace(277,305, 100)  # 设置等值线间隔
# print(len(levels))
# plt.contourf(lon, lat,t_up, levels=levels, cmap="coolwarm",transform=ccrs.PlateCarree())
plt.pcolormesh(lon, lat, t_up,cmap="RdYlBu", transform=ccrs.PlateCarree())

# 添加海岸线
ax.coastlines()
# 添加颜色条
cbar = plt.colorbar()
# cbar.set_label('Sea Surface Temperature')
# 设置标题和坐标轴标签
# plt.title('Sea Surface Temperature Distribution')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
#设置网格
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.01, linestyle='--')
# plt.savefig(r'C:\Users\31860\Desktop\v2rayN\a.jpg')
# 显示图像
plt.show()


