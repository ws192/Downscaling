import netCDF4
import numpy
import torch
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
np.set_printoptions(threshold=np.inf)
from netCDF4 import Dataset
# nc_file = Dataset(r"K:\model_from_hlt\gridone_new.nc",'a')
# # nc_file_lsm=Dataset(r'D:\lsm.nc')
# # lsm_data = nc_file_lsm.variables['lsm'][:]
# print(nc_file.variables['depth'])
#
# lon = nc_file.variables['longitude'][:]
# lat = nc_file.variables['latitude'][:]
#
# # 确定目标区域的索引
# # 经度从-180°到180°，纬度从90°到-90°
# lon_start_index = np.where(lon >= 99)[0][0]
# lon_end_index = np.where(lon <= 124)[0][-1]
# lat_start_index = np.where(lat <= 25)[0][0]
# lat_end_index = np.where(lat >= 0)[0][-1]
#
# # 提取指定区域的depth数据
# depth = nc_file.variables['depth'][lon_start_index:lon_end_index, lat_start_index:lat_end_index]
# depth=np.transpose(depth,[1,0])
# depth=np.flipud(depth)
# tmp=np.zeros((1500,1500))
# tmp[:]=depth
# print(depth)
# np.save(r'D:\Ckpt\ssh.npy',tmp)
# print(depth.max())
# print(depth.min())
depth=torch.Tensor(np.load(r'D:\Ckpt\ssh.npy').reshape(1,1,1500,1500))
depth=torch.nn.functional.interpolate(depth,size=(256,256),mode='nearest').numpy().reshape(256,256)
depth=np.where(depth<0,0,depth)
np.save(r'D:\Ckpt\ssh256.npy',2*depth/6176-1)

print(np.load(r'D:\Ckpt\ssh256.npy').max())
print(np.load(r'D:\Ckpt\ssh256.npy').min())
# # 获取纬度和经度坐标
# lat = nc_file.variables["latitude"][:]
# # print(lat)
# lon = nc_file.variables['longitude'][:]
# print(lon)
# 创建绘图窗口
fig = plt.figure(figsize=(20, 16))
ax = plt.axes(projection=ccrs.PlateCarree())
# 设置绘图的范围
# ax.set_extent([140, 155, 30, 45], crs=ccrs.PlateCarree())
# ax.set_extent([-170, -154, -46, -30], crs=ccrs.PlateCarree())
# 绘制数据
# levels = np.arange(10, sosstsst.max(), 0.1)  # 设置等值线间隔
# levels = np.linspace(277,305, 40)  # 设置等值线间隔
# levels = np.linspace(277,305, 500)  # 设置等值线间隔
# k=[277,280,285,285.2,285.25,285.30,285.35,285.40,285.45,285.50,285.55,285.60,285.65,285.70,285.75,285.80,286,290,295,297,298,299,300,300.1,300.11,300,,305]
# a=numpy.array(k)
# print(len(levels))
# saaa= np.ma.masked_where(lsm_data[0] != 0, sst[1001])
# plt.contourf(lon, lat, sst[1001], levels=levels, cmap="RdYlBu",transform=ccrs.PlateCarree())
# plt.imshow(aaaa)
# plt.pcolormesh(lon, lat, sst[501], cmap="jet",transform=ccrs.PlateCarree())
lat = np.linspace(start=0, stop=25, num=256)
lon = np.linspace(start=99, stop=124, num=256)
cmap = plt.cm.get_cmap('jet')
cmap.set_under('w')
plt.pcolormesh(lon, lat, depth, cmap=cmap, transform=ccrs.PlateCarree(), vmin=1, vmax=500)
# 添加海岸线
ax.coastlines()
# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('Sea Surface Temperature')
# 设置标题和坐标轴标签
plt.title('Sea Surface Temperature Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#设置网格
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.01, linestyle='--')
# plt.savefig(r'C:\Users\31860\Desktop\v2rayN\a.jpg')
# 显示图像
plt.show()
#
# #
