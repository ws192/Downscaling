import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
nc_file = Dataset(r"C:\Users\31860\PycharmProjects\DeepR\my_SR\data\lsm.nc")
print(nc_file.variables["lsm"])

lsm = nc_file.variables["lsm"][0,:,:]
# 获取纬度和经度坐标
lat = nc_file.variables["latitude"][:]
# print(lat.shape)
lon = nc_file.variables['longitude'][:]
# print(lon.shape)
# 创建绘图窗口
fig = plt.figure(figsize=(20, 16))

ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([60, 160, -20, 60], crs=ccrs.PlateCarree())
levels = np.array([-0.5,0.5,1.5])  # 设置等值线间隔
# levels = np.linspace(lsm.min(), lsm.max(), 2)  # 设置等值线间隔
# print(len(levels))
cmap = plt.cm.get_cmap('jet')
cmap.set_under('w')
plt.pcolormesh(lon, lat, lsm, cmap=cmap, transform=ccrs.PlateCarree(), vmin=16, vmax=30)
# 添加海岸线
# ax.coastlines(resolution='50m')
rect_extent = [108, 120, 16, 24]
rect = plt.Rectangle((rect_extent[0], rect_extent[2]), rect_extent[1]-rect_extent[0], rect_extent[3]-rect_extent[2],
                     linewidth=2, edgecolor='black', facecolor='none', transform=ccrs.PlateCarree())
ax.add_patch(rect)
# 添加颜色条
# cbar = plt.colorbar()
# cbar.set_label('Land Sea Mask')
# 设置标题和坐标轴标签
# plt.title('Land Sea Mask Distribution')
plt.xlabel('Longitude', fontdict={'size': 16, 'family': 'Times New Roman'})
plt.ylabel('Latitude', fontdict={'size': 16, 'family': 'Times New Roman'})
#设置网格
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.01, linestyle='--')
plt.savefig(r'C:\Users\31860\Desktop\coastline.png')
# 显示图像
plt.show()


