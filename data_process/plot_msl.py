import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
nc_file = Dataset(r"C:\Users\31860\PycharmProjects\DeepR\my_SR\newdata\msl.nc")
print(nc_file.variables["msl"])

msl = nc_file.variables["msl"][:]
# 获取纬度和经度坐标
lat = nc_file.variables["latitude"][:]
lon = nc_file.variables['longitude'][:]

for i in range(36):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # 设置绘图的范围
    # ax.set_extent([40, 80, -10, 30], crs=ccrs.PlateCarree())
    # 绘制数据
    # levels = np.arange(10, sosstsst.max(), 0.1)  # 设置等值线间隔
    levels = np.linspace(msl.min(), msl.max(), 100)  # 设置等值线间隔
    # print(len(levels))
    plt.contourf(lon, lat, msl[i], levels=levels, cmap="coolwarm", transform=ccrs.PlateCarree())
    # 添加海岸线
    ax.coastlines()
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('Mean sea level pressure')
    # 设置标题和坐标轴标签
    plt.title('Mean sea level pressure')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # 设置网格
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='black', alpha=0.01, linestyle='--')
    # plt.savefig(r'C:\Users\31860\Desktop\v2rayN\a.jpg')
    # 显示图像
    plt.show()


