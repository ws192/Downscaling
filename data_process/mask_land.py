import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



#根据land sea mask除去陆地数据，只保留海洋数据



# 读取netCDF文件
nc_file_hf = nc.Dataset(r'D:\e.nc','a')
nc_file_lsm=nc.Dataset(r'D:\lsm.nc')
# 读取平均表面压力数据和land sea mask数据
hf = nc_file_hf.variables['e'][:]
lsm_data = nc_file_lsm.variables['lsm'][:]
# print(msl)
# print(lsm_data)

# with open(r'C:\Users\31860\Desktop\a.txt', 'w') as file:
#     # 遍历数据并将其写入文件
#     for i in range(721):
#         for j in range(1440):
#             file.write("{:.2f} ".format(lsm_data[0,i,j]))
#         file.write("\n")

#
# 创建一个全零数组，与平均表面压力数据的维度相同
masked_slhf_data = np.zeros_like(hf)

for i in range(1003):
    # 将平均表面压力数据按照land sea mask进行掩码操作
    masked_slhf_data[i] = np.ma.masked_where(lsm_data[0] != 0, hf[i])

nc_file_hf.variables['e'][:]=masked_slhf_data
print(masked_slhf_data)
print(masked_slhf_data.shape)

lat = nc_file_hf.variables["latitude"][:]
# print(lat.shape)
lon = nc_file_hf.variables['longitude'][:]
# print(lon.shape)
for i in range(36):
    # 创建绘图窗口
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # 设置绘图的范围
    # ax.set_extent([40, 80, -10, 30], crs=ccrs.PlateCarree())
    # 绘制数据
    # levels = np.arange(10, sosstsst.max(), 0.1)  # 设置等值线间隔
    levels = np.linspace(hf.min(), hf.max(), 10)  # 设置等值线间隔
    # print(len(levels))
    plt.contourf(lon, lat, hf[i], levels=levels, cmap="Paired", transform=ccrs.PlateCarree())
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
    # gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.01, linestyle='--')
    # plt.savefig(r'C:\Users\31860\Desktop\v2rayN\a.jpg')
    # 显示图像
    plt.show()

