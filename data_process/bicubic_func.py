from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
np.set_printoptions(threshold=np.inf)

# 产生16个像素点不同的权重
def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0
# 双三次插值算法
# dstH为目标图像的高，dstW为目标图像的宽
def BiCubic_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    # img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg = np.zeros((dstH, dstW), dtype=np.float64)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= scrH or y + jj >= scrW:
                        continue
                    tmp += img[x + ii, y + jj] * BiBubic(ii - u) * BiBubic(jj - v)
            retimg[i, j] = np.clip(tmp, 0, 307)
    return retimg

from netCDF4 import Dataset
nc_file = Dataset(r"D:\2mtmp_forbi.nc",'a')
print(nc_file)
# print(nc_file.variables['sst'][:,0,:,:][0].shape)
sst=nc_file.variables["t2m"][1001]
# sst=sst.data
image = sst
# print(image.shape[1])
# 举例：将图片统一转换为256*256的图片
image2 = BiCubic_interpolation(image, 160, 160)
sst=image2
print(sst.shape)
lat = nc_file.variables["latitude"][:]
print(lat)
lon = nc_file.variables['longitude'][:]
print(lon)
# 创建绘图窗口
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# 设置绘图的范围
# ax.set_extent([120, 160, 10, 50], crs=ccrs.PlateCarree())
# ax.set_extent([-170, -154, -46, -30], crs=ccrs.PlateCarree())
print(sst.max())
# 绘制数据
# levels = np.arange(10, sosstsst.max(), 0.1)  # 设置等值线间隔
levels = np.linspace(270,304.0897936611791, 100)  # 设置等值线间隔

# print(len(levels))
plt.contourf(lon, lat, sst, levels=levels, cmap="coolwarm",transform=ccrs.PlateCarree())
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


