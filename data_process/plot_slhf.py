import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
nc_file = Dataset(r'D:\2mtmp_forbi.nc','a')
print(nc_file)
slhf=nc_file.variables['t2m'][1001]
print(slhf)
# print(slhf.min())

lat = nc_file.variables["latitude"][:]
print(lat)
lon = nc_file.variables['longitude'][:]
print(lon)
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
# ax.set_extent([40, 80, -10, 30], crs=ccrs.PlateCarree())
# levels = np.array([-0.5,0.5,1.5])  # 设置等值线间隔
levels = np.linspace(slhf.min(), slhf.max(), 50)  # 设置等值线间隔
# print(len(levels))
plt.contourf(lon, lat, slhf, levels=levels, cmap="coolwarm", transform=ccrs.PlateCarree())
ax.coastlines()
cbar = plt.colorbar()
cbar.set_label('Surface latent heat flux')
plt.title('Surface latent heat flux Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.01, linestyle='--')
# plt.savefig(r'C:\Users\31860\Desktop\v2rayN\a.jpg')
plt.show()



