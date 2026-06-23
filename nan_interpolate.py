import matplotlib
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker

np.set_printoptions(threshold=np.inf)


def nan_interpolate(low_sst, high_mask):
    lat_src = np.linspace(0, 8, 80)  # 纬度
    lon_src = np.linspace(0, 8, 80)  # 经度
    lon_src_grid, lat_src_grid = np.meshgrid(lon_src, lat_src)  # 源网格
    sst_src = low_sst  # 假定的海温数据
    sst_src = np.where(sst_src < 0.5, np.nan, sst_src)
    land_mask_src = np.isnan(sst_src)  # 假定陆地为 nan

    # 2. 创建目标网格 (1/12° 分辨率，128×128网格)
    lat_tgt = np.linspace(0, 8, 400)
    lon_tgt = np.linspace(0, 8, 400)
    lon_tgt_grid, lat_tgt_grid = np.meshgrid(lon_tgt, lat_tgt)

    # 假设新的高分辨率海陆掩膜是海洋为 0，陆地为 np.nan
    land_mask_tgt = high_mask

    # 3. 使用 scipy 的 griddata 进行双线性插值，仅插值海洋区域
    # 构造插值点和目标点
    points_src = np.array([lon_src_grid[~land_mask_src].flatten(),
                           lat_src_grid[~land_mask_src].flatten()]).T  # 非陆地的点
    values_src = sst_src[~land_mask_src].flatten()  # 非陆地点的值

    points_tgt = np.array([lon_tgt_grid.flatten(), lat_tgt_grid.flatten()]).T

    # 双线性插值
    sst_tgt = griddata(points_src, values_src, points_tgt, method='cubic').reshape(144, 144)
    sst_tgt[np.isnan(land_mask_tgt)] = np.nan
    sst_tgt = np.nan_to_num(sst_tgt)
    # # 4. 确保陆地填充为 np.nan
    # # sst_tgt[np.isnan(land_mask_tgt)] = np.nan  # 在目标掩膜中陆地区域强制设置为 nan
    #
    # # 5. 对剩余可能的空值使用最近邻插值来填充，保证整个海洋区域都有值
    sst_tmp = griddata(points_src, values_src, points_tgt, method='nearest').reshape(144, 144)
    sst_tmp[np.isnan(land_mask_tgt)] = np.nan
    sst_tmp = np.nan_to_num(sst_tmp)
    #
    mask = (sst_tgt == 0) & (sst_tmp != 0)
    sst_tgt[mask] = sst_tmp[mask]

    # print(sst_tgt)
    return sst_tgt


import numpy as np


# def bilinear_interpolation(lon_src, lat_src, sst_src, lon_tgt, lat_tgt):
#     """
#     手动实现双线性插值。
#     lon_src, lat_src: 源网格的经纬度坐标（一维数组）
#     sst_src: 源网格的值，2D 数组 (纬度 x 经度)
#     lon_tgt, lat_tgt: 目标网格的经纬度坐标（一维数组）
#     返回：在目标网格上的插值值 (2D 数组)
#     """
#     # 创建结果数组
#     sst_tgt = np.full((len(lat_tgt), len(lon_tgt)), np.nan)  # 初始化为 nan
#
#     for i, lat in enumerate(lat_tgt):  # 遍历目标网格的每个纬度点
#         for j, lon in enumerate(lon_tgt):  # 遍历目标网格的每个经度点
#             # 找到 lat_src 和 lon_src 中小于且最近和大于且最近的点
#             if lon < lon_src[0] or lon > lon_src[-1] or lat < lat_src[0] or lat > lat_src[-1]:
#                 continue  # 超出源网格范围则跳过
#             lon1_idx = np.searchsorted(lon_src, lon) - 1
#             lon2_idx = lon1_idx + 1
#             lat1_idx = np.searchsorted(lat_src, lat) - 1
#             lat2_idx = lat1_idx + 1
#
#             # 获取源网格的 4 个相邻点的经纬度
#             lon1, lon2 = lon_src[lon1_idx], lon_src[lon2_idx]
#             lat1, lat2 = lat_src[lat1_idx], lat_src[lat2_idx]
#
#             # 获取 4 个点的值（考虑 nan 的区域）
#             q11 = sst_src[lat1_idx, lon1_idx]
#             q12 = sst_src[lat2_idx, lon1_idx]
#             q21 = sst_src[lat1_idx, lon2_idx]
#             q22 = sst_src[lat2_idx, lon2_idx]
#
#             # 如果任何 corner 值为 nan，则跳过此点的插值
#             if np.isnan(q11) or np.isnan(q12) or np.isnan(q21) or np.isnan(q22):
#                 continue
#
#             # 双线性插值公式
#             sst_tgt[i, j] = (
#                                     q11 * (lon2 - lon) * (lat2 - lat) +
#                                     q21 * (lon - lon1) * (lat2 - lat) +
#                                     q12 * (lon2 - lon) * (lat - lat1) +
#                                     q22 * (lon - lon1) * (lat - lat1)
#                             ) / ((lon2 - lon1) * (lat2 - lat1))
#
#     return sst_tgt
#
#
# def nan_interpolate(low_sst, high_mask):
#     lat_src = np.linspace(0, 8, 32)  # 纬度
#     lon_src = np.linspace(0, 8, 32)  # 经度
#     lon_src_grid, lat_src_grid = np.meshgrid(lon_src, lat_src)  # 源网格
#     sst_src = low_sst  # 假定的海温数据
#     sst_src = np.where(sst_src < 0.5, np.nan, sst_src)
#     land_mask_src = np.isnan(sst_src)  # 假定陆地为 nan
#
#     # 2. 创建目标网格
#     lat_tgt = np.linspace(0, 8, 128)
#     lon_tgt = np.linspace(0, 8, 128)
#     lon_tgt_grid, lat_tgt_grid = np.meshgrid(lon_tgt, lat_tgt)
#
#     # 假设新的高分辨率海陆掩膜是海洋为 0，陆地为 np.nan
#     land_mask_tgt = high_mask
#
#     # 3. 使用手写的双线性插值，仅对海洋区域处理
#     sst_tgt = bilinear_interpolation(lon_src, lat_src, sst_src, lon_tgt, lat_tgt)
#
#     # 4. 确保陆地填充为 np.nan
#     sst_tgt[np.isnan(land_mask_tgt)] = np.nan  # 在目标掩膜中填充陆地区域为 nan
#
#     return sst_tgt


def plottt(sst):
    for i in range(1):
        lat = np.linspace(start=6, stop=22, num=sst.shape[0])
        lon = np.linspace(start=107, stop=123, num=sst.shape[1])
        fig = plt.figure(figsize=(8, 8))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_xticks([107, 115, 123], crs=ccrs.PlateCarree())
        ax.set_yticks([6, 14, 22], crs=ccrs.PlateCarree())
        cmap = plt.cm.get_cmap('jet')
        cmap.set_under('w')
        plt.pcolormesh(lon, lat, sst, cmap=cmap, transform=ccrs.PlateCarree(),
                       norm=matplotlib.colors.BoundaryNorm(np.arange(14, 30, 0.1), cmap.N))
        # plt.pcolormesh(lon, lat, sst, cmap=cmap, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('SST(°C)', size=14, family='Times New Roman')

        # ax.coastlines()
        # ax.add_feature(cartopy.feature.LAND)
        # ax.add_feature(cartopy.feature.OCEAN)
        # # ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.3)
        # ax.add_feature(cartopy.feature.BORDERS, linestyle=':', linewidth=0.3)
        # ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
        # ax.add_feature(cartopy.feature.RIVERS)

        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # # 设置横纵坐标文本的格式
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()

        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.tight_layout()

        # 调整坐标文本的位置和间距
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_tick_params(pad=10)
        ax.yaxis.set_tick_params(pad=10)
        plt.show()

# hla=np.load(r'C:\Users\31860\Desktop\hla.npy')
# print(hla.min())
# print(hla.max())

low_res_sst_stack = np.load(r'C:\Users\31860\Desktop\tmplow36.npy')
high_res_sst_stack = np.load(r"D:\dataStack.npy")[:, :, 24:144 + 24, 0:144 + 0]
result = np.zeros((5479, 12, 144, 144), dtype=np.float16)
for m in range(5479):
    for n in range(12):
        low_sst = low_res_sst_stack[m][n][:]
        high_res_sst = high_res_sst_stack[m][n][:]
        high_mask = np.zeros((144, 144))  # 目标海陆掩膜
        high_mask = np.where(high_res_sst < 0.5, np.nan, high_mask)
        rr = nan_interpolate(low_sst, high_mask)

        result[m][n][:] = rr

        # plottt(rr)
    print(m)

np.save(r'D:\cubicLow144.npy', result)
# plottt(high_res_sst)
# plottt(nan_interpolate(low_sst, high_mask))
