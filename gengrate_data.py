import numpy as np
from util import *

# np.random.seed(3407)

# noised_len = 7000
# # 生成4000个[0, 9860)之间的随机整数
# random_integers = np.random.randint(0, 10227, size=noised_len)


# low_res_sst = np.load(r'D:\Ckpt\data_GLO64_noise.npy')
# high_res_sst = np.load(r'D:\Ckpt\data_GLO256.npy')
#
# whole_low=np.zeros((10227,64,64))
# whole_high=np.zeros((10227,256,256))
#
#
#
# for i in range(10227):
#     index = i
#     tmp_low = np.zeros((64, 64))
#     tmp_high = np.zeros((256, 256))
#
#     tmp_low[:] = low_res_sst[index]
#     tmp_high[:] = high_res_sst[index]
#
#     if np.random.rand() < 0.5:
#         tmp_low[:] = add_gaussian_noise(tmp_low)
#     else:
#         tmp_low[:] = add_poisson_noise(tmp_low)
#
#     whole_low[i][:] = tmp_low
#     whole_high[i][:] = tmp_high
#
#
# np.save(r'D:\Ckpt\whole_256.npy',whole_high)
# np.save(r'D:\Ckpt\whole_64.npy',whole_low)


# np.random.seed(23333)
# noised_len = 5000
# # 生成4000个[0, 9860)之间的随机整数
# random_integers = np.random.randint(0, 10227, size=noised_len)


low_res_sst = np.load(r'D:\Ckpt\data_GLO64_masked.npy')
high_res_sst = np.load(r'D:\Ckpt\data_GLO256.npy')

whole_low=np.zeros((9862*3+365,64,64))
whole_high=np.zeros((9862*3+365,256,256))

for i in range(9862):
    whole_low[i][:]=low_res_sst[i]
    whole_high[i][:]=high_res_sst[i]

k=9862
for i in range(1):
    index_a = 160
    index_b = 9861-index_a

    tmp_low_a = np.zeros((64, 64))
    tmp_low_b = np.zeros((64, 64))
    tmp_high_a = np.zeros((256, 256))
    tmp_high_b = np.zeros((256, 256))

    tmp_low_a[:] = low_res_sst[index_a]
    tmp_low_b[:] = low_res_sst[index_b]
    tmp_high_a[:] = high_res_sst[index_a]
    tmp_high_b[:] = high_res_sst[index_b]

    a = np.zeros((15, 15))
    b = np.zeros((15, 15))
    a[:] = tmp_low_a[11:26, 20:35]
    b[:] = tmp_low_b[34:49, 30:45]
    tmp_low_a[11:26, 20:35][:] = b
    tmp_low_b[34:49, 30:45][:] = a

    ha = np.zeros((60, 60))
    hb = np.zeros((60, 60))
    ha[:] = tmp_high_a[44:104, 80:140]
    hb[:] = tmp_high_b[136:196, 120:180]
    tmp_high_a[44:104, 80:140][:] = hb
    tmp_high_b[136:196, 120:180][:] = ha

    # a = np.zeros((15, 15))
    # b = np.zeros((15, 15))
    # a[:] = tmp_low_a[34:49, 30:45]
    # b[:] = tmp_low_b[42:57, 39:54]
    # tmp_low_a[34:49, 30:45][:] = b
    # tmp_low_b[42:57, 39:54][:] = a
    # ha = np.zeros((60, 60))
    # hb = np.zeros((60, 60))
    # ha[:] = tmp_high_a[136:196, 120:180]
    # hb[:] = tmp_high_b[168:228, 156:216]
    # tmp_high_a[136:196, 120:180][:] = hb
    # tmp_high_b[168:228, 156:216][:] = ha

    a = np.zeros((15, 15))
    b = np.zeros((15, 15))
    a[:] = tmp_low_a[42:57, 39:54]
    b[:] = tmp_low_b[11:26, 20:35]
    tmp_low_a[42:57, 39:54][:] = b
    tmp_low_b[11:26, 20:35][:] = a
    ha = np.zeros((60, 60))
    hb = np.zeros((60, 60))
    ha[:] = tmp_high_a[168:228, 156:216]
    hb[:] = tmp_high_b[44:104, 80:140]
    tmp_high_a[168:228, 156:216][:] = hb
    tmp_high_b[44:104, 80:140][:] = ha

    lat = np.linspace(start=0, stop=25, num=tmp_high_b.shape[0])
    lon = np.linspace(start=99, stop=124, num=tmp_high_b.shape[1])
    fig = plt.figure(figsize=(16, 10))
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_xticks([99, 104, 108, 112, 116, 120], crs=ccrs.PlateCarree())
    ax.set_yticks([0, 5, 10, 15, 20, 25], crs=ccrs.PlateCarree())
    cmap = plt.cm.get_cmap('jet')
    cmap.set_under('w')
    plt.pcolormesh(lon, lat, tmp_high_b, cmap=cmap, transform=ccrs.PlateCarree(),
                   norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), cmap.N))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=24)

    ax.coastlines()

    ax.xaxis.set_tick_params(labelsize=28)
    ax.yaxis.set_tick_params(labelsize=28)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置横纵坐标文本的格式
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # 调整坐标文本的位置和间距
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(pad=10)
    ax.yaxis.set_tick_params(pad=10)

    plt.show()


    whole_low[k][:] = tmp_low_a
    whole_high[k][:] = tmp_high_a
    k += 1

    whole_low[k][:] = tmp_low_b
    whole_high[k][:] = tmp_high_b
    k += 1

for i in range(365):
    whole_low[i+9862*3][:] =low_res_sst[i+9862]
    whole_high[i+9862*3][:] =high_res_sst[i+9862]


# np.save(r'D:\Ckpt\whole_256_cutMix.npy',whole_high)
# np.save(r'D:\Ckpt\whole_64_cutMix.npy',whole_low)