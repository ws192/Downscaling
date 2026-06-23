import random
import h5py
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
import cmaps

np.set_printoptions(threshold=np.inf)
from netCDF4 import Dataset


#
# file=Dataset(r'K:\model_from_hlt\METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_99.03E-123.97E_0.03N-24.98N_1993-01-01-2020-12-31.nc','a')
# # print(file.variables['sst'][0])
# asst=file.variables['analysed_sst']
# print(asst.shape)
# kkk=np.zeros((10227,500,500))
# for i in range(1):
#     a=asst[i]-273.15
#
#     a=np.where(np.isnan(a),0,a)
#     a=np.where(a<5,0,a)
#
#     kkk[i][:]=a
# np.save(r'GLO-SST.npy',kkk)
import cartopy.mpl.ticker as cticker

def bigsst(sst,size=512):
    for i in range(1):
        lat = np.linspace(start=0, stop=25, num=size)
        lon = np.linspace(start=99, stop=124, num=size)
        fig = plt.figure(figsize=(16, 10))

        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_xticks([0, 5, 10, 15, 20, 25], crs=ccrs.PlateCarree())
        ax.set_yticks([99, 104, 108, 112, 116, 120], crs=ccrs.PlateCarree())
        cmap = plt.cm.get_cmap('jet')
        cmap.set_under('w')
        plt.pcolormesh(lon, lat, sst, cmap=cmap,
                       transform=ccrs.PlateCarree(),
                       norm=matplotlib.colors.BoundaryNorm(np.arange(18, 30, 0.1), cmap.N))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=24)

        # ax.coastlines()

        ax.xaxis.set_tick_params(labelsize=28)
        ax.yaxis.set_tick_params(labelsize=28)
        # labels = ax.get_xticklabels() + ax.get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]

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
        plt.xlabel('Longitude', fontdict={'size': 16, 'family': 'Times New Roman'})
        plt.ylabel('Latitude', fontdict={'size': 16, 'family': 'Times New Roman'})
        plt.show()
def smallsst(sst):
    for i in range(1):

        lat = np.linspace(start=0, stop=25, num=sst.shape[0])
        lon = np.linspace(start=99, stop=124, num=sst.shape[1])
        fig = plt.figure(figsize=(16, 10))
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_xticks([99, 104, 108, 112, 116, 120], crs=ccrs.PlateCarree())
        ax.set_yticks([0, 5, 10, 15, 20, 25], crs=ccrs.PlateCarree())
        cmap = plt.cm.get_cmap('jet')
        cmap.set_under('w')
        plt.pcolormesh(lon, lat, sst, cmap=cmap, transform=ccrs.PlateCarree(),
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



data = np.load(r'K:\Data\GLO64.npy')[9131]
# data2 = np.load(r'D:\Ckpt\hla.npy')[2922:9131]


print(data.max())
print(data.min())



# for i in range(10227):
#     data2[i][:]=np.where(data[i]<2,0,data2[i])
#
#
# np.save(r'D:\Ckpt\noaa256_patched.npy',data2)
#
#
# low_res_sst=data2[8410]
# print(low_res_sst)



# np.save(r'D:\Ckpt\data_GLO256.npy',data)




# for i in range(6211):
#     a=low_res_sst[2922+i]
#     aa1=low_res_sst[2922+i-1]
#     aa2=low_res_sst[2922+i-2]
#     aa3=low_res_sst[2922+i+1]
#     aa4=low_res_sst[2922+i+2]
#     aaaa[i,0][:]=aa2
#     aaaa[i,1][:]=aa1
#     aaaa[i,2][:]=a
#     aaaa[i,3][:]=aa3
#     aaaa[i,4][:]=aa4
# np.save(r'D:\Ckpt\data_GLO128noisemd.npy',aaaa)



# low_res_sst1 = np.load(r'D:\Ckpt\data_GLO256.npy')[10000]
# low_res_sst2 = np.load(r'D:\Ckpt\data_GLO256noise.npy')[10000]


low_res_sst1 = np.load(r'C:\Users\31860\Desktop\DiffI2I_diffusionfixed\dfixed.npy')
low_res_sst2 = np.load(r'D:\Ckpt\data_GLO256.npy')[10000]
from util import *
# plot_density_scatter(low_res_sst1,low_res_sst2,'DIFFIR')
# plot_hist(low_res_sst1,low_res_sst2,'DIFFIR')
plot_bias(low_res_sst1,low_res_sst2,'DIFFIRM+SLA')
# smallsst(low_res_sst1)
# smallsst(low_res_sst2)





# for i in range(10227):
#     aa = low_res_sst[i][:]
#     mean = 0  # 噪声的平均值
#     std = 0.2  # 噪声的标准差
#     # 生成与 downsampled_data 形状相同的高斯噪声数组
#     gaussian_noise = np.random.normal(mean, std, aa.shape)
#     bbbbb=aa+gaussian_noise
#     bbbbb=np.where(aa<3,0,bbbbb)
#     aaaa[i][:]=bbbbb


# np.save(r'D:\Ckpt\data_GLO64_noise.npy',aaaa)
# low_res_sst = np.load(r'D:\Ckpt\data_GLO256.npy')
# aaaa=np.zeros((10227,256,256))
# b=low_res_sst[0][:]
# for i in range(10227):
#     a=low_res_sst[i][:]
#     b=np.where(a<2,0,b)
#
# for i in range(10227):
#     c=low_res_sst[i][:]
#     c=np.where(b<2,0,c)
#     aaaa[i][:]=c
#
# np.save(r'D:\Ckpt\data_GLO256.npy',aaaa)


# low_res_sst2 = np.load(r'D:\Ckpt\noaal.npy')
# # # a=np.where(low_res_sst2<2,0,low_res_sst)
# np.save(r'D:\Ckpt\GLO2562.npy',np.where(low_res_sst<2,0,low_res_sst2))


# a=np.load(r'K:\model_from_hlt\2668.npy')
# # print(a.shape)
# bigsst(a,size=256)
#
# a=torch.nn.functional.mse_loss(torch.Tensor(low_res_sst[9856].reshape(1,1,320, 320)),torch.Tensor(low_res_sst2[9856].reshape(1,1,320, 320)))
# print(torch.sqrt(a))
# with torch.no_grad():
#
#     # high_res_sst = np.load(r'D:\Ckpt\GLOlatent.npy').reshape(-1,1)
#     autoencoder = torch.load(r'K:\1.31Autoencoderkl\akl368000.pt')
#     # sst = autoencoder.decode(torch.Tensor(low_res_sst[9856].reshape(1, 4, 40, 40)).cuda()).sample.cpu().numpy()
#     v=low_res_sst[9856].reshape(1, 1, 320, 320)
#     v=v/34
#     v=2*v-1
#     # sst, posterior = autoencoder(sample=torch.Tensor(v).cuda(), return_dict=False)
#     latent = autoencoder.encode(torch.Tensor(v).cuda()).latent_dist.mode()
#     sst = autoencoder.decode(latent).sample
#
#     # sst=(scaler_for_high.inverse_transform(image.reshape(1, -1))).reshape(128, 128)
#     # sst = ((((sst.reshape(320, 320)) + 1) / 2)) * 34
#     # sst=(scaler_for_high.inverse_transform(image.reshape(1, -1))).reshape(128, 128)
#     sst =((sst.cpu().numpy().reshape(320, 320))/2 + 0.5)  * 34
#     bigsst(sst, size=320)

# smallsst(bbbbb)

# bigsst(bbb)
# smallsst(file.variables['sst'][0]-273.15)
# print((file.variables['sst'][0].filled(0)))
# smallsst(np.load(r'K:\model_from_hlt\ERA5_daily2.npy')[0])
# print(np.load(r'K:\model_from_hlt\ERA5_daily.npy')[0])
# print(np.load(r'K:\model_from_hlt\NOAA_OI_V22.npy').shape)
# a=np.load(r'K:\model_from_hlt\GLO80.npy')
# b=np.load(r'K:\model_from_hlt\NOAA-80.npy')
# c=np.where(a<2,0,b)
# np.save(r'K:\model_from_hlt\NOAA-802.npy',c)

# dc=Dataset(r'K:\model_from_hlt\19930105-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIARAN.nc','a')
# bigsst(dc.variables['analysed_sst'][0][1800:2300, 5580:6080]-273.15)


