import os
import random
import h5py
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy
import torch.nn.functional as F
import numpy as np
import torch
import warnings

from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from dataclasses import dataclass
from netCDF4 import Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

warnings.filterwarnings("ignore")

high_res_sst = np.load(r"D:\dataStack.npy")[4380:4380+365*3, 0:10, 24:144 + 24, 0:144 + 0]
np.save(r"D:\jgr\highdatagt.npy", high_res_sst)
print(high_res_sst.shape)
# high_res_sst=torch.Tensor(high_res_sst)
# high_resolution = 144
# low_resolution = 36
#
#
# def sr(inn, height, mode):
#     innn = torch.Tensor(inn).reshape((1, 1, height, height))
#     print(innn.shape)
#     SR = torch.nn.functional.interpolate(innn, size=(low_resolution, low_resolution), mode=mode).cpu().numpy()
#     return SR.reshape(low_resolution, low_resolution)
#
#
# # low_res_sst = np.load(r"D:\SST\tmplow.npy")[:]
#
# def smooth_sea_temperature(sea_temperature, sigma=0.5):
#     # 创建一个掩码，标识海洋区域
#     ocean_mask = ~np.isnan(sea_temperature)
#     # 用0填充NaN以进行滤波
#     filled_sea_temperature = np.where(ocean_mask, sea_temperature, 0)
#     # 应用高斯滤波
#     smoothed_temperature = gaussian_filter(filled_sea_temperature, sigma=sigma)
#     # 计算权重
#     weight_mask = gaussian_filter(ocean_mask.astype(float), sigma=sigma)
#     # 正常化平滑后的温度，仅在海洋区域有效
#     with np.errstate(invalid='ignore'):
#         smoothed_temperature /= weight_mask
#     # 保持陆地区域为NaN
#     smoothed_temperature[~ocean_mask] = np.nan
#     return smoothed_temperature
#
#
# with torch.no_grad():
#     tmp = np.zeros((5479, 12, low_resolution, low_resolution), dtype=np.float16)
#     for m in range(5479):
#         for n in range(12):
#             # # lowwwwwwwwwwwwwwwwwwwwwwwwwwww dataset
#             gt = high_res_sst[m][n].reshape(1, 1, high_resolution, high_resolution).cuda()
#             nearSST = F.interpolate(gt, mode='nearest', size=(low_resolution, low_resolution)).cpu().numpy()
#             nearSST = nearSST.reshape(low_resolution, low_resolution)
#             nearSST = np.where(nearSST < 1, np.nan, nearSST)
#             smoothed_temperature = smooth_sea_temperature(nearSST, sigma=1.0)
#             smoothed_temperature[np.isnan(smoothed_temperature)] = 0
#             z = smoothed_temperature.reshape((low_resolution, low_resolution))
#             tmp[m][n][:] = z
#             print(str(m) + r':' + str(n))
#     np.save(r"K:\lowdata4xf10.npy", tmp)
#     print(tmp[0][0])
#
# # low_res_sst = np.load(r"D:\dataStack.npy")[:, :, :, :]
