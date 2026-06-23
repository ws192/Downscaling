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
from tqdm.auto import tqdm
from diffusers import AutoencoderKL
from dataclasses import dataclass
from netCDF4 import Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")

warnings.filterwarnings("ignore")

high_res_sst = np.load(r"D:\dataStack.npy")[:, :, 36:128 + 36, 12:128 + 12]
low_res_sst = np.load(r"D:\SST\tmplow.npy")[:]

with torch.no_grad():
    for m in range(5479):
        for n in range(12):
            gt = high_res_sst[m][n].reshape((128, 128))
            smoothed_temperature = low_res_sst[m][n].reshape((128, 128))
            mask = (smoothed_temperature == 0) & (gt != 0)
            smoothed_temperature[mask] = gt[mask]
            smoothed_temperature = np.where(gt < 0.5, 0, smoothed_temperature)
            low_res_sst[m][n][:] = smoothed_temperature
            print(str(m) + r':' + str(n))
    np.save(r"D:\SST\tmplowMMM.npy", low_res_sst)
