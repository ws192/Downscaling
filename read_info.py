import netCDF4
data=netCDF4.Dataset(r'C:\Users\31860\Desktop\4865e01751bb2cbbdf0f009aaf57b445.nc')
# print(data.variables['u10'].shape)
print(data.variables)

# import numpy
# import numpy as np
#
# np.set_printoptions(threshold=np.inf)
# from netCDF4 import Dataset
#
#
# def read_nc():
#     prefix = (
#         r"/200T/wangsh_pytorch/download/")
#     postfix = r"_u10.nc"
#
#     sst = numpy.zeros((365, 1, 64, 64), dtype=np.float16)
#     base = 1993
#     for i in range(13):
#         pivot = base + i
#         path = prefix + str(pivot) + postfix
#         print(path)
#         sstMatrix = Dataset(path).variables['u10'].reshape(-1, 1, 65, 65)[:, :, :64, :64].filled(0).astype(np.float16)
#         if i == 0:
#             sst = sstMatrix
#         else:
#             sst = np.concatenate((sst, sstMatrix), axis=0)
#     np.save("u10.npy", sst.reshape((5479, 64, 64)))
#
#
# read_nc()
