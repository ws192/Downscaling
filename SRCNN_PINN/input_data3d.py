import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')  # 或 'Qt6Agg'（如果安装的是 PyQt6/PySide6）
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机
setup_seed(888888)
############################################标签数据############################################################

# 1. 加载并验证原始数据
with np.load('DWQ_pinn2.npz') as loaded:
    # 打印基本信息
    print("加载的数组:", list(loaded.keys()))  # 输出['Q', 'tm', 'td', 'um', 'vm',  'wd', 'mld']
    print("单个数组形状 - Q:", loaded['Q'].shape)  # 输出 (1500, 96, 96)

    # 2. 创建数据副本并处理NaN值
    arrays = {name: loaded[name].copy() for name in loaded.files}  # 使用copy()避免修改原始数据

    tmnan = arrays['tm']
    # 3. 处理每个数组（除了xx和yy）
    for name in arrays:
        if name not in ['xx', 'yy']:  # 跳过不需要处理的数组
            arr = arrays[name]

            # 计算非NaN的平均值
            mean_val = np.nanmean(arr)

            # 用平均值填充NaN
            arr_filled = np.where(np.isnan(arr), mean_val, arr)
            # arr_filled = np.where(np.isnan(arr), np.nan, arr)
            # 更新数组
            arrays[name] = arr_filled

            # 打印处理信息
            print(f"{name}: 填充值={mean_val:.4f}, 处理后NaN数量={np.isnan(arr_filled).sum()}")

    # 4. 将处理后的数组赋值给变量
    Q = arrays['Q']
    tm = arrays['tm']
    td = arrays['td']
    um = arrays['um']
    vm = arrays['vm']
    wd = arrays['wd']
    mld = arrays['mld']

# 验证最终变量
print("\n处理后的数组形状验证:")
print("Q.shape:", Q.shape)
print("tm.shape:", tm.shape)
print("Q是否有NaN:", np.isnan(Q).any())  # 应为False

################################################## 观测数据##############################################
tm_lr = tm[1:, ::2, ::2]  # (1499，48,48)

tm_hr = tm[1:]
wd_old = wd[:-1]  # (1499，96,96)
um_old = um[:-1]
vm_old = vm[:-1]
tm_old = tm[:-1]
Q_old = Q[:-1]
mld_old = mld[:-1]
td_old = td[:-1]
#################################################数据展开##################################################
# pic=tmnan[20].reshape(96,96)
# plt.imshow(pic, cmap='jet',vmin=22,vmax=32)
# plt.colorbar()
# plt.show()

# plt.savefig('origin.png')  # 保存图像而不是显示
# plt.close()
###################################### 剔除nan ################################
# valid_mask = ~np.isnan(tm_lr).flatten()  # 获取非 NaN 的布尔掩码（变成 1D）
# td_lr = td_lr[valid_mask]
# wd_lr = wd_lr[valid_mask]
# um_lr = um_lr[valid_mask]
# vm_lr = vm_lr[valid_mask]
# tm_lr = tm_lr[valid_mask]
# Q_lr = Q_lr[valid_mask]
# mld_lr = mld_lr[valid_mask]
# tm_olr=tm_olr[valid_mask]
######################################################
tm_lr = torch.Tensor(tm_lr)
tm_hr = torch.Tensor(tm_hr)

tmmax = tm_lr.max()
tmmin = tm_lr.min()
print("最大值:", tmmax)  # 直接打印最大值
print("最小值:", tmmin)
tm_lr = (tm_lr - tmmin) / (tmmax - tmmin)
tm_hr = (tm_hr - tmmin) / (tmmax - tmmin)

#############################################################
um = torch.Tensor(um_old)
vm = torch.Tensor(vm_old)
wd = torch.Tensor(wd_old)
tm_old = torch.Tensor(tm_old)
td = torch.Tensor(td_old)
Q = torch.Tensor(Q_old)
mld = torch.Tensor(mld_old)
################################################################

old = torch.stack((Q, mld, um, vm, td, wd, tm_old), dim=1)  # B,C,H,W
lr = tm_lr.reshape(-1, 1, 48, 48)  # B,H,W
hr = tm_hr.reshape(-1, 1, 96, 96)

lr_train, old_train, hr_train = lr[0:1200], old[0:1200], hr[0:1200]
lr_valid, old_valid, hr_valid = lr[1200:1350], old[1200:1350], hr[1200:1350]
lr_test, old_test, hr_test = lr[1350:], old[1350:], hr[1350:]
print(lr_train.shape, lr_valid.shape, lr_test.shape)


class MyDataset(Dataset):
    def __init__(self, data, var, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)
        self.var = torch.Tensor(var)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.var[idx], self.label[idx]


batch_size1 = 16
batch_size2 = 16
batch_size3 = 1

trainset = MyDataset(lr_train, old_train, hr_train)
trainloader = DataLoader(trainset, batch_size=batch_size1, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

validset = MyDataset(lr_valid, old_valid, hr_valid)
validloader = DataLoader(validset, batch_size=batch_size2, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

testset = MyDataset(lr_test, old_test, hr_test)
testloader = DataLoader(testset, batch_size=batch_size3, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
