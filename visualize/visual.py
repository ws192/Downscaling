# 在文件末尾添加特征可视化工具
import matplotlib
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
import numpy as np
import torch
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from einops import rearrange
from matplotlib.patches import Rectangle

# visual.py (正确的方式)
from arch_util import Attention, TransformerBlock


def visualize_super_res_features(features, save_dir="feature_visualization"):
    """
    可视化超分过程中的中间特征图
    :param features: DIRformer返回的特征字典
    :param save_dir: 图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, feat_map in features.items():
        # 处理特征图: [B, C, H, W] -> 取第一个样本 (B=0)
        feat = feat_map[0].detach().cpu()

        # 归一化到 [0, 1]
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

        # 调整特征图尺寸到统一大小 (48x48)
        feat = F.interpolate(feat.unsqueeze(0), size=(48, 48), mode='bilinear').squeeze(0)

        # 选择前16个通道可视化 (4x4网格)
        num_channels = min(16, feat.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(num_channels):
            flipped_feat = np.flipud(feat[i].numpy())
            # flipped_feat = ((flipped_feat + 1) / 2) * 35
            axes[i].imshow(flipped_feat, cmap='jet')
            axes[i].set_title(f"Channel {i + 1}")
            axes[i].axis('off')

        # 保存特征图
        plt.suptitle(f"Feature Map: {name}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
        plt.savefig(f"{save_dir}/{name}_features.png", dpi=300)
        plt.close()

    print(f"特征图已保存至: {os.path.abspath(save_dir)}")


def compute_pixel_contribution(model, lr_input, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算低分辨率输入图像中每个像素对高分辨率输出的贡献度
    :param model: 训练好的超分模型(DiffIRS1)
    :param lr_input: 低分辨率输入图像，形状为 [1, 1, H, W]
    :param device: 计算设备
    :return: 贡献度热力图(归一化到[0,1])和高分辨率输出
    """
    # 设置输入需要梯度
    lr_input = lr_input.to(device).requires_grad_(True)
    model.eval()

    # 前向传播获取高分辨率输出
    with torch.set_grad_enabled(True):
        sr_output, _, _ = model(lr_input, None, None)  # high和hhh设为None（根据模型实际需求调整）

        # 计算输出对输入的梯度（对输出所有像素求和，关注整体贡献）
        grad = autograd.grad(
            outputs=sr_output.sum(),  # 输出像素值之和作为目标
            inputs=lr_input,
            create_graph=False,
            retain_graph=False
        )[0]  # 获取梯度张量 [1, 1, H, W]

    # 处理梯度得到贡献度热力图
    contribution_map = grad.abs().squeeze().cpu().detach().numpy()  # 取绝对值并转为numpy
    contribution_map = np.flipud(contribution_map)
    contribution_map = (contribution_map - contribution_map.min()) / (
            contribution_map.max() - contribution_map.min() + 1e-8)  # 归一化到[0,1]

    return sr_output, contribution_map


def visualize_pixel_contribution(lr_input, contribution_map, save_path="pixel_contribution.png"):
    """
    可视化输入像素贡献度：叠加热力图到原始低分辨率图像
    :param lr_input: 原始低分辨率图像 [1, 1, H, W]
    :param contribution_map: 贡献度热力图 [H, W]
    :param save_path: 图像保存路径
    """
    lr_np = lr_input.squeeze().cpu().detach().numpy()
    lr_np = np.flipud(lr_np)
    h, w = lr_np.shape

    # 创建画布
    plt.figure(figsize=(10, 5))

    # 原始低分辨率图像
    plt.subplot(1, 2, 1)
    plt.imshow(lr_np, cmap='jet', norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))
    plt.title("Low-Resolution Input")
    plt.axis('off')

    # 贡献度热力图叠加
    plt.subplot(1, 2, 2)
    plt.imshow(lr_np, cmap='jet',
               norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))  # 半透明显示原始图像
    heatmap = plt.imshow(contribution_map, cmap='jet')  # 叠加热力图（红=高贡献，蓝=低贡献）
    plt.colorbar(heatmap, label='Contribution Score')
    plt.title("Pixel Contribution Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"像素贡献度可视化结果已保存至: {save_path}")


import torch
import numpy as np
import matplotlib.pyplot as plt
from S1_arch import DiffIRS1
import os

# 在文件末尾添加特征可视化工具
import matplotlib
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
import numpy as np
import torch
from einops import rearrange

from visualize.arch_util import Attention, TransformerBlock


def visualize_super_res_features(features, save_dir="feature_visualization"):
    """
    可视化超分过程中的中间特征图
    :param features: DIRformer返回的特征字典
    :param save_dir: 图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, feat_map in features.items():
        # 处理特征图: [B, C, H, W] -> 取第一个样本 (B=0)
        feat = feat_map[0].detach().cpu()

        # 归一化到 [0, 1]
        # feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

        # 调整特征图尺寸到统一大小 (48x48)
        feat = F.interpolate(feat.unsqueeze(0), size=(48, 48), mode='bilinear').squeeze(0)

        # 选择前16个通道可视化 (4x4网格)
        num_channels = min(16, feat.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(num_channels):
            flipped_feat = np.flipud(feat[i].numpy())
            # flipped_feat = ((flipped_feat + 1) / 2) * 35
            axes[i].imshow(flipped_feat, cmap='jet')
            axes[i].set_title(f"Channel {i + 1}")
            axes[i].axis('off')

        # 保存特征图
        plt.suptitle(f"Feature Map: {name}", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
        plt.savefig(f"{save_dir}/{name}_features.png", dpi=300)
        plt.close()

    print(f"特征图已保存至: {os.path.abspath(save_dir)}")


def compute_pixel_contribution(model, lr_input, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算低分辨率输入图像中每个像素对高分辨率输出的贡献度
    :param model: 训练好的超分模型(DiffIRS1)
    :param lr_input: 低分辨率输入图像，形状为 [1, 1, H, W]
    :param device: 计算设备
    :return: 贡献度热力图(归一化到[0,1])和高分辨率输出
    """
    # 设置输入需要梯度
    lr_input = lr_input.to(device).requires_grad_(True)
    model.eval()

    # 前向传播获取高分辨率输出
    with torch.set_grad_enabled(True):
        sr_output, _, _ = model(lr_input, None, None)  # high和hhh设为None（根据模型实际需求调整）

        # 计算输出对输入的梯度（对输出所有像素求和，关注整体贡献）
        grad = autograd.grad(
            outputs=sr_output.sum(),  # 输出像素值之和作为目标
            inputs=lr_input,
            create_graph=False,
            retain_graph=False
        )[0]  # 获取梯度张量 [1, 1, H, W]

    # 处理梯度得到贡献度热力图
    contribution_map = grad.abs().squeeze().cpu().detach().numpy()  # 取绝对值并转为numpy
    contribution_map = np.flipud(contribution_map)
    contribution_map = (contribution_map - contribution_map.min()) / (
            contribution_map.max() - contribution_map.min() + 1e-8)  # 归一化到[0,1]

    return sr_output, contribution_map


def visualize_pixel_contribution(lr_input, contribution_map, save_path="pixel_contribution.png"):
    """
    可视化输入像素贡献度：叠加热力图到原始低分辨率图像
    :param lr_input: 原始低分辨率图像 [1, 1, H, W]
    :param contribution_map: 贡献度热力图 [H, W]
    :param save_path: 图像保存路径
    """
    lr_np = lr_input.squeeze().cpu().detach().numpy()
    lr_np = np.flipud(lr_np)
    h, w = lr_np.shape

    # 创建画布
    plt.figure(figsize=(10, 5))

    # 原始低分辨率图像
    plt.subplot(1, 2, 1)
    plt.imshow(lr_np, cmap='jet', norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))
    plt.title("Low-Resolution Input")
    plt.axis('off')

    # 贡献度热力图叠加
    plt.subplot(1, 2, 2)
    plt.imshow(lr_np, cmap='jet',
               norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))  # 半透明显示原始图像
    heatmap = plt.imshow(contribution_map, cmap='jet')  # 叠加热力图（红=高贡献，蓝=低贡献）
    plt.colorbar(heatmap, label='Contribution Score')
    plt.title("Pixel Contribution Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"像素贡献度可视化结果已保存至: {save_path}")


# 在文件末尾添加特征可视化工具
import matplotlib
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.autograd as autograd
import numpy as np
import torch
from einops import rearrange

from visualize.arch_util import Attention, TransformerBlock

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import matplotlib.ticker as mticker
import netCDF4 as nc

u_wind_file = r'K:\SCS\era5_10m_u_component_of_wind_2013.nc'
v_wind_file = r'K:\SCS\era5_10m_v_component_of_wind_2013.nc'
day = 169

# --- 读取和准备数据 ---
try:
    with nc.Dataset(u_wind_file, 'r') as u_data_nc:
        lats_wind = u_data_nc.variables['latitude'][:]
        lons_wind = u_data_nc.variables['longitude'][:]
        u10 = u_data_nc.variables['u10'][day, :, :]
    with nc.Dataset(v_wind_file, 'r') as v_data_nc:
        v10 = v_data_nc.variables['v10'][day, :, :]
except FileNotFoundError:
    print(f"错误：请确保文件路径 '{u_wind_file}' 和 '{v_wind_file}' 是正确的。")
    exit()

wind_speed = np.sqrt(u10**2 + v10**2)
lon2d_wind, lat2d_wind = np.meshgrid(lons_wind, lats_wind)
print(lon2d_wind.shape)
print(lat2d_wind.shape)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch.nn.functional as F
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def visualize_super_res_features(features, save_dir="feature_visualization"):
    """
    可视化超分过程中的中间特征图（解决子图坐标重叠问题）
    :param features: DIRformer返回的特征字典
    :param save_dir: 图像保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, feat_map in features.items():
        if name in ['decoder_level2']:
            # 处理特征图: [B, C, H, W] -> 取第一个样本 (B=0)
            feat = feat_map[0].detach().cpu()

            # 调整特征图尺寸到统一大小 (48x48)
            feat = F.interpolate(feat.unsqueeze(0), size=(48, 48), mode='bicubic').squeeze(0)
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)  # 归一化到[0,1]

            # 选择要可视化的通道索引
            channels_to_visualize = [0, 6, 14, 16]

            # 创建子图时增加figsize，给刻度留出更多空间
            fig, axes = plt.subplots(1, 4, figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            axes = axes.flatten()

            # 定义网格坐标（适配48x48特征图）
            x = np.arange(107, 119, step=0.25)
            y = np.arange(8, 20, step=0.25)

            # 用于创建colorbar的mappable对象
            im = None
            # 统一所有子图的colorbar范围（避免每个子图范围不同导致颜色无意义）
            vmin_global = np.nanmin(feat.numpy())
            vmax_global = np.nanmax(feat.numpy())

            for i, channel_idx in enumerate(channels_to_visualize):
                # 获取通道特征（保留原方向，无需翻转，因为投影坐标系已定义方向）
                channel_feat = feat[channel_idx].numpy()

                # 使用统一的颜色范围，确保不同通道特征可对比
                im = axes[i].pcolormesh(
                    x, y, channel_feat,
                    cmap='jet',
                    shading='auto',
                    vmin=np.nanmin(channel_feat),
                    vmax=np.nanmax(channel_feat)
                )
                vmin_global = np.nanmin(channel_feat)
                vmax_global = np.nanmax(channel_feat)
                # 设置子图标题和刻度
                axes[i].set_title(f"Channel {i}", fontsize=16)
                axes[i].set_xticks([107, 113, 119])
                axes[i].set_yticks([8, 14, 20])
                # 调整刻度标签字体大小，避免过大导致重叠
                axes[i].xaxis.set_major_formatter(LongitudeFormatter())
                axes[i].yaxis.set_major_formatter(LatitudeFormatter())
                axes[i].tick_params(axis='both', labelsize=14)  # 缩小刻度字体

                if i in [0]:
                    rect = Rectangle((107, 10), 4, 4, linewidth=2, edgecolor='r', facecolor='none',
                                     transform=ccrs.PlateCarree())
                    axes[i].add_patch(rect)

                # 加粗边框
                for spine in axes[i].spines.values():
                    spine.set_linewidth(1.5)

            # 【核心修改1：调整子图间距】增加水平和垂直间距，避免刻度重叠
            fig.subplots_adjust(
                left=0.05,  # 左边界
                right=0.95,  # 右边界
                bottom=0.1,  # 下边界
                top=0.9,  # 上边界
                wspace=0.3,  # 水平子图间距（关键：调大避免左右刻度重叠）
                hspace=0.2  # 垂直子图间距（关键：调大避免上下刻度重叠）
            )

            # 添加colorbar
            if im:
                cbar2 = fig.colorbar(
                    im,
                    ax=axes.ravel().tolist(),
                    orientation='vertical',
                    aspect=30,
                    shrink=0.4
                )
                ticks2 = np.linspace(vmin_global, vmax_global, num=5)
                cbar2.set_ticks(ticks2)
                cbar2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                cbar2.ax.tick_params(labelsize=14)  # 调整colorbar刻度字体

            plt.show()
            # plt.savefig(f"{save_dir}/{name}_features.png", dpi=300, bbox_inches='tight')  # 保存时裁剪空白
            plt.close()
        if name in ['encoder_level3']:
            # 处理特征图: [B, C, H, W] -> 取第一个样本 (B=0)
            feat = feat_map[0].detach().cpu()

            # 调整特征图尺寸到统一大小 (48x48)
            feat = F.interpolate(feat.unsqueeze(0), size=(48, 48), mode='bicubic').squeeze(0)
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)  # 归一化到[0,1]

            # 选择要可视化的通道索引
            # channels_to_visualize = [0, 1,2,3,4,5,6,7]
            # channels_to_visualize = [8, 9, 10, 11, 12, 13, 14, 15]
            # channels_to_visualize = [16, 17, 18, 19, 20, 21, 22, 23]
            # channels_to_visualize = [24, 25, 26, 27, 28, 29, 30, 31]
            # channels_to_visualize = [32, 33, 34, 35, 36, 37, 38, 39]
            # channels_to_visualize = [40, 41, 42, 43, 44, 45, 46, 47]
            # channels_to_visualize = [48, 49, 50, 51, 52, 53, 54, 55]
            # channels_to_visualize = [56, 57, 58, 59, 60, 61, 62, 63]
            # channels_to_visualize = [64, 65, 66, 67, 68, 69, 70, 71]
            # channels_to_visualize = [72, 73, 74, 75, 76, 77, 78, 79]
            # channels_to_visualize = [80, 81, 82, 83, 84, 85, 86, 87]
            # channels_to_visualize = [88, 89, 90, 91, 92, 93, 94, 95]
            # channels_to_visualize = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
            # channels_to_visualize = [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
            # channels_to_visualize = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]
            start=0
            end=15
            for i in range(1):

                channels_to_visualize = [280,281]

                # 创建子图时增加figsize，给刻度留出更多空间
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()})
                axes = axes.flatten()

                # 定义网格坐标（适配48x48特征图）
                x = np.arange(107, 119, step=0.25)
                y = np.arange(8, 20, step=0.25)

                # 用于创建colorbar的mappable对象
                im = None
                # 统一所有子图的colorbar范围（避免每个子图范围不同导致颜色无意义）
                vmin_global = np.nanmin(feat.numpy())
                vmax_global = np.nanmax(feat.numpy())
                t = ["DSSR3*","DSSR3"]
                for i, channel_idx in enumerate(channels_to_visualize):
                    # 获取通道特征（保留原方向，无需翻转，因为投影坐标系已定义方向）
                    channel_feat = feat[channel_idx].numpy()

                    # 使用统一的颜色范围，确保不同通道特征可对比
                    im = axes[i].pcolormesh(
                        x, y, channel_feat,
                        cmap='jet',
                        shading='auto',
                        vmin=np.nanmin(channel_feat),
                        vmax=np.nanmax(channel_feat)
                    )
                    vmin_global = np.nanmin(channel_feat)
                    vmax_global = np.nanmax(channel_feat)
                    # 设置子图标题和刻度
                    axes[i].set_title(f"{t[i]}", fontsize=16)
                    axes[i].set_xticks([107, 113, 119])
                    axes[i].set_yticks([8, 14, 20])
                    # 调整刻度标签字体大小，避免过大导致重叠
                    axes[i].xaxis.set_major_formatter(LongitudeFormatter())
                    axes[i].yaxis.set_major_formatter(LatitudeFormatter())
                    axes[i].tick_params(axis='both', labelsize=14)  # 缩小刻度字体

                    # if i in [0]:
                    rect = Rectangle((107, 10), 4, 4, linewidth=2, edgecolor='r', facecolor='none',
                                     transform=ccrs.PlateCarree())
                    axes[i].add_patch(rect)

                    # 加粗边框
                    for spine in axes[i].spines.values():
                        spine.set_linewidth(1.5)

                # 【核心修改1：调整子图间距】增加水平和垂直间距，避免刻度重叠
                fig.subplots_adjust(
                    left=0.05,  # 左边界
                    right=0.95,  # 右边界
                    bottom=0.1,  # 下边界
                    top=0.9,  # 上边界
                    wspace=0.3,  # 水平子图间距（关键：调大避免左右刻度重叠）
                    hspace=0.2  # 垂直子图间距（关键：调大避免上下刻度重叠）
                )

                # 添加colorbar
                if im:
                    cbar2 = fig.colorbar(
                        im,
                        ax=axes.ravel().tolist(),
                        orientation='vertical',
                        aspect=30
                        # shrink=0.8
                    )
                    ticks2 = np.linspace(vmin_global, vmax_global, num=5)
                    cbar2.set_ticks(ticks2)
                    cbar2.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
                    cbar2.ax.tick_params(labelsize=14)  # 调整colorbar刻度字体

                plt.show()

    print(f"特征图已保存至: {os.path.abspath(save_dir)}")






def compute_pixel_contribution(model, lr_input, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    计算低分辨率输入图像中每个像素对高分辨率输出的贡献度
    :param model: 训练好的超分模型(DiffIRS1)
    :param lr_input: 低分辨率输入图像，形状为 [1, 1, H, W]
    :param device: 计算设备
    :return: 贡献度热力图(归一化到[0,1])和高分辨率输出
    """
    # 设置输入需要梯度
    lr_input = lr_input.to(device).requires_grad_(True)
    model.eval()

    # 前向传播获取高分辨率输出
    with torch.set_grad_enabled(True):
        sr_output, _, _ = model(lr_input, None, None)  # high和hhh设为None（根据模型实际需求调整）

        # 计算输出对输入的梯度（对输出所有像素求和，关注整体贡献）
        grad = autograd.grad(
            outputs=sr_output.sum(),  # 输出像素值之和作为目标
            inputs=lr_input,
            create_graph=False,
            retain_graph=False
        )[0]  # 获取梯度张量 [1, 1, H, W]

    # 处理梯度得到贡献度热力图
    contribution_map = grad.abs().squeeze().cpu().detach().numpy()  # 取绝对值并转为numpy
    contribution_map = np.flipud(contribution_map)
    contribution_map = (contribution_map - contribution_map.min()) / (
            contribution_map.max() - contribution_map.min() + 1e-8)  # 归一化到[0,1]

    return sr_output, contribution_map


def visualize_pixel_contribution(lr_input, contribution_map, save_path="pixel_contribution.png"):
    """
    可视化输入像素贡献度：叠加热力图到原始低分辨率图像
    :param lr_input: 原始低分辨率图像 [1, 1, H, W]
    :param contribution_map: 贡献度热力图 [H, W]
    :param save_path: 图像保存路径
    """
    lr_np = lr_input.squeeze().cpu().detach().numpy()
    lr_np = np.flipud(lr_np)
    h, w = lr_np.shape

    # 创建画布
    plt.figure(figsize=(10, 5))

    # 原始低分辨率图像
    plt.subplot(1, 2, 1)
    plt.imshow(lr_np, cmap='jet', norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))
    plt.title("Low-Resolution Input")
    plt.axis('off')

    # 贡献度热力图叠加
    plt.subplot(1, 2, 2)
    plt.imshow(lr_np, cmap='jet',
               norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), plt.cm.jet.N))  # 半透明显示原始图像
    heatmap = plt.imshow(contribution_map, cmap='jet')  # 叠加热力图（红=高贡献，蓝=低贡献）
    plt.colorbar(heatmap, label='Contribution Score')
    plt.title("Pixel Contribution Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"像素贡献度可视化结果已保存至: {save_path}")


import torch
import numpy as np
import matplotlib.pyplot as plt
from S1_arch import DiffIRS1
import os


class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}  # 存储注意力图
        self.hooks = []  # 存储hook句柄
        self._register_hooks()
        # 添加调试信息
        print("AttentionVisualizer初始化完成，已注册hook数量:", len(self.hooks))

    def _register_hooks(self):
        """为所有注意力模块注册前向hook"""

        def hook_fn(module, input, output):
            try:
                # 尝试从不同位置获取输入特征
                if isinstance(input, tuple) and len(input) > 0:
                    if isinstance(input[0], list):
                        x = input[0][0]  # 处理列表形式的输入
                    else:
                        x = input[0]  # 默认取第一个元素
                else:
                    x = input

                # 确保x是4D张量 (B, C, H, W)
                if x.dim() != 4:
                    print(f"警告: 输入张量维度不正确，期望4D，实际{x.dim()}D")
                    return

                b, c, h, w = x.shape
                num_heads = module.attn.num_heads

                # 重新执行注意力计算过程以获取权重
                qkv = module.attn.qkv_dwconv(module.attn.qkv(x))
                q, k, v = qkv.chunk(3, dim=1)

                q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=num_heads)
                k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=num_heads)

                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)

                attn = (q @ k.transpose(-2, -1)) * module.attn.temperature
                attn = attn.softmax(dim=-1)

                # 存储注意力权重
                module_id = id(module.attn)
                self.attention_maps[module_id] = attn.detach()
                print(f"成功捕获注意力权重，模块ID: {module_id}")

            except Exception as e:
                print(f"hook函数执行错误: {str(e)}")

        # 遍历模型所有模块，查找包含'attn'属性的模块
        found = 0
        for name, module in self.model.named_modules():
            # 更灵活的模块识别方式
            # if hasattr(module, 'attn') and isinstance(module.attn, Attention):
            #     hook = module.register_forward_hook(hook_fn)
            #     self.hooks.append(hook)
            #     found += 1
            #     print(f"已为模块注册hook: {name}")
            if hasattr(module, 'attn') and module.attn.__class__.__name__ == 'Attention':
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                found += 1
                print(f"已为模块注册hook: {name}")

        if found == 0:
            print("警告: 未找到任何包含Attention的模块")

    def get_attention_maps(self):
        """获取捕获的注意力图"""
        return self.attention_maps

    def visualize_attention(self, input_tensor, layer_idx=0, head_idx=0, save_path=None):
        """
        可视化注意力权重

        Args:
            input_tensor: 输入模型的张量 (batch, channel, h, w)
            layer_idx: 要可视化的注意力层索引
            head_idx: 要可视化的注意力头索引
            save_path: 保存图像的路径，None则直接显示
        """
        # 重置注意力图存储
        self.attention_maps = {}

        # 检查输入和模型设备是否一致
        model_device = next(self.model.parameters()).device
        input_device = input_tensor.device
        if model_device != input_device:
            print(f"警告: 模型和输入设备不一致，模型在{model_device}，输入在{input_device}")
            input_tensor = input_tensor.to(model_device)

        # 前向传播以触发hook
        with torch.no_grad():
            # 尝试不同的模型调用方式
            try:
                self.model(input_tensor, None, None)
            except Exception as e:
                print(f"模型前向传播错误: {str(e)}")
                # 尝试其他可能的调用方式
                try:
                    self.model(input_tensor)
                except Exception as e2:
                    print(f"模型调用方式2错误: {str(e2)}")
                    return

        # 获取所有注意力层的权重
        attention_layers = list(self.attention_maps.values())
        if not attention_layers:
            # 增加更多调试信息
            print("未捕获到注意力权重，调试信息:")
            print(f"模型模块总数: {len(list(self.model.named_modules()))}")
            print(f"已注册hook数量: {len(self.hooks)}")
            raise ValueError("未捕获到注意力权重，请检查模型结构和hook注册情况")

        # ... 其余代码保持不变 ...







        # 获取指定层和头的注意力权重
        attn_weights = attention_layers[layer_idx][0, head_idx]  # (h*w, h*w)

        # 计算平均注意力权重 (对查询维度求平均)
        avg_attn = attn_weights.mean(dim=0)  # (h*w,)

        # 获取空间维度
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        avg_attn = avg_attn.reshape(h, w)  # (h, w)

        # 归一化到0-1
        avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)

        # 上采样到输入图像大小
        avg_attn = avg_attn.reshape(1, 1, h, w)  # 添加 batch 和 channel 维度
        attn_map = F.interpolate(avg_attn, size=(h, w), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

        # 绘制原始图像和注意力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 显示原始图像
        input_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        ax1.imshow(input_img, cmap='gray')
        ax1.set_title('Input Image')
        ax1.axis('off')

        # 显示注意力图
        im = ax2.imshow(attn_map, cmap='jet')
        ax2.set_title(f'Attention Map (Layer {layer_idx}, Head {head_idx})')
        ax2.axis('off')

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label('Attention Weight')

        # 保存或显示图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"注意力图已保存至: {save_path}")
        else:
            plt.show()

        plt.close()

    def __del__(self):
        """清理hook"""
        for hook in self.hooks:
            hook.remove()