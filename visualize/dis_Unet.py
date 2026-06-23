from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
# class add_attn(nn.Module):
#
#     def __init__(self, x_channels, g_channels=256):
#         super(add_attn, self).__init__()
#         self.W = nn.Sequential(
#             nn.Conv2d(x_channels, x_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(x_channels))
#         self.theta = nn.Conv2d(x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False)
#
#         self.phi = nn.Conv2d(g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         self.psi = nn.Conv2d(x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
#
#     def forward(self, x, g):
#         input_size = x.size()
#         batch_size = input_size[0]
#         assert batch_size == g.size(0)
#
#         theta_x = self.theta(x)
#         theta_x_size = theta_x.size()
#         print(theta_x_size[2:])
#         print(self.phi(g).shape)
#         phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='nearest')
#         f = F.relu(theta_x + phi_g, inplace=True)
#
#         sigm_psi_f = torch.sigmoid(self.psi(f))
#         sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='nearest')
#
#         y = sigm_psi_f.expand_as(x) * x
#         W_y = self.W(y)
#         return W_y
# class unetCat(nn.Module):
#
#     def __init__(self, dim_in, dim_out):
#         super(unetCat, self).__init__()
#         norm = spectral_norm
#         self.convU = norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))
#
#     def forward(self, input_1, input_2):
#         # Upsampling
#         input_2 = F.interpolate(input_2, scale_factor=2, mode='nearest')
#
#         output_2 = F.leaky_relu(self.convU(input_2), negative_slope=0.2, inplace=True)
#
#         offset = output_2.size()[2] - input_1.size()[2]
#         padding = 2 * [offset // 2, offset // 2]
#         output_1 = F.pad(input_1, padding)
#         y = torch.cat([output_1, output_2], 1)
#         return y
# # @ARCH_REGISTRY.register()
# class UNetDiscriminatorAesrgan(nn.Module):
#     """Defines a U-Net discriminator with spectral normalization (SN)"""
#
#     def __init__(self, num_in_ch=1, num_feat=64, skip_connection=True):
#         super(UNetDiscriminatorAesrgan, self).__init__()
#         norm = spectral_norm
#
#         self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
#
#         self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False))
#         self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False))
#
#         # Center
#         self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False))
#
#         self.gating = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))
#
#         # attention Blocks
#         self.attn_1 = add_attn(x_channels=num_feat * 4, g_channels=num_feat * 4)
#         self.attn_2 = add_attn(x_channels=num_feat * 2, g_channels=num_feat * 4)
#         self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)
#
#         # Cat
#         self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
#         self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
#         self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)
#
#         # upsample
#         self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
#         self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
#         self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
#
#         # extra
#         self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
#         self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
#         self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
#
#     def forward(self, x):
#         x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
#         x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
#         x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
#         x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
#
#         gated = F.leaky_relu(self.gating(x3), negative_slope=0.2, inplace=True)
#
#         # Attention
#         attn1 = self.attn_1(x2, gated)
#         attn2 = self.attn_2(x1, gated)
#         attn3 = self.attn_3(x0, gated)
#
#         # upsample
#         x3 = self.cat_1(attn1, x3)
#         x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
#         x4 = self.cat_2(attn2, x4)
#         x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
#         x5 = self.cat_3(attn3, x5)
#         x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
#
#         # extra
#         out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
#         out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
#         out = self.conv9(out)
#
#         return out



# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor, nn
from torch.nn import functional as F_torch
from torch.nn.utils import spectral_norm

# __all__ = [
#     "DiscriminatorForUNet",
#     "discriminator_for_unet",
# ]


class DiscriminatorForUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            channels: int = 96,
            upsample_method: str = "bilinear",
    ) -> None:
        r"""Discriminator for UNet.

        Args:
            in_channels (int, optional): Number of channels in the input image. Default is 3.
            out_channels (int, optional): Number of channels in the output image. Default is 1.
            channels (int, optional): Number of channels in the intermediate layers. Default is 64.
            upsample_method (str, optional): The upsample method. Default is "bilinear".
        """
        super(DiscriminatorForUNet, self).__init__()
        if out_channels != 1:
            raise ValueError("The output channels must be 1.")

        self.upsample_method = upsample_method

        self.conv_1 = nn.Conv2d(in_channels, 96, 3, stride=1, padding=1)
        self.down_block_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, int(channels * 2), 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), int(channels * 4), 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 8), 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 8), int(channels * 4), 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 2), 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), channels, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_4 = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)


        self.upsamppp1=Upsample(n_feat=int(channels * 8))
        self.upsamppp2=Upsample(n_feat=int(channels * 4))
        self.upsamppp3=Upsample(n_feat=int(channels * 2))


    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)

        # Down-sampling
        down_1 = self.down_block_1(conv_1)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)

        # Up-sampling
        # down_3 = F_torch.interpolate(down_3, scale_factor=2, mode=self.upsample_method, align_corners=False)
        down_3 = self.upsamppp1(down_3)
        up_1 = self.up_block_1(down_3)

        up_1 = torch.add(up_1, down_2)
        # up_1 = F_torch.interpolate(up_1, scale_factor=2, mode=self.upsample_method, align_corners=False)
        up_1 = self.upsamppp2(up_1)
        up_2 = self.up_block_2(up_1)

        up_2 = torch.add(up_2, down_1)
        # up_2 = F_torch.interpolate(up_2, scale_factor=2, mode=self.upsample_method, align_corners=False)
        up_2 = self.upsamppp3(up_2)
        up_3 = self.up_block_3(up_2)

        up_3 = torch.add(up_3, conv_1)

        x = self.conv_2(up_3)
        x = self.conv_3(x)
        return self.conv_4(x)
#
#
# def discriminator_for_unet(**kwargs) -> DiscriminatorForUNet:
#     return DiscriminatorForUNet(**kwargs)

# if __name__ == "__main__":
#     from torchsummary import summary
#     uNet = DiscriminatorForUNet(1).cuda()
#     summary(uNet, (1, 192, 192), batch_size=1)