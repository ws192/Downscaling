import common as common
import torch.nn as nn
import numbers
from einops import rearrange
from arch_util import *
import torch.nn.functional as F
import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class DIRformer(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 scale=1,
                 dim=32,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[4, 4, 4, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(DIRformer, self).__init__()
        self.scale = scale
        inp_channels = 1
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # self.SSHE=HAT()
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        # common.Upsampler(common.default_conv, 1, int(dim * 2 ** 1), act=False),
        modules_tail = [common.Upsampler(common.default_conv, 3, int(dim * 2 ** 1), act=False),
                        nn.LeakyReLU(0.2, inplace=True), common.default_conv(int(dim * 2 ** 1), out_channels, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, low, k_v, SSH, hhh):
        # Is better than feat=low, modules_tail_scale=4
        # Maybe feat = F.interpolate(low, scale_factor=4, mode='nearest'),modules_tail_scale=1 is even better
        SSHF = None
        # SSHF=SSHF.repeat(low.shape[0],1,1,1)
        #feat = low
        feat = F.interpolate(low, size=(48, 48), mode='nearest')
        inp_enc_level1 = self.patch_embed(feat)
        out_enc_level1, _, _ = self.encoder_level1([inp_enc_level1, k_v, SSHF])
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2, _, _ = self.encoder_level2([inp_enc_level2, k_v, SSHF])
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3, _, _ = self.encoder_level3([inp_enc_level3, k_v, SSHF])
        inp_enc_level4 = self.down3_4(out_enc_level3)

        latent, _, _ = self.latent([inp_enc_level4, k_v, SSHF])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _, _ = self.decoder_level3([inp_dec_level3, k_v, SSHF])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _, _ = self.decoder_level2([inp_dec_level2, k_v, SSHF])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _, _ = self.decoder_level1([inp_dec_level1, k_v, SSHF])

        out_dec_level1, _, _ = self.refinement([out_dec_level1, k_v, SSHF])

        out_dec_level1 = self.tail(out_dec_level1)

        out_dec_level1 = out_dec_level1 + F.interpolate(low, size=(144,144), mode='nearest')

        # 收集需要可视化的中间特征 (新增代码)
        visualized_features = {
            "encoder_level1": out_enc_level1,  # 编码器第一层输出
            "encoder_level2": out_enc_level2,  # 编码器第二层输出
            "encoder_level3": out_enc_level3,  # 编码器第三层输出
            "latent": latent,  # 潜在特征层
            "decoder_level3": out_dec_level3,  # 解码器第三层输出
            "decoder_level2": out_dec_level2,  # 解码器第二层输出
            "decoder_level1": out_dec_level1  # 解码器第一层输出
        }

        return out_dec_level1, visualized_features  # 返回超分结果和特征字典

        # return out_dec_level1


class CPEN(nn.Module):
    def __init__(self, n_feats=288, n_encoder_res=4):
        super(CPEN, self).__init__()
        E1 = [nn.Conv2d(32, n_feats * 2, kernel_size=3, padding=1),
              nn.LeakyReLU(0.2, inplace=True),
              ]
        E2 = [
            common.ResBlock(
                common.default_conv, n_feats * 2, kernel_size=3, gn=True
            ) for _ in range(4)
        ]

        E5 = [
            nn.Conv2d(n_feats * 2, 1, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        E = E1 + E2 + E5
        self.E = nn.Sequential(
            *E
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(4)

    def forward(self, x, gt):
        gt0 = self.pixel_unshuffle(gt)
        feat = self.pixel_unshuffle(F.interpolate(x, scale_factor=4, mode='nearest'))
        x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).reshape(-1, 1, 48, 48)
        S1_IPR = []
        S1_IPR.append(fea)
        return fea, S1_IPR


# class CPENSSH(nn.Module):
#     def __init__(self, n_feats=64, n_encoder_res=4):
#         super(CPENSSH, self).__init__()
#         E1 = [nn.Conv2d(1, n_feats , kernel_size=kksize, padding=ppading),
#               nn.LeakyReLU(0.2, inplace=True),
#               nn.Conv2d(n_feats, 96, kernel_size=kksize, padding=ppading, stride=2),
#               nn.LeakyReLU(0.2, inplace=True),
#               nn.Conv2d(96, 96, kernel_size=kksize, padding=ppading, stride=2),
#               nn.LeakyReLU(0.2, inplace=True)]
#
#         E2 = [
#             common.ResBlock(
#                 common.default_conv, 96 , kernel_size=kksize, gn=True
#             ) for _ in range(n_encoder_res)
#         ]
#
#
#         E3 = [
#             nn.Conv2d(96, 96, kernel_size=kksize, padding=ppading),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(96, 64, kernel_size=kksize, padding=ppading,stride=2),
#             nn.LeakyReLU(0.2, inplace=True)
#             # nn.Conv2d(n_feats , n_feats, kernel_size=kksize, padding=ppading,stride=2),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.AdaptiveAvgPool2d(1),
#         ]
#         E = E1 + E2 + E3
#         self.E = nn.Sequential(
#             *E
#         )
#
#     def forward(self, x):
#         fea = self.E(x).squeeze(-1).squeeze(-1)
#         fea = rearrange(fea, 'b c h w -> b (h w) c')
#         return fea


class DiffIRS1(nn.Module):
    def __init__(self,
                 n_encoder_res=3,
                 inp_channels=1,
                 out_channels=1,
                 scale=1,
                 dim=96,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[4, 4, 4, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(DiffIRS1, self).__init__()

        self.G = DIRformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            scale=scale,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,  ## Other option 'BiasFree'
        )

        self.E = CPEN(n_feats=64, n_encoder_res=n_encoder_res)

    def forward(self, low, high, hhh):
        IPRS1, S1_IPR = None, None
        SSH1 = None
        # 获取超分结果和中间特征 (修改此行)
        sr, visualized_features = self.G(low, IPRS1, SSH1, hhh)
        return sr, S1_IPR, visualized_features  # 返回特征字典
