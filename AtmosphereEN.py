import torch
import torch.nn as nn
from arch_util import TransformerBlock


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.SiLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += shortcut
        return self.relu(x)


class FeatureFusionLayer(nn.Module):
    def __init__(self, in_channels, out_features):
        super(FeatureFusionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.SiLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(in_channels
                      * 11 * 36 * 36, out_features),
            nn.SiLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.Linear(out_features, out_features),
            nn.SiLU(inplace=True),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        print(x.shape)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(b, -1)
        x = self.fc(x)
        x = x.reshape(b, 1, -1)
        return x


class EFN(nn.Module):
    def __init__(self):
        super(EFN, self).__init__()

        self.ResBlocks1 = nn.Sequential(
            ResnetBlock3D(1, 128), ResnetBlock3D(128, 128),
            ResnetBlock3D(128, 128), ResnetBlock3D(128, 256)
        )

        self.TransBlocks1 = nn.Sequential(
            *[TransformerBlock(dim=256, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias') for
              _ in range(4)]
        )
        self.ResBlocks2 = nn.Sequential(
            ResnetBlock3D(256, 256), ResnetBlock3D(256, 384),
            # ResnetBlock3D(384, 384), ResnetBlock3D(384, 384)
        )
        self.TransBlocks2 = nn.Sequential(
            *[TransformerBlock(dim=384, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias') for
              _ in range(2)]
        )

        self.fusion_layer = FeatureFusionLayer(384, 1024)

    def forward(self, x):
        x = self.ResBlocks1(x)
        x = self.TransBlocks1(x)
        x = self.ResBlocks2(x)
        x = self.TransBlocks2(x)

        x = self.fusion_layer(x)
        return x


if __name__ == "__main__":
    model = EFN().cuda()
    input_tensor = torch.randn(1, 1, 11, 36, 36).cuda()
    output = model(input_tensor)

    print("Output shape:", output.shape)
