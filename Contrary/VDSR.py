import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            ResBlock(in_channels),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            ResBlock(out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = nn.Sequential(
                ResBlock(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.SiLU(inplace=True),
                ResBlock(out_channels)
            )


        self.res_block = ResBlock(out_channels)
        self.res_block2 = ResBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.res_block(x)
        x = self.res_block2(x)
        return x


class SuperResolutionUNet(nn.Module):
    def __init__(self, in_channels=81, out_channels=81, base_dim=128):
        super(SuperResolutionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = nn.Sequential(
            ResBlock(in_channels),
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            ResBlock(base_dim)
        )

        self.down1 = Down(base_dim, base_dim * 2)  # 256 ch
        self.down2 = Down(base_dim * 2, base_dim * 4)  # 512 ch
        self.down3 = Down(base_dim * 4, base_dim * 8)  # 1024 ch

        self.up1 = Up(base_dim * 8 + base_dim * 4, base_dim * 4, bilinear=True)
        self.up2 = Up(base_dim * 4 + base_dim * 2, base_dim * 2, bilinear=True)
        self.up3 = Up(base_dim * 2 + base_dim, base_dim, bilinear=True)

        self.final_conv = nn.Sequential(
            ResBlock(base_dim),
            nn.Conv2d(base_dim, out_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            ResBlock(out_channels)
        )

    def forward(self, x):
        target_size = (599, 580)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = F.interpolate(x, size=target_size, mode='bilinear')

        out = self.final_conv(x)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SuperResolutionUNet(in_channels=81, out_channels=81, base_dim=256).to(device)

    batch_size = 1
    input_tensor = torch.randn(batch_size, 81, 253, 241).to(device)

    output_tensor = model(input_tensor)

    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output_tensor.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
