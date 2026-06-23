from torch import nn
from torch.nn import functional as F


def bilinear(data):
    target_data = F.interpolate(data, scale_factor=2, mode='bilinear', align_corners=False)  # .cpu().detach().numpy()
    return target_data


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)  # (3,64,9,4,1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)  # (64 32 5 2 1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=5 // 2)  # (32 3  5 2 1)

        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 加一层 3 1 1
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x
