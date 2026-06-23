from model import *
from input_data3d import *
from tqdm import tqdm
import matplotlib

print(matplotlib.get_backend())  # 查看当前后端
matplotlib.use('TkAgg')  # 切换到交互式后端（如TkAgg）
import matplotlib.pyplot as plt

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载模型
model = SRCNN().to(device)
checkpoint = torch.load('PINN2.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 初始化存储
losses = []
results = []


def rmse(y1, y2):
    y1 = y1.numpy()
    y2 = y2.numpy()
    y1 = np.where(np.isnan(tmnan[10]), np.nan, y1)
    y2 = np.where(np.isnan(tmnan[10]), np.nan, y2)

    loss = ((y1 - y2) ** 2) ** 0.5
    loss = np.nanmean(loss)

    return loss


# 处理数据
with torch.no_grad():
    for data, var, label in tqdm(testloader, desc="Processing"):
        data, label = data.to(device), label.to(device)  # Move to device
        data = bilinear(data)
        out = model(data)

        # Denormalize
        out = out * (tmmax - tmmin) + tmmin
        label = label * (tmmax - tmmin) + tmmin

        # Calculate RMSE
        loss = rmse(out.cpu().squeeze(), label.cpu().squeeze())

        losses.append(loss)
        results.append(out.cpu().squeeze())

# Convert to numpy arrays
result_array = np.array([t.numpy() for t in results])
loss_array = np.array(losses)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# 可视化
result_array = np.where(np.isnan(tmnan[-149:]), np.nan, result_array)
# plt.imshow(result_array[10], cmap='jet',vmin=22,vmax=32)
# plt.colorbar()
# plt.show()
# plt.savefig('srpinn.png')  # 保存图像而不是显示
# plt.close()

# 保存损失
np.savez('pinn_testloss.npz', loss=loss_array)
np.savez('pinn_result.npz', result=result_array)  # 299,96,96
