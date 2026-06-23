import matplotlib.pyplot as plt
import numpy as np

# data = np.load('srcnn_trainloss.npz')
# loss1 = data['loss'][2:]
# loss1=list(loss1)
####################################################
data = np.load('pinn_trainloss.npz')
loss1 = data['loss'][2:]
loss1 = list(loss1)

loss2 = data['dl'][2:]
loss2 = list(loss2)

loss3 = data['pl'][2:]
loss3 = list(loss3)
####################################################
x = range(1, len(loss1) + 1)

loss4 = data['loss2'][2:]
loss4 = list(loss4)
x2 = range(1, len(loss4) + 1)

# 创建折线图

plt.plot(x, loss1, marker='o', linestyle='-', color='r', label='loss')
plt.plot(x, loss2, marker='o', linestyle='-', color='g', label='data')
plt.plot(x, loss3, marker='o', linestyle='-', color='b', label='pde')
# plt.plot(x2, loss4, marker='o', linestyle='-', color='k', label='validloss')
plt.title('HS pred')
plt.xlabel('epoch')
plt.ylabel('RMSE(°C)')
plt.legend()
plt.show()

data1 = np.load('pinn_testloss.npz')
loss1 = data1['loss'][2:]
loss1 = list(loss1)

data2 = np.load('srcnn_testloss.npz')
loss2 = data2['loss'][2:]
loss2 = list(loss2)

x = range(1, len(loss1) + 1)
plt.figure(figsize=(10, 6))

# 绘制原始曲线
plt.plot(x, loss1, marker='o', linestyle='-', color='r', label='PINN', alpha=0.7)
plt.plot(x, loss2, marker='o', linestyle='-', color='g', label='CNN', alpha=0.7)

# 计算并绘制均值线
mean1 = sum(loss1) / len(loss1)
mean2 = sum(loss2) / len(loss2)
plt.axhline(y=mean1, color='r', linestyle='--', linewidth=2, alpha=0.5, label=f'{mean1:.4f}°C')
plt.axhline(y=mean2, color='g', linestyle='--', linewidth=2, alpha=0.5, label=f'{mean2:.4f}°C')

# 添加标注文本
# plt.text(len(loss1)+0.5, mean1, f'{mean1:.3f}', color='r', va='center')
# plt.text(len(loss2)+0.5, mean2, f'{mean2:.3f}', color='g', va='center')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE (°C)', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14)

# 优化图例显示
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
