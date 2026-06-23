import torch
from model import *
from torch.autograd import grad
from input_data3d import *
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端（无 GUI）

# 设备设置
device = torch.device('cuda')

# 模型转移到设备
model = SRCNN().to(device)
checkpoint = torch.load('train.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 300
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (1200 / 16))

ls4 = []
ls1 = []
ls2 = []
ls3 = []
best_score = 10000

model_weights = './PINN.pth'
# 训练循环修改
for epoch in range(epochs):
    model.train()
    losses = 0
    loss11 = 0
    loss22 = 0

    print(f"epoch:{epoch}/{epochs}")
    with tqdm(trainloader, dynamic_ncols=True, colour="#ff924a") as loader:
        for data, var, label in loader:
            # 确保所有数据都在GPU上
            data = data.to(device)  # 16 1 48 48
            var = var.to(device)  # 16 7 48 48
            label = label.to(device)  # 16 1 96 96

            optimizer.zero_grad()

            data = bilinear(data)  # 16 1 96 96
            out = model(data)
            # 数据残差
            loss1 = mse(out, label)

            grad_y, grad_x = torch.gradient(out, dim=(2, 3))  # 16 1 96 96

            # PDE残差计算old = torch.stack((Q,mld,um,vm,td,wd,tm_old), dim=1) #B,C,H,W
            Q = var[:, [0]].float().to(device)
            mld = var[:, [1]].float().to(device)
            um = var[:, [2]].float().to(device)
            vm = var[:, [3]].float().to(device)
            td = var[:, [4]].float().to(device)
            wd = var[:, [5]].float().to(device)
            sst = var[:, [6]].float().to(device)

            residual = sst + Q * 3600 / (4000 * 1027 * mld) - um * grad_x - vm * grad_y - wd * (sst - td) / mld
            pred = (residual - tmmin) / (tmmax - tmmin)
            loss2 = mse(pred, out)

            a = loss1 / (loss1 + loss2)

            loss = a * loss1 + (1 - a) * loss2
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss11 += loss1
            loss22 += loss2
            losses += loss  # 使用.item()避免内存累积
        print('loss11:{:5f}'.format(loss1), 'loss22:{:.5f}'.format(loss2))

        train_loss = losses / len(trainloader)
        loss111 = loss11 / len(trainloader)
        loss222 = loss22 / len(trainloader)

        ls1.append(train_loss.detach().cpu().numpy())
        ls2.append(loss111.detach().cpu().numpy())
        ls3.append(loss222.detach().cpu().numpy())

    with torch.no_grad():
        valid_losses = 0
        with tqdm(validloader, dynamic_ncols=True, colour="#ff924a") as loader:
            for data, var, label in loader:
                # 确保所有数据都在GPU上
                data = data.to(device)  # 16 1 48 48
                var = var.to(device)  # 16 7 48 48
                label = label.to(device)  # 16 1 96 96

                data = bilinear(data)  # 16 1 96 96
                out = model(data)
                # 数据残差
                vloss = mse(out, label)
                valid_losses += vloss
            valid_loss = valid_losses / len(validloader)
            ls4.append(valid_loss.detach().cpu().numpy())

    # save model
    if valid_loss < best_score:
        best_score = valid_loss
        checkpoint = {'best_score': valid_loss,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, model_weights)

l1 = np.array(ls1)
l2 = np.array(ls2)
l3 = np.array(ls3)
l4 = np.array(ls4)
np.savez(r'pinn_trainloss.npz', loss=l1, dl=l2, pl=l3, loss2=l4)
