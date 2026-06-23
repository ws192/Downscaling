import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy
import torch.nn.functional as F
import numpy as np
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from netCDF4 import Dataset
from torch.utils.data import TensorDataset
from accelerate import notebook_launcher
from sklearn.preprocessing import MinMaxScaler
from SRCNN import SRCNN
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")




@dataclass
class TrainingConfig:
    data_num = 1002,
    num_inference_times = 200,
    image_size = 160,
    image_width = 160,
    image_height = 160,
    train_batch_size = 4
    eval_batch_size = 4
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 100
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = r'C:\Users\31860\PycharmProjects\DeepR\my_SR\stable_diffusion_vae_model\fine_tuning_vae'
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 3407


config = TrainingConfig()

file_sst1 = Dataset(r"D:\2mtmp_low_up.nc").variables['t2m'][:]
file_sst_gt1 = Dataset(r"D:\2mtmp_normal.nc").variables['t2m'][:]




#归一化
# minmaxscaler=MinMaxScaler(feature_range=(-1,1))

# reshape_gt = file_sst_gt1.reshape(1002, -1)
# reshape_sst = file_sst1.reshape(1002, -1)
# concatarray = numpy.concatenate((reshape_sst, reshape_gt), axis=0)
#
# minmaxscaler.fit(concatarray)
# file_sst1 = (minmaxscaler.fit_transform(reshape_sst)).reshape(1002, 160, 160)
# file_sst_gt1 = (minmaxscaler.transform(reshape_gt)).reshape(1002, 160, 160)


temp_array_l = np.zeros((1, 1, 160, 160), dtype=float)
temp_array_gt = np.zeros((1, 1, 160, 160), dtype=float)
temp_array_gt[0, 0, :] = file_sst_gt1[1001]
gt_rmse = torch.Tensor(temp_array_gt)
temp_array_l[0, 0, :] = file_sst1[1001]
l_rmse = torch.Tensor(temp_array_l)


dataset = []
dataset_gt = []

for i in range(1000):
    tarray = np.zeros((1, 160, 160), dtype=float)
    tarray_gt = np.zeros((1, 160, 160), dtype=float)

    tarray[0, :] = file_sst1[i]
    dataset.append(tarray)

    tarray_gt[0, :] = file_sst_gt1[i]
    dataset_gt.append(tarray_gt)



dataset = [torch.Tensor(matrix) for matrix in dataset]
dataset_gt = [torch.Tensor(matrix) for matrix in dataset_gt]


train_dataloader = torch.utils.data.DataLoader(batch_size=config.train_batch_size, shuffle=False, dataset=dataset)
train_dataloader_gt = torch.utils.data.DataLoader(batch_size=config.train_batch_size, shuffle=False, dataset=dataset_gt)


print("data prepare finished")


model=SRCNN()


# train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        # mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with="tensorboard",
        # project_dir=os.path.join(config.output_dir, "logs"),
    )


    if accelerator.is_main_process:

        # if config.output_dir is not None:
        #     os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")
        data_ft = iter(train_dataloader)
        for step, batch in enumerate(train_dataloader_gt):
            ft = next(data_ft).cuda()
            batch = batch.cuda()
            with accelerator.accumulate(model):
                noise_pred = model(ft)
                loss = F.mse_loss(noise_pred, batch)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                with torch.no_grad():
                    image = model(l_rmse.cuda())
                    image = image.cpu().numpy()
                    sst = (image[0, 0, :])
                    # sst = (minmaxscaler.inverse_transform(sst.reshape(1, -1))).reshape(160, 160)
                    mse2 = F.mse_loss(torch.Tensor(Dataset(r"D:\2mtmp_normal.nc").variables['t2m'][1001]),
                                  torch.Tensor(sst))
                    print("diffusion_rmse:" + str(torch.sqrt(mse2)))
                lat = Dataset(r"D:\2mtmp_low_up.nc").variables["latitude"][:]
                lon = Dataset(r"D:\2mtmp_low_up.nc").variables['longitude'][:]
                fig = plt.figure(figsize=(10, 8))
                ax = plt.axes(projection=ccrs.PlateCarree())
                print(sst.min())
                print(sst.max())
                levels = np.linspace(277, 305, 100)  # 设置等值线间隔
                plt.contourf(lon, lat, sst, levels=levels, cmap="coolwarm", transform=ccrs.PlateCarree())
                ax.coastlines()
                cbar = plt.colorbar()
                cbar.set_label('t2m')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.show()

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                torch.save(model,r'D:\Ckpt\SRCNN\model.pt')


# args = ()
# notebook_launcher(train_loop, args, num_processes=1)
train_loop(config, model,optimizer, train_dataloader, lr_scheduler)