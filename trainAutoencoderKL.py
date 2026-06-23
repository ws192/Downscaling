import random
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch.nn.functional as F
import numpy as np
import torch
from accelerate import Accelerator
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader
from diffusers import AutoencoderKL
import os
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TrainingConfig:
    train_batch_size = 16
    num_epochs = 400
    gradient_accumulation_steps = 1
    learning_rate = 8e-5
    lr_warmup_steps = 0
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 3407


config = TrainingConfig()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


high_res_sst = np.load(r'/200T/wangsh_pytorch/3DSR/highResData.npy')

sample_length = 10227*5
low_resolution = 24
high_resolution = 144
max_pixel = 35
train_length = 10227*5
# valid_length = 365
plot_day = 10110

scaled_high_res_sst = (high_res_sst / max_pixel).reshape(sample_length, 1, high_resolution, high_resolution)

high_train = torch.Tensor(scaled_high_res_sst[10227 * 4:10227 * 5, :, :, :]).reshape(10227, 1, high_resolution,
                                                                                     high_resolution)
print(high_train.shape)
high_valid = torch.Tensor(scaled_high_res_sst[10227 * 4:10227 * 5, :, :, :]).reshape(10227, 1, high_resolution,
                                                                                     high_resolution)

train_dataloader_high = DataLoader(batch_size=config.train_batch_size, shuffle=True,
                                   dataset=high_train, pin_memory=True)
valid_dataloader_high = DataLoader(batch_size=config.train_batch_size, shuffle=True,
                                   dataset=high_valid, pin_memory=True)

base_filters = 128
setup_seed(3407)
model = AutoencoderKL(in_channels=1, out_channels=1, sample_size=192,
                      block_out_channels=(128, base_filters * 2, base_filters * 4, base_filters * 4),
                      down_block_types=(
                          'DownEncoderBlock2D',
                          'DownEncoderBlock2D',
                          'AttnDownEncoderBlock2D',
                          'AttnDownEncoderBlock2D'
                      ),
                      up_block_types=(
                          'AttnUpDecoderBlock2D',
                          'AttnUpDecoderBlock2D',
                          'UpDecoderBlock2D',
                          'UpDecoderBlock2D'
                      ),
                      ).cuda()
# model = torch.load(r'./model/akl27612.pt')
#model=AutoencoderKL.from_pretrained(r'./model/3iter_57348')
# model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load(r'./model/akl27612.pt'))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=(len(train_dataloader_high) * 0),
    num_training_steps=(len(train_dataloader_high) * config.num_epochs),
)

valid_1 = True

xxx = []
yyy = []
index = 0

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader_high, lr_scheduler
)
accelerator.register_for_checkpointing(model)
progress_bar = tqdm(total=len(train_dataloader_high), disable=not accelerator.is_main_process)
global_step = 0

# train
for epoch in range(config.num_epochs):
    progress_bar.set_description(f"Epoch {epoch}")
    for step, highh in enumerate(train_dataloader):
        highh = highh
        optimizer.zero_grad()
        reconstruction = model(sample=highh, return_dict=False)[0]
        # posterior=model.encode(highh).latent_dist
        # kl_loss = posterior.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss1 = F.l1_loss(highh, reconstruction)
        accelerator.backward(loss1)
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)
        logs = {"loss": loss1.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        global_step += 1

    # After some epoch calculate the valid samples loss and save the model
    if accelerator.is_main_process:

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            with torch.no_grad():
                whole_loss = 0
                times = 0
                for step, highh in enumerate(valid_dataloader_high):
                    highh = highh.cuda()
                    reconstruction = model(sample=highh, return_dict=False)[0]
                    loss = F.mse_loss(highh, reconstruction)
                    whole_loss = whole_loss + loss
                    times = times + 1

                validloss = (whole_loss / times).item()
                xxx.append(index)
                index += 1
                yyy.append(validloss)
                print('valid samples loss:' + str(validloss))
                with open(r'./validloss.txt', 'a') as file:
                    file.write(
                        'epoch:' + str(epoch) + ';' + 'globalstep:' + str(global_step) + ';' + 'validloss:' + str(
                            validloss) + '\n')

                sr = model(high_valid[plot_day][:].reshape(1, 1, high_resolution, high_resolution).cuda(),
                           return_dict=False)[0]
                image = sr.cpu().numpy().reshape(high_resolution, high_resolution)
                print(image.shape)
                sst = ((image + 1) / 2) * max_pixel
                gt = (((high_valid[plot_day, 0, :, :].numpy()) + 1) / 2) * max_pixel
                mse = F.mse_loss(torch.Tensor(gt), torch.Tensor(sst))
                print("diffusion_rmse:" + str(torch.sqrt(mse)))

                lat = np.linspace(start=16, stop=24, num=sst.shape[0])
                lon = np.linspace(start=112, stop=120, num=sst.shape[1])
                fig = plt.figure(figsize=(16, 16))
                ax = plt.axes(projection=ccrs.PlateCarree())
                cmap = plt.cm.get_cmap('jet')
                cmap.set_under('w')
                plt.pcolormesh(lon, lat, sst, cmap=cmap, transform=ccrs.PlateCarree(),
                               norm=matplotlib.colors.BoundaryNorm(np.arange(24, 32, 0.1), cmap.N))
                # ax.coastlines()
                cbar = plt.colorbar()
                cbar.set_label('SST')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.savefig(r'./img/img_' + str(global_step) + r'.jpg')

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # plt.figure()
            #
            # plt.plot(xxx, yyy, color='cyan')
            #
            # plt.show(block = True)
            path=r'./model/iter_'+str(global_step)
            os.mkdir(path)
            # torch.save(model.state_dict(), r'./model/2akl' + str(global_step) + r'.pt')
            accelerator.unwrap_model(model).save_pretrained(  # unwrap先解包，然后再使用save_pretrained的方式进行保存模型
                save_directory=path,
                is_main_process=accelerator.is_main_process,
                state_dict=accelerator.get_state_dict(model),
                save_func=accelerator.save)
