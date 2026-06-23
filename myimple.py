import os
import random
import torch.nn.functional as F
import numpy as np
import torch
from diffusers import UNet3DConditionModel
from diffusers import DDIMScheduler, AutoencoderKL
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from dataclasses import dataclass
from torch.utils.data import DataLoader

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


@dataclass
class TrainingConfig:
    train_batch_size = 1
    num_epochs = 600
    gradient_accumulation_steps = 1
    learning_rate = 1e-6
    lr_warmup_steps = 0
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 3407


config = TrainingConfig()
single_layer = 5
sample_length = 5479
layers = 12 * single_layer
high_resolution = 36
max_pixel = 35
train_length = 4383
valid_length = 365
plot_day = 5
hat = 11
floor = -8
sstHighRes = np.load(r'dataStack.npy')[:, :, 24:144 + 24, 0:144 + 0]
print(sstHighRes.shape)
print("data loaded")

l_latent = np.load(r'lla.npy')[:] / hat
h_latent = np.load(r'hla.npy')[:] / hat
concatla = np.concatenate((l_latent, h_latent), axis=1)
ae = AutoencoderKL.from_pretrained(r'./ae_mg/ae_iter_312322').cuda()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(3407)

noise_list = np.zeros((15, valid_length, layers, high_resolution, high_resolution))
for i in range(15):
    for n in range(valid_length):
        noise = np.random.randn(layers, high_resolution, high_resolution)
        noise_list[i][n] = noise

noise_list = torch.Tensor(noise_list)

print("noise generated")

high_sst = sstHighRes

scaled_high_res_sst = concatla

scaled_high_res_sst = scaled_high_res_sst.reshape(
    (sample_length, 1, layers * 2, high_resolution, high_resolution))

ltrain = torch.Tensor(scaled_high_res_sst[0:train_length, :, :, :, :])
lvalid = torch.Tensor(scaled_high_res_sst[train_length:train_length + 365, :, :, :, :])

print(ltrain.shape)
print(lvalid.shape)

train_dataloader = DataLoader(batch_size=config.train_batch_size, shuffle=False,
                              dataset=ltrain, pin_memory=True)
valid_dataloader = DataLoader(batch_size=config.train_batch_size, shuffle=False,
                              dataset=lvalid, pin_memory=True)

print("dataloader generated")

model = UNet3DConditionModel(
    in_channels=2, out_channels=1, layers_per_block=2,
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D"),
    block_out_channels=(320, 480, 640)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

noise_scheduler = DDIMScheduler(beta_schedule='squaredcos_cap_v2')


def train_loop(model, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(model)

    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar.set_description(f"Epoch {epoch}")

        for step, lowhigh in enumerate(train_dataloader):
            loww = lowhigh[:, :, 0:layers, :, :].cuda()
            # loww = torch.tile(loww, (1, 1, 5, 1, 1))
            highh = lowhigh[:, :, layers:layers * 2, :, :].cuda()
            noise = torch.randn(highh.shape).to('cuda')
            batchsize = config.train_batch_size
            # sample timestep
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batchsize,), device='cuda'
            ).long()
            noisy_images = noise_scheduler.add_noise(highh, noise, timesteps).cuda()
            with accelerator.accumulate(model):
                latent = torch.cat([loww, noisy_images], dim=1)
                optimizer.zero_grad()
                constant_value = 1
                encoder_hidden_states = torch.full((batchsize, 4, 1024), constant_value, dtype=torch.float16).cuda()

                noise_pred = model(latent, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:

            with torch.no_grad():
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    noise_scheduler.set_timesteps(50)

                    for c in range(1):
                        low_reference = (torch.Tensor(
                            lvalid[plot_day, :, 0:layers].reshape(1, 1, layers, high_resolution,
                                                                  high_resolution))).cuda()
                        # low_reference = torch.tile(low_reference, (1, 1, 5, 1, 1))
                        image = (
                            noise_list[c][plot_day].reshape(1, 1, layers, high_resolution, high_resolution).cuda())
                        for t in noise_scheduler.timesteps:
                            latent = torch.cat([low_reference, image], dim=1)
                            constant_value = 1
                            encoder_hidden_states = torch.full((1, 4, 1024), constant_value,
                                                               dtype=torch.float16).cuda()
                            model_output = \
                                model(latent, t, encoder_hidden_states=encoder_hidden_states, return_dict=False)[
                                    0]
                            image = noise_scheduler.step(model_output, t, image).prev_sample

                        image = image.reshape(-1, layers, high_resolution, high_resolution)
                        latent = image * hat
                    gtlatent = (lvalid[plot_day, :, layers:layers * 2].reshape(-1, layers, high_resolution,
                                                                               high_resolution)) * hat
                    for v in range(12):
                        sst = ae.decode(latent[:, v * single_layer:v * single_layer + single_layer, :,
                                        :]).sample.cpu().numpy().reshape(144, 144)
                        gt = ae.decode(gtlatent[:, v * single_layer:v * single_layer + single_layer, :,
                                       :].cuda()).sample.cpu().numpy().reshape(144, 144)
                        mse1 = F.mse_loss(
                            torch.Tensor(gt * 35),
                            torch.Tensor(sst * 35))
                        thereis = torch.sqrt(mse1)
                        print("diffusion_rmse:" + str(torch.sqrt(mse1)))


            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                path = r'./3dm/iter_' + str(global_step)
                if not os.path.exists(path):
                    os.mkdir(path)
                accelerator.unwrap_model(model).save_pretrained(
                    save_directory=path,
                    is_main_process=accelerator.is_main_process,
                    state_dict=accelerator.get_state_dict(model),
                    save_func=accelerator.save)


train_loop(model=model, optimizer=optimizer, train_dataloader=train_dataloader, lr_scheduler=lr_scheduler)
