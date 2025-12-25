import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator



IMAGE_SIZE = 32
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 2e-4
GRAD_ACCUM_STEPS = 4
CHECKPOINT_DIR = "./checkpoints_2"
SAVE_EVERY_EPOCHS = 8

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


accelerator = Accelerator(
    mixed_precision="bf16", 
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
)
device = accelerator.device

#loading the dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),                       # [0,1]
    transforms.Normalize((0.5,)*3, (0.5,)*3),    # → [-1,1]
    transforms.RandomHorizontalFlip(p=0.1),
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

#creating the model 
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=3,
    block_out_channels=(256, 256, 256),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
    attention_head_dim=256,
    dropout=0.2,
)


#defining the noise schedule, the loss and optimizer 
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1024,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="v_prediction",
)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model, optimizer, trainloader = accelerator.prepare(
    model, optimizer, trainloader
)

#Resuming training
start_epoch = 0
global_step = 0

checkpoint_files = sorted(
    [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
)

if checkpoint_files:
    latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    accelerator.print(f" Resuming from {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]



accelerator.print("Training started")
#final training loop
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    progress_bar = tqdm(
        trainloader,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}",
    )

    for images, _ in progress_bar:
        with accelerator.accumulate(model):
            images = images.to(device)

            # Sample timesteps
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
                dtype=torch.long,
            )

            # Gaussian noise
            noise = torch.randn_like(images)

            # Add noise
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )

            # Predict v
            v_pred = model(noisy_images, timesteps).sample

            # Ground-truth v
            target_v = noise_scheduler.get_velocity(
                images, noise, timesteps
            )

            #loss 
            loss = criterion(v_pred, target_v)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.set_postfix(loss=loss.item())


        #saving checkpoint
    if epoch % SAVE_EVERY_EPOCHS == 0 and accelerator.is_main_process:
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        ckpt_path = os.path.join(
            CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(ckpt, ckpt_path)
        accelerator.print(f"Saved {ckpt_path}")


accelerator.print("Training complete")
