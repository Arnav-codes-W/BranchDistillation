import os
import torch
from diffusers import DDIMScheduler,UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm

# -------------------------------------------------
# Config
# -------------------------------------------------
TOTAL_IMAGES = 50_000
BATCH_SIZE = 128           # adjust to GPU memory
IMAGE_SIZE = 32
CHANNELS = 3
NUM_STEPS = 100           # DDIM steps
SAVE_DIR = "samples_fake"

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet2DModel.from_pretrained('/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet')
model.eval()
#creating the scheduler 
scheduler_info = DDPMScheduler.from_pretrained('/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/scheduler')
scheduler = DDIMScheduler.from_config(scheduler_info.config)
scheduler.set_timesteps(NUM_STEPS)

model = model.to(device)
model.eval()


scheduler.timesteps = scheduler.timesteps.to(device)

# -------------------------------------------------
# Sampling loop
# -------------------------------------------------
img_id = 0

with torch.no_grad():
    while img_id < TOTAL_IMAGES:
        batch = min(BATCH_SIZE, TOTAL_IMAGES - img_id)

        # Start from pure noise
        samples = torch.randn(
            (batch, CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
            device=device
        )

        # Reverse diffusion
        for t in scheduler.timesteps:
            model_output = model(samples, t).sample
            samples = scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=samples
            ).prev_sample

        # [-1, 1] → [0, 1]
        samples = (samples.clamp(-1, 1) + 1) / 2

        # Save images
        for i in range(batch):
            save_image(
                samples[i],
                f"{SAVE_DIR}/{img_id:06d}.png"
            )
            img_id += 1

        if img_id % 5000 == 0:
            print(f"✅ Generated {img_id}/{TOTAL_IMAGES} images")

print("🎉 Finished generating 50k images")
