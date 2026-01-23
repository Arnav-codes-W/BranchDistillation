import os
from sre_parse import BRANCH
import torch
from diffusers import DDIMScheduler,UNet2DModel, DDPMScheduler
from torchvision.utils import save_image
from tqdm import tqdm


#EMA helper 
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# -------------------------------------------------
# Config
# -------------------------------------------------
TOTAL_IMAGES = 10_000
BATCH_SIZE = 128         # adjust to GPU memory
IMAGE_SIZE = 32
CHANNELS = 3
NUM_STEPS = 128       # DDIM steps
SAVE_DIR = "samples_fake"
Branch= 'False'
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = UNet2DModel.load_config('/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet')

if Branch == "True":
    config['out_channels'] =6 
else:
    config['out_channels'] =3

model = UNet2DModel.from_config(config=config)

#state_dict = load_file('config = UNet2DModel.load_config('/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet')
#teacher = UNet2DModel.from_config(config=config)
#teacher.eval()

checkpoint = torch.load(
    "/teamspace/studios/this_studio/checkpoints/pd_250_to_125/ckpt_25000.pt",
    map_location="cpu",
)

model.load_state_dict(checkpoint["student"])
ema=EMA(model)
ema.shadow = checkpoint["ema"]
ema.apply_shadow(model)
model.to(device)
model.eval()

#creating the scheduler 

scheduler = DDIMScheduler(num_train_timesteps=(NUM_STEPS),beta_start=0.0001, beta_end= 0.02, beta_schedule='linear',prediction_type='epsilon')
scheduler.set_timesteps(NUM_STEPS)
scheduler.timesteps= scheduler.timesteps.to(device=device)



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
            eps = model_output[:,-3:,:,:]
            samples = scheduler.step(
                model_output=eps,
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

        if img_id % 512 == 0:
            print(f"✅ Generated {img_id}/{TOTAL_IMAGES} images")

print("🎉 Finished generating 10k images")
