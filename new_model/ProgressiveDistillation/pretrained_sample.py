import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"



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

# -------------------------
# Load model config
# -------------------------
config = UNet2DModel.load_config(
    "/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet"
)

model = UNet2DModel.from_config(config)

# -------------------------
# Load STUDENT checkpoint
# -------------------------

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

'''
checkpoint = load_file('/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet/diffusion_pytorch_model.safetensors')
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
'''
# ------------------------
# Load scheduler
# -------------------------

scheduler = DDIMScheduler(num_train_timesteps=(125),beta_start=0.0001, beta_end= 0.02, beta_schedule='linear',prediction_type='epsilon')
scheduler.set_timesteps(100)
scheduler.timesteps= scheduler.timesteps.to(device=device)

# -------------------------
# Sampling
# -------------------------
samples = torch.randn((16, 3, 32, 32), device=device)

with torch.no_grad():
    for t in scheduler.timesteps:
        model_output = model(samples, t).sample
        samples = scheduler.step(
            model_output=model_output,
            timestep=t,
            sample=samples,
        ).prev_sample

# -------------------------
# Visualize
# -------------------------
# -------------------------
samples = (samples.clamp(-1, 1) + 1) / 2

# -------------------------
# Visualize
# -------------------------
grid = make_grid(samples, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()

# -------------------------
# Save
# -------------------------
save_image(
    samples,
    "images_generated/student_125.png",
    nrow=4,
)