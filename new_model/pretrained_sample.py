#using the actual paper weights converted to diffusers 
import matplotlib.pyplot as plt
from torchvision.utils import make_grid,save_image
import torch
from diffusers import UNet2DModel,DDPMScheduler,DDIMScheduler
from safetensors.torch import load_file

device = 'cuda'if torch.cuda.is_available() else 'cpu'

config = UNet2DModel.load_config('/teamspace/studios/this_studio/pretrained/ddpm_ema_cifar10/unet')   #loads a configuration dictionary 
model = UNet2DModel.from_config(config)                      #loads the actual model


state_dict = load_file('/teamspace/studios/this_studio/pretrained/ddpm_ema_cifar10/unet/diffusion_pytorch_model.safetensors', device="cpu") #loads the weights dictionary
model.load_state_dict(state_dict=state_dict)                       #loads the weights into the model

#creating the scheduler 
scheduler_info = DDPMScheduler.from_pretrained('/teamspace/studios/this_studio/pretrained/ddpm_ema_cifar10/scheduler')
scheduler = DDIMScheduler.from_config(scheduler_info.config)
scheduler.set_timesteps(50)

#loading model to device
model.to(device)
model.eval()

#creating noise 
samples = torch.randn((16,3,32,32)).to(device)
scheduler.timesteps = scheduler.timesteps.to(device)
#sampling loop
with torch.no_grad():
    for t in scheduler.timesteps:
        t=t.to(device)
        model_output = model(samples, t).sample
        samples = scheduler.step(
            model_output, t, samples
        ).prev_sample


samples = (samples.clamp(-1, 1) + 1) / 2
grid = make_grid(samples, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()

#saving the image
save_image(
    samples,
    "images/samples_pretrained_original.png",
    nrow=4,
)