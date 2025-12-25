import torch
from diffusers import DDIMScheduler,DDPMScheduler,UNet2DModel
import matplotlib.pyplot as plt  
from torchvision.utils import make_grid, save_image

#creating a model 
model = UNet2DModel(
    sample_size=32,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=3,  # how many ResNet layers to use per UNet block
    block_out_channels=(256, 256, 256),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
    ),
    attention_head_dim=256,
    dropout= 0.2
)
device = "cuda" if torch.cuda.is_available() else 'cpu'

#loading a saved checkpoint 
checkpoint = torch.load(
    "/teamspace/studios/this_studio/checkpoints_2/checkpoint_epoch_16.pt",
    map_location=device
)
state_dict = checkpoint["model"]

# Handle torch.compile checkpoints (_orig_mod.)
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

model.load_state_dict(state_dict)
model.to(device)
model.eval()


#creating scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1024,
    beta_schedule="squaredcos_cap_v2",      
    prediction_type="v_prediction",
)
#creating DDIM sampler 
sampler = DDIMScheduler.from_config(noise_scheduler.config)
sampler.set_timesteps(200)

#samples 
samples = torch.randn((16,3,32,32)).to(device)
sampler.timesteps = sampler.timesteps.to(device)
#sampling loop
with torch.no_grad():
    for t in sampler.timesteps:
        t=t.to(device)
        model_output = model(samples, t).sample
        samples = sampler.step(
            model_output, t, samples
        ).prev_sample



#visualizing the samples 
samples = (samples.clamp(-1, 1) + 1) / 2
grid = make_grid(samples, nrow=4)
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis("off")
plt.show()

#saving the image
save_image(
    samples,
    "images/samples_cosine.png",
    nrow=4,
)