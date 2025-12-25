from diffusers import UNet2DModel,DDPMScheduler
import torch
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import torch.nn as nn 
from torch.optim import AdamW 
from tqdm import tqdm 
from accelerate import Accelerator
import os 

#PREPROCESSING
num_epochs = 128  # total number of parameter updates = 100,000
batch_size =64 #(781 iterations per epoch )


transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image to Tensor (HWC -> CHW, values 0-255 to 0.0-1.0)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to range [-1, 1]
    transforms.RandomHorizontalFlip(p=0.1)
    ])


trainset = torchvision.datasets.CIFAR10(root='/teamspace/studios/this_studio/dataset ', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='/teamspace/studios/this_studio/dataset ', train=False, download=True, transform=transform)


trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def unnormalize(img):
    return img * 0.5 + 0.5  # [-1,1] → [0,1]

# Get a batch
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Make grid
from torchvision.utils import make_grid
img_grid = make_grid(images)
img_grid = unnormalize(img_grid)

# Convert CHW → HWC for matplotlib
npimg = img_grid.permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(6,6))
plt.imshow(npimg)
plt.axis("off")
plt.show()

print("Labels:", labels)

#creating accelerator to speed up training

device ="cuda" if torch.cuda.is_available( ) else "cpu"

# MODEL DEFINED

#model matching the original paper 
#model params =49.5 million
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

#CALCULATE THE VARIOUS SCHEDULES 
#we use the original ddpm paper scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    beta_schedule="linear",
    prediction_type='v_prediction'
)

#THE TRAINING LOOP FINALLY 
model.train()
criterion = nn.MSELoss()

model.to(device)
optim = AdamW(model.parameters(), lr=2e-4)



#checkpoints saving code
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

#training loop finally final
print("The training has started")

for epoch in range(num_epochs):

    progress_bar = tqdm(
        trainloader,
        desc=f"Epoch {epoch}",
        #disable=not accelerator.is_local_main_process,
    )

    for images, labels in progress_bar:
        images = images.to(device)

        # Sample timestep (int64)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (images.shape[0],),
            device=images.device,
            dtype=torch.int64,
        )

        # Sample Gaussian noise (IMPORTANT)
        noise = torch.randn_like(images)

        # Add noise
        noisy_images = noise_scheduler.add_noise(
            original_samples=images,
            noise=noise,
            timesteps=timesteps,
        )

        # Model prediction (v)
        v_pred = model(noisy_images, timesteps).sample

        # Ground-truth velocity
        target_v = noise_scheduler.get_velocity(
            images, noise, timesteps
        )

        # Loss
        loss = criterion(v_pred, target_v)

        # Optimization step
        optim.zero_grad()  
        loss.backward() #backpropogation calculates the gradients dl/dw for each model parameter
        optim.step() #optimizer takes a step in the direction for each model weight where the loss decreases, the size of the step is calculated by the learning rate 

        # Update tqdm bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.13f}"
        )

        #save model checkpoints for each epoch divisible by 16
        