import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader 
from accelerate import Accelerator
from diffusers import UNet2DModel, DDIMScheduler
import os
import copy
import argparse

# 1. CONFIGURATION 
parser = argparse.ArgumentParser()
parser.add_argument('--teacher_checkpoint', type=str, default="./checkpoints/checkpoint_teacher.pt", help="Path to the teacher model checkpoint")
parser.add_argument('--student_checkpoint', type=str, default="./checkpoints/checkpoint_student.pt", help="Path to the student model checkpoint")
parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for the optimizer")
parser.add_argument('--N_steps', type=int, default=1024, help="Current teacher steps")
parser.add_argument('--M_steps', type=int, default=512, help="Target student steps")

args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate 
teacher_checkpoint = args.teacher_checkpoint 
N_steps = args.N_steps  
M_steps = args.M_steps 

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

# 2. MODELS & SCHEDULERS
# Load Teacher
teacher_model = UNet2DModel(
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

# Load your pre-trained weights
checkpoint = torch.load(teacher_checkpoint, map_location="cpu")

state_dict = checkpoint.get("model", checkpoint) 
teacher_model.load_state_dict(state_dict, strict=True)
teacher_model.to(device)
teacher_model.eval()

# Freeze teacher parameters
teacher_model.requires_grad_(False)

# Student is initialized from Teacher weights  
student_model = copy.deepcopy(teacher_model).train().to(device) # change in branch distillation

# We use DDIM for deterministic distillation   {beta value check karna hai}
scheduler = DDIMScheduler(
    num_train_timesteps=N_steps, 
    beta_start=1e-4, 
    beta_end=0.02, 
    beta_schedule="linear"
    prediction_type='v_prediction'
)

# 3. DATASET LOADING
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


optimizer = AdamW(student_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Prepare for acceleration
student_model, optimizer , trainloader = accelerator.prepare(student_model, optimizer, trainloader)

# 4. THE DISTILLATION MATH  
                    # check karna hai
def get_teacher_target(z_t, t, teacher, scheduler, step_size):
    """
    Implements the teacher's two-step jump to find the student's target.
    """
    with torch.no_grad():
        # Step 1: Teacher predicts v at t, move to mid-point (t - 0.5 * step_size)
        t_mid = t - (step_size // 2)
        v_mid = teacher(z_t, t).sample
        
        # Calculate z_mid using DDIM formula
        alpha_t = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_mid = scheduler.alphas_cumprod[t_mid].view(-1, 1, 1, 1)
        
        # In v-prediction: x = sqrt(alpha)*z - sqrt(1-alpha)*v
        #                  eps = sqrt(1-alpha)*z + sqrt(alpha)*v
        x_pred = (alpha_t*0.5 * z_t) - ((1 - alpha_t)*0.5 * v_mid)
        eps_pred = ((1 - alpha_t)*0.5 * z_t) + (alpha_t*0.5 * v_mid)

        z_mid = (alpha_mid*0.5 * x_pred) + ((1 - alpha_mid)*0.5 * eps_pred)
        
        # Step 2: Teacher predicts v at t_mid, move to target (t - step_size)
        t_target = t - step_size
        # Ensure we don't go below 0
        t_target = torch.where(t_target < 0, torch.zeros_like(t_target), t_target)

        v_target = teacher(z_mid, t_target).sample

        return v_target

# 5. TRAINING LOOP
for epoch in range(num_epochs):
    progress_bar = tqdm(
        trainloader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
    )
    
    for images, _ in progress_bar:
        images = images.to(device)
        noise = torch.randn_like(images)
        
        # Sample t from the set of steps allowed for this stage
        step_size = N_steps // M_steps
        t_indices = torch.randint(1, M_steps + 1, (images.shape[0],), device=device)
        t = t_indices * step_size - 1  # Convert to teacher's timestep
        
        # Add noise to images
        z_t = scheduler.add_noise(images, noise, t)
        
        # Get Teacher Target (z_{t-step_size})
        v_target = get_teacher_target(z_t, t, teacher_model, scheduler, step_size)
        
        # Student attempts to predict the jump in one step
        v_pred = student_model(z_t, t).sample
        
        
        # Calculate loss
        loss = criterion(v_pred, v_target)
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        progress_bar.set_postfix(loss=loss.item())

    # Save model checkpoints for each epoch divisible by 10
    if (epoch % 10 == 0):
        if accelerator.is_main_process:
            checkpoint = {
                "epoch": epoch,
                "model": accelerator.unwrap_model(student_model).state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            os.makedirs("distill_checkpoints", exist_ok=True)
            torch.save(checkpoint, f"distill_checkpoints/checkpoint_student_epoch_{epoch}.pt")