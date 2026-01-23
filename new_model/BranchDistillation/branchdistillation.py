
# as batch size 256, the checkpoints are only till 25000

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.xpu import device
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDIMScheduler, DDPMScheduler
from safetensors.torch import load_file
from tqdm import tqdm
import os

#ddim helper functions 
def get_alpha_sigma(scheduler, t, device):
    """
    Returns alpha_t, sigma_t with broadcastable shape.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sigma = (1.0 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    return alpha, sigma


def ddim_step_explicit(z_t, eps, t, scheduler, device, clip=True):
    """
    Deterministic DDIM step: t -> t-1 (eta = 0)
    """
    t_prev = torch.clamp(t - 1, min=0)

    alpha_t, sigma_t = get_alpha_sigma(scheduler, t, device)
    alpha_prev, sigma_prev = get_alpha_sigma(scheduler, t_prev, device)

    # reconstruct x0
    x0_hat = (z_t - sigma_t * eps) / alpha_t
    if clip:
        x0_hat = x0_hat.clamp(-1, 1)

    # deterministic DDIM update
    z_prev = alpha_prev * x0_hat + sigma_prev * eps
    return z_prev




#gamma weighing 
def gamma_weight(alpha, sigma, gamma):
    """
    EDM-style loss weighting
    """
    if gamma == 0:
        return 1.0
    return (1.0 + alpha / sigma).pow(gamma)





class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)


RESUME = True
CKPT_PATH = "/teamspace/studios/this_studio/pd_128_to_64_branch/ckpt_34000.pt"




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 140
LR = 2e-4
GAMMA= 1.0
TEACHER_STEPS = 128
STUDENT_STEPS = 64

SAVE_DIR = "./pd_128_to_64_branch"
os.makedirs(SAVE_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    transforms.RandomHorizontalFlip()
])

dataset = datasets.CIFAR10(
    root="./cifar",
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers = 8,
    pin_memory=True,
    persistent_workers=True,
)


config = UNet2DModel.load_config(
    "/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet"
)
config['out_channels']= 6
teacher = UNet2DModel.from_config(config)
checkpoint = torch.load(
    "/teamspace/studios/this_studio/checkpoints/pd_250_to_125_branch/ckpt_50000.pt",
    map_location="cpu",
)
teacher.load_state_dict(checkpoint['student'])
teacher.to(DEVICE).eval() #teacher fixed to eval 
#config for branching out 


student = UNet2DModel.from_config(config)

#loading the weights 
teacher_state = teacher.state_dict()
'''
# remove output layer
for k in ["conv_out.weight", "conv_out.bias"]:
    teacher_state.pop(k)
'''

student.load_state_dict(teacher_state, strict=False) #init student w tteacher without the last layer kernels 

'''
#load the teacher last layer weights twice into the student last layer
with torch.no_grad():
    # Teacher output
    w_t = teacher.conv_out.weight.data    # [3, 128, 3, 3]
    b_t = teacher.conv_out.bias.data      # [3]

    # Student output
    w_s = student.conv_out.weight.data    # [6, 128, 3, 3]
    b_s = student.conv_out.bias.data      # [6]

    # Copy teacher weights twice
    w_s[0:3].copy_(w_t)
    w_s[3:6].copy_(w_t)

    b_s[0:3].copy_(b_t)
    b_s[3:6].copy_(b_t)
'''
student.to(DEVICE).train() #student fixed to train 
  

ema = EMA(student)

#ddpm scheduler
ddpm = DDPMScheduler(num_train_timesteps=TEACHER_STEPS)

#ddim teacher scheduler 
ddim_teacher = DDIMScheduler(
    num_train_timesteps=TEACHER_STEPS,
    prediction_type="epsilon",
)
ddim_teacher.set_timesteps(TEACHER_STEPS)

#ddim student scheduler
ddim_student = DDIMScheduler(
    num_train_timesteps=STUDENT_STEPS,
    prediction_type="epsilon",
)
ddim_student.set_timesteps(STUDENT_STEPS)

optimizer = AdamW(student.parameters(), lr=LR) #we only update student parameters 
loss_fn = nn.MSELoss()

start_step = 0
start_epoch = 0

if RESUME and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    student.load_state_dict(ckpt["student"])
    student.to(DEVICE).train()
    ema.shadow = {k: v.to(DEVICE) for k, v in ckpt["ema"].items()}

    start_step = ckpt["step"]

    # recover epoch index approximately
    steps_per_epoch = len(loader)
    start_epoch = start_step // steps_per_epoch

    print(f"✅ Resumed from step {start_step} (epoch {start_epoch})")


global_step = start_step

for epoch in range(start_epoch,EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x0, _ in pbar:
        x0 = x0.to(DEVICE)
        bsz = x0.size(0)

        # sample even timestep: t = 2k
        k = torch.randint(1, STUDENT_STEPS, (bsz,), device=DEVICE)
        t = 2 * k
        #t_vec = torch.full((bsz,), t, device=DEVICE, dtype=torch.long)

        # DDPM forward
        eps = torch.randn_like(x0).to(device=DEVICE)
        zt = ddpm.add_noise(x0, eps, t)
        zt= zt.to(device=DEVICE)

        with torch.no_grad():
            # t -> t-1

            #we only have to use the last teacher channels for both of the z predictions 
            eps_t = teacher(zt, t).sample
            eps_t = eps_t[:,-3:,:,:]
            zt_1 = ddim_step_explicit(
                z_t=zt,
                eps=eps_t,
                t=t,
                scheduler=ddim_teacher,
                device=DEVICE,
            ) #this is just paremeterization and no learning is taking place 

            # t-1 -> t-2
            eps_t1 = teacher(zt_1, t - 1).sample
            eps_t1= eps_t1[:,-3:,:,:]
            zt_2 = ddim_step_explicit(
                z_t=zt_1,
                eps=eps_t1,
                t=t - 1,
                scheduler=ddim_teacher,
                device=DEVICE,
            )

      
        eps_s = student(zt, k).sample
        eps_s2= eps_s[:,-3:,:,:] #actual noise prediction
        eps_s1= eps_s[:,:3,:,:] #intermediate step 

        zt_student_1 = ddim_step_explicit(
            z_t=zt,
            eps=eps_s1,
            t=k,
            scheduler=ddim_student,
            device=DEVICE,
        )
        zt_student_2 = ddim_step_explicit(
            z_t=zt,
            eps=eps_s2,
            t=k,
            scheduler=ddim_student,
            device=DEVICE,
        )

        alpha_s, sigma_s = get_alpha_sigma(ddim_student, k, DEVICE)
        w = gamma_weight(alpha_s, sigma_s, GAMMA) #we use gamma weighing 

                
        loss_1 = (w * (zt_student_2 - zt_2) ** 2).mean()
        loss_2 = (w * (zt_student_1 - zt_1) ** 2).mean()
        loss = loss_1 + loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(student)

        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

        if global_step % 1000 == 0:
            torch.save(
                {
                    "step": global_step,
                    "student": student.state_dict(),
                    "ema": ema.shadow,
                },
                f"{SAVE_DIR}/ckpt_{global_step}.pt"
            )

print("✅ Distillation 500 → 250 finished")
