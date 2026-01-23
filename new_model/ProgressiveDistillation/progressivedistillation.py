import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
EPOCHS = 140
LR = 2e-5
GAMMA= 1.0
TEACHER_STEPS = 250
STUDENT_STEPS = 125

SAVE_DIR = "./pd_250_to_125"
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
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)


config = UNet2DModel.load_config(
    "/teamspace/studios/this_studio/new_model/pretrained/ddpm_ema_cifar10/unet"
)

teacher = UNet2DModel.from_config(config)
checkpoint = torch.load(
    "/teamspace/studios/this_studio/pd_500_to_250/ckpt_54000.pt",
    map_location="cpu",
)
teacher.load_state_dict(checkpoint['student'])
teacher.to(DEVICE).eval()

student = UNet2DModel.from_config(config)
student.load_state_dict(teacher.state_dict())  #init student w tteacher 
student.to(DEVICE).train()

ema = EMA(student)

ddpm = DDPMScheduler(num_train_timesteps=TEACHER_STEPS)

ddim_teacher = DDIMScheduler(
    num_train_timesteps=TEACHER_STEPS,
    prediction_type="epsilon",
)
ddim_teacher.set_timesteps(TEACHER_STEPS)

ddim_student = DDIMScheduler(
    num_train_timesteps=STUDENT_STEPS,
    prediction_type="epsilon",
)
ddim_student.set_timesteps(STUDENT_STEPS)

optimizer = AdamW(student.parameters(), lr=LR)
loss_fn = nn.MSELoss()



global_step = 0

for epoch in range(EPOCHS):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x0, _ in pbar:
        x0 = x0.to(DEVICE)
        bsz = x0.size(0)

        # sample even timestep: t = 2k
        k = torch.randint(1, STUDENT_STEPS, (bsz,), device=DEVICE)
        t = 2 * k
        #t_vec = torch.full((bsz,), t, device=DEVICE, dtype=torch.long)

        # DDPM forward
        eps = torch.randn_like(x0)
        zt = ddpm.add_noise(x0, eps, t)

        with torch.no_grad():
            # t -> t-1
            eps_t = teacher(zt, t).sample
            zt_1 = ddim_step_explicit(
                z_t=zt,
                eps=eps_t,
                t=t,
                scheduler=ddim_teacher,
                device=DEVICE,
            )

            # t-1 -> t-2
            eps_t1 = teacher(zt_1, t - 1).sample
            zt_2 = ddim_step_explicit(
                z_t=zt_1,
                eps=eps_t1,
                t=t - 1,
                scheduler=ddim_teacher,
                device=DEVICE,
            )

      
        eps_s = student(zt, k).sample
        zt_student = ddim_step_explicit(
            z_t=zt,
            eps=eps_s,
            t=k,
            scheduler=ddim_student,
            device=DEVICE,
        )

        alpha_s, sigma_s = get_alpha_sigma(ddim_student, k, DEVICE)
        w = gamma_weight(alpha_s, sigma_s, GAMMA)

        loss = torch.mean(w * (zt_student - zt_2) ** 2)
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

print("✅ Distillation 250 → 125 finished")
