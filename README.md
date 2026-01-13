 The significant computational cost of the iterative sampling
process in diffusion models hinders their practical applica-
tion. Progressive Distillation accelerates this process but dis-
cards valuable information from the teacher model’s interme-
diate steps, leading to knowledge loss. To mitigate this issue,
we propose a novel framework,
BranchDistillation, where the student model is trained using
branches that simultaneously map to the entire sequence of
the teacher’s designated timesteps. By capturing the com-
plete denoising trajectory, our method ensures comprehen-
sive knowledge transfer. Experimental results demonstrate
that BranchDistillation achieves superior performance com-
pared to Progressive Distillation while maintaining com-
parable computational efficiency. Our findings suggest that
BranchDistillation offers a more effective approach to knowl-
edge distillation, significantly reducing sampling time in dif-
fusion models

The results of our experiments are shown below 



<img width="618" height="499" alt="image" src="https://github.com/user-attachments/assets/c7b274e9-3994-4402-875c-79e21c8fbb08" />


In our framework, if the teacher model T performs N steps
that we want to distill into a single student step, we re-
configure the final layer in the student model S to output
N*C channels, where C is the number of channels in the
output image. These output channels are then conceptually
reshaped into N distinct ’branches’, each producing a full-
resolution tensor of shape [C,H,W]. N is set to 2 for our
experiments
Each branch is responsible for predicting one of the teacher’s
intermediate denoised images.The BranchDistillation loss is
then calculated as the sum of the reconstruction losses be-
tween each student branch prediction and its corresponding
teacher state.We use MSE loss. This has the effect of essen-
tially absorbing knowledge from all the timesteps used to
denoise , instead of just mapping the concerned outputs.
