import os

import numpy as np
import torch

from distributions.discrete import DiscretizedQuadratic, ProductOfLocalUniformOrdinals
from samplers import regular_samplers, auxiliary_samplers
from utils.utils import numpify, unique_vectors_and_counts

###### SETUP ######
torch.set_default_dtype(torch.float32)
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
state_space_size = 6
data_dim = 2

save_dir = "C:/Users/benja/Code/ebm/results/general_figures"
print("saving to:", save_dir)
os.makedirs(save_dir, exist_ok=True)

# STATE SPACE
# color = 'cyan'
color = 'aliceblue'
state_space_min, state_space_max = -1.0, 1.0
state_space = torch.linspace(state_space_min, state_space_max, state_space_size, device=device)
length_scale = (state_space[1] - state_space[0]).item()
x = y = numpify(state_space)
X, Y = np.meshgrid(x, y)
full_ss = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
full_ss = torch.from_numpy(full_ss).to(device)
full_ss_grid = full_ss.view(*X.shape, 2)


###### OVERVIEW FIGURE ##########

e = torch.tensor([2.0, 50.0], device='cuda:0')
Q = torch.tensor([[0.7071, 0.7071],
                  [-0.7071, 0.7071]], device='cuda:0')
QeQ = Q @ torch.diag_embed(e) @ Q.T
b = torch.tensor([6.3378e-05, 6.3378e-05], device='cuda:0')
model = DiscretizedQuadratic(H=QeQ, b=b, state_space_1d=state_space, data_dim=data_dim)
model.to(device)

fig, axs = model.plot_2d(logspace=True, subplots=(1, 3), figsize=(18, 7), alpha=1.0, use_cbar=True)
for ax in axs:
    ax.axis('off')
for ax in axs:
    ax.scatter(numpify(full_ss[:, 0]), numpify(full_ss[:, 1]), facecolors='none', edgecolors='k', linewidth=0.2)
i, j = 3, 2
s0 = full_ss_grid[i, j].unsqueeze(0)  # (1, 2)
s0tiled = s0.tile((10000, 1))
samplers = [
    regular_samplers.OrdinalGWGSampler(2, state_space, radius=1),
    regular_samplers.NCGSampler(2, 0.5, state_space=state_space, var_type="ordinal"),
    auxiliary_samplers.PAVGSampler(n_dims=2,
                                   epsilon=1000.0,
                                   adaptive_update_freq=100,
                                   init_adapt_stepsize=0.25,
                                   adapt_stepsize_decay=0.99,
                                   init_precon_matrix=QeQ,
                                   variable_type="ordinal",
                                   state_space=state_space),
]
names = ["Gibbs-with-Gradients", "Norm-Constrained Gradient", "Preconditioned AVG"]
for i, sampler in enumerate(samplers):
    s_prop = sampler.sample_proposal(s0tiled, model)
    s_prop = s_prop.detach().cpu().numpy()
    s_prop, counts = unique_vectors_and_counts(s_prop)
    axs[i].scatter(s_prop[:, 0], s_prop[:, 1], label="samples", s=2*counts**0.75, c=color, edgecolors='k', linewidth=1.0)
    axs[i].scatter(numpify(s0[:, 0]), numpify(s0[:, 1]), c='k', marker='x', s=200.0, linewidth=1.5)
    axs[i].set_title(names[i], fontsize=15)

spath = os.path.join(save_dir, "overview_figure.png")
fig.savefig(spath, bbox_inches="tight")
spath = os.path.join(save_dir, "overview_figure.pdf")
fig.savefig(spath, bbox_inches="tight")


###### APPENDIX FIGURE ##########
e = torch.tensor([2.0, 10.0], device='cuda:0')
Q = torch.tensor([[0.7071, 0.7071],
                  [-0.7071, 0.7071]], device='cuda:0')
QeQ = Q @ torch.diag_embed(e) @ Q.T
b = torch.tensor([6.3378e-05, 6.3378e-05], device='cuda:0')
model = DiscretizedQuadratic(H=QeQ, b=b, state_space_1d=state_space, data_dim=data_dim)
model.to(device)
fig, axs = model.plot_2d(logspace=True, subplots=(1, 3), figsize=(18, 7), alpha=1.0, use_cbar=True)
for ax in axs:
    ax.axis('off')
for ax in axs:
    ax.scatter(numpify(full_ss[:, 0]), numpify(full_ss[:, 1]), facecolors='none', edgecolors='k', linewidth=0.2)

i, j = 4, 2
s0 = full_ss_grid[i, j].unsqueeze(0)  # (1, 2)
s0tiled = s0.tile((10000, 1))

def prop(x, model):
    return ProductOfLocalUniformOrdinals(x, state_space=state_space, radius=2)

samplers = [
    regular_samplers.OrdinalGWGSampler(2, state_space, radius=1),
    regular_samplers.OrdinalGWGSampler(2, state_space, radius=3),
    regular_samplers.MHSampler(prop, length_scale=(state_space[1] - state_space[0]).item()),
]
names = ["GWG", "Ordinal-GWG", "MH Uniform"]
for i, sampler in enumerate(samplers):
    s_prop = sampler.sample_proposal(s0tiled, model)
    s_prop = s_prop.detach().cpu().numpy()
    s_prop, counts = unique_vectors_and_counts(s_prop)
    axs[i].scatter(s_prop[:, 0], s_prop[:, 1], label="samples", s=2*counts**0.75, c=color, edgecolors='k', linewidth=1.0)
    axs[i].scatter(numpify(s0[:, 0]), numpify(s0[:, 1]), c='k', marker='x', s=200.0, linewidth=1.5)
    axs[i].set_title(names[i], fontsize=15)

spath = os.path.join(save_dir, "ordinal_baselines.png")
fig.savefig(spath, bbox_inches="tight")
spath = os.path.join(save_dir, "ordinal_baselines.pdf")
fig.savefig(spath, bbox_inches="tight")
