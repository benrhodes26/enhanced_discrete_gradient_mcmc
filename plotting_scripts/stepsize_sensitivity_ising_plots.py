import os
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import save_fig
import matplotlib.ticker as mticker


load_path = "C:/Users/benja/Code/ebm/results/ising_lattice_sigma0.2_stepsize_sensitivity/"
names = ["NCG", "AVG", "PAVG model-agnostic"]
ncg_epvals = np.array([(0.5 * (3/2)**i) for i in range(-5, 6)])
avg_epvals = np.array([(0.2 * (3/2)**i) for i in range(-5, 6)])
pavg_epvals = np.array([(0.2 * (3/2)**i) for i in range(-5, 6)])
all_eps = [ncg_epvals, avg_epvals, pavg_epvals]
chosen_ep_idx = 5
gwg_error = ((0.138**2)/100)**0.5

all_errors = {}
for name in names:
    for i in range(len(ncg_epvals)):
        path = os.path.join(load_path, f"ep{i}", name, "10", "reestimated", "rmses_per_iter.npz")
        rmses = np.load(path)["rmses"]
        all_errors.setdefault(name, []).append(rmses)


fig, axs = plt.subplots(1, 3, figsize=(12, 5))
axs = axs.ravel()
for ax, name, eps in zip(axs, all_errors, all_eps):
    chosen_ep = eps[chosen_ep_idx]

    final_errors = np.array([e[-1] for e in all_errors[name]])
    chosen_error = final_errors[chosen_ep_idx]
    x = eps / chosen_ep
    y = final_errors / chosen_error

    ax.plot(x, y, label=f"{name}")
    ax.plot((x.min(), x.max()), (gwg_error / chosen_error, gwg_error / chosen_error), linestyle='--', label="GWG")

    ax.set_xscale('log')
    ax.set_yscale('log')
    if name.startswith("AVG"):
        ax.set_ylim((0.01, 3.0))
        ax.set_yticks([0.01, 0.1, 1])
    else:
        ax.set_ylim((0.8, 10))
        ax.set_yticks([1, 2, 5, 10])

    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels([], minor=True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.set_xlabel(r"Relative step-size", fontsize=13)
    if name == "NCG": ax.set_ylabel(r"Relative error", fontsize=13)
    ax.set_title(name, fontsize=14)
    ax.legend()


fig.suptitle("Effect of step-size on estimation error", fontsize=15, y=1.03)
save_fig(fig, load_path, "step_size_sensitivity")
