import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from utils.utils import save_fig

save_path = "C:/Users/benja/Code/ebm/results/ising/sigma0.2/dim16/"

strengths = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1 , 0.2, 0.3, 0.4, 0.5])
all_pairwise_errors = {}
for s in strengths:
    target_type = f"higher_order_strength_{s}"
    path = f"C:/Users/benja/Code/ebm/results/ising/sigma0.2/dim16/{target_type}/final_5min"

    with open(os.path.join(path, "results.pkl"), "rb") as f:
        res = pickle.load(f)

    method_names = [m["name"] for m in res["method_dicts"]]
    for m in method_names:
        all_pairwise_errors.setdefault(m, []).append(res["metrics"][m]["pairwise_error"])

fig, ax = plt.subplots(1, 1)
for name in all_pairwise_errors:
    final_errors = [e[-1] for e in all_pairwise_errors[name]]
    final_mean_errors = np.array([e[0] for e in final_errors])
    final_std_errors = np.array([e[1] for e in final_errors])
    lower = (final_mean_errors - final_std_errors)
    upper = (final_mean_errors + final_std_errors)

    ax.plot(strengths, final_mean_errors, label=name)
    ax.fill_between(strengths, lower, upper, alpha=0.2)

    ax.legend()
    ax.set_xlabel(r"coefficient of third-order terms $\alpha$")
    ax.set_ylabel("Estimation error")


ax.set_title("Effect of higher order terms on estimation error")
save_fig(fig, save_path, "effect_of_higher_order_terms")
