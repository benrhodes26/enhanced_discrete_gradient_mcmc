from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from utils import mmd
from utils.utils import torchify, save_fig


def add_method_to_fig(method_name, K, metric, path, axs):
    with open(path + f"{method_name}/{K}/{metric}_per_iter.npz", 'rb') as f:
        res_dict = dict(np.load(f))
        y = res_dict[f"{metric}"]
        time, n_iters = res_dict["total_time"] / 60, res_dict["total_itrs"]
        axs[0].plot(np.linspace(0, n_iters, len(y)), y, label=f"{method_name} K={K}")
        axs[1].plot(np.linspace(0, time, len(y)), y, label=f"{method_name} K={K}")


def make_fro_error_latex_tables(methods, all_K_vals, n_dims, num_runs, col_names, dirpath):
    all_rmses = []
    for i in range(1, num_runs + 1):
        path_i = dirpath + str(i)
        rmses_for_run = []
        for method, K_values in zip(methods, all_K_vals):

            rmses_for_method = []
            for k in K_values:
                fpath = os.path.join(path_i, method, str(k), "reestimated")
                if os.path.isdir(fpath):
                    fpath = os.path.join(fpath, "rmse.txt")
                else:
                    fpath = os.path.join(path_i, method, str(k), "rmse.txt")

                with open(fpath, 'r') as f:
                    line = f.readline()
                    rmses_for_method.append(float(line))

            rmses_for_run.append(rmses_for_method)

        all_rmses.append(rmses_for_run)

    all_rmses = np.array(all_rmses)  # (n_runs, n_methods, n_Kvals)
    frobenius_norms = (n_dims * np.array(all_rmses) ** 2) ** 0.5

    print("Mean frobenius norm for each method & K values")
    mean_frobenius_norms = frobenius_norms.mean(0)  # (n_methods, n_Kvals)
    df = pd.DataFrame(mean_frobenius_norms, index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("standard deviation of frobenius norm for each method & K values")
    std_frobenius_norms = frobenius_norms.std(0)  # (n_methods, n_Kvals)
    df = pd.DataFrame(std_frobenius_norms, index=methods, columns=col_names).round(4)
    print(df.to_latex())

    return frobenius_norms.mean(0)

def marginal_cov_error(x, y):
    marginal_error = torch.abs(x.mean(0) - y.mean(0)).mean().item()
    cov_diff = torch.cov(x.T) - torch.cov(y.T)  # (D, D)
    cov_error = (cov_diff**2).sum().sqrt().item()
    return marginal_error, cov_error


def mean_std_mmd_via_subsampling(x, y, size=1000, n_subsamples=5, max=False):
    mmds = []
    for i in range(n_subsamples):
        x_sub = x[torch.randperm(len(x))][:size]
        y_sub = y[torch.randperm(len(y))][:size]
        MMDKernel = mmd.MMD(mmd.exp_avg_hamming, use_ustat=False)
        mmd_est = MMDKernel.compute_mmd(x_sub, y_sub)
        mmds.append(mmd_est)

    mmds = torch.Tensor(mmds)  # (n_subsamples,)
    if max:
        return mmds.max()  # take worst-case, to be conservative
    else:
        return mmds.mean(), mmds.std()


def single_run_mmd_table(methods, all_K_vals, run_id, col_names, dirpath, true_fpath):

    with open(true_fpath, 'rb') as f:
        true_buffer = torchify(np.load(true_fpath)["buffer"])

    path_i = dirpath + str(run_id)
    mmds_for_run = []
    stds_for_run = []
    for method, K_values in zip(methods, all_K_vals):

        mmds_for_method = []
        stds_for_method = []
        for k in K_values:
            fpath = os.path.join(path_i, method, str(k), "reestimated")
            if os.path.isdir(fpath):
                fpath = os.path.join(fpath, "buffer.npz")
            else:
                fpath = os.path.join(path_i, method, str(k), "buffer.npz")

            buffer = np.load(fpath)["buffer"]
            mmd, mmd_std = mean_std_mmd_via_subsampling(torchify(buffer, device='cpu'), torchify(true_buffer, device='cpu'))
            mmds_for_method.append(mmd)
            stds_for_method.append(mmd_std)

        mmds_for_run.append(mmds_for_method)
        stds_for_run.append(stds_for_method)

    df = pd.DataFrame(np.array(mmds_for_run), index=methods, columns=col_names)
    df = df.round(3)
    with open(os.path.join(path_i, "mmd_latex_table.txt"), 'w') as f:
        print(df.to_latex())
        df.to_latex(f)

    print("Mean MMD for each method & K values")
    df = pd.DataFrame(mmds_for_run, index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("Std across subsamples for MMD for each method & K values")
    df = pd.DataFrame(stds_for_run, index=methods, columns=col_names).round(4)
    print(df.to_latex())


def make_other_metric_tables(methods, all_K_vals, num_runs, col_names, dirpath, true_fpath, n_samples=1000):

    with open(true_fpath, 'rb') as f:
        true_buffer = torchify(np.load(true_fpath)["buffer"])

    all_mmds = []
    all_marginal_error = []
    all_cov_error = []
    for i in range(1, num_runs + 1):
        path_i = dirpath + str(i)
        mmds_for_run = []
        marg_error_for_run = []
        cov_error_for_run = []
        for method, K_values in zip(methods, all_K_vals):

            mmds_for_method = []
            marg_error_for_method = []
            cov_error_for_method = []
            for k in K_values:
                fpath = os.path.join(path_i, method, str(k), "reestimated")
                if os.path.isdir(fpath):
                    fpath = os.path.join(fpath, "buffer.npz")
                else:
                    fpath = os.path.join(path_i, method, str(k), "buffer.npz")

                buffer = torchify(np.load(fpath)["buffer"])
                mmd = mean_std_mmd_via_subsampling(buffer, true_buffer, size=n_samples, max=True)
                marginal_error, cov_error = marginal_cov_error(buffer, true_buffer)

                mmds_for_method.append(mmd)
                marg_error_for_method.append(marginal_error)
                cov_error_for_method.append(cov_error)

            mmds_for_run.append(mmds_for_method)
            marg_error_for_run.append(marg_error_for_method)
            cov_error_for_run.append(cov_error_for_method)

        all_mmds.append(mmds_for_run)
        all_marginal_error.append(marg_error_for_run)
        all_cov_error.append(cov_error_for_run)

    all_mmds = np.array(all_mmds)  # (n_runs, n_methods, n_Kvals)
    all_marginal_error = np.array(all_marginal_error)  # (n_runs, n_methods, n_Kvals)
    all_cov_error = np.array(all_cov_error)  # (n_runs, n_methods, n_Kvals)

    print("####################################")
    print("Mean MMD for each method & K values")
    df = pd.DataFrame(all_mmds.mean(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("####################################")
    print("standard deviation of MMD for each method & K values")
    df = pd.DataFrame(all_mmds.std(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("####################################")
    print("Mean marginal-error for each method & K values")
    df = pd.DataFrame(all_marginal_error.mean(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("####################################")
    print("standard deviation of marginal-error for each method & K values")
    df = pd.DataFrame(all_marginal_error.std(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("####################################")
    print("Mean cov-error for each method & K values")
    df = pd.DataFrame(all_cov_error.mean(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    print("####################################")
    print("standard deviation of cov-error for each method & K values")
    df = pd.DataFrame(all_cov_error.std(0), index=methods, columns=col_names).round(4)
    print(df.to_latex())

    return all_mmds.mean(0), all_marginal_error.mean(0), all_cov_error.mean(0)


def make_buffer_plots(methods, all_K_vals, n_dims, dirpath, true_fpath):

    dirpath = dirpath + str(1)
    dim_sqrt = int(n_dims ** 0.5)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, dim_sqrt, dim_sqrt), p, normalize=False, nrow=8)
    save_dir = os.path.join(dirpath, "buffer_img_plots")
    os.makedirs(save_dir, exist_ok=True)

    for method, K_values in zip(methods, all_K_vals):
        for k in K_values:
            fpath = os.path.join(dirpath, method, str(k), "reestimated", "buffer.npz")
            with open(fpath, 'rb') as f:
                buffer = torchify(np.load(fpath)["buffer"][:64])  # (we create an 8x8 grid of images)
            save_path = os.path.join(save_dir, method + "_" + str(k) + ".pdf")
            plot(save_path, buffer)

    # plot real USPS data & ground-truth EBM data
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = tr.Compose([tr.Resize(dim_sqrt), tr.ToTensor(), lambda x: (x > .5).float().view(-1)])
    train_data = torchvision.datasets.USPS(root="C:/Users/benja/Code/ebm/data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, 64, shuffle=True, drop_last=True)
    with open(true_fpath, 'rb') as f:
        buffer = torchify(np.load(true_fpath)["buffer"])  # (we create an 8x8 grid of images)
    for i in range(4):
        save_path = os.path.join(save_dir, "GWG" + "_" + str(50) + "_" + str(i) + ".pdf")
        plot(save_path, buffer[i*64:(i+1)*64])
        save_path = os.path.join(save_dir, "USPS" + "_" + str(i) + ".pdf")
        plot(save_path, torchify(next(iter(train_loader))[0]))


def main():

    # plot_type = "ising_lattice_table"
    plot_type = "usps_table"

    if plot_type == "ising_lattice_table":

        methods = ["NCG", "AVG", "Block-Gibbs (PAVG model-specific)", "PAVG model-agnostic", "GWG", "Gibbs"]
        # different K values needed to ensure roughly equal wallclock time
        all_K_vals = [
            [1, 6, 12, 18, 24],
            [1, 5, 10, 15, 20],
            [3, 15, 30, 45, 60],
            [1, 5, 10, 15, 20],
            [1, 6, 12, 18, 24],
            [2, 10, 20, 30, 40]
        ]
        n_dims = 100
        num_runs = 3
        col_names = ["1", "5", "10", "15", "20"]
        dirpath = "C:/Users/benja/Code/ebm/results/ising_lattice_sigma0.2_run"
        make_fro_error_latex_tables(methods, all_K_vals, n_dims, num_runs, col_names, dirpath)

    if plot_type == "usps_table":

        methods = ["NCG", "AVG", "PAVG", "GWG", "Gibbs"]
        # different K values needed to ensure roughly equal wallclock time
        all_K_vals = [
            [5, 10, 15, 20],
            [5, 10, 15, 20],
            [5, 10, 15, 20],
            [5, 10, 15, 20],
            [8, 16, 24, 32]
        ]
        n_dims = 256
        num_runs = 3
        col_names = ["5", "10", "15", "20"]
        dirpath = "C:/Users/benja/Code/ebm/results/ising_neural_frozen_usps_run"
        true_fpath = os.path.join("C:/Users/benja/Code/ebm/results/ising_neural_usps/gwg/50/buffer.npz")

        make_buffer_plots(methods, all_K_vals, n_dims, dirpath, true_fpath)
        fro_errors = make_fro_error_latex_tables(methods, all_K_vals, n_dims, num_runs, col_names, dirpath)
        metrics = make_other_metric_tables(methods, all_K_vals, 3, col_names, dirpath, true_fpath)
        fig, axs = plt.subplots(1, 3, figsize=(22, 6))
        axs = axs.ravel()
        fro_errors = fro_errors.ravel()
        for ax, metric, name in zip(axs, metrics, ["MMD", "marginal error", "covariance error"]):
            metric = metric.ravel()

            regr = linear_model.LinearRegression()
            regr.fit(fro_errors.reshape(-1, 1), metric.reshape(-1, 1))
            preds = regr.predict(fro_errors.reshape(-1, 1)).reshape(-1)
            R2 = r2_score(metric.reshape(-1, 1), preds.reshape(-1, 1))

            ax.scatter(fro_errors, metric, c="k")
            ax.plot(fro_errors, preds, label=f"R^2 = {R2:.4f}")

            ax.set_ylabel(name)
            ax.set_xlabel("Parameter error")
            ax.legend(fontsize=14)

        save_fig(fig, dirpath + "1", "metric_comparisons")


    if plot_type == "train_curves":
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs = axs.ravel()
        path = "C:/Users/benja/Code/ebm/results/ising_lattice_sigma0.2/"
        metric = "rmses"

        add_method_to_fig("NCG", 5, metric, path, axs)
        add_method_to_fig("AVG", 20, metric, path, axs)
        add_method_to_fig("Block-Gibbs (PAVG model-specific)", 1, metric, path, axs)
        add_method_to_fig("PAVG model-agnostic", 10, metric, path, axs)
        add_method_to_fig("GWG", 10, metric, path, axs)
        add_method_to_fig("Gibbs", 15, metric, path, axs)

        axs[0].set_xlabel("Training iteration")
        axs[0].set_ylabel("RMSE")
        axs[1].set_xlabel("Minutes")
        for ax in axs:
            ax.legend()
            ax.set_ylim((0.0, 0.1))

        fig.suptitle("Ising graph structure estimation")
        fig.savefig(path + f"{metric}_comparison", bbox_inches="tight")
        fig.savefig(path + f"{metric}_comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
