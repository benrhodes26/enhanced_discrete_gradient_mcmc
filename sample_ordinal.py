import argparse
from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from samplers.run_sample import run_sampling_procedure
import pickle
from distributions.discrete import ProductOfDiscretized1dSimpleFunctions, Ordinal, \
    MixtureOfProductOfDiscretized1dSimpleFunctions, DiscretizedQuadratic
from utils.mcmc_utils import plot_ess_boxplots, plot_acc_rates, plot_hops
from utils.utils import set_default_rcparams, unique_vectors_and_counts, MyTimer, get_list_of_markers, save_fig

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_default_rcparams()


def plot_target_dist(data_dim, model, save_dir):
    fig, _ = model.plot()
    save_fig(fig, save_dir, "1D_ordinal_dist")
    if data_dim == 2:
        fig, ax = model.plot_2d(use_cbar=False)
        save_fig(fig, save_dir, "2d_ordinal_logdist")

        fig, ax = model.plot_2d(logspace=False, use_cbar=False)
        samples = model.sample((1000,)).detach().cpu().numpy()
        unique_samples, counts = unique_vectors_and_counts(samples)
        ax.scatter(unique_samples[:, 0], unique_samples[:, 1], label="samples", s=counts, c='y')
        save_fig(fig, save_dir, "2d_ordinal_dist")


def plot_mcmc_progress(model, emp_marginals, kls, x, iter, path):

    if len(kls) > 1:
        fig, ax = plt.subplots(1, 1)
        x_axis = np.linspace(0, int(iter) + 1, len(kls))

        kls_np = np.array(kls)
        ax.plot(x_axis, kls_np)
        second_largest = np.partition(kls_np.flatten(), -2)[-2]
        ax.set_ylim([0.0, 2 * second_largest])

        ax.set_xlabel("iteration")
        ax.set_ylabel("KL")
        ax.set_title("KL between true & empirical distribution of MCMC samples")
        fig.savefig(os.path.join(path, "KL.png"))
        plt.close(fig)

    ss = model.state_space_1d.detach().cpu()
    tiled_ss = torch.tile(model.state_space_1d.unsqueeze(-1), (1, model.data_dim))
    emp_logps = emp_marginals.log_prob(tiled_ss).detach().cpu()  # (n_states, n_dims)
    fig, axs = model.plot(legend=False)
    for i in range(min(10, emp_logps.shape[1])):
        axs[0].scatter(ss, emp_logps[:, i], alpha=0.5, label=f"{i}th log emp marginal")
        axs[1].scatter(ss, emp_logps[:, i].exp(), alpha=0.5, label=f"{i}th emp marginal")
        axs[0].legend()
        axs[1].legend()
    fig.savefig(os.path.join(path, f"dists_iter_{iter}.png"))
    plt.close(fig)

    if model.data_dim == 2:
        fig, ax = model.plot_2d(logspace=False)
        xplot = x.detach().cpu().numpy()
        xplot, counts = unique_vectors_and_counts(xplot)
        ax.scatter(xplot[:, 0], xplot[:, 1], label="samples", s=counts, c='y')
        fig.savefig(os.path.join(path, f"dist_2d_{iter}.png"))
        plt.close(fig)


def plot_kl_comparison(kls, method_names, save_dir, n_iters=None, times=None, legsize=12):

    # plot KL graphs
    markers = get_list_of_markers(len(method_names))
    for i in range(2):

        fig, ax = plt.subplots(1, 1)
        all_means = []
        for method, marker in zip(method_names, markers):
            means, stds = zip(*kls[method])
            all_means.append(means)
            means, stds = np.array(means), np.array(stds)
            if n_iters is not None:
                x_axis = np.linspace(0, n_iters[method], len(means))
            else:
                x_axis = np.linspace(0, times[method][-1] / 60, len(means))

            ax.plot(x_axis, means, label=method, marker=marker)
            ax.fill_between(x_axis, means - stds, means + stds, alpha=0.2)

        all_means = np.array(all_means)
        upper_lim = 2 * np.partition(all_means.flatten(), -2)[-2]
        lower_lim = all_means.min() / 2
        ax.set_ylim([lower_lim, upper_lim])

        xlab = "Iterations" if n_iters is not None else "Minutes"
        ax.set_xlabel(xlab)
        ylab = "Average KL" if i == 0 else "Average KL (log-scale)"
        ax.set_ylabel(ylab)
        if i == 1: ax.set_yscale('log')
        ax.set_title("Marginal estimation error")
        ax.legend(loc="best", fontsize=legsize)

        name = "KL" if i == 0 else "log-KL"
        name += " - iters" if n_iters is not None else " - wallclock"
        save_fig(fig, save_dir, name)


def plot_cov_error_comparison(save_dir, cov_errors, method_names, n_iters=None, times=None, legsize=12):

    markers = get_list_of_markers(len(method_names))
    # plot empirical covariance matrix error
    for i in range(2):
        name = "Covariance estimation error"
        fig, ax = plt.subplots(1, 1)
        for method, marker in zip(method_names, markers):
            means, stds = zip(*cov_errors[method])
            means, stds = np.array(means), np.array(stds)

            if n_iters is not None:
                x_axis = np.linspace(0, n_iters[method], len(means))
            else:
                x_axis = np.linspace(0, times[method][-1] / 60, len(means))

            ax.plot(x_axis, means, label=method, marker=marker)
            ax.fill_between(x_axis, means - stds, means + stds, alpha=0.2)

        xlab = "Iterations" if n_iters is not None else "Minutes"
        ax.set_xlabel(xlab)
        ylab = "Frobenius norm" if i == 0 else "Frobenius norm (log-scale)"
        ax.set_ylabel(ylab)
        if i == 1: ax.set_yscale('log')
        ax.set_title(name)
        ax.legend(loc="best", fontsize=legsize)
        if i == 1: name += " - logscale"
        name += " - iters" if n_iters is not None else " - wallclock"
        save_fig(fig, save_dir, name)


def plot_and_save(args, method_dicts, chains, hops, accept_rates, metrics, ess, n_iters, times, save_dir):

    # save arguments passed into this function (to make it trivial to recreate figures later)
    to_save = locals()
    with open(os.path.join(save_dir, "results.pkl"), 'wb') as f:
        pickle.dump(to_save, f)

    method_names = [m["name"] for m in method_dicts]

    plot_ess_boxplots(args, ess, method_names, save_dir, times)
    plot_acc_rates(accept_rates, method_names, n_iters, save_dir)
    plot_hops(method_names, hops, save_dir)

    kls = {name: metrics[name]["marginal_kl"] for name in method_names}
    plot_kl_comparison(kls, method_names, save_dir, n_iters=n_iters)
    plot_kl_comparison(kls, method_names, save_dir, times=times)

    cov_errors = {name: metrics[name]["cov_error"] for name in method_names}
    plot_cov_error_comparison(save_dir, cov_errors, method_names, n_iters=n_iters)
    plot_cov_error_comparison(save_dir, cov_errors, method_names, times=times)


def compute_metrics(args, method, x_all, metrics_dict, sampler, target_dist, true_cov_mat):

    n_chains, n_steps, n_dims = x_all.shape
    x_all_c = x_all - x_all.mean(dim=1, keepdims=True)
    est_cov_mat = (1/n_steps) * torch.bmm(x_all_c.swapaxes(1, 2), x_all_c)  # (n_chains, n_dims, n_dims)
    cov_error = torch.linalg.norm(est_cov_mat - true_cov_mat, dim=(1, 2))  # (n_chains,)

    av_cov_error, std_cov_error = cov_error.mean().item(), cov_error.std().item() / n_chains ** 0.5
    metrics_dict[method].setdefault("cov_error", []).append([av_cov_error, std_cov_error])
    print(f"len chain: {len(x_all)}. COV error: {av_cov_error:.4f}")

    # compute empirical marginal distributions (over a sliding window of samples from the MCMC chains)
    if x_all.numel() == 0:
        emp_marginals = deepcopy(target_dist.init_dist)
        emp_marginals.product = False
    else:
        counts = (x_all.unsqueeze(-1) == target_dist.state_space_1d.to(x_all.device)).sum(1).float()  # (n_chains, n_dims, n_states)
        counts += 1e-5
        freqs = counts / counts.sum(-1, keepdim=True)  # (n_chains, n_dims, n_states)
        emp_marginals = Ordinal(logits=torch.log(freqs), state_space=target_dist.state_space_1d.to(x_all.device))

    tiled_ss = torch.tile(target_dist.state_space_1d[:, None, None], (1, n_chains, n_dims))  # (n_states, n_chains, n_dims)
    logq_ss = emp_marginals.log_prob(tiled_ss.to(x_all.device))  # (n_states, n_chains, n_dims)
    marginals = target_dist.log_prob_1d(tile=True).to(x_all.device)  # (n_states, n_dims)
    cross_entropy = (-marginals.unsqueeze(1).exp() * logq_ss).sum(0).mean(-1)  # (n_chains,)
    kls = target_dist.neg_entropy_1d().to(x_all.device) + cross_entropy  # (n_chains,)

    av_marginal_kl, std_marginal_kl = kls.mean().item(), kls.std().item() / n_chains**0.5
    metrics_dict[method].setdefault("marginal_kl", []).append([av_marginal_kl, std_marginal_kl])
    print(f"KL(p_true, p_empirical) = {av_marginal_kl}")


def define_target_dist(args, save_dir):

    ss_min = args.state_space_min if args.state_space_min else -args.state_space_max
    state_space = torch.linspace(ss_min, args.state_space_max, args.state_space_size, device=device)

    if "mixture" in args.model_name:
        model = MixtureOfProductOfDiscretized1dSimpleFunctions(model_type=args.model_name,
                                                               state_space_1d=state_space,
                                                               data_dim=args.data_dim,
                                                               point_init=args.point_init)
    elif "quadratic" in args.model_name:
        # e = torch.tensor([-5.0, 0.1], device='cuda:0')
        e = torch.tensor([-50.0, 1.0], device='cuda:0')
        Q = torch.tensor([[0.7071, 0.7071],
                          [-0.7071, 0.7071]], device='cuda:0')
        QeQ = Q @ torch.diag_embed(e) @ Q.T
        H = QeQ
        b = torch.tensor([6.3378e-05, 6.3378e-05], device='cuda:0')
        model = DiscretizedQuadratic(H=H, b=b, state_space_1d=state_space, data_dim=args.data_dim,
                                     point_init=args.point_init)
    else:
        model = ProductOfDiscretized1dSimpleFunctions(model_type=args.model_name,
                                                      state_space_1d=state_space,
                                                      data_dim=args.data_dim,
                                                      point_init=args.point_init)
    model.to(device)
    plot_target_dist(args.data_dim, model, save_dir)
    true_samples = model.sample((100000,))
    true_cov_mat = torch.cov(true_samples.T)  # (D, D)
    del true_samples

    chain_init = model.init_sample((args.n_samples,)).to(device)

    return model, chain_init, true_cov_mat.cpu()


def main(args):

    target_dist, chain_init, true_cov_mat = define_target_dist(args, args.save_dir)

    def metric_fn(*x, td=target_dist, tc=true_cov_mat):
        return compute_metrics(*x, target_dist=td, true_cov_mat=tc)

    base_radius = int(args.state_space_size / 50)
    if args.data_dim == 20:
        if "poly4" in args.model_name:
            methods = [
                # {'name': 'NCG', 'epsilon': 0.05},
                # {'name': 'AVG', 'epsilon': 0.02},
                {'name': 'PAVG', 'epsilon': 0.02, 'postadapt_epsilon': 0.06},
                {'name': 'Gibbs', 'random_scan': True},
                {'name': 'GWG', 'radius': 1},
                {'name': 'GWG-ordinal', 'radius': 8 * base_radius},
                {'name': 'MH-uniform', 'radius': base_radius},
            ]
        elif "poly2" in args.model_name:
            methods = [
                {'name': 'NCG', 'epsilon': 0.05},
                {'name': 'AVG',  'epsilon': 0.02},
                {'name': 'PAVG', 'epsilon': 0.02, 'postadapt_epsilon': 1000},
                {'name': 'Gibbs', 'random_scan': True},
                {'name': 'GWG', 'radius': 1},
                {'name': 'GWG-ordinal', 'radius': 16 * base_radius},
                {'name': 'MH-uniform', 'radius': 2 * base_radius},
            ]
        else:
            raise NotImplementedError("need to specify methods and hyperparameters for: ", args.model_name)
    else:
        raise NotImplementedError("need to specify methods and hyperparameters for: ", args.model_name)


    run_sampling_procedure(args, methods, target_dist, chain_init, metric_fn, plot_and_save)


def parse_args():

    dim = 20
    model_name = 'mixture50_poly4'
    # model_name = 'mixture50_poly2'
    state_space_min, state_space_max = -1.5, 3.0
    n_chains = 200
    state_space_size = 50
    wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 10.5, 0.01, 1.0  # measured in minutes
    # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 0, 1000, 10, 100  # measured in iterations

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "results", "ordinal"))
    parser.add_argument('--seed', type=int, default=123437)
    parser.add_argument('--model_name', type=str, default=model_name)
    parser.add_argument('--var_type', type=str, default="ordinal")
    parser.add_argument('--state_space_size', type=int, default=state_space_size)
    parser.add_argument('--state_space_min', type=float, default=state_space_min)
    parser.add_argument('--state_space_max', type=float, default=state_space_max)
    parser.add_argument('--point_init', type=int, default=1, choices={0, 1})
    parser.add_argument('--data_dim', type=int, default=dim)
    parser.add_argument('--debug_sampler', type=int, default=0, choices={0, 1})

    # sampling steps + diagnostics
    parser.add_argument('--wallclock_mode', type=float, default=wallclock_mode,
                        help="if 1, then --max_runtime --save_freq & --metric_tracking_freq are measured in minutes. "
                             "Otherwise they are measured in iterations.")
    parser.add_argument('--max_runtime', type=float, default=max_runtime,
                        help="number of minutes to run each sampler for")
    parser.add_argument('--save_freq', type=int, default=save_freq,
                        help="Frequency at which we save the state of the chain (expressed as number of minutes)."
                             "A value of 0 means we save every iteration.")
    parser.add_argument('--metric_tracking_freq', type=float, default=metric_tracking_freq,
                        help="how often we compute metrics in mins (note: metric computation can be time-consuming")

    parser.add_argument('--n_samples', type=int, default=n_chains, help="number of parallel chains")
    parser.add_argument('--sliding_window_size', type=int, default=10000, help="maximum length of chain stored in memory")
    parser.add_argument('--burn_in', type=float, default=.1, help="fraction of max_runtime spent for burn-in")
    parser.add_argument('--no_ess', action="store_true")

    args = parser.parse_args()

    subdir = os.path.join(args.save_dir, f"dim{args.data_dim}")
    os.makedirs(subdir, exist_ok=True)
    save_dir = os.path.join(subdir, args.model_name + f"_ssize{args.state_space_size}")
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
