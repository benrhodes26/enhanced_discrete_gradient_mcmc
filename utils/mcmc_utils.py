import os

import numpy as np
import torch
import torch.distributions as dists
import tensorflow_probability as tfp
import tensorflow as tf
from matplotlib import pyplot as plt

from distributions.discrete import ProductOfLocalUniformOrdinals
from utils.utils import numpify, torchify, find_idxs_of_rows_of_B_in_A, get_list_of_markers, save_fig


def difference_function(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        x_pert = x.clone()
        x_pert[:, i] = 1. - x[:, i]
        delta = model(x_pert).squeeze() - orig_out
        d[:, i] = delta
    return d


def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()


def difference_function_multi_dim(x, model):
    d = torch.zeros_like(x)
    orig_out = model(x).squeeze()
    for i in range(x.size(1)):
        for j in range(x.size(2)):
            x_pert = x.clone()
            x_pert[:, i] = 0.
            x_pert[:, i, j] = 1.
            delta = model(x_pert).squeeze() - orig_out
            d[:, i, j] = delta
    return d


def approx_difference_function_multi_dim(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    gx_cur = (gx * x).sum(-1)[:, :, None]
    return gx - gx_cur


def short_run_mcmc(logp_net, x_init, k, sigma, step_size=None):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    if step_size is None:
        step_size = (sigma ** 2.) / 2.
    for i in range(k):
        f_prime = torch.autograd.grad(logp_net(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + sigma * torch.randn_like(x_k)

    return x_k


def get_empirical_prob_over_statespace(state_space, x):
    ss, xx = numpify(state_space).astype(np.int32), numpify(x).astype(np.int32)
    idxs = find_idxs_of_rows_of_B_in_A(ss, xx)
    u_idxs, counts = np.unique(idxs, return_counts=True)
    count_vec = np.zeros(len(ss))
    count_vec[u_idxs] = counts
    count_vec /= count_vec.sum()
    return torchify(count_vec)


def batched_empirical_prob_over_statespace(state_space, x):
    n, k, d = x.shape
    ss, xx = numpify(state_space).astype(np.int32), numpify(x).astype(np.int32).reshape(n * k, d)
    idxs = find_idxs_of_rows_of_B_in_A(ss, xx)
    idxs = idxs.reshape(n, k)
    count_vecs = np.zeros((n, len(ss)))
    for i in range(n):
        u_idxs, counts = np.unique(idxs[i], return_counts=True)
        count_vecs[i][u_idxs] = counts
        count_vecs[i] /= count_vecs[i].sum()
    return torchify(count_vecs)


def binary_marginals_and_pairwise_mis(n_dims, logp_fn, device=None, n_model_samples=None):
    assert n_dims <= 20, "This function is extremely expensive in memory for n_dims >= 20"
    if device is None: device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    true_marginals = []
    binary = torch.tensor([0., 1.], device=device)
    state_space = torch.cartesian_prod(*[binary for _ in range(n_dims)])
    if n_dims <= 16:
        probs = logp_fn(state_space).exp()
    else:
        probs = []
        for i in range(0, 2 ** n_dims, 50000):
            stop = min(2 ** n_dims, i + 50000)
            probs.append(logp_fn(state_space[i:stop]).exp())
        probs = torch.cat(probs, dim=0)
    probs /= probs.sum()

    for i in range(n_dims):
        is_one = (state_space[:, i] == 1)
        marginal_i = probs[is_one].sum().item()
        true_marginals.append(marginal_i)
        print(f"p(x_{i} = 1) = ", marginal_i)
    true_marginals = torchify(true_marginals)

    if n_dims <= 4:
        prob_table = numpify(torch.cat([state_space, probs.unsqueeze(-1)], dim=1))
        print(prob_table)

    true_mis = compute_all_pairwise_mis(n_dims, state_space, probs, plot=False)

    samples = None
    if n_model_samples:
        mi_diffs = []
        for i in range(10):
            idxs = torch.distributions.Categorical(probs=probs).sample((n_model_samples,))
            samples = state_space[idxs]
            emp_dist = get_empirical_prob_over_statespace(state_space, samples)
            est_mis = compute_all_pairwise_mis(n_dims, state_space, emp_dist, plot=False)
            mi_diffs.append(np.abs(true_mis - est_mis)[np.triu_indices(n_dims, 1)])
        print(f"Optimal MI with {n_model_samples} samples (mean/std): ", np.mean(mi_diffs), np.std(mi_diffs))

    return state_space, probs, true_marginals, true_mis, samples


def compute_all_pairwise_mis(n_dims, state_space, prob_per_state, plot=False):
    ss = state_space
    pp = prob_per_state
    mis = []
    for i in range(n_dims):
        pi0 = pp[ss[:, i] == 0].sum() + 1e-8
        pi1 = pp[ss[:, i] == 1].sum() + 1e-8
        for j in range(n_dims):
            pj0 = pp[ss[:, j] == 0].sum() + 1e-8
            pj1 = pp[ss[:, j] == 1].sum() + 1e-8

            p11 = pp[(ss[:, i] == 1) & (ss[:, j] == 1)].sum() + 1e-8
            p10 = pp[(ss[:, i] == 1) & (ss[:, j] == 0)].sum() + 1e-8
            p01 = pp[(ss[:, i] == 0) & (ss[:, j] == 1)].sum() + 1e-8
            p00 = pp[(ss[:, i] == 0) & (ss[:, j] == 0)].sum() + 1e-8

            mi = p11 * (p11.log() - pi1.log() - pj1.log())
            mi += p10 * (p10.log() - pi1.log() - pj0.log())
            mi += p01 * (p01.log() - pi0.log() - pj1.log())
            mi += p00 * (p00.log() - pi0.log() - pj0.log())
            mis.append(mi.item())

    mis = np.array(mis).reshape(n_dims, n_dims)
    if plot:
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(mis)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        ax.set_title("pairwise MI between dimensions")
        plt.show()

    return mis


def pairwise_prob_differences(n_dims, state_space, prob_per_state1, prob_per_state_2):
    ss = state_space
    pp, qq = prob_per_state1, prob_per_state_2
    pairwise_errors = []
    for i in range(n_dims):
        for j in range(i, n_dims):
            p11 = pp[(ss[:, i] == 1) & (ss[:, j] == 1)].sum().item()
            p10 = pp[(ss[:, i] == 1) & (ss[:, j] == 0)].sum().item()
            p01 = pp[(ss[:, i] == 0) & (ss[:, j] == 1)].sum().item()
            p00 = pp[(ss[:, i] == 0) & (ss[:, j] == 0)].sum().item()

            q11 = qq[(ss[:, i] == 1) & (ss[:, j] == 1)].sum().item()
            q10 = qq[(ss[:, i] == 1) & (ss[:, j] == 0)].sum().item()
            q01 = qq[(ss[:, i] == 0) & (ss[:, j] == 1)].sum().item()
            q00 = qq[(ss[:, i] == 0) & (ss[:, j] == 0)].sum().item()

            error = np.abs(p11 - q11) + np.abs(p10 - q10) + np.abs(p01 - q01) + np.abs(p00 - q00)
            pairwise_errors.append(error)

    errors = np.mean(pairwise_errors)
    return errors


def batched_pairwise_prob_differences(n_dims, state_space, prob_per_state1, prob_per_state_2):
    ss = state_space
    pp, qq = prob_per_state1, prob_per_state_2.T  # qq has shape (len(state_space), n_chains)
    pairwise_errors = []
    for i in range(n_dims):
        for j in range(i, n_dims):
            p11 = pp[(ss[:, i] == 1) & (ss[:, j] == 1)].sum().item()
            p10 = pp[(ss[:, i] == 1) & (ss[:, j] == 0)].sum().item()
            p01 = pp[(ss[:, i] == 0) & (ss[:, j] == 1)].sum().item()
            p00 = pp[(ss[:, i] == 0) & (ss[:, j] == 0)].sum().item()

            q11 = qq[(ss[:, i] == 1) & (ss[:, j] == 1)].sum(0)  # (n_chains,)
            q10 = qq[(ss[:, i] == 1) & (ss[:, j] == 0)].sum(0)  # (n_chains,)
            q01 = qq[(ss[:, i] == 0) & (ss[:, j] == 1)].sum(0)  # (n_chains,)
            q00 = qq[(ss[:, i] == 0) & (ss[:, j] == 0)].sum(0)  # (n_chains,)

            error = torch.abs(p11 - q11) + torch.abs(p10 - q10) + torch.abs(p01 - q01) + torch.abs(
                p00 - q00)  # (n_chains,)
            pairwise_errors.append(error)

    errors = torch.mean(torch.stack(pairwise_errors, dim=0), dim=0)  # (n_chains,)
    return errors


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    """ESS code as used in GWG repo, but with the addition of
     filter_beyond_positive_pairs=True, which appears to be best-practice.
     """
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    with tf.device('/cpu:0'):
        cv = tfp.mcmc.effective_sample_size(c, filter_beyond_positive_pairs=True).numpy()
    cv[np.isnan(cv)] = 1.  # keeping this because gwg repo uses it, but it looks dodgy
    return cv


def plot_ess_boxplots(args, ess, method_names, save_dir, times):
    # plot ESS boxplots
    short_method_names = get_short_method_names(method_names)
    if not args.no_ess:
        fig, ax = plt.subplots(1, 1)
        ax.boxplot([ess[temp] for temp in method_names], labels=short_method_names, showfliers=False)
        ax.set_xlabel("Method")
        ax.set_ylabel("ESS")
        ax.set_title("ESS")
        save_fig(fig, save_dir, "ess")

        fig, ax = plt.subplots(1, 1)
        ax.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in method_names],
                    labels=short_method_names, showfliers=False)
        ax.set_xlabel("Method")
        ax.set_ylabel("ESS per second")
        ax.set_title("ESS per second")
        save_fig(fig, save_dir, "ess_per_sec")


def get_short_method_names(method_names):
    short_method_names = []
    for name in method_names:
        if name == "GWG-ordinal":
            short_method_names.append("GWG-ord")
        elif name == "MH-uniform":
            short_method_names.append("Uni")
        else:
            short_method_names.append(name)
    return short_method_names


def plot_acc_rates(accept_rates, method_names, n_iters, save_dir):
    fig, ax = plt.subplots(1, 1)
    for method in method_names:
        x_axis = np.linspace(0, n_iters[method], len(accept_rates[method]))
        ax.plot(x_axis, np.array(accept_rates[method]), label=method)
    ax.set_xlabel("iteration")
    ax.set_ylabel("accept rate")
    ax.set_title("Acceptance rates of MH samplers")
    fig.legend()
    save_fig(fig, save_dir, "accept_rates")


def plot_hops(method_names, hops, save_dir):
    # Plot hops
    fig, ax = plt.subplots(1, 1)
    for method in method_names:
        ax.plot(hops[method], label="{}".format(method))
    fig.legend()
    save_fig(fig, save_dir, "hops")


def sampling_time_barchart(method_names, per_step_times, save_dir):
    # bar chart of average time per sampling step for each method
    fig, ax = plt.subplots(1, 1)
    x = method_names
    average_times = list(per_step_times.values())
    x_pos = [i for i, _ in enumerate(x)]
    ax.bar(x_pos, average_times, color='green')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Average time per sampling step")
    fig.legend()
    fig.savefig(os.path.join(save_dir, "times.pdf"))
    fig.savefig(os.path.join(save_dir, "times.png"))
    plt.close(fig)


def plot_marginal_error(method_names, marginal_errors, save_dir, n_iters=None, times=None):

    markers = get_list_of_markers(len(method_names))

    for i in range(2):
        fig, ax = plt.subplots(1, 1)
        all_means = []
        for method, marker in zip(method_names, markers):
            means, stds = zip(*marginal_errors[method])
            all_means.append(means)
            mean_err = np.array(means)  # (n, d-2)
            stds = np.array(stds)
            if n_iters is not None:
                x_axis = np.linspace(0, n_iters[method], len(mean_err))
            else:
                assert times is not None, "must provide either 'n_iters' or 'times' when calling this method"
                x_axis = np.linspace(0, times[method][-1] / 60, len(mean_err))

            ax.plot(x_axis, mean_err, label=method, marker=marker)
            ax.fill_between(x_axis, mean_err - stds, mean_err + stds, alpha=0.2)

        all_means = np.array(all_means)
        lower, upper = all_means.min() / 1.5, 3e-2
        ax.set_ylim([lower, upper])
        if i == 1: ax.set_yscale('log')
        ax.set_xlabel("Iteration") if n_iters is not None else ax.set_xlabel("Minutes")
        ylab = "absolute error" if i == 0 else f"absolute error (log)"
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=12)
        ax.set_title(f"Marginal estimation error")

        name = "marginal error"
        if i == 1: name += "_log"
        name += " - iters" if n_iters is not None else " - wallclock"
        save_fig(fig, save_dir, name)



def plot_mi_error(method_names, mi_errors, save_dir=None, n_iters=None, times=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))

    for method in method_names:
        mi_error, mi_weighted_error = list(zip(*mi_errors[method]))
        if n_iters is not None:
            x_axis = np.linspace(0, n_iters[method], len(mi_error))
        else:
            assert times is not None, "must specify either n_iters or times"
            x_axis = np.linspace(0, times[method][-1] / 60, len(mi_error))

        ax.plot(x_axis, mi_error, label=method)
        ax1.plot(x_axis, mi_error, label=method)
        ax1.set_yscale('log')
        ax2.plot(x_axis, mi_weighted_error, label=method)
        ax3.plot(x_axis, mi_weighted_error, label=method)
        ax3.set_yscale('log')

    for i, a in enumerate([ax, ax1, ax2, ax3]):
        a.set_xlabel("Iteration") if n_iters is not None else a.set_xlabel("Minutes")
        if i % 2 == 0:
            a.set_ylabel("absolute error")
        else:
            a.set_ylabel("(log) absolute error")
        a.legend(loc="best", fontsize=12)

    if save_dir:
        fig.suptitle(f"Average absolute error for all pairwise MIs")
        fig.savefig(os.path.join(save_dir, f"mi_error {'(iters)' if n_iters is not None else '(wallclock)'}.png"))
        plt.close(fig)
        fig1.suptitle(f"Average absolute error for all pairwise (log) MIs")
        fig1.savefig(os.path.join(save_dir, f"log_mi_error {'(iters)' if n_iters is not None else '(wallclock)'}.png"))
        plt.close(fig1)

        fig2.suptitle(f"Weighted average absolute error for all pairwise MIs")
        fig2.savefig(
            os.path.join(save_dir, f"weighted_mi_error {'(iters)' if n_iters is not None else '(wallclock)'}.png"))
        plt.close(fig2)
        fig3.suptitle(f"Weighted average absolute error for all pairwise (log) MIs")
        fig3.savefig(
            os.path.join(save_dir, f"log_weighted_mi_error {'(iters)' if n_iters is not None else '(wallclock)'}.png"))
        plt.close(fig3)
    else:
        return [(fig, ax, x_axis), (fig1, ax1, x_axis), (fig2, ax2, x_axis), (fig3, ax3, x_axis)]


def plot_pairwise_errors(method_names, errors, save_dir=None, n_iters=None, times=None):
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)

    markers = get_list_of_markers(len(method_names))
    all_means = []
    for method, marker in zip(method_names, markers):
        mean_err, stds = zip(*errors[method])
        all_means.append(mean_err)
        mean_err, stds = np.array(mean_err), np.array(stds)
        if n_iters is not None:
            x_axis = np.linspace(0, n_iters[method], len(mean_err))
        else:
            assert times is not None, "must specify either n_iters or times"
            x_axis = np.linspace(0, times[method][-1] / 60, len(mean_err))

        ax1.plot(x_axis, mean_err, label=method, marker=marker)
        ax1.fill_between(x_axis, mean_err - stds, mean_err + stds, alpha=0.2)
        ax2.plot(x_axis, mean_err, label=method, marker=marker)
        ax2.fill_between(x_axis, mean_err - stds, mean_err + stds, alpha=0.2)
        ax2.set_yscale('log')

    all_means = np.array(all_means)
    for i, a in enumerate([ax1, ax2]):
        a.set_ylim([all_means.min() / 1.5, 3e-1])
        a.set_xlabel("Iteration") if n_iters is not None else a.set_xlabel("Minutes")
        if i == 0:
            a.set_ylabel("absolute error")
        else:
            a.set_ylabel("(log) absolute error")
        a.legend(loc="best", fontsize=12)

    ax1.set_title(f"Pairwise estimation error")
    ax2.set_title(f"Pairwise estimation error")
    if save_dir:
        save_fig(fig1, save_dir, f"pairwise_error {'(iters)' if n_iters is not None else '(wallclock)'}")
        save_fig(fig2, save_dir, f"log_pairwise_error {'(iters)' if n_iters is not None else '(wallclock)'}")
    else:
        return [(fig1, ax1, x_axis), (fig2, ax2, x_axis)]


def make_discrete_neighbourhood_sampling_fns(
        n_dims,
        variable_type,
        default_radius=1,
        n_categorical_states=None,
        ordinal_state_space=None
):
    """Returns functions that, given a batch of vectors x, generate points from the neighbourhoods of each x.

    sampler_neighbour(x) stochastically generates a random point in the Hamming ball of radius r
    onehop_neighbours(x) generates all points in the Hamming ball of radius 1

    """

    if variable_type == "binary":

        def sample_neighbour(x, weights=None, r=None):
            if r is None: r = default_radius
            if weights is None: weights = torch.zeros_like(x)
            for _ in range(int(r)):
                cd_forward = dists.OneHotCategorical(logits=weights * -(2. * x - 1))
                changes = cd_forward.sample()
                x = (1. - x) * changes + x * (1. - changes)
            return x

        def onehop_neigbours(x):
            one_hop_neighbours = torch.eye(x.size(1), device=x.device).unsqueeze(1).tile((1, x.size(0), 1))
            return (x - one_hop_neighbours).abs()

    elif variable_type == "categorical":

        def sample_neighbour(x, w=None, r=None):
            x = x.view(-1, n_dims, n_categorical_states)
            if r is None: r = default_radius
            if w is None: w = torch.zeros_like(x)
            for _ in range(int(r)):
                forward_logits = w - 1e9 * x  # don't sample current state
                flat_forward_logits = forward_logits.view(x.size(0), -1)
                cd_forward = dists.OneHotCategorical(logits=flat_forward_logits, validate_args=False)
                changes = cd_forward.sample()
                changes_r = changes.view(x.size())  # reshape to (bs, dim, nout)
                changed_ind = changes_r.sum(-1)
                x = x.clone() * (1. - changed_ind[:, :, None]) + changes_r
            return x.view(-1, n_dims * n_categorical_states)

        def onehop_neigbours(x):
            x_tiled = x.unsqueeze(0).tile((n_dims * n_categorical_states, 1, 1))  # (dk, n, dk)
            for i in range(n_dims - 1):
                sl = slice(i * n_categorical_states, (i + 1) * n_categorical_states)
                x_tiled[sl, :, sl] = torch.eye(n_categorical_states, device=x.device).unsqueeze(1).tile((1, x.size(0), 1))
            return x_tiled  # (dk, n, dk)

    elif variable_type == "ordinal":

        length_scale = (ordinal_state_space[1] - ordinal_state_space[0]).item()

        def sample_neighbour(x, w=None, r=None):
            if r is None: r = default_radius
            return ProductOfLocalUniformOrdinals(x, state_space=ordinal_state_space, radius=r).sample()

        def onehop_neigbours(x):
            n, d = x.shape
            x_tiled = torch.tile(x.unsqueeze(-1), (1, 1, 2 * d))  # (n, d, 2d)
            right = torch.ones(d, device=x.device) * length_scale
            left = -right
            x_tiled[:, np.arange(d), np.arange(d)] += right
            x_tiled[:, np.arange(d), np.arange(d, 2 * d)] += left
            x_tiled = x_tiled.clamp(ordinal_state_space.min(), ordinal_state_space.max())
            return x_tiled.permute(2, 0, 1)  # (2d, n, d)
    else:
        raise NotImplementedError

    return sample_neighbour, onehop_neigbours
