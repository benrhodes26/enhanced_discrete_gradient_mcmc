import argparse
import numpy as np
import os
import pickle
import torch

from samplers.run_sample import run_sampling_procedure
from torch.distributions import Bernoulli, Beta, Independent
from utils.mcmc_utils import binary_marginals_and_pairwise_mis, plot_marginal_error, \
    plot_acc_rates, plot_ess_boxplots, plot_pairwise_errors, \
    batched_empirical_prob_over_statespace, batched_pairwise_prob_differences, plot_hops
from utils.utils import set_default_rcparams, numpify

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_default_rcparams()


def plot_and_save_fn(args,
                     method_dicts,
                     chains,
                     hops,
                     accept_rates,
                     metrics,
                     ess,
                     n_iters,
                     times,
                     save_dir):

    # save arguments passed into this function (to make it trivial to recreate figures later)
    to_save = locals()
    with open(os.path.join(save_dir, "results.pkl"), 'wb') as f:
        pickle.dump(to_save, f)

    method_names = [m["name"] for m in method_dicts]
    plot_hops(method_names, hops, save_dir)
    plot_ess_boxplots(args, ess, method_names, save_dir, times)
    plot_acc_rates(accept_rates, method_names, n_iters, save_dir)

    if "marginal_error" in metrics[list(metrics.keys())[0]]:
        marginal_errors = {name: metrics[name]["marginal_error"] for name in method_names}
        plot_marginal_error(method_names, marginal_errors, save_dir, n_iters=n_iters)
        plot_marginal_error(method_names, marginal_errors, save_dir, times=times)

        pairwise_errors = {name: metrics[name]["pairwise_error"] for name in method_names}
        plot_pairwise_errors(method_names, pairwise_errors, save_dir, n_iters=n_iters)
        plot_pairwise_errors(method_names, pairwise_errors, save_dir, times=times)


def compute_metrics(args, method, x_all, metrics_dict, sampler, full_state_space, true_probs, true_marginals):

    if full_state_space is None: return  # can't enumerate space, which we rely on to normalize posterior and compute metrics

    n_chains = len(x_all)
    marginal_errors = numpify(torch.abs(x_all.mean(1) - true_marginals))
    av_marginal_error, std_marginal_error = marginal_errors.mean(), marginal_errors.std() / n_chains**0.5
    metrics_dict[method].setdefault("marginal_error", []).append([av_marginal_error, std_marginal_error])
    print("marginal error (mean/std): ", av_marginal_error, std_marginal_error)

    all_pairwise_errors = []
    for i in range(args.num_target_copies):
        empirical_probs = batched_empirical_prob_over_statespace(full_state_space, x_all[:, :, i * args.D: (i + 1) * args.D])
        perrors = batched_pairwise_prob_differences(args.D, full_state_space, true_probs[i], empirical_probs)
        all_pairwise_errors.append(perrors)

    all_pairwise_errors = torch.stack(all_pairwise_errors, dim=1)  # (n_chains, num_target_copies)
    av_pairwise_error = all_pairwise_errors.mean().item()
    std_pairwise_error = all_pairwise_errors.mean(-1).std().item() / n_chains**0.5

    metrics_dict[method].setdefault("pairwise_error", []).append([av_pairwise_error, std_pairwise_error])
    print(f"pairwise errors (mean/std-across-chains/n_samples_per_chain): "
          f"{av_pairwise_error}, {std_pairwise_error}, {x_all.shape[1]}")


def define_prior_and_posterior(n_chains, y, Z, alpha_pi0=1e-3, beta_pi0=1.0, alpha_sigma=0.1,
                               beta_sigma=0.1, g=None, lamb=1e-3, temp=1.0):
    N, D = Z.shape
    if g is None: g = N

    pi0_prior = Beta(alpha_pi0, beta_pi0)
    pi0 = pi0_prior.sample((n_chains,)).to(device)

    bernoulli_prob = torch.ones(D, device=device).unsqueeze(0) * pi0.unsqueeze(1)
    x_prior = Independent(Bernoulli(probs=bernoulli_prob), 1)
    x_prior_samples = x_prior.sample()

    y_squared = (y ** 2).sum()

    def posterior(x):
        """x is a batch of binary vectors with shape [B, D]"""
        B, D = x.shape
        Dx = x.sum(-1)  # (B,)
        log_gamma_term = torch.lgamma(Dx + alpha_pi0) + torch.lgamma(D - Dx + beta_pi0)

        x_tiled = torch.tile(x.unsqueeze(1), (1, N, 1))  # (B, N, D)
        Zx = x_tiled * Z.to(device=x.device)  # (B, N, D)
        Zx_trans = Zx.swapaxes(1, 2)  # (B, D, N)

        I = torch.eye(D, device=x.device).unsqueeze(0)
        covar = torch.bmm(Zx_trans, Zx)
        reg_covar = ((1 / g) * covar) + (lamb * I)  # (B, D, D)
        gscaled_reg_covar = (((1 + g) / g) * covar) + (lamb * I)  # (B, D, D)

        reg_covar_chol = torch.linalg.cholesky(reg_covar)  # (B, D, D)
        gscaled_reg_covar_chol = torch.linalg.cholesky(gscaled_reg_covar)  # (B, D, D)

        logdet1 = reg_covar_chol[:, range(D), range(D)].log().sum(-1)  # (B,)
        logdet2 = gscaled_reg_covar_chol[:, range(D), range(D)].log().sum(-1)  # (B,)
        logdet_term = logdet1 - logdet2  # (B,)

        Zxt_y = Zx_trans @ y.to(device=x.device)  # (B, D)
        ridge_ols = torch.cholesky_solve(Zxt_y.unsqueeze(-1), gscaled_reg_covar_chol).squeeze(-1)  # (B, D)
        quad_term = (Zxt_y * ridge_ols).sum(-1)  # (B,)
        beta = (2 * beta_sigma) + y_squared.to(device=x.device) - quad_term
        alpha = - alpha_sigma - (N / 2)
        alpha_log_beta = alpha * beta.log()  # (B,)

        return temp * (log_gamma_term + logdet_term + alpha_log_beta)  # (B,)

    return x_prior_samples, posterior


def define_target_dist(args, seed=123438):

    # always use same target distributions (args.seed only controls randomness in the samplers)
    torch.manual_seed(seed)
    np.random.seed(seed)

    relevant_indices = []
    non_relevant_indices = []
    all_prior_samples = []
    posteriors = []
    all_probs = []
    all_true_marginals = []
    all_true_mis = []
    state_space = None
    for i in range(args.num_target_copies):
        n_groups = args.num_relevant_groups
        nd_per_group = args.D // n_groups
        relevant_indices += [(i * args.D) + k * nd_per_group + l for k in range(n_groups) for l in range(nd_per_group)]
        non_relevant_indices += [k for k in range(i * args.D, (i+1) * args.D) if k not in relevant_indices]
        Z = torch.randint(low=0, high=3, size=(args.N, args.D // n_groups), device=device, dtype=torch.float32)
        Z = torch.cat([Z.clone() for _ in range(n_groups)], dim=1)  # (N, D)
        y = Z[:, :nd_per_group].sum(-1)

        prior_samples, posterior = define_prior_and_posterior(n_chains=args.n_samples, y=y, Z=Z, alpha_pi0=1e-3, beta_pi0=10.0)
        all_prior_samples.append(prior_samples)
        posteriors.append(posterior)

        if args.D <= 20:
            state_space, probs, true_marginals, true_mis, _ = binary_marginals_and_pairwise_mis(args.D, posterior)
            all_true_marginals += true_marginals
            all_true_mis.append(true_mis)
            all_probs.append(probs)

    all_probs = torch.stack(all_probs, dim=0)

    n_grouped_dims = args.num_target_copies * args.D
    n_bernoulli_dims = n_grouped_dims * args.irrelevant_mult_factor
    irrelevant_prob = 0.001

    prior_samples = torch.cat(all_prior_samples, dim=-1)
    bernoulli_probs = torch.ones(len(prior_samples), n_bernoulli_dims, device=prior_samples.device) * irrelevant_prob
    prior_samples = torch.cat([prior_samples, Bernoulli(probs=bernoulli_probs).sample()], dim=-1)
    if args.D <= 20:
        all_true_marginals += [irrelevant_prob for _ in range(n_bernoulli_dims)]
        all_true_marginals = torch.as_tensor(all_true_marginals)

    def posterior(x):
        posterior_probs = [logp(x[:, i * args.D: (i + 1) * args.D]) for i, logp in enumerate(posteriors)]
        bernoulli_probs = x[:, n_grouped_dims:] * np.log(irrelevant_prob) + (1 - x[:, n_grouped_dims:]) * np.log(1-irrelevant_prob)
        logps = torch.stack(posterior_probs + [bernoulli_probs.sum(-1)], dim=-1)
        return logps.sum(-1)

    if args.init_type == "prior":
        chain_init = prior_samples
    elif args.init_type == "mode":
        mode = torch.zeros_like(prior_samples)
        for i in range(args.num_target_copies):
            mode[:, i*args.D + 1] = 1
        chain_init = mode
    else:
        raise ValueError

    all_model_samples = []
    for i in range(len(all_probs)):
        model_samples_idxs = torch.distributions.Categorical(probs=all_probs[i]).sample((10000,)).squeeze()
        model_samples = state_space[model_samples_idxs]
        all_model_samples.append(model_samples)
    all_model_samples = torch.cat(all_model_samples, dim=1)
    bernoulli_probs = torch.ones(len(all_model_samples), n_bernoulli_dims, device=prior_samples.device) * irrelevant_prob
    model_samples = torch.cat([all_model_samples, Bernoulli(probs=bernoulli_probs).sample()], dim=-1)
    true_cov_mat = torch.cov(model_samples.T)  # (D, D)
    true_cov_mat[n_grouped_dims:, n_grouped_dims:] += 0.01*torch.eye(n_bernoulli_dims, device=device)  # avoid instability
    del model_samples

    return posterior, chain_init, state_space.cpu(), all_probs.cpu(), all_true_marginals.cpu(), true_cov_mat.cpu()


def main(args):

    target_dist, chain_init, state_space, true_probs, true_marginals, true_cov_mat = define_target_dist(args)

    def metric_fn(*x, ss=state_space, tp=true_probs, tm=true_marginals):
        return compute_metrics(*x, full_state_space=ss, true_probs=tp, true_marginals=tm)

    methods = [
        {'name': 'NCG', 'epsilon': 0.3},
        {'name': 'AVG', 'epsilon': 1000.0},
        {'name': 'PAVG', 'epsilon': 1000.0},
        {'name': 'Gibbs', 'random_scan': True},
        {'name': 'GWG'},
    ]

    run_sampling_procedure(args, methods, target_dist, chain_init, metric_fn, plot_and_save_fn)


def parse_args():

    N, D = 20, 20
    num_groups = 4  # 4 groups of replicated covariates, each of size 5
    irrelevant_mult_factor = 4  # 4*20 = 80 `irrelevant' Bernoulli dims

    wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 10.5, 0.001, 1.0  # measured in minutes
    # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 0, 5000, 1, 500  # measured in iterations

    data_dim = D * (irrelevant_mult_factor + 1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=N)
    parser.add_argument('--D', type=int, default=D,
                        help="dimensionality of a bayesian posterior")
    parser.add_argument('--var_type', type=str, default="binary")
    parser.add_argument('--num_target_copies', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "results", f"sbl/{N}_{data_dim}"))
    parser.add_argument('--seed', type=int, default=654912)

    # sampling steps + diagnostics
    parser.add_argument('--init_type', type=str, default="mode")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--sliding_window_size', type=int, default=10000,
                        help="maximum number of samples stored from a single chain")
    parser.add_argument('--wallclock_mode', type=float, default=wallclock_mode,
                        help="if 1, then --max_runtime --save_freq & --metric_tracking_freq are measured in minutes. "
                             "Otherwise they are measured in iterations.")
    parser.add_argument('--max_runtime', type=float, default=max_runtime, help="how long to run each sampler for")
    parser.add_argument('--save_freq', type=float, default=save_freq, help="how often we save samples")
    parser.add_argument('--metric_tracking_freq', type=float, default=metric_tracking_freq,
                        help="how often we compute metrics (note: metric computation can be time-consuming")

    parser.add_argument('--regression_type', type=int, default=2)
    parser.add_argument('--num_relevant_groups', type=int, default=num_groups)
    parser.add_argument('--irrelevant_mult_factor', type=int, default=irrelevant_mult_factor)

    parser.add_argument('--debug_sampler', type=int, default=0, choices={0, 1})
    parser.add_argument('--burn_in', type=float, default=.2)
    parser.add_argument('--no_ess', type=int, default=0, choices={0, 1})
    parser.add_argument('--num_ess_dims', type=int, default=20)
    args = parser.parse_args()

    assert args.D % args.num_relevant_groups == 0, "we require D to be a multiple of the number of relevant indices"
    args.data_dim = data_dim

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
