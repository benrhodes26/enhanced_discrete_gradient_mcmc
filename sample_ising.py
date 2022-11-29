import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torchvision

from distributions.discrete import LatticeIsingModel
from samplers.run_sample import run_sampling_procedure
from utils.mcmc_utils import binary_marginals_and_pairwise_mis, plot_acc_rates, plot_ess_boxplots, plot_marginal_error,\
    plot_pairwise_errors, batched_empirical_prob_over_statespace, batched_pairwise_prob_differences
from utils.utils import set_default_rcparams, numpify

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_default_rcparams()


def plot_and_save(args,
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
    for method in method_names: plt.plot(hops[method], label="{}".format(method))
    plt.legend()
    plt.savefig(os.path.join(save_dir, "hops.pdf"))
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


def define_single_target_dist(args, third_order_strength):

    # always use same target distributions (args.seed only controls randomness in the samplers)
    torch.manual_seed(123438)
    np.random.seed(123438)
    model = LatticeIsingModel(args.D,
                              init_sigma=args.sigma,
                              init_bias=args.bias,
                              third_order_interaction_strength=third_order_strength
                              )
    model.to(device)
    im = plt.imshow(model.G.detach().cpu().numpy())
    plt.colorbar(im)
    plt.savefig(os.path.join(args.save_dir, "ising_matrix.png"))

    dim_sqrt = int(args.D ** 0.5)
    if dim_sqrt ** 2 == args.D:
        plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, dim_sqrt, dim_sqrt),
                                                         p, normalize=False, nrow=int(x.size(0) ** .5))
    else:
        plot = None
    init_samples = model.init_sample(args.n_samples).to(device)

    state_space, ss_probs, true_marginals, true_mis, true_samples = None, None, None, None, None
    if args.D <= 20:
        state_space, ss_probs, true_marginals, true_mis, true_samples = \
            binary_marginals_and_pairwise_mis(args.D, model, n_model_samples=args.n_model_samples)

    return model, plot, init_samples, state_space, ss_probs, true_marginals, true_mis, true_samples


def define_target_dist(args, seed=123438, third_order_strength=None):

    # always use same target distributions (args.seed only controls randomness in the samplers)
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_samples = []
    all_probs = []
    all_true_marginals = []
    state_space = None
    for i in range(args.num_target_copies):
        _, plot, init_samples, state_space, ss_probs, true_marginals, _, true_samples =\
            define_single_target_dist(args, third_order_strength)
        all_samples.append(init_samples)
        all_true_marginals += true_marginals
        all_probs.append(ss_probs)

    all_samples = torch.cat(all_samples, dim=-1)
    all_true_marginals = torch.as_tensor(all_true_marginals, device='cpu')
    all_probs = torch.stack(all_probs, dim=0)

    model = LatticeIsingModel(args.D,
                              init_sigma=args.sigma,
                              init_bias=args.bias,
                              num_repeats=args.num_target_copies,
                              third_order_interaction_strength=third_order_strength,
                              )
    model = model.to(device=device)

    return model, all_samples, state_space, all_probs, all_true_marginals, plot


def main(args):

    target_dist, chain_init, state_space, true_probs, true_marginals, plot_ising_matrix_fn = define_target_dist(args)

    def metric_fn(*x, ss=state_space, tp=true_probs, tm=true_marginals):
        return compute_metrics(*x, full_state_space=ss, true_probs=tp, true_marginals=tm)

    if args.num_target_copies == 1:
        methods = [
            {'name': 'Gibbs', 'random_scan': True},
            {'name': 'GWG'},
        ]
    else:
        raise ValueError

    run_sampling_procedure(args, methods, target_dist, chain_init, metric_fn, plot_and_save)
    

def parse_args():

    N, D = 20, 16
    num_target_copies = 1
    block_diag = 0
    sigma, neg, abs = 0.2, 0, 0
    savepath = "blockdiag" if block_diag else f"sigma{sigma}_negweights{neg}_abs{abs}"
    # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 1.0, 0.001, 0.05
    wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 10.0, 0.005, 0.5
    # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 0, 5000, 10, 1000
    n_chains = 100
    window_size = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=N)
    parser.add_argument('--D', type=int, default=D)
    parser.add_argument('--var_type', type=str, default="binary")
    parser.add_argument('--num_target_copies', type=int, default=num_target_copies)
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "results", "ising", savepath))
    parser.add_argument('--seed', type=int, default=123439)

    # runtime + metrics
    parser.add_argument('--n_samples', type=int, default=n_chains)
    parser.add_argument('--sliding_window_size', type=int, default=window_size)
    parser.add_argument('--wallclock_mode', type=float, default=wallclock_mode,
                        help="if 1, then --max_runtime --save_freq & --metric_tracking_freq are measured in minutes. "
                             "Otherwise they are measured in iterations.")
    parser.add_argument('--max_runtime', type=float, default=max_runtime, help="How long to run each sampler for")
    parser.add_argument('--save_freq', type=float, default=save_freq, help="how often we save samples")
    parser.add_argument('--metric_tracking_freq', type=float, default=metric_tracking_freq, help="how often we compute metrics")

    # model params
    parser.add_argument('--model', type=str, default="ising")
    parser.add_argument('--use_negative_weights', type=int, default=neg, choices={0, 1})
    parser.add_argument('--sigma', type=float, default=sigma)
    parser.add_argument('--abs_eig', type=float, default=abs, choices={0, 1})
    parser.add_argument('--block_diag', type=int, default=block_diag, choices={0, 1})
    parser.add_argument('--bias', type=float, default=0.)

    parser.add_argument('--debug_sampler', type=int, default=1, choices={0, 1})
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--no_ess', type=int, default=0, choices={0, 1})
    parser.add_argument('--n_model_samples', type=int, default=None,
                        help="Only specify if you want to sample from true model (possible when d is small e.g. <= 20)."
                             "This can be useful for e.g. computing MI estimates with true model samples "
                             "(which sets an optimal performance limit for our samplers)")

    args = parser.parse_args()
    args.data_dim = args.D * args.num_target_copies
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
