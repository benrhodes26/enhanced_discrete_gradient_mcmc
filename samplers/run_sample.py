import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from distributions.discrete import ProductOfLocalUniformOrdinals
from samplers import auxiliary_samplers, regular_samplers
from utils.mcmc_utils import get_ess
from utils.utils import set_default_rcparams

set_default_rcparams()
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def get_sampler(args, method_dict, save_dir):
    method = method_dict["name"]

    var_type = args.var_type
    assert var_type in ["binary", "ordinal", "categorical"]

    state_space, n_categorical_states = None, None
    if var_type == "ordinal":
        ss_min = args.state_space_min if args.state_space_min else -args.state_space_max
        state_space = torch.linspace(ss_min, args.state_space_max, args.state_space_size, device=device)
    if var_type == "categorical":
        assert "n_categorical_states" in args, "must specify n_categorical_states in args"
        n_categorical_states = args.n_categorical_states

    if method.lower().startswith("avg"):
        print("Using AVG sampler")
        sampler = auxiliary_samplers.AVGSampler(
            n_dims=args.data_dim,
            epsilon=method_dict["epsilon"],
            variable_type=var_type,
            n_categorical_states=n_categorical_states,
            state_space=state_space,
            save_dir=save_dir
        )

    elif method.lower().startswith("pavg"):
        print("Using PAVG sampler")
        sampler = auxiliary_samplers.PAVGSampler(
            n_dims=args.data_dim,
            epsilon=method_dict["epsilon"],
            adaptive_update_freq=method_dict.get("adaptive_update_freq", 100),
            init_adapt_stepsize=method_dict.get("init_adapt_stepsize", 0.25),
            adapt_stepsize_decay=method_dict.get("adaptation_decay_factor", 0.99),
            init_precon_matrix=method_dict.get("init_precon_mat", None),
            allow_adaptation_of_precon_matrix=method_dict.get("allow_adaptation_of_precon_matrix", True),
            num_iteration_before_fitting_M=method_dict.get("num_iteration_before_fitting_M", 1000),
            postadapt_epsilon=method_dict.get("postadapt_epsilon", None),
            variable_type=var_type,
            n_categorical_states=n_categorical_states,
            state_space=state_space,
            save_dir=save_dir)

    elif method.lower().startswith("block-gibbs"):
        ### Block-Gibbs auxiliary-variable Sampler---Martens & Sutskever (2010). Also see Zhang et al. (2012).
        sampler = auxiliary_samplers.BGAVSampler(
            n_dims=args.data_dim,
            model_name=args.model,
            variable_type=var_type,
            n_categorical_states=n_categorical_states,
            state_space=state_space)

    elif method.lower().startswith("ncg"):
        sampler = regular_samplers.NCGSampler(
            args.data_dim,
            epsilon=method_dict["epsilon"],
            n_forward_copies=method_dict.get("n_forward_copies", 1),
            state_space=state_space,
            var_type=var_type,
            use_simple_mod=method_dict.get("use_simple_mod", False),
            reg_lambda=method_dict.get("reg_lambda", 0.0)
        )


    elif method.lower().startswith("gwg") | method.lower().startswith("lb"):
        if var_type == "binary":
            sampler = regular_samplers.BinaryGWGSampler(
                args.data_dim,
                approx=method_dict.get("approx", True),
                temp=2.
            )
        elif var_type == "ordinal":
            sampler = regular_samplers.OrdinalGWGSampler(
                args.data_dim,
                state_space,
                use_gradient=method_dict.get("approx", True),
                radius=method_dict.get("radius", 1)
            )
        else:
            sampler = regular_samplers.CategoricalGWGSampler(
                args.data_dim,
                approx=method_dict.get("approx", True),
                temp=2.
            )

    elif method.lower().startswith("gibbs"):
        if var_type == "binary":
            sampler = regular_samplers.BinaryGibbsSampler(
                args.data_dim,
                rand=method_dict.get("random_scan", True)
            )
        elif var_type == "ordinal":
            sampler = regular_samplers.OrdinalGibbsSampler(
                args.data_dim,
                state_space,
                rand=method_dict.get("random_scan", True))

        else:
            sampler = regular_samplers.CategoricalGibbsSampler(
                args.data_dim,
                rand=method_dict.get("random_scan", True)
            )

    elif "uniform" in method.lower():
        def prop(x, model):
            return ProductOfLocalUniformOrdinals(x, state_space=state_space, radius=method_dict["radius"])

        sampler = regular_samplers.MHSampler(prop, length_scale=(state_space[1] - state_space[0]).item())

    else:
        raise ValueError(f"Invalid sampler: {method}")

    return sampler


def run_sampling_procedure(args, methods, target_dist, chain_init, compute_metrics, plot_and_save):

    np.set_printoptions(precision=3)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subdir = os.path.join(args.save_dir)
    os.makedirs(subdir, exist_ok=True)
    savename = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    save_dir = os.path.join(subdir, savename)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    print(f"Saving results to {save_dir}")

    ess_comparison_point = chain_init[np.random.randint(0, len(chain_init))][None]
    num_ess_dims = args.num_ess_dims if "num_ess_dims" in args else args.data_dim

    # dictionaries whose keys will be the name of a method e.g. 'gibbs'
    prop_hops_dict = {}
    hops_dict = {}
    accept_rates_dict = {}
    ess_dict = {}
    n_iters_dict = {}
    times_dict = {}
    chains_dict = {}
    metrics_dict = {}

    for method_dict in methods:

        method = method_dict["name"]
        method_save_dir = os.path.join(save_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)

        sampler = get_sampler(args, method_dict, method_save_dir)

        n_iters_dict[method] = 0
        times_dict[method] = []
        prop_hops_dict[method] = []
        hops_dict[method] = []
        metrics_dict[method] = {}

        trace_plot_chains = []
        ess_statistic = []
        sliding_window_empirical_dist = []
        cur_time = now = 0.
        last_sample_save_time = 0.0
        last_metric_save_time = 0.0
        burnin_finished = False
        x = chain_init.clone()

        while now < args.max_runtime:
            now = (cur_time / 60) if args.wallclock_mode else n_iters_dict[method]
            n_iters_dict[method] += 1

            # TAKE MCMC STEP
            st = time.time()
            x = sampler.step(x.detach(), target_dist).detach()
            time_per_step = time.time() - st
            cur_time += time_per_step

            if now - last_sample_save_time > args.save_freq:
                last_sample_save_time = now
                times_dict[method].append(cur_time)
                cur_hops = sampler.acc_hops[-1]
                hops_dict[method].append(cur_hops)

                if burnin_finished and len(trace_plot_chains) < 5000:
                    trace_plot_chains.append(x.cpu().numpy()[:10, 0][None])  # track first dim of 10 chains

                dims_to_sum = (1, 2) if len(x.shape) == 3 else 1
                ess_stat = (x[:, :num_ess_dims] - ess_comparison_point[:, :num_ess_dims]).abs().float().sum(dims_to_sum)  # (n_chains, )
                ess_statistic.append(ess_stat.cpu().numpy()[None])

                # ACCUMULATE SAMPLES
                if now > args.burn_in * args.max_runtime and not burnin_finished:
                    burnin_finished = True
                    sliding_window_empirical_dist = []  # reset our buffer of samples
                sliding_window_empirical_dist.append(x.clone().detach().cpu())
                if len(sliding_window_empirical_dist) > args.sliding_window_size:
                    del sliding_window_empirical_dist[0]  # limit size of buffer

                # COMPUTE METRICS
                if now - last_metric_save_time > args.metric_tracking_freq:
                    last_metric_save_time = now
                    print("#" * 20)
                    print_str = f"method {method}, " + \
                                f"itr = {n_iters_dict[method]}," + \
                                f" av acc-rate {np.mean(sampler.acc_rates)}" + \
                                f" prop-hop = {sampler.proposed_hops[-1]:.4f}," + \
                                f" acc-hop = {cur_hops:.4f}," + \
                                f" av acc-hop = {np.array(hops_dict[method]).mean():.4f}," + \
                                f" recent-av acc-hop = {np.array(hops_dict[method])[-100:].mean():.4f}," + \
                                f"time = {cur_time / n_iters_dict[method]:.4f}"
                    if hasattr(sampler, "epsilon"):
                        print_str += f"Epsilon: {sampler.epsilon:.4f}"
                    print(print_str)

                    if now > args.burn_in * args.max_runtime:
                        metric_samples = torch.stack(sliding_window_empirical_dist, dim=1)  # (n_chains, n_steps, n_dims)
                    else:
                        metric_samples = x.unsqueeze(1).detach().cpu()
                    compute_metrics(args, method, metric_samples, metrics_dict, sampler)

        accept_rates_dict[method] = list(sampler.acc_rates)
        prop_hops_dict[method] = list(sampler.proposed_hops)
        trace_plot_chains = np.vstack(trace_plot_chains)  # (chain_length, n_chains)
        ess_statistic = np.vstack(ess_statistic)  # (chain_length, chain_length)
        chains_dict[method] = trace_plot_chains

        # TRACE PLOTS
        plt.clf()
        fig, axs = plt.subplots(int(np.ceil(trace_plot_chains.shape[1] / 2)), 2, figsize=(8, 12), sharex=True, sharey=True)
        axs = axs.ravel()
        for i in range(trace_plot_chains.shape[1]):
            axs[i].plot(trace_plot_chains[:, i], label=f"chain {i}")
        fig.savefig(os.path.join(save_dir, "trace_{}.png".format(method)))
        plt.close(fig)

        # COMPUTE & PRINT ESS
        if not args.no_ess:
            ess_dict[method] = get_ess(ess_statistic, args.burn_in)
            print(f"ESS = {ess_dict[method].mean()} +/- {ess_dict[method].std()} (mean +/- std over {ess_statistic.shape[1]} chains)")

    # Plot comparisons across methods, and save all results
    plot_and_save(args=args,
                  method_dicts=methods,
                  chains=chains_dict,
                  hops=hops_dict,
                  accept_rates=accept_rates_dict,
                  metrics=metrics_dict,
                  ess=ess_dict,
                  n_iters=n_iters_dict,
                  times=times_dict,
                  save_dir=save_dir)
