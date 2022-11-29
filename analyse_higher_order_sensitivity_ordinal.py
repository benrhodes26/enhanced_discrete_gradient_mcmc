import argparse
import torch
import os

from samplers.run_sample import run_sampling_procedure
from utils.utils import set_default_rcparams
from sample_ising import define_target_dist, plot_and_save, compute_metrics

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_default_rcparams()


"""This script can be use to generate Figure 11 in the appendix of the paper"""

def main(args):
    subdir = os.path.join(args.save_dir, f"dim{args.data_dim}")
    os.makedirs(subdir, exist_ok=True)

    hos = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    avg_eps = [1000.0] * 11
    pavg_eps = [1000.0] * 11

    for ho, avg_ep, pavg_ep in zip(hos, avg_eps, pavg_eps):

        save_dir = os.path.join(subdir, f"higher_order_strength_{ho}")
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir
        args.higher_order_strength = ho

        target_dist, chain_init, state_space, true_probs, true_marginals, plot_ising_matrix_fn = \
            define_target_dist(args, third_order_strength=ho)
        def metric_fn(*x, ss=state_space, tp=true_probs, tm=true_marginals):
            return compute_metrics(*x, full_state_space=ss, true_probs=tp, true_marginals=tm)

        methods = [
            {'name': 'PAVG', 'epsilon': avg_ep, 'postadapt_epsilon': pavg_ep,
             'allow_adaptation_of_precon_matrix': False, 'init_precon_mat': 8 * target_dist.J.clone()},
            {'name': 'AVG', 'epsilon': avg_ep},
        ]

        run_sampling_procedure(args, methods, target_dist, chain_init, metric_fn, plot_and_save)


def parse_args(ising=True):

    if ising:
        D = 16
        sigma = 0.2
        wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 5.5, 0.005, 1.0
        # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 10.5, 0.005, 1.0

        parser = argparse.ArgumentParser()
        parser.add_argument('--D', type=int, default=D)
        parser.add_argument('--var_type', type=str, default="binary")
        parser.add_argument('--num_target_copies', type=int, default=1)
        parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "results", "ising", f"sigma{sigma}"))

        # model params
        parser.add_argument('--model', type=str, default="ising")
        parser.add_argument('--sigma', type=float, default=sigma)
        parser.add_argument('--bias', type=float, default=0.)
        parser.add_argument('--n_model_samples', type=int, default=None,
                            help="Only specify if you want to sample from true model (possible when d is small e.g. <= 20)."
                                 "This can be useful for e.g. computing MI estimates with true model samples "
                                 "(which sets an optimal performance limit for our samplers)")
    else:
        dim = 20
        state_space_size = 50
        # wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 0.5, 0.001, 0.2
        wallclock_mode, max_runtime, save_freq, metric_tracking_freq = 1, 5.5, 0.01, 1.0

        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir', type=str, default=os.path.join(os.getcwd(), "results", "ordinal"))
        parser.add_argument('--var_type', type=str, default="ordinal")
        parser.add_argument('--state_space_size', type=int, default=state_space_size)
        parser.add_argument('--state_space_min', type=float, default=-0.5)
        parser.add_argument('--state_space_max', type=float, default=2)
        parser.add_argument('--point_init', type=int, default=1, choices={0, 1})
        parser.add_argument('--data_dim', type=int, default=dim)


    parser.add_argument('--seed', type=int, default=123437)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--sliding_window_size', type=int, default=10000)
    parser.add_argument('--wallclock_mode', type=float, default=wallclock_mode,
                        help="if 1, then --max_runtime --save_freq & --metric_tracking_freq are measured in minutes. "
                             "Otherwise they are measured in iterations.")
    parser.add_argument('--max_runtime', type=float, default=max_runtime, help="How long to run each sampler for")
    parser.add_argument('--save_freq', type=float, default=save_freq, help="how often we save samples")
    parser.add_argument('--metric_tracking_freq', type=float, default=metric_tracking_freq,
                        help="how often we compute metrics")
    parser.add_argument('--debug_sampler', type=int, default=1, choices={0, 1})
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--no_ess', type=int, default=0, choices={0, 1})

    args = parser.parse_args()
    args.data_dim = args.D * args.num_target_copies if "num_target_copies" in args else 1
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
