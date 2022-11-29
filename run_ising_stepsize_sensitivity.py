#! /usr/bin/env python

import sys
import subprocess

# dryrun = True
dryrun = False
sigma = 0.2
run_id = 0  # different ids use different random seeds and will save to different directories

shared_args = [
    "--model=ising_lattice_2d",
    f"--sigma={sigma}",
    f"--data_file=data/ising_lattice_sigma{sigma}/data.pkl",
    f"--seed={run_id}23456",
    f"--sampling_steps_per_iter=10"
]
if dryrun:
    shared_args += ["--n_iters=10", f"--save_dir=results/ising_lattice_sigma{sigma}/dryrun/"]
    n_ep_vals = 1
else:
    shared_args += ["--n_iters=2000"]
    n_ep_vals = 11

NCG_args = ["--sampler=NCG"]
eps_values = [(0.5 * (3/2)**i) for i in range(-5, 6)][:n_ep_vals]
for i, ep in enumerate(eps_values):
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *NCG_args,
                     f"--epsilon={ep}", f"--save_dir=results/ising_lattice_sigma{sigma}_stepsize_sensitivity/ep{i}/"], shell=False)

AVG_args = ["--sampler=AVG"]
eps_values = [(0.2 * (3/2)**i) for i in range(-5, 6)][:n_ep_vals]
for i, ep in enumerate(eps_values):
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *AVG_args,
                     f"--epsilon={ep}", f"--save_dir=results/ising_lattice_sigma{sigma}_stepsize_sensitivity/ep{i}/"], shell=False)

PAVG_args = ["--sampler=PAVG model-agnostic"]
eps_values = [(0.2 * (3/2)**i) for i in range(-5, 6)][:n_ep_vals]
for i, ep in enumerate(eps_values):
    print(ep)
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *PAVG_args,
                     f"--epsilon={ep}", f"--save_dir=results/ising_lattice_sigma{sigma}_stepsize_sensitivity/ep{i}/"], shell=False)
