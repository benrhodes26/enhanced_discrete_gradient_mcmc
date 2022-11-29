#! /usr/bin/env python
import os
import sys
import subprocess

# dryrun = True
dryrun = False
dataset = "usps"
run_id = 3  # changes the random seed and saves results to a different directory

shared_args = [
    "--model=ising_neural_frozen",
    f"--model_load_path=results/ising_neural_{dataset}/gwg/50/ising_neural_model_2022-07-10_10-41-16",
    f"--data_file=results/ising_neural_{dataset}/gwg/50/buffer.npz",
    f"--seed={run_id}23456"
]
if dryrun:
    shared_args += ["--n_iters=10", f"--save_dir=results/ising_neural_frozen_{dataset}/dryrun/"]
    n_kvals = 1
else:
    shared_args += ["--n_iters=2100", f"--save_dir=results/ising_neural_frozen_{dataset}_run{run_id}/"]
    n_kvals = 4

NCG_args = ["--sampler=NCG", "--epsilon=0.2"]
K_values = [5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *NCG_args, f"--sampling_steps_per_iter={k}"],
                    shell=False)

AVG_args = ["--sampler=AVG", "--epsilon=0.08"]
K_values = [5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *AVG_args, f"--sampling_steps_per_iter={k}"],
                    shell=False)

PAVG_args = ["--sampler=PAVG", "--epsilon=0.08"]
K_values = [5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *PAVG_args, f"--sampling_steps_per_iter={k}"],
                    shell=False)

GWG_args = ["--sampler=GWG"]
K_values = [5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *GWG_args, f"--sampling_steps_per_iter={k}"],
                    shell=False)

# larger K vals to balance fact that Gibbs is faster
Gibbs_args = ["--sampler=Gibbs"]
K_values = [8, 16, 24, 32][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *Gibbs_args, f"--sampling_steps_per_iter={k}"],
                    shell=False)




# # Fit Boltzmann machine with block-gibbs sampler (so we can use it as a preconditioner for PAVG)
# BG_args = [
#     "--model=boltzmann",
#     f"--data_file=usps",
#     "--n_iters=2000"
#     f"--save_dir=results/boltzmann_{dataset}/",
#     "--sampler=Block-Gibbs",
#     "--sampling_steps_per_iter=1",
#     "--l1=0.0"  # block-gibbs seems stable without regularisation :)
# ]
# subprocess.call([sys.executable, 'train_ising.py', *BG_args], shell=False)
#
# # Note: the above Block-Gibbs learning of the preconditioning matrix takes 18.68 seconds, so we reduce the number
# # of iterations used here so that PAVG doesn't get an unfair advantage
# n_iters = [1666, 1872, 1938, 1972]
# precon_path = os.path.join(os.getcwd(), "results", f"boltzmann_usps", "block-gibbs", "1", "J.npz")
# PAVG_args = ["--sampler=PAVG v2", "--epsilon=0.04", f"--precon_load_path={precon_path}"]
# K_values = [5, 10, 15, 20][:n_kvals]
# for k, n_iter in zip(K_values, n_iters):
#     subprocess.call([sys.executable, 'train_ising.py', *shared_args, *PAVG_args,
#                      f"--sampling_steps_per_iter={k}", f"--n_iters={n_iter}"], shell=False)
