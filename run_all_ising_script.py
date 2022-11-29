#! /usr/bin/env python

import sys
import subprocess

dryrun = True
# dryrun = False
sigma = 0.2
run_id = 1  # different ids use different random seeds and will save to different directories

shared_args = [
    "--model=ising_lattice_2d",
    f"--sigma={sigma}",
    f"--data_file=data/ising_lattice_sigma{sigma}/data.pkl",
    f"--seed={run_id}23456"
]
if dryrun:
    shared_args += ["--n_iters=10", f"--save_dir=results/ising_lattice_sigma{sigma}/dryrun/"]
    n_kvals = 1
else:
    shared_args += ["--n_iters=2000", f"--save_dir=results/ising_lattice_sigma{sigma}_run{run_id}/"]
    n_kvals = 5

# NCG is approx 1.2x faster than AVG
NCG_args = ["--sampler=NCG", "--epsilon=0.5"]
K_values = [1, 6, 12, 18, 24][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *NCG_args, f"--sampling_steps_per_iter={k}"], shell=False)

AVG_args = ["--sampler=AVG", "--epsilon=0.2"]
K_values = [1, 5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *AVG_args, f"--sampling_steps_per_iter={k}"], shell=False)

# BG is approx 3x faster than AVG
BG_args = ["--sampler=Block-Gibbs (PAVG model-specific)"]
K_values = [3, 15, 30, 45, 60][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *BG_args, f"--sampling_steps_per_iter={k}"], shell=False)

# PAVG is approx same speed as AVG
PAVG_args = ["--sampler=PAVG model-agnostic", "--epsilon=0.2"]
K_values = [1, 5, 10, 15, 20][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *PAVG_args, f"--sampling_steps_per_iter={k}"], shell=False)

# NCG is approx 1.2x faster than AVG
GWG_args = ["--sampler=GWG"]
K_values = [1, 6, 12, 18, 24][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *GWG_args, f"--sampling_steps_per_iter={k}"], shell=False)

# Gibbs is approx 2x faster than AVG
Gibbs_args = ["--sampler=Gibbs"]
K_values = [2, 10, 20, 30, 40][:n_kvals]
for k in K_values:
    subprocess.call([sys.executable, 'train_ising.py', *shared_args, *Gibbs_args, f"--sampling_steps_per_iter={k}"], shell=False)
