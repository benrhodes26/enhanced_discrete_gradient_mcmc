import numpy as np
import os
import pandas as pd
import random
import re
import torch
import time
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
from matplotlib.lines import Line2D


def set_default_rcparams():
    global SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIG_SIZE = 14
    HUGE_SIZE = 16
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=HUGE_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=HUGE_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=18)  # fontsize of the figure title
    # plt.rc('text', usetex=True)  # fontsize of the figure title


def reset_all_fontsizes():
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title


def set_all_fontsizes(size):
    plt.rc('font', size=size)  # controls default text sizes
    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size)  # legend fontsize
    plt.rc('figure', titlesize=size)  # fontsize of the figure title


def save_fig(fig, save_dir, name="", both_formats=True, bbox=True, close=True):
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches='tight' if bbox else None)
    if both_formats:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight' if bbox else None)
    if close:
        plt.close(fig)


def get_list_of_markers(num=None):
    markers = [
        ".", "o", "v", "^", "*", "s", "+", "D", "<", ">", "1", "2", "3", "4", "8", "p", "P",
        "h", "H", "x", "X", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ","
    ]
    if num: markers = markers[:num]
    return markers


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def numpify(x):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    return x


def torchify(x, device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')):
    if isinstance(x, float):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    elif isinstance(x, list):
        return torch.from_numpy(np.array(x)).to(dtype=torch.float32, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=torch.float32, device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32, device=device)
    else:
        raise ValueError


def fn_and_grad(model, s):
    s.requires_grad = True
    with torch.enable_grad():
        f = model(s)
        g = torch.autograd.grad(f.sum(), s)[0]  # Gradient
    s.requires_grad = False
    return f, g


def exp_moving_average(x, n=100, mult=0.99):
    x_recent = x[-n:]  # at most 100 elements
    weights = np.array([mult**(n - i) for i in range(n)])
    return (x_recent * weights[-len(x_recent):]).mean()


def mask_invalid(H, n_dims, n_cat_states):
    """For categorical variables, we mask cross-terms for states belonging to same dim"""
    Ik = torch.eye(n_cat_states, device=H.device)
    mask = 1 - torch.block_diag(*[1 - Ik for _ in range(n_dims)])  # (ndims*nstates, ndims*nstates)
    return H * mask


def xor(a, b):
    return bool(a) != bool(b)


def all_pairwise_combinations(a, b):
    aa = a.repeat((1, b.shape[0])).reshape(-1, a.shape[1])
    bb = b.repeat((a.shape[0], 1))
    return torch.cat([aa, bb], dim=1)


def find_idxs_of_b_in_a(a, b):
    sort_idx = a.argsort()
    return sort_idx[np.searchsorted(a, b, sorter=sort_idx)]


def find_idxs_of_rows_of_B_in_A(A, B, mult=None):
    """Example:
    A = np.array([[4,  2],
                  [9,  3],
                  [8,  5],
                  [3,  3],
                  [5,  6]])
    B = np.array([[4, 2],
                  [3, 3],
                  [5, 6]])
    returns: [0,3,4]
    """
    if mult:
        A, B = (A*mult).astype(np.int32), (B*mult).astype(np.int32)
    dims = A.max(0) + 1
    X1D = np.ravel_multi_index(A.T, dims)
    searched_valuesID = np.ravel_multi_index(B.T, dims)
    sidx = X1D.argsort()
    return sidx[np.searchsorted(X1D, searched_valuesID, sorter=sidx)]


def bmv(A, x, dtype=None):
    """Batched matrix-vector multiply.
    A.shape = [B, N, M]
    x.shape = [B, M]
    """
    if A.shape[0] == 1:
        assert x.shape[0] == 1
        return (A[0] @ x[0]).unsqueeze(0)

    if dtype:
        x_type = x.dtype
        A = A.to(dtype=dtype)
        x = x.to(dtype=dtype)

    ret = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
    if dtype: ret = ret.to(x_type)
    return ret


def tile(x, n):
    assert isinstance(n, int) and n > 0, 'Argument \'n\' must be an integer.'
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def unique_vectors_and_counts(x):
    unique_x = np.unique(x, axis=0)
    idxs = x.T[None, ...] == unique_x[..., None]
    counts = np.all(idxs, axis=1).sum(-1)
    return unique_x, counts


def get_unique_elements_and_idxs(a):
    """compute 'au', the unique elements of a.
     Also return the (first) indices of au in a, and the indices of a in au
    Example:
        a = [4, 3, 1, 3, 1, 5, 4]
        return [4, 3, 1, 5], [0, 1, 2, 5], [0, 1, 2, 1, 2, 3, 0]

    @param a: np.ndarray
    @return: (np.ndarray, np.ndarray)
    """
    au = pd.unique(a)
    idx1 = find_idxs_of_b_in_a(a, au)
    idx2 = find_idxs_of_b_in_a(au, a)
    return au, idx1, idx2


def get_matrix_from_poly2_coefs(coefs, d):
    mat = np.zeros((d, d))
    inds = np.triu_indices(d)
    if len(coefs) == 1 + len(inds[0]):
        mat[inds] = coefs[1:]  # throw-away bias
    elif len(coefs) == len(inds[0]):
        mat[inds] = coefs
    else:
        raise ValueError
    mat += mat.T
    mat /= 2
    return mat


def get_tensor_mem_in_gb(H):
    """https://discuss.pytorch.org/t/how-to-know-the-memory-allocated-for-a-tensor-on-gpu/28537/9"""
    return (H.element_size() * H.nelement()) / 1e9


def set_random(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    # torch.cuda.manual_seed_all(seed=seed)  # Not needed since I only use 1 GPU.
    random.seed(seed)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class MyTimer(object):
    """Measures time and prints it to stdout for katib to read"""
    def __init__(self, name, unit="hours"):
        self.name = name
        if unit == "seconds":
            self.divisor = 1
        elif unit == "minutes":
            self.divisor = 60
        elif unit == "hours":
            self.divisor = 60 * 60
        else:
            raise ValueError(f"don't recognise unit: {unit}")

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        total_time = (time.time() - self.start_time) / self.divisor
        print(f"{self.name}={total_time:.4f}")

    def get_time(self):
        return (time.time() - self.start_time) / self.divisor
