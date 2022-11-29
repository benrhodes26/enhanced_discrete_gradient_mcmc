import igraph as ig
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dists
from bisect import bisect_left
from networks import MLP, USPSConvNet
from tqdm import tqdm
from utils.utils import find_idxs_of_b_in_a, numpify

import utils.utils as util


class MixtureDist:
    """Bad for-loop implementation of mixture dist - improve later"""

    def __init__(self, dists, device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')):
        self.dists = dists
        self.k = len(dists)
        self.log_mix_weights = (torch.ones(self.k, device=device) / self.k).log()

    def log_prob(self, x):
        logprobs = torch.zeros((len(x), self.k))  # (n_batch, n_components)
        for i, d in enumerate(self.dists):
            logprobs[:, i] = d.log_prob(x)
        logprobs += self.log_mix_weights
        return torch.logsumexp(logprobs, dim=-1)  # (n_batch)

    def sample(self, n=torch.Size()):
        idxs = dists.Categorical(logits=self.log_mix_weights).sample(n)
        counts = torch.bincount(idxs)
        samples = []
        for c, d in zip(counts, self.dists):
            if c > 0:
                samples.append(d.sample((c,)))
        return torch.vstack(samples)

    def log_prob_1d(self, x):
        logprobs = torch.zeros((len(x), self.k), device=x.device)  # (n_batch, n_components)
        for i, d in enumerate(self.dists):
            logprobs[:, i] = d.log_prob_1d(x)
        logprobs += self.log_mix_weights
        return torch.logsumexp(logprobs, dim=-1)  # (n_batch)


class Ordinal:
    """Variant of categorical distribution with domain of 'state_space' rather than {0, ..., k-1}"""

    def __init__(self, logits, state_space, tile_n=None, product=False, validate_args=True):

        if tile_n:
            self.logits = torch.tile(logits, (tile_n, 1))
        else:
            self.logits = logits
        self.state_space = state_space
        self.product = product

        self.dist = dists.Categorical(logits=self.logits, validate_args=validate_args)

    def log_prob(self, x):
        x_idxs = self.idx_lookup(x)
        logp = self.dist.log_prob(x_idxs)  # (n_batch, )
        if self.product:
            assert len(x.shape) > 1
            return logp.sum(-1)
        else:
            return logp

    def prob(self, x):
        return self.log_prob(x).exp()

    def sample(self, n=torch.Size()):
        sample_idxs = self.dist.sample(n)  # (n_batch, n_dims)
        return self.state_space[sample_idxs]

    def idx_lookup(self, x):
        """Return tensor with same shape as x, but with each element replaced by its index in self.state_space"""
        one_hot = (x.unsqueeze(-1) == self.state_space)
        assert one_hot.any(-1).all().bool(), "Not all elements of x belong to the state space"
        return one_hot.max(-1)[1].long()


class ProductOfLocalUniformOrdinals(nn.Module):

    def __init__(self, x, state_space, radius):
        super().__init__()
        self.state_space = state_space  # (k, )
        self.radius = radius
        self.shape = x.shape
        with torch.no_grad():
            self.logits = self.logits(x)  # (n_batch, n_dims, n_states)

    def log_prob(self, s):
        return Ordinal(logits=self.logits, state_space=self.state_space, product=True).log_prob(s)  # (n_batch, )

    def sample(self, n=torch.Size()):
        return Ordinal(logits=self.logits, state_space=self.state_space).sample(n)  # (n_batch, n_dims)

    def logits(self, x):
        logits = torch.ones(x.size(0)*x.size(1), len(self.state_space), device=x.device) * -np.inf
        x_coords, y_coords = self.get_valid_coords(x, self.state_space, self.radius)
        logits[x_coords, y_coords] = 1
        return logits.view(*self.shape, len(self.state_space))

    @staticmethod
    def get_valid_coords(x, state_space, radius, exclude_centre=False):
        x = x.flatten()
        idxs = util.torchify(find_idxs_of_b_in_a(a=numpify(state_space), b=util.numpify(x)), x.device)
        x_coords = torch.arange(len(x), device=x.device).repeat_interleave(int((2 * radius) + 1))
        y_coords = (idxs.unsqueeze(-1) + torch.arange(-radius, radius + 1, device=x.device))
        valid = (y_coords >= 0) & (y_coords < len(state_space))
        if exclude_centre:
            valid = valid & (y_coords != idxs.unsqueeze(-1))
        y_coords, valid = y_coords.flatten(), valid.flatten()
        x_coords, y_coords = x_coords[valid].long(), y_coords[valid].long()
        return x_coords, y_coords


class LocalPseudoMarginal(nn.Module):

    def __init__(self, x, model, dim, state_space, radius, temp=2.0):
        super().__init__()
        self.dim = dim
        self.state_space = state_space
        self.n_states = len(state_space)
        self.radius = radius
        self.length_scale = state_space[1] - state_space[0]
        self.temp = temp
        with torch.no_grad():
            self.logits = self.logits(x, model)  # (n_batch, n_dims, n_states)

    def log_prob(self, s):
        return Ordinal(logits=self.logits, state_space=self.state_space, product=True).log_prob(s)  # (n_batch, )

    def sample(self, n=torch.Size()):
        return Ordinal(logits=self.logits, state_space=self.state_space).sample(n)  # (n_batch, n_dims)

    def logits(self, x, model):
        n_batch = x.size(0)
        window = (torch.arange(-self.radius, self.radius + 1, device=x.device) * self.length_scale)
        window_size = len(window)

        x_tiled = torch.tile(x[:, None, None, :], (1, self.dim, window_size, 1))
        mask = torch.eye(self.dim, device=x.device)
        mask = torch.tile(mask.unsqueeze(-1), (1, 1, window_size))
        x_tiled += (mask * window).swapaxes(1, 2)  # (n_batch, dim, window, dim)
        window_logits = model(x_tiled.view(-1, self.dim)).view(n_batch, self.dim, window_size)
        valid_window_logits = window_logits[torch.all((x_tiled >= self.state_space.min()) & (x_tiled <= self.state_space.max()), dim=-1)]

        logits = torch.ones(x.size(0)*x.size(1), self.n_states, device=x.device) * -np.inf
        x_coords, y_coords = ProductOfLocalUniformOrdinals.get_valid_coords(x, self.state_space, self.radius)
        logits[x_coords, y_coords] = valid_window_logits

        return logits.view(n_batch, self.dim, self.n_states) / self.temp


class OrdinalTargetDist(nn.Module):

    def __init__(self, data_dim, state_space_1d, point_init=True):
        super().__init__()
        self.data_dim = data_dim
        self.state_space_1d = state_space_1d
        self.num_states = len(state_space_1d)

        self.device = state_space_1d.device

        if point_init:
            init_logits = -1e6 * torch.ones_like(state_space_1d, device=state_space_1d.device)
            start = self.num_states // 5
            init_logits[start] = init_logits[start+1] = init_logits[start+2] = 1  # fairly arbitrary points
        else:
            init_logits = torch.ones_like(state_space_1d, device=state_space_1d.device)

        self.init_dist = Ordinal(logits=init_logits, state_space=state_space_1d, tile_n=data_dim, product=True)

    def init_sample(self, n=torch.Size()):
        return self.init_dist.sample(n)

    def forward(self):
        raise NotImplementedError


class MixtureOfProductOfDiscretized1dSimpleFunctions(OrdinalTargetDist):
    """Needs refactoring so that it extends Ordinal + Independent + MixtureSameFamily"""
    def __init__(self, model_type="mixture3_poly2", state_space_1d=torch.linspace(-1, 1, 20), learn=False, data_dim=1, point_init=True):
        super().__init__(data_dim, state_space_1d, point_init=point_init)
        self.model_type = model_type
        self.list_of_dists = self.get_component_dists(model_type, learn, state_space_1d, data_dim)
        self.n_components = len(self.list_of_dists)
        self.distribution = MixtureDist(self.list_of_dists, device=state_space_1d.device)

    def get_component_dists(self, model_type, learn, state_space_1d, data_dim):
        dists = []
        if "poly2" in model_type:
            n_mixture = int(model_type.split("mixture")[1].split("_")[0])
            shifts = np.linspace(0, 2, num=n_mixture)
            for i in range(n_mixture):
                dists += [ProductOfDiscretized1dSimpleFunctions(f"poly2{shifts[i]}", state_space_1d, learn, data_dim, normalize=True)]

        if "poly3" in model_type:
            assert model_type.split("mixture")[1].split("_")[0] == "3"
            dists += [ProductOfDiscretized1dSimpleFunctions("poly3", state_space_1d, learn, data_dim, normalize=True)]
            dists += [ProductOfDiscretized1dSimpleFunctions("poly31", state_space_1d, learn, data_dim, normalize=True)]
            dists += [ProductOfDiscretized1dSimpleFunctions("poly32", state_space_1d, learn, data_dim, normalize=True)]

        if "poly4" in model_type:
            n_mixture = int(model_type.split("mixture")[1].split("_")[0])
            shifts = np.linspace(0, 3, num=n_mixture)
            for i in range(n_mixture):
                dists += [ProductOfDiscretized1dSimpleFunctions(f"poly4{shifts[i]}", state_space_1d, learn, data_dim, normalize=True)]

        elif "polytanh4" in model_type:
            n_mixture = int(model_type.split("mixture")[1].split("_")[0])
            shifts = np.linspace(-1, 2, num=n_mixture)
            for i in range(n_mixture):
                dists += [ProductOfDiscretized1dSimpleFunctions(f"poly4{shifts[i]}", state_space_1d,
                                                                learn, data_dim, tanh=True, normalize=True)]

        if not dists: raise ValueError
        return dists

    def forward(self, x):
        assert len(x.shape) == 2
        f = torch.zeros((len(x), self.n_components), device=x.device)  # (n_batch, n_components)
        for i, dist in enumerate(self.list_of_dists):
            f[:, i] = dist.forward(x)
        f += self.distribution.log_mix_weights.to(x.device)
        return torch.logsumexp(f, dim=-1)  # (n_batch)

    def log_prob(self, x):
        assert len(x.shape) == 2
        return self.distribution.log_prob(x)

    def sample(self, n=torch.Size()):
        return self.distribution.sample(n)

    def log_prob_1d(self, x=None, tile=False):
        if x is None: x = self.state_space_1d
        logp_1d = self.distribution.log_prob_1d(x)  # (n_states, )
        if tile:
            return torch.tile(logp_1d.unsqueeze(-1), (1, self.data_dim))  # (n_states, n_dims)
        else:
            return logp_1d

    def kl(self, q_dist):
        """Average D_KL(p_i || q_i) across marginals of p & q"""

        logp_1d = self.log_prob_1d(self.state_space_1d)  # (n_states, n_dims)
        neg_entropy = (logp_1d.exp() * logp_1d).sum(0).mean()

        tiled_ss = torch.tile(self.state_space_1d.unsqueeze(-1), (1, self.data_dim))  # (n_states, d)
        logq_ss = q_dist.log_prob(tiled_ss)  # (n_states, n_dims)
        cross_entropy = (-logp_1d.exp() * logq_ss).sum(0).mean()

        return neg_entropy + cross_entropy

    def neg_entropy_1d(self):
        logp_1d = self.log_prob_1d(self.state_space_1d)  # (n_states, n_dims)
        neg_entropy_1d = (logp_1d.exp() * logp_1d).sum(0).mean()
        return neg_entropy_1d

    def plot(self, legend=True):
        """Plot marginal distribution"""
        logps = self.log_prob_1d(self.state_space_1d)
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs = axs.ravel()
        ax = axs[0]
        ax.scatter(self.state_space_1d.detach().cpu(), logps.detach().cpu(), color='k', alpha=0.5, label="log pmf")
        if legend: ax.legend()
        ax = axs[1]
        ax.scatter(self.state_space_1d.detach().cpu(), logps.exp().detach().cpu(), color='k', alpha=0.5, label="pmf")
        if legend: ax.legend()
        return fig, axs

    def plot_2d(self, logspace=True, subplots=(1, 1), figsize=(8, 6), alpha=0.8, use_cbar=True):

        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(*subplots)
        axs = [fig.add_subplot(gs[i, j], aspect="equal") for i in range(subplots[0]) for j in range(subplots[1])]

        x = y = np.linspace(self.state_space_1d[0].item()-0.1, self.state_space_1d[-1].item()+0.1, 100)
        X, Y = np.meshgrid(x, y)
        grid = numpy.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.forward(torch.from_numpy(grid).to(self.device))
        if not logspace: Z = Z.exp()
        percentiles = np.percentile(Z.detach().cpu().numpy(), np.arange(101))
        percentiles = np.concatenate([percentiles[:90:5], percentiles[90:101:2]])

        for ax in axs:
            cntr = ax.contourf(X, Y, Z.reshape(*X.shape).detach().cpu().numpy(),
                               levels=np.unique(percentiles), cmap='RdBu_r', alpha=alpha, antialiased=True)
            for c in cntr.collections:
                c.set_linewidth(0.0)

        if use_cbar:
            fig.colorbar(cntr, ax=axs, pad=0.01, shrink=0.6)

        if len(axs) == 1: axs = axs[0]
        return fig, axs

    def plot_3d(self, logspace=True):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = y = np.linspace(self.state_space_1d[0].item(), self.state_space_1d[-1].item(), 100)
        X, Y = np.meshgrid(x, y)
        grid = numpy.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.forward(torch.from_numpy(grid).to(self.device))
        Z = torch.maximum(Z, -10 * torch.ones_like(Z))
        if not logspace: Z = Z.exp()
        percentiles = np.percentile(np.unique(Z.detach().cpu().numpy()), np.linspace(0, 100, 50))
        ax.contour3D(X, Y, Z.reshape(*X.shape).detach().cpu().numpy(), levels=np.unique(percentiles), cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        return fig, ax


class ProductOfDiscretized1dSimpleFunctions(OrdinalTargetDist):
    """Needs refactoring so that it extends Ordinal + uses Independent"""

    def __init__(self,
                 model_type="poly2",
                 state_space_1d=torch.linspace(-1, 1, 20),
                 learn=False,
                 data_dim=1,
                 logits=None,
                 tanh=False,
                 normalize=False,
                 point_init=True,
                 higher_order_strength=1.0,
                 ):

        super().__init__(data_dim, state_space_1d, point_init=point_init)
        self.model_type = model_type
        self.tanh = tanh
        self.set_model_weights(learn, model_type, higher_order_strength)

        if normalize:
            self.log_normalizer = torch.logsumexp(self.forward(self.state_space_1d.unsqueeze(-1)), dim=0)

        if logits is None:
            self.logits = self.forward(self.state_space_1d.unsqueeze(-1))
        else:
            self.logits = logits

    def set_model_weights(self, learn, model_type, higher_order_strength):

        model_known = False
        if "poly" in model_type:
            self.shift = 0  # polynomial will have form \sum_k (x+shift)^k
            self.scale = 1.0
            if "poly2" in model_type:
                self.scale = 2.0
                self.weights = nn.Parameter(torch.as_tensor([0.75, -1.0, -1.5]), requires_grad=learn)
                shift = model_type.split("poly2")[-1]
                if shift:
                    self.shift = -float(shift)

            elif "poly4" in model_type:
                self.weights = 2.0 * torch.as_tensor([0.0, -1, 1, -1, -1])
                self.scale = 1.0
                shift = model_type.split("poly4")[-1]
                if shift:
                    self.shift = -float(shift)

            elif "polyho" in model_type:
                self.weights = 2.0 * torch.as_tensor([0.0, 1, -1, -higher_order_strength, -higher_order_strength])
                self.scale = 1.0
                shift = model_type.split("polyho")[-1]
                if shift:
                    self.shift = -float(shift)

            model_known = True

        if "cos" in model_type:
            self.freq = 5
            self.amp = 0.5
            model_known = True

        if not model_known:
            raise ValueError(f"Unknown model: {model_type}")

    def forward(self, x):
        assert len(x.shape) == 2
        f = torch.zeros_like(x, device=x.device)  # (n_batch, n_dims)
        if "poly" in self.model_type:
            for i in range(len(self.weights)):
                f += self.weights[i] * (((x+self.shift)*self.scale)**i)
        if "cos" in self.model_type:
            f += self.amp * torch.cos(self.freq * x)

        if hasattr(self, "log_normalizer"):
            f -= self.log_normalizer.to(x.device)

        return f.sum(-1)  # (n_batch, )

    def log_prob(self, x):
        assert len(x.shape) == 2
        return Ordinal(logits=self.logits, state_space=self.state_space_1d, tile_n=self.data_dim, product=True).log_prob(x)

    def sample(self, n=torch.Size()):
        return Ordinal(logits=self.logits, state_space=self.state_space_1d, tile_n=self.data_dim, product=True).sample(n)

    def log_prob_1d(self, x=None, tile=False):
        if x is None: x = self.state_space_1d
        assert len(x.shape) == 1
        logp_1d = Ordinal(logits=self.logits, state_space=self.state_space_1d).log_prob(x)
        if tile:
            return torch.tile(logp_1d.unsqueeze(-1), (1, self.data_dim))  # (n_states, n_dims)
        else:
            return logp_1d

    def kl(self, q_dist):
        """per-dimension D_KL(p || q) computed exactly for fully factorised p & q"""

        logp_1d = self.log_prob_1d(self.state_space_1d)  # (n_states, n_dims)
        neg_entropy = (logp_1d.exp() * logp_1d).sum(0).mean()

        tiled_ss = torch.tile(self.state_space_1d.unsqueeze(-1), (1, self.data_dim))
        logq_ss = q_dist.log_prob(tiled_ss)  # (n_states, n_dims)
        cross_entropy = (-logp_1d.exp() * logq_ss).sum(0).mean()

        return neg_entropy + cross_entropy

    def neg_entropy_1d(self):
        logp_1d = self.log_prob_1d(self.state_space_1d)  # (n_states, n_dims)
        neg_entropy_1d = (logp_1d.exp() * logp_1d).sum(0).mean()
        return neg_entropy_1d

    # def kl(self, q_dist):
    #     """D_KL(p || q) computed exactly if possible, otherwise with a monte-carlo estimator"""
    #     logp_1d = self.log_prob_1d(self.state_space_1d)  # (n_states,)
    #     neg_entropy_1d = (logp_1d.exp() * logp_1d).sum()
    #     neg_entropy = neg_entropy_1d * self.data_dim
    #
    #     if hasattr(self, "full_state_space"):
    #         pss = self.log_prob(self.full_state_space).exp()
    #         logq_ss = q_dist.log_prob(self.full_state_space)
    #         cross_entropy = (-pss * logq_ss).sum()
    #     else:
    #         samples = self.sample(torch.Size(int(1e5)))
    #         logq_ss = q_dist.log_prob(samples)
    #         cross_entropy = -logq_ss.mean()
    #
    #     return neg_entropy + cross_entropy

    def plot(self, legend=True):
        smin, smax = self.state_space_1d.min(), self.state_space_1d.max()
        srange = smax - smin
        x = torch.linspace(smin - (srange/5), smax + (srange/5), 512, device=self.device)
        fx = self.forward(x.unsqueeze(-1)).squeeze()
        fs = self.forward(self.state_space_1d.unsqueeze(-1)).squeeze()
        logps = self.log_prob_1d(self.state_space_1d)

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs = axs.ravel()
        ax = axs[0]
        ax.plot(x.detach().cpu(), fx.detach().cpu(), label="underlying continuous function", c='b')
        ax.scatter(self.state_space_1d.detach().cpu(), fs.detach().cpu(),
                   color='b', alpha=0.5, label="unnormalised log pmf")
        ax.scatter(self.state_space_1d.detach().cpu(), logps.detach().cpu(),
                   color='k', alpha=0.5, label="log pmf")
        if legend: ax.legend()
        ax = axs[1]
        ax.scatter(self.state_space_1d.detach().cpu(), logps.exp().detach().cpu(),
                   color='k', alpha=0.5, label="pmf")
        if legend: ax.legend()
        return fig, axs

    def plot_2d(self, logspace=True, subplots=(1, 1), figsize=(8, 6), alpha=0.8, use_cbar=True):

        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(*subplots)
        axs = [fig.add_subplot(gs[i, j], aspect="equal") for i in range(subplots[0]) for j in range(subplots[1])]

        x = y = np.linspace(self.state_space_1d[0].item()-0.1, self.state_space_1d[-1].item()+0.1, 100)
        X, Y = np.meshgrid(x, y)
        grid = numpy.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.forward(torch.from_numpy(grid).to(self.device))
        if not logspace: Z = Z.exp()
        percentiles = np.percentile(Z.detach().cpu().numpy(), np.arange(101))
        percentiles = np.concatenate([percentiles[:90:5], percentiles[90:101:2]])

        for ax in axs:
            cntr = ax.contourf(X, Y, Z.reshape(*X.shape).detach().cpu().numpy(),
                               levels=np.unique(percentiles), cmap='RdBu_r', alpha=alpha, antialiased=True)
            for c in cntr.collections:
                c.set_linewidth(0.0)

        if use_cbar:
            fig.colorbar(cntr, ax=axs, pad=0.01, shrink=0.6)

        if len(axs) == 1: axs = axs[0]
        return fig, axs

    def plot_3d(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x = y = np.linspace(self.state_space_1d[0].item(), self.state_space_1d[-1].item(), 50)
        X, Y = np.meshgrid(x, y)
        grid = numpy.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.forward(torch.from_numpy(grid).to(self.device))
        percentiles = np.percentile(Z.detach().cpu().numpy(), np.linspace(0, 100, 50))
        ax.contour3D(X, Y, Z.reshape(*X.shape).detach().cpu().numpy(), levels=percentiles, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        return fig, ax


class DiscretizedQuadratic(OrdinalTargetDist):

    def __init__(self, H, b, state_space_1d, data_dim=2, point_init=True):

        super().__init__(data_dim, state_space_1d, point_init=point_init)
        self.H = H
        self.b = b

        full_ss_size = len(self.state_space_1d) ** self.data_dim
        if full_ss_size > 1e7: raise NotImplementedError("state space is too large to construct!")
        if data_dim > 1:
            self.full_state_space = torch.cartesian_prod(*[self.state_space_1d for _ in range(self.data_dim)])
        else:
            self.full_state_space = self.state_space_1d.unsqueeze(-1)

        self.np_ss = numpify(self.full_state_space)
        full_ss_logits = self.forward(self.full_state_space)  # (n_full_state_space)
        self.full_ss_logprobs = full_ss_logits - torch.logsumexp(full_ss_logits, dim=0)
        self.log_marginals = self.get_log_marginals()  # (n_dims, n_states)

    def forward(self, x):
        return 0.5 * (x * (x @ self.H)).sum(-1) + (self.b*x).sum(-1)

    def log_prob(self, s):
        idxs = find_idxs_of_b_in_a(self.np_ss, numpify(s))
        return self.full_ss_logprobs[idxs]

    def sample(self, n=torch.Size()):
        return Ordinal(logits=self.full_ss_logprobs, state_space=self.full_state_space).sample(n)

    def get_log_marginals(self):
        log_marginals = torch.zeros(self.data_dim, self.num_states, device=self.device)
        for i in range(self.data_dim):
            for j, state in enumerate(self.state_space_1d):
                mask = (self.full_state_space[:, i] == state)
                log_marginals[i, j] = torch.logsumexp(self.full_ss_logprobs[mask], dim=0)
        return log_marginals

    def log_prob_1d(self, x=None, tile=True):
        if x is None:
            return self.log_marginals.T  # (n_states, n_dims)
        else:
            raise NotImplementedError

    def kl(self, q_dist):
        """Average D_KL(p_i || q_i) across marginals of p & q"""
        neg_entropy = (self.log_marginals.exp() * self.log_marginals).sum(-1)  # (n_dims, )
        tiled_ss = torch.tile(self.state_space_1d.unsqueeze(-1), (1, self.data_dim))
        logq_marginals = q_dist.log_prob(tiled_ss)  # (n_states, n_dims)
        p_marginals = self.log_marginals.exp().T  # (n_states, n_dims)
        cross_entropy = (-p_marginals * logq_marginals).sum(0)  # (n_dims)
        return (neg_entropy + cross_entropy).mean()

    def neg_entropy_1d(self):
        return (self.log_marginals.exp() * self.log_marginals).sum(-1).mean()

    def plot(self, legend=True):
        """Plot marginal distribution"""

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs = axs.ravel()
        for i in range(self.data_dim):
            logps = self.log_marginals[i]  # (n_states)
            ax = axs[0]
            ax.scatter(self.state_space_1d.detach().cpu(), logps.detach().cpu(), color='k', alpha=0.5, label=f"log marginal {i}")
            if legend: ax.legend()
            ax = axs[1]
            ax.scatter(self.state_space_1d.detach().cpu(), logps.exp().detach().cpu(), color='k', alpha=0.5, label=f"marginal {i}")
        if legend: ax.legend()
        return fig, axs

    def plot_2d(self, logspace=True, subplots=(1, 1), figsize=(8, 6), alpha=0.8, use_cbar=True):
        # fig, axs = plt.subplots(*subplots, figsize=figsize)
        # axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(*subplots)
        axs = [fig.add_subplot(gs[i, j], aspect="equal") for i in range(subplots[0]) for j in range(subplots[1])]

        x = y = np.linspace(self.state_space_1d[0].item()-0.1, self.state_space_1d[-1].item()+0.1, 100)
        X, Y = np.meshgrid(x, y)
        grid = numpy.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.forward(util.torchify(grid))
        if not logspace: Z = Z.exp()
        percentiles = np.percentile(util.numpify(Z), np.arange(101))
        percentiles = np.concatenate([percentiles[:90:5], percentiles[90:101:2]])

        # cmap = plt.get_cmap('Greens')
        cmap = plt.get_cmap('Blues')
        # cmap = plt.get_cmap('viridis_r')
        # new_cmap = util.truncate_colormap(cmap, 0.5, 0.8)
        new_cmap = util.truncate_colormap(cmap, 0.3, 0.9)
        for ax in axs:
            # for i in range(3):
            cntr = ax.contourf(X, Y, Z.reshape(*X.shape).detach().cpu().numpy(),
                               levels=np.unique(percentiles), cmap=new_cmap, alpha=alpha, antialiased=True)
            for c in cntr.collections:
                c.set_edgecolor("face")

            ax.set_xlabel('x')
            ax.set_ylabel('y')

        if use_cbar:
            # fig.colorbar(cntr, ax=ax)
            fig.colorbar(cntr, ax=axs, pad=0.01, shrink=0.6)

        if len(axs) == 1: axs = axs[0]
        return fig, axs


class BernoulliRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, data_mean=None):
        super().__init__()
        linear = nn.Linear(n_visible, n_hidden)
        self.W = nn.Parameter(linear.weight.data)
        self.b_h = nn.Parameter(torch.zeros(n_hidden,))
        self.b_v = nn.Parameter(torch.zeros(n_visible,))
        if data_mean is not None:
            init_val = (data_mean / (1. - data_mean)).log()
            self.b_v.data = init_val
            self.init_dist = dists.Bernoulli(probs=data_mean)
        else:
            self.init_dist = dists.Bernoulli(probs=torch.ones((n_visible,)) * .5)
        self.data_dim = n_visible

    def p_v_given_h(self, h):
        logits = h @ self.W + self.b_v[None]
        return dists.Bernoulli(logits=logits)

    def p_h_given_v(self, v):
        logits = v @ self.W.t() + self.b_h[None]
        return dists.Bernoulli(logits=logits)

    def logp_v_unnorm(self, v):
        sp = torch.nn.Softplus()(v @ self.W.t() + self.b_h[None]).sum(-1)
        vt = (v * self.b_v[None]).sum(-1)
        return sp + vt

    def logp_v_unnorm_beta(self, v, beta):
        if len(beta.size()) > 0:
            beta = beta[:, None]
        vW = v @ self.W.t() * beta
        sp = torch.nn.Softplus()(vW + self.b_h[None]).sum(-1) - torch.nn.Softplus()(self.b_h[None]).sum(-1)
        # vt = (v * self.b_v[None]).sum(-1)
        ref_dist = torch.distributions.Bernoulli(logits=self.b_v)
        vt = ref_dist.log_prob(v).sum(-1)
        return sp + vt

    def forward(self, x):
        return self.logp_v_unnorm(x)

    def _gibbs_step(self, v):
        h = self.p_h_given_v(v).sample()
        v = self.p_v_given_h(h).sample()
        return v

    def gibbs_sample(self, v=None, n_steps=2000, n_samples=None, plot=False):
        if v is None:
            assert n_samples is not None
            v = self.init_dist.sample((n_samples,)).to(self.W.device)
        if plot:
           for i in tqdm(range(n_steps)):
               v = self._gibbs_step(v)
        else:
            for i in range(n_steps):
                v = self._gibbs_step(v)
        return v

# if block_diag:
        #     print("Creating block diagonal...")
        #     num_repeats = int(n_dims / 2)
        #     e = util.torchify(np.array([1.0, 10.0]))
        #     Q = util.torchify(np.array([[0.7071, 0.7071],
        #                       [-0.7071, 0.7071]]))
        #     block = Q @ torch.diag_embed(e) @ Q.T
        #     A = (1/(8*init_sigma)) * torch.block_diag(*[block.clone() for _ in range(num_repeats)])
        # else:
        #     g = ig.Graph.Lattice(dim=[int(n_dims**0.5)] * 2, circular=True)  # Boundary conditions
        #     A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        #     if negative_weights:
        #         print("Adding negative weights to Ising matrix...")
        #         g = ig.Graph.Lattice(dim=[int(n_dims**0.5)] * 2, circular=True)  # Boundary conditions
        #         A += -np.roll(np.asarray(g.get_adjacency().data), 5)  # g.get_sparse_adjacency()
        #     A = util.torchify(A).float()
        #     if abs_eig:
        #         print("Taking absolute value of eigenvalues...")
        #         e, Q = torch.linalg.eigh(A)
        #         A = Q @ torch.diag_embed(torch.abs(e)) @ Q.T

class LatticeIsingModel(nn.Module):

    def __init__(self, n_dims, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False,
                 learn_bias=False, num_repeats=1, third_order_interaction_strength=None):
        super().__init__()

        g = ig.Graph.Lattice(dim=[int(n_dims ** 0.5)] * 2, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        A = util.torchify(A).float()

        b = torch.ones((n_dims,)).float() * init_bias
        if num_repeats > 1:
            b = torch.repeat_interleave(num_repeats)
            A = torch.block_diag(*[A for _ in range(num_repeats)])

        self.G = nn.Parameter(A, requires_grad=learn_G)
        self.bias = nn.Parameter(b, requires_grad=learn_bias)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)

        self.init_dist = dists.Bernoulli(logits=2 * b)
        self.data_dim = n_dims * num_repeats
        self.third_order_interaction_strength = third_order_interaction_strength
        if third_order_interaction_strength:
            self.third_order_indices = np.random.randint(0, n_dims, size=(50, 3))

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.sigma * 0.5 * (self.G + self.G.T)

    def forward(self, y):
        x = y.clone()
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        val = xgx + b

        if self.third_order_interaction_strength:
            val += self.third_order_interaction_strength * x[:, self.third_order_indices].prod(-1).sum(-1)  # (batch_size, )

        return val


class BoltzmannMachine(nn.Module):

    def __init__(self, n_dims, init_bias=None, learn_G=False, learn_bias=False):
        super().__init__()

        A = torch.randn((n_dims, n_dims)) * 0.01
        self.G = nn.Parameter(A, requires_grad=learn_G)

        if init_bias is None:
            init_bias = torch.ones((n_dims,)).float() / n_dims
        self.bias = nn.Parameter((init_bias + 1e-2).log(), requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(probs=init_bias)
        self.data_dim = n_dims

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G + self.G.T

    def forward(self, y):
        x = y.clone()
        if len(x.size()) > 2: x = x.view(x.size(0), -1)
        xJ = x @ self.J
        xJx = (xJ * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return b + (0.5 * xJx)


class QuadraticNeuralModel(nn.Module):

    def __init__(self, data_dim, n_layers=3, hsize=128, learn_G=False, learn_bias=False,
                 init_bias=None, learn_net=False, net_weight=0.0, dataset=None):
        super().__init__()
        self.data_dim = data_dim
        G = torch.randn((data_dim, data_dim)) * .01
        self.G = nn.Parameter(G.float(), requires_grad=learn_G)
        if init_bias is None: init_bias = torch.ones((data_dim,)).float() / data_dim
        self.bias = nn.Parameter((init_bias+1e-2).log(), requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(probs=init_bias)
        self.net_weight = net_weight
        if data_dim == 256:
            self.net = USPSConvNet()
        else:
            self.net = MLP(insize=data_dim, outsize=1, hsize=hsize, fc_num_layers=n_layers, activation=torch.nn.SiLU)

        for p in self.net.parameters():
            p.requires_grad = learn_net

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return 0.5 * (self.G + self.G.T)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1
        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)

        f_quad = xgx + b
        f_neural = self.net_weight * self.net(x).squeeze(-1)

        return f_quad + f_neural

    # def forward(self, x):
    #     if len(x.size()) > 2:
    #         x = x.view(x.size(0), -1)
    #
    #     x = (2 * x) - 1
    #     # xg = x @ self.J
    #     # xgx = (xg * x).sum(-1)
    #     # b = (self.bias[None, :] * x).sum(-1)
    #     #
    #     # f_quad = xgx + b
    #     f_neural = self.net_weight * self.net(x).squeeze(-1)
    #
    #     return f_neural

    def unfreeze_net(self):
        for p in self.net.parameters():
            p.requires_grad = True


class ERIsingModel(nn.Module):
    def __init__(self, n_node, avg_degree=2, init_bias=0., learn_G=False, learn_bias=False):
        super().__init__()
        g = ig.Graph.Erdos_Renyi(n_node, float(avg_degree) / float(n_node))
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        A = torch.tensor(A).float()
        weights = torch.randn_like(A) * ((1. / avg_degree) ** .5)
        weights = weights * (1 - torch.tril(torch.ones_like(weights)))
        weights = weights + weights.t()

        self.G = nn.Parameter(A * weights, requires_grad=learn_G)
        self.bias = nn.Parameter(torch.ones((n_node,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = n_node

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return 0.5 * (self.G + self.G.T)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b


class LatticePottsModel(nn.Module):
    def __init__(self, dim, n_out=3, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim, dim], circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** 2, n_out)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.OneHotCategorical(logits=self.bias)
        self.dim = dim
        self.n_out = n_out
        self.data_dim = dim ** 2

    @property
    def mix(self):
        off_diag = -(torch.ones((self.n_out, self.n_out)) - torch.eye(self.n_out)).to(self.G) * self.sigma
        diag = torch.eye(self.n_out).to(self.G) * self.sigma
        return off_diag + diag

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    def forward2(self, x):
        assert list(x.size()[1:]) == [self.dim ** 2, self.n_out]

        xr = x.view(-1, self.n_out)
        xr_mix = (xr @ self.mix).view(x.size(0), -1, self.n_out)

        xr_mix_xr = (xr_mix[:, :, None, :] * x[:, None, :, :]).sum(-1)

        pairwise = (xr_mix_xr * self.G[None]).sum(-1).sum(-1)
        indep = (x * self.bias[None]).sum(-1).sum(-1)

        return pairwise + indep

    def forward(self, x):
        assert list(x.size()[1:]) == [self.dim ** 2, self.n_out]
        xr = x.view(-1, self.n_out)
        xr_mix = (xr @ self.mix).view(x.size(0), -1, self.n_out)

        TEST = torch.einsum("aik,ij->ajk", xr_mix, self.G)
        TEST2 = torch.einsum("aik,aik->a", TEST, x)

        indep = (x * self.bias[None]).sum(-1).sum(-1)

        # return pairwise + indep
        return TEST2 + indep


class DensePottsModel(nn.Module):
    def __init__(self, dim, n_out=20, init_bias=0., learn_J=False, learn_bias=False):
        super().__init__()
        self.J = nn.Parameter(torch.randn((dim, dim, n_out, n_out)) * .01, requires_grad=learn_J)
        self.bias = nn.Parameter(torch.ones((dim, n_out)).float() * init_bias, requires_grad=learn_bias)
        self.dim = dim
        self.n_out = n_out
        self.data_dim = dim * n_out

    @property
    def init_dist(self):
        return dists.OneHotCategorical(logits=self.bias)

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    def forward(self, x, beta=1.):
        assert list(x.size()[1:]) == [self.dim, self.n_out]
        Jx = torch.einsum("ijkl,bjl->bik", self.J, x)
        xJx = torch.einsum("aik,aik->a", Jx, x)
        bias = (self.bias[None] * x).sum(-1).sum(-1)
        return xJx * beta + bias


class MyOneHotCategorical:
    def __init__(self, mean):
        self.dist = torch.distributions.OneHotCategorical(probs=mean)

    def sample(self, x):
        return self.dist.sample(x).float()

    def log_prob(self, x):
        logits = self.dist.logits
        lp = torch.log_softmax(logits, -1)
        return (x * lp[None]).sum(-1)


class OneHotCategoricalLengthMixture:

    def __init__(self, data, pad_idx, num_bins):

        self.pad_idx = pad_idx
        max_seq_len, n_tokens = data.shape[1], data.shape[2]

        # if a pad token occurs, it must be part of a block of pad tokens at end of sequence
        pad_mask = (data[..., pad_idx] == 1).float()
        for i in range(1, max_seq_len):
            assert torch.all(pad_mask[:, i] >= pad_mask[:, i - 1]), \
                f"There exists a pad token at dimension {i - 1} that is followed by a non-pad token"

        # empirical sequence length distribution
        lengths = torch.argmax(pad_mask, dim=1)
        lengths[lengths == 0] = max_seq_len
        freqs = torch.bincount(lengths) / len(lengths)
        self.length_dist = torch.distributions.Categorical(probs=freqs)

        # empirical mean over each bin
        if num_bins == max_seq_len:
            assignments, binwidths = lengths-1, np.array([i for i in range(max_seq_len)] + [max_seq_len])
        else:
            try:
                assignments, binwidths = pd.qcut(lengths.numpy(), num_bins, labels=False, retbins=True)
            except Exception as e:
                print("Probably need to reduce num_bins")
                raise e

        sample_sizes = []
        self.means = torch.zeros((num_bins, max_seq_len, n_tokens))

        for j in range(num_bins):

            bin_data = data[assignments == j].float()
            bin_pad_mask = pad_mask[assignments == j]
            max_len = int(binwidths[j + 1])
            sample_size_per_bin = []
            for i in range(max_len):
                seqs = bin_data[bin_pad_mask[:, i] == 0]  # seqs with length >= i
                mean = seqs[:, i, :].mean(0) + (1e-2 / n_tokens)
                self.means[j][i] = mean / mean.sum()
                sample_size_per_bin.append(len(seqs))

            self.means[j][max_len:, pad_idx] = 1.0  # pad rest of sequence
            sample_sizes.append(sample_size_per_bin)

        self.binwidths = binwidths
        self.len_to_bin_dict = {i: bisect_left(binwidths[1:], i) for i in range(max_seq_len + 1)}

    def sample(self, n):

        # sample seq length
        length = self.length_dist.sample((n,)).cpu().numpy().astype(np.int32)

        # get mean vector from corresponding bins
        bin_idx = [self.len_to_bin_dict[l] for l in length]
        mean = self.means[bin_idx]

        # set any dims beyond length to be the padding token
        for i, l in enumerate(length):
            mean[i, l:, :] = 0
            mean[i, l:, self.pad_idx] = 1

        dist = torch.distributions.OneHotCategorical(probs=mean)
        return dist.sample((1,)).float().squeeze(0)

    def log_prob(self, x):
        raise NotImplementedError

