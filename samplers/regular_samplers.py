import time

import torch
import torch.nn as nn
import torch.distributions as dists
import utils.mcmc_utils as utils
import numpy as np

from distributions.discrete import Ordinal, ProductOfLocalUniformOrdinals
from utils.utils import find_idxs_of_b_in_a, numpify, torchify, fn_and_grad


class BinaryGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self.proposed_hops = []
        self.acc_hops = []
        self.acc_rates = [1.0]
        self._i = 0
        self.rand = rand
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()
        sample = x.clone()
        lp_keep = model(sample)
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change)

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])

        self.acc_hops.append((x != sample).float().sum(-1).mean().item())
        self.proposed_hops.append((x != sample).float().sum(-1).mean().item())
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for binary data
class BinaryGWGSampler(nn.Module):

    def __init__(self, dim, approx=True, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0, use_cache=True):
        super().__init__()
        self.dim = dim
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self.use_cache = use_cache
        self.cache = None
        self.acc_rates = []
        self.acc_hops = []
        self.proposed_hops = []
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.av_time, self.time_n = 0.0, 0
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

    def step(self, x, model):
        start_time = time.time()
        x_cur = x

        if self.use_cache and self.cache is not None:
            logp, logits = self.cache
        else:
            logp = model(x_cur)
            logits = self.diff_fn(x_cur, model)
            self.cache = logp, logits

        cd_forward = dists.OneHotCategorical(logits=logits, validate_args=False)
        changes = cd_forward.sample()

        lp_forward = cd_forward.log_prob(changes)
        x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

        logp_rev = model(x_delta)
        reverse_logits = self.diff_fn(x_delta, model)
        cd_reverse = dists.OneHotCategorical(logits=reverse_logits, validate_args=False)
        lp_reverse = cd_reverse.log_prob(changes)

        m_term = (logp_rev - logp)
        la = m_term + lp_reverse - lp_forward
        a = (la.exp() > torch.rand_like(la)).float()
        x_acc = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        if self.use_cache:
            logp_acc = logp_rev * a + (logp * (1. - a))
            logits_acc = reverse_logits * a[:, None] + (logits * (1. - a[:, None]))
            self.cache = logp_acc, logits_acc

        self.acc_rates.append(a.mean().item())
        self.acc_hops.append(torch.abs(x - x_acc).sum(-1).mean().item())
        self.proposed_hops.append(torch.abs(x - x_delta).sum(-1).mean().item())

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return x_acc


class OrdinalGibbsSampler(nn.Module):
    def __init__(self, dim, state_space, rand=False):
        super().__init__()
        self.dim = dim
        self.state_space = state_space
        self.length_scale = (state_space[1] - state_space[0]).item()
        self.n_states = len(state_space)
        self._i = 0
        self.rand = rand
        self.acc_rates = []
        self.proposed_hops = []
        self.acc_hops = []
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()
        n_batch = x.size(0)

        if self.rand:
            logits = torch.zeros(self.dim)
            mask = dists.OneHotCategorical(logits=logits).sample((n_batch,)).to(x.device)
            mask = torch.tile(mask.unsqueeze(-1), (1, 1, self.n_states))
        else:
            mask = torch.zeros((n_batch, self.dim, self.n_states)).to(x.device)
            mask[:, self._i, :] = 1

        x_tiled = torch.tile(x[:, None, :], (1, self.n_states, 1))
        x_tiled = (x_tiled * (1 - mask).swapaxes(1, 2)) + ((mask * self.state_space).swapaxes(1, 2))
        x_tiled = x_tiled.view(n_batch * self.n_states, self.dim)

        conditional_logprobs = model(x_tiled)
        conditional_logprobs = conditional_logprobs.view(n_batch, self.n_states)

        updates = Ordinal(logits=conditional_logprobs, state_space=self.state_space).sample()
        mask = mask[:, :, 0]
        xhat = (x * (1 - mask)) + (updates.unsqueeze(-1) * mask)

        self._i = (self._i + 1) % self.dim
        self.acc_rates.append(1.0)
        hop_dist = torch.abs(x - xhat).sum(-1).mean().item() / self.length_scale
        self.proposed_hops.append(hop_dist)
        self.acc_hops.append(hop_dist)

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return xhat

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class OrdinalGWGSampler(nn.Module):

    def __init__(self, dim, state_space, radius=1, use_gradient=True, temp=2., use_cache=True):

        super().__init__()
        self.dim = dim
        self.state_space = state_space
        self.length_scale = (state_space[1] - state_space[0]).item()
        self.radius = radius
        self.temp = temp
        self.use_cache = use_cache
        self.cache = None
        # self.window = torch.arange(-self.radius, self.radius + 1).to(state_space.device) * self.length_scale
        self.window = torch.cat([torch.arange(-self.radius, 0), torch.arange(1, self.radius+1)], dim=0).to(state_space.device) * self.length_scale
        self.window_size = len(self.window)
        self.length_scale = (state_space[1] - state_space[0]).item()
        self.n_states = len(state_space)
        self.acc_rates = []
        self.acc_hops = []
        self.proposed_hops = []
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()

        if self.use_cache and self.cache is not None:
            logp, logits = self.cache
        else:
            logp = model(x)
            logits = self.get_logits(x, model)  # (n, d*n_states)
            self.cache = logp, logits

        fwd_prop = dists.Categorical(logits=logits, validate_args=False)
        idxs = fwd_prop.sample()  # (n, )
        d = torch.div(idxs, self.n_states, rounding_mode='floor')
        state_idxs = idxs - d * self.n_states

        xrev = x.clone()
        xrev[torch.arange(x.size(0)), d] = self.state_space[state_idxs]

        logp_rev = model(xrev)
        reverse_logits = self.get_logits(xrev, model)  # (n, d*n_states)
        reverse_prop = dists.Categorical(logits=reverse_logits, validate_args=False)

        x_state_idxs = torchify(
            find_idxs_of_b_in_a(a=numpify(self.state_space), b=numpify(x[torch.arange(x.size(0)), d])))
        x_idxs = (d * self.n_states) + x_state_idxs
        prop_diff = reverse_prop.log_prob(x_idxs) - fwd_prop.log_prob(idxs)
        model_diff = logp_rev - logp
        log_accept_prob = model_diff + prop_diff

        acc_mask = (log_accept_prob.exp() > torch.rand_like(log_accept_prob)).float()  # acceptance mask
        xacc = xrev * acc_mask[:, None] + x * (1. - acc_mask[:, None])
        if self.use_cache:
            logp = logp_rev * acc_mask + (logp * (1. - acc_mask))
            logits = reverse_logits * acc_mask[:, None] + (logits * (1. - acc_mask[:, None]))
            self.cache = logp, logits

        self.acc_rates.append(acc_mask.mean().item())
        self.acc_hops.append(torch.abs(x - xacc).sum(-1).mean().item() / self.length_scale)
        self.proposed_hops.append(torch.abs(x - xrev).sum(-1).mean().item() / self.length_scale)

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return xacc

    def get_logits(self, x, model):
        """note: this implementation is probably more baroque than it needs to be,
        but its awkward to write a vectorised implementation that takes into account
        the boundaries of the ordinal state-space
        """
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        x_tiled = torch.tile(x.unsqueeze(-1), (1, 1, self.window_size))  # (n, d, w)
        x_tiled += self.window
        window_logits = torch.tile(gx.unsqueeze(-1), (1, 1, self.window_size)) * self.window  # (n, d, w)

        # mask out any invalid logits (which occur because we are at the edge of the state-space)
        valid_moves = (x_tiled >= self.state_space.min()) & (x_tiled <= self.state_space.max())  # (n, d, w)
        valid_window_logits = window_logits[valid_moves]  # (n_valid, )

        # pad these valid_window_logits with -infs representing states outside of our window
        logits = torch.ones(x.size(0) * x.size(1), self.n_states, device=x.device) * -np.inf  # (n*d, n_states)
        x_valid, y_valid = ProductOfLocalUniformOrdinals.get_valid_coords(x, self.state_space, self.radius, exclude_centre=True)
        logits[x_valid, y_valid] = valid_window_logits

        return torch.nan_to_num(logits.view(x.size(0), -1) / self.temp)  # (n, d*n_states)

    def sample_proposal(self, x, model):
        logits = self.get_logits(x, model)  # (n, d*n_states)
        fwd_prop = dists.Categorical(logits=logits)
        idxs = fwd_prop.sample()  # (n, )
        d = torch.div(idxs, self.n_states, rounding_mode='floor')
        state_idxs = idxs - d * self.n_states
        xrev = x.clone()
        xrev[torch.arange(x.size(0)), d] = self.state_space[state_idxs]
        return xrev


class CategoricalGibbsSampler(nn.Module):

    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand
        self.acc_rates = [1.0]
        self.acc_hops = []
        self.proposed_hops = []

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        n_chains, n_dims, n_states = x.shape
        """ for-loop version of the code below
        # logits = []
        # for k in range(n_states):
        #     sample = x.clone()
        #     sample_i = torch.zeros((n_states,))
        #     sample_i[k] = 1.
        #     sample[:, i, :] = sample_i
        #     lp_k = model(sample).squeeze()
        #     logits.append(lp_k[:, None])    
        # logits = torch.cat(logits, 1)
        """
        x_all = x.unsqueeze(2).tile(1, 1, n_states, 1)
        x_all = x_all.permute(1, 0, 2, 3)  # (n_dims, n_chains, k, k)
        x_all[i] = torch.eye(n_states).unsqueeze(0).tile(n_chains, 1, 1)
        x_all = x_all.permute(1, 2, 0, 3)  # (n_chains, k, n_dims, k)
        logits = model(x_all.reshape(n_chains*n_states, n_dims, n_states)).view(n_chains, n_states)

        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        x_prop = x.clone()
        x_prop[:, i, :] = updates

        self.proposed_hops.append(torch.abs(x - x_prop).sum((1, 2)).mean().item() / 2.0)
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != x_prop).float().sum(-1) / 2.).sum(-1).mean().item()
        self.acc_hops.append(((x != x_prop).float().sum(-1) / 2.).sum(-1).mean().item())
        self._ar = self._hops
        return x_prop

    def logp_accept(self, xhat, x, model):
        # only true if what was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class CategoricalGWGSampler(nn.Module):

    def __init__(self, dim, approx=True, temp=2., use_cache=True):
        super().__init__()
        self.dim = dim
        self.use_cache = use_cache
        self.cache = None
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self._approx_errors = np.array([0., 0.])
        self._normalizer_diffs = 0.
        self.acc_rates = []
        self.acc_hops = []
        self.proposed_hops = [1.0]

        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: CategoricalGWGSampler.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: CategoricalGWGSampler.difference_function_multi_dim(x, m) / self.temp

    @torch.no_grad()
    def step(self, x, model):
        x_cur = x

        if self.use_cache and self.cache is not None:
            logp, flat_forward_logits = self.cache
        else:
            logp = model(x_cur)
            forward_logits = self.diff_fn(x_cur, model)
            forward_logits = forward_logits - 1e9 * x_cur # forbid sampler from proposing current state
            flat_forward_logits = forward_logits.view(x_cur.size(0), -1)
            self.cache = logp, flat_forward_logits

        # FORWARD
        cd_forward = dists.OneHotCategorical(logits=flat_forward_logits, validate_args=False)
        changes = cd_forward.sample()
        fwd_proposal_logp = cd_forward.log_prob(changes)

        # REVERSE
        changes_r = changes.view(x_cur.size())  # reshape to (bs, dim, nout)
        changed_ind = changes_r.sum(-1)  # get mask indicating which dim was changed
        x_prop = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r
        logp_rev = model(x_prop).squeeze()

        reverse_logits = self.diff_fn(x_prop, model)
        reverse_logits = reverse_logits - 1e9 * x_prop
        flat_reverse_logits = reverse_logits.view(x_prop.size(0), -1)
        cd_reverse = dists.OneHotCategorical(logits=flat_reverse_logits, validate_args=False)
        reverse_changes = x_cur * changed_ind[:, :, None]
        rev_proposal_logp = cd_reverse.log_prob(reverse_changes.view(x_prop.size(0), -1))

        # ACCEPT/REJECT
        m_term = (logp_rev - logp).squeeze()
        la = m_term + rev_proposal_logp - fwd_proposal_logp
        acc_mask = (la.exp() > torch.rand_like(la)).float()
        x_cur = x_prop * acc_mask[:, None, None] + x_cur * (1. - acc_mask[:, None, None])

        if self.use_cache:
            logp = logp_rev * acc_mask + (logp * (1. - acc_mask))
            logits = flat_reverse_logits * acc_mask[:, None] + (flat_forward_logits * (1. - acc_mask[:, None]))
            self.cache = logp, logits

        # METRICS
        self.acc_rates.append(acc_mask.mean().item())
        av_hops = (x != x_cur).float().sum(-1).sum(-1).mean().item() / 2
        self.acc_hops.append(av_hops)
        self._pt = ((rev_proposal_logp - fwd_proposal_logp).mean().item())

        # # track inaccuracy of gradient estimator
        # m_term_est1 = (forward_logits * changes_r).sum(dim=(1, 2))
        # m_term_est2 = (reverse_logits * reverse_changes).sum(dim=(1, 2))
        # gt_error = torch.abs(m_term - m_term_est1).mean().item()
        # rel_error = torch.abs(m_term_est1 - m_term_est2).mean().item()
        # approx_errors.append([gt_error, rel_error])
        #
        # # track diffs in normalizers of cd_forward & reverse
        # norm1 = flat_forward_logits.logsumexp(dim=-1)  # (batch_size, )
        # norm2 = flat_reverse_logits.logsumexp(dim=-1)  # (batch_size, )
        # norm_diff = torch.abs(norm1 - norm2).mean().item()
        # normalizer_diffs.append(norm_diff)

        return x_cur

    @staticmethod
    def approx_difference_function_multi_dim(x, model):
        with torch.enable_grad():
            x = x.requires_grad_()
            gx = torch.autograd.grad(model(x).sum(), x)[0]
            gx_cur = (gx * x).sum(-1)[:, :, None]
        return gx - gx_cur

    @staticmethod
    def difference_function_multi_dim(x, model):
        n_chains, n_dims, n_states = x.shape
        d = torch.zeros_like(x).permute(1, 0, 2)  # (n_dims, n_chains, n_states)
        orig_out = model(x).squeeze()
        for i in range(x.size(1)):
            x_all = x.unsqueeze(2).tile(1, 1, n_states, 1)
            x_all = x_all.permute(1, 0, 2, 3)  # (n_dims, n_chains, k, k)
            x_all[i] = torch.eye(n_states).unsqueeze(0).tile(n_chains, 1, 1)
            x_all = x_all.permute(1, 2, 0, 3)  # (n_chains, k, n_dims, k)
            d[i] = model(x_all.reshape(n_chains * n_states, n_dims, n_states)).view(n_chains, n_states) - orig_out[:, None]
        return d.permute(1, 0, 2)  # (n_chains, n_dims, n_states)


class SymmetricMHSampler(nn.Module):

    def __init__(self, proposal_dist, length_scale=1.0):
        super().__init__()
        self.prop_dist = proposal_dist
        self.acc_rates = []
        self.proposed_hops = []
        self.length_scale = length_scale
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()
        xhat = self.prop_dist(x).sample()
        log_accept_prob = model(xhat) - model(x)

        acc_mask = (log_accept_prob.exp() > torch.rand_like(log_accept_prob)).float()  # acceptance mask
        x_acc = xhat * acc_mask[:, None] + x * (1. - acc_mask[:, None])
        self.acc_rates.append(acc_mask.mean().item())

        # how many states were hopped in the proposals?
        self.proposed_hops.append(torch.abs(x - xhat).sum(-1).mean().item() / self.length_scale)

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return x_acc


class MHSampler(nn.Module):

    def __init__(self, proposal_dist, length_scale):
        super().__init__()
        self.prop_dist = proposal_dist
        self.acc_rates = []
        self.acc_hops = []
        self.proposed_hops = []
        self.length_scale = length_scale
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()
        prop_dist = self.prop_dist(x, model)
        xhat = prop_dist.sample()
        reverse_prop_dist = self.prop_dist(xhat, model)

        log_accept_prob = model(xhat) - model(x) + reverse_prop_dist.log_prob(x) - prop_dist.log_prob(xhat)

        acc_mask = (log_accept_prob.exp() > torch.rand_like(log_accept_prob)).float()  # acceptance mask
        x_acc = xhat * acc_mask[:, None] + x * (1. - acc_mask[:, None])
        self.acc_rates.append(acc_mask.mean().item())

        # how many states were hopped in the proposals?
        self.acc_hops.append(torch.abs(x - x_acc).sum(-1).mean().item() / self.length_scale)
        self.proposed_hops.append(torch.abs(x - xhat).sum(-1).mean().item() / self.length_scale)

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return x_acc

    def sample_proposal(self, x, model):
        prop_dist = self.prop_dist(x, model)
        return prop_dist.sample()

class NCGSampler(nn.Module):

    def __init__(self,
                 n_dims,
                 epsilon=1.0,
                 n_forward_copies=1,
                 state_space=None,
                 var_type="binary",
                 use_simple_mod=False,
                 reg_lambda=0.0,
                 support_dist=2.0,
                 use_cache=True,
                 device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.n_dims = n_dims
        self.epsilon = epsilon
        self.n_forward_copies = n_forward_copies
        self.use_simple_mod = use_simple_mod
        self.reg_lambda = reg_lambda
        self.support_dist = support_dist
        self.use_cache = use_cache
        if state_space is None: state_space = torch.as_tensor([0.0, 1.0], device=device)
        self.state_space = state_space
        self.var_type = var_type

        if var_type == "categorical":
            self.length_scale = 2
            self.dist = self.ProductOfConditionalCategoricals
        else:
            self.length_scale = (state_space[1] - state_space[0]).item()
            self.dist = lambda x: self.ProductOfConditionalOrdinals(x, state_space)

        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self.cache = None
        self.acc_rates = []
        self.acc_hops = []
        self.proposed_hops = []
        self.av_time, self.time_n = 0.0, 0

    def step(self, x, model):
        start_time = time.time()
        input_shape = x.shape
        if self.var_type == "categorical":
            # reshaping to [n_batch, seq_len*n_states]. If model requires input to have
            # shape [n_batch, seq_len, n_states], then it should do the reshaping itself.
            x = x.view(x.size(0), -1)

        if self.use_cache and self.cache is not None:
            logp, logits = self.cache
        else:
            logp, g = fn_and_grad(model, x)
            logits = self.logits(x, g)
            self.cache = logp, logits

        # FORWARD
        fwd_dist = self.dist(logits)
        x_rev = fwd_dist.sample()

        # REVERSE
        logp_rev, g_rev = fn_and_grad(model, x_rev)
        reverse_logits = self.logits(x_rev, g_rev)
        rev_dist = self.dist(reverse_logits)

        # ACCEPTANCE PROBABILITY
        model_logratio = logp_rev - logp
        proposal_logratio = rev_dist.logp(x) - fwd_dist.logp(x_rev)
        la = model_logratio + proposal_logratio

        # ACCEPT/REJECT
        a = (la.exp() > torch.rand_like(la)).float()
        x_acc = x_rev * a[:, None] + x * (1. - a[:, None])

        if self.use_cache:
            logp = logp_rev * a + (logp * (1. - a))
            logits = reverse_logits * a[:, None, None] + (logits * (1. - a[:, None, None]))
            self.cache = logp, logits

        # self.acc_rates.append(a.mean().item())
        self.acc_rates.append((torch.abs(x - x_acc).sum(-1) > 0).float().mean().item())
        self.proposed_hops.append(torch.abs(x - x_rev).sum(-1).mean().item() / self.length_scale)
        self.acc_hops.append(torch.abs(x - x_acc).sum(-1).mean().item() / self.length_scale)

        new_time = time.time() - start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1

        return x_acc.view(input_shape)

    def logits(self, s, g, temp=2.0):

        if self.var_type == "categorical":
            logits = (g/temp) + (s/self.epsilon) - (1/(2*self.epsilon))  # (n, d*k)
            logits = logits.view(logits.size(0), self.n_dims, -1)  # (n, d, k)
        else:
            logits = (g/temp) + (s/self.epsilon)  # (n, d)
            logits = logits.unsqueeze(-1) * self.state_space  # (n, d, k)
            logits -= (1/(2*self.epsilon)) * self.state_space ** 2  # (n, d, k)

        return logits

    def sample_proposal(self, s, model):
        logp, g = fn_and_grad(model, s)
        logits = self.logits(s, g)
        fwd_dist = self.ProductOfConditionalOrdinals(logits, self.state_space)
        xrev = fwd_dist.sample()
        return xrev

    class ProductOfConditionalOrdinals(nn.Module):

        def __init__(self, logits, state_space):
            super().__init__()
            self.state_space = state_space  # (k, )
            self.logits = logits

        def logp(self, s):
            return Ordinal(logits=self.logits, state_space=self.state_space, product=True).log_prob(s)  # (n_batch, )

        def sample(self, n=torch.Size()):
            return Ordinal(logits=self.logits, state_space=self.state_space).sample(n)  # (n_batch, n_dims)

    class ProductOfConditionalCategoricals(nn.Module):

        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.dist = dists.OneHotCategorical(logits=self.logits, validate_args=False)
            self.shp = self.logits.shape

        def logp(self, s):
            return self.dist.log_prob(s.view(self.shp)).sum(-1)  # (n_batch, )

        def sample(self, n=torch.Size()):
            samples = self.dist.sample(n)  # (n_batch, n_dims*n_states)
            if n == torch.Size():
                return samples.view(self.logits.size(0), -1)
            else:
                return samples.view(n[0], self.logits.size(0), -1)
