import numpy as np
import time
import torch
import torch.distributions as dists
import torch.nn as nn

from distributions.discrete import Ordinal
from utils.utils import fn_and_grad, mask_invalid, numpify, torchify, get_matrix_from_poly2_coefs

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class AbstractAuxiliarySampler:

    def __init__(self,
                 n_dims,
                 epsilon,
                 variable_type="binary",
                 n_categorical_states=None,
                 state_space=None,
                 use_cache=True,
                 device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu'),
                 save_dir=None
                 ):

        self.n_dims = n_dims
        self._epsilon = epsilon
        self.n_categorical_states = n_categorical_states
        self.save_dir = save_dir
        self.device = device
        self.use_cache = use_cache

        self.iter = 0
        self.av_time, self.time_n = 0.0, 0  # track average time of self.step()
        self.cache = None
        self.acc_rates = []
        self.proposed_hops = []
        self.acc_hops = []

        # Define p(z|s) and q(s|z)s
        self.pz_given_s = self.IdentityCovGaussian

        self.variable_type = variable_type
        if variable_type == "binary":
            self.state_space = torch.as_tensor([0.0, 1.0], device=device)
            self.length_scale = 1
            self.ps_given_z = self.ProductOfConditionalBernoullis

        elif variable_type == "ordinal":
            self.state_space = state_space
            self.length_scale = (state_space[1] - state_space[0]).item()
            def pco(*args):
                return self.ProductOfConditionalOrdinals(*args, state_space=state_space)
            self.ps_given_z = pco

        elif variable_type == "categorical":
            self.state_space = torch.as_tensor([0.0, 1.0], device=device)
            self.length_scale = 2.
            self.ps_given_z = self.ProductOfConditionalCategoricals

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    def get_discrete_conditional_logits(self):
        raise NotImplementedError

    def _mask_invalid(self, H):
        """For categorical variables, we mask cross-terms for states belonging to same dim"""
        nb, nd, nk = H.size(0), self.n_dims, self.n_categorical_states
        H = H.reshape(nb, nd * nk, nd * nk)
        Ik = torch.eye(nk, device=H.device)
        mask = 1 - torch.block_diag(*[1 - Ik for _ in range(nd)])  # (ndims*nstates, ndims*nstates)
        return H * mask

    def track_metrics(self, acc_mask, s, s_prop, s_accept, model_logratio=None, gauss_logratio=None, prop_logratio=None):

        new_time = time.time() - self.start_time
        self.av_time = (new_time + self.time_n * self.av_time) / (self.time_n + 1)
        self.time_n += 1
        self.iter += 1

        dims_to_sum = (1, 2) if len(s.shape) == 3 else 1
        self.acc_rates.append(acc_mask.mean().item())
        # self.acc_rates.append((torch.abs(s - s_accept).sum(dims_to_sum) > 0).float().mean().item())
        self.proposed_hops.append(torch.abs(s - s_prop).sum(dims_to_sum).mean().item() / self.length_scale)
        self.acc_hops.append(torch.abs(s - s_accept).sum(dims_to_sum).mean().item() / self.length_scale)

    class ProductOfConditionalBernoullis(nn.Module):

        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.dist = dists.Bernoulli(logits=self.logits)

        def log_prob(self, s):
            return self.dist.log_prob(s).sum(-1)

        def sample(self, n=torch.Size()):
            return self.dist.sample(n)

    class ProductOfConditionalOrdinals(nn.Module):

        def __init__(self, logits, state_space):
            super().__init__()
            self.logits = logits
            self.state_space = state_space  # (k, )
            self.dist = Ordinal(logits=self.logits, state_space=self.state_space, product=True)

        def log_prob(self, s):
            return self.dist.log_prob(s)  # (n_batch, )

        def sample(self, n=torch.Size()):
            return self.dist.sample(n)  # (n_batch, n_dims)

    class ProductOfConditionalCategoricals(nn.Module):

        def __init__(self, logits):
            super().__init__()
            self.logits = logits
            self.dist = dists.OneHotCategorical(logits=self.logits, validate_args=False)
            self.shp = self.logits.shape

        def log_prob(self, s):
            return self.dist.log_prob(s.view(self.shp)).sum(-1)  # (n_batch, )

        def sample(self, n=torch.Size()):
            samples = self.dist.sample(n)  # (n_batch, n_dims*n_states)
            if n == torch.Size():
                return samples.view(*self.logits.shape[:-2], -1)
            else:
                return samples.view(n, *self.logits.shape[:-2], -1)

    class IdentityCovGaussian(nn.Module):

        def __init__(self, mu):
            super().__init__()
            self.mu = mu
            self.scale_tril = torch.eye(mu.size(-1), device=mu.device)
            self.dist = dists.MultivariateNormal(loc=self.mu, scale_tril=self.scale_tril, validate_args=False)

        def logp(self, x):
            return self.dist.log_prob(x)

        def sample(self, n=torch.Size()):
            return self.dist.sample(n)


class AVGSampler(AbstractAuxiliarySampler):

    def __init__(self,
                 n_dims,
                 epsilon=1000,
                 variable_type="binary",
                 n_categorical_states=None,  # only needed if variable_type=="categorical"
                 state_space=None,
                 use_cache=True,
                 device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu'),
                 save_dir=None
                 ):

        super().__init__(n_dims=n_dims,
                         epsilon=epsilon,
                         variable_type=variable_type,
                         n_categorical_states=n_categorical_states,
                         state_space=state_space,
                         use_cache=use_cache,
                         device=device,
                         save_dir=save_dir)

    @torch.no_grad()
    def step(self, s, model):

        self.start_time = time.time()
        input_shape = s.shape
        if self.variable_type == "categorical":
            # reshaping to [n_batch, seq_len*n_states]. If model requires input to have
            # shape [n_batch, seq_len, n_states], then it should do the reshaping itself.
            s = s.view(s.size(0), -1)

        if self.use_cache and self.cache is not None:
            logp, g = self.cache
        else:
            logp, g = fn_and_grad(model, s)
            if self.use_cache: self.cache = logp, g

        ### FORWARD ###
        pz_given_s = self.pz_given_s(s * (2/self.epsilon)**0.5)
        z = pz_given_s.sample()
        fwd_logits = self.get_discrete_conditional_logits(z, g)
        ps_given_z = self.ps_given_z(fwd_logits)
        s_rev = ps_given_z.sample()

        ### REVERSE ###
        logp_rev, g_rev = fn_and_grad(model, s_rev)
        reverse_pz_given_s = self.pz_given_s(s_rev * (2/self.epsilon)**0.5)
        rev_logits = self.get_discrete_conditional_logits(z, g_rev)
        reverse_ps_given_z = self.ps_given_z(rev_logits)

        ### ACCEPT/REJECT ###
        model_logratio = logp_rev - logp
        gauss_logratio = reverse_pz_given_s.logp(z) - pz_given_s.logp(z)
        proposal_logratio = reverse_ps_given_z.log_prob(s) - ps_given_z.log_prob(s_rev)
        log_accept_prob = model_logratio + gauss_logratio + proposal_logratio

        acc_mask, s_accept = self.accept_reject(log_accept_prob, s, s_rev, logp_rev, g_rev)

        self.track_metrics(acc_mask, s, s_rev, s_accept, model_logratio, gauss_logratio, proposal_logratio)

        return s_accept.view(input_shape).clone()

    def get_discrete_conditional_logits(self, z, g):
        zprime = z * (2/self.epsilon)**0.5
        epsinv = -(1/self.epsilon)

        if self.variable_type == "binary":
            logits = g + zprime + epsinv  # (n_batch, n_dims)

        elif self.variable_type == "ordinal":
            quadratic_term = epsinv * (self.state_space ** 2)  # (k, )
            linear_term = (g + zprime).unsqueeze(-1) * self.state_space  # (n, d, k)
            logits = linear_term + quadratic_term  # (n, d, k)

        elif self.variable_type == "categorical":
            logits = g + zprime + epsinv
            logits = logits.view(*logits.shape[:-1], self.n_dims, self.n_categorical_states)  # (n_batch, n_dims, n_states)
        else:
            raise ValueError

        return logits

    def accept_reject(self, log_accept_prob, s, s_rev, logp_rev, g_rev):

        acc_mask = (log_accept_prob.exp() > torch.rand_like(log_accept_prob)).float()  # acceptance mask

        def update(old, new, a=acc_mask, dim=0):
            a_shape = [1 if i != dim else -1 for i in range(len(old.shape))]
            a = a.view(*a_shape)
            return (old * (1. - a)) + (new * a)

        s_accept = update(s, s_rev)

        if self.use_cache:
            logp, g = self.cache
            logp = update(logp, logp_rev)
            g = update(g, g_rev)
            self.cache = logp, g

        return acc_mask.squeeze(), s_accept


class PAVGSampler(AbstractAuxiliarySampler):

    def __init__(self,
                 n_dims,
                 epsilon,
                 adaptive_update_freq,
                 init_adapt_stepsize,
                 adapt_stepsize_decay,
                 num_iteration_before_fitting_M=1000,
                 init_precon_matrix=None,
                 allow_adaptation_of_precon_matrix=True,
                 postadapt_epsilon=None,
                 variable_type="binary",
                 n_categorical_states=None,
                 state_space=None,
                 use_cache=True,
                 device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu'),
                 save_dir=None
                 ):

        super().__init__(n_dims=n_dims,
                         epsilon=epsilon,
                         variable_type=variable_type,
                         n_categorical_states=n_categorical_states,
                         state_space=state_space,
                         use_cache=use_cache,
                         device=device,
                         save_dir=save_dir)

        self.adaptive_update_freq = adaptive_update_freq
        self.adapt_stepsize = init_adapt_stepsize
        self.adapt_stepsize_decay = adapt_stepsize_decay
        self.num_iteration_before_fitting_M = num_iteration_before_fitting_M
        self.postadapt_epsilon = postadapt_epsilon if postadapt_epsilon else epsilon

        if init_precon_matrix is None:
            if variable_type == "categorical":
                init_precon_matrix = torch.zeros((n_dims*n_categorical_states, n_dims*n_categorical_states))
            else:
                init_precon_matrix = torch.zeros((n_dims, n_dims))

        self.data_for_fitting_precon_mat = []  # chains, grads & logprobs
        self._precon_multiplier = 1.0  # adaptively learned
        self.cache_precon_eigendecomp(init_precon_matrix)
        self.allow_adaptation_of_precon_matrix = allow_adaptation_of_precon_matrix

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value
        self.cache_precon_params()

    @property
    def precon_multiplier(self):
        return self._precon_multiplier

    @precon_multiplier.setter
    def precon_multiplier(self, value):
        self._precon_multiplier = value
        self.cache_precon_params()

    def cache_precon_eigendecomp(self, precon_mat):
        # cache preconditioning matrix along with its eigendecomposition
        precon_mat = precon_mat.to(device=self.device)
        e, Q = torch.linalg.eigh(precon_mat)  # ascending order of eigvals
        self.precon_eigendecomp = (precon_mat, e, Q)
        self.cache_precon_params()

    def cache_precon_params(self):
        precon_mat, e, Q = self.precon_eigendecomp
        M = precon_mat * self.precon_multiplier
        escaled = e * self.precon_multiplier
        deps = (2 / self.epsilon) - min(escaled.min(), 0)
        Esqrt = torch.diag_embed((escaled + deps) ** 0.5)
        Meps_sqrt = Q @ (Esqrt @ Q.T)
        self.precon_params = M, Meps_sqrt, deps

    @torch.no_grad()
    def step(self, s, model, verbose=False):
        self.start_time = time.time()
        input_shape = s.shape
        if self.variable_type == "categorical":
            # reshaping to [n_batch, seq_len*n_states]. If model requires input to have
            # shape [n_batch, seq_len, n_states], then it should do the reshaping itself.
            s = s.view(s.size(0), -1)

        #### FORWARD #####
        M, Meps_sqrt, deps = self.precon_params
        if self.use_cache and self.cache is not None:
            logp, g = self.cache
        else:
            logp, g = fn_and_grad(model, s)
            self.cache = logp, g

        b = g - (s @ M.T)

        # continuous auxiliary distribution
        pz_given_s = self.pz_given_s(s @ Meps_sqrt.T)
        z = pz_given_s.sample()
        fwd_logits = self.get_discrete_conditional_logits(z, b, Meps_sqrt, deps)
        ps_given_z = self.ps_given_z(fwd_logits)
        s_prop = ps_given_z.sample()

        #### REVERSE #####
        logp_prop, g_prop = fn_and_grad(model, s_prop)
        b_prop = g_prop - (s_prop @ M.T)

        # reverse distributions
        reverse_pz_given_s = self.pz_given_s(s_prop @ Meps_sqrt.T)
        rev_logits = self.get_discrete_conditional_logits(z, b_prop, Meps_sqrt, deps)
        reverse_ps_given_z = self.ps_given_z(rev_logits)

        #### ACCEPT/REJECT ####
        model_logratio = logp_prop - logp
        gauss_logratio = reverse_pz_given_s.logp(z) - pz_given_s.logp(z)
        proposal_logratio = reverse_ps_given_z.log_prob(s) - ps_given_z.log_prob(s_prop)
        log_accept_prob = model_logratio + gauss_logratio + proposal_logratio

        acc_mask, s_accept = self.accept_reject(
            model=model,
            log_accept_prob=log_accept_prob,
            s=s,
            s_prop=s_prop,
            logp_prop=logp_prop,
            g_prop=g_prop
        )
        self.track_metrics(acc_mask, s, s_prop, s_accept, model_logratio, gauss_logratio, proposal_logratio)

        # ADAPTATION
        keep_adapting = self.adapt_stepsize > 1e-4
        if keep_adapting:
            if self.iter % self.adaptive_update_freq == 0:
                self.update_precon_multiplier()
            self.maybe_update_precon_mat(model, s_accept)

        if verbose:
            return s_accept.view(input_shape), s_prop.view(input_shape), acc_mask
        else:
            return s_accept.view(input_shape)

    def get_discrete_conditional_logits(self, z, b, Meps_sqrt, deps):
        zHsqrt = torch.matmul(z, Meps_sqrt.T)  # (n, d*k)
        if self.variable_type == "binary":
            logits = b + zHsqrt - 0.5 * deps  # (n, d)
        elif self.variable_type == "ordinal":
            linear_term = (b + zHsqrt).unsqueeze(-1) * self.state_space  # (n, d, k)
            quadratic_term = -0.5 * deps * (self.state_space ** 2)  # (k,)
            logits = linear_term + quadratic_term  # (n_batch, n_dims, k)
        elif self.variable_type == "categorical":
            logits = (b + zHsqrt - 0.5 * deps)
            logits = logits.view(*logits.shape[:-1], self.n_dims, self.n_categorical_states)  # (n, d, k)
        else:
            raise ValueError

        return logits

    def accept_reject(self, model, log_accept_prob, s, s_prop, logp_prop, g_prop):
        acc_mask = (log_accept_prob.exp() > torch.rand_like(log_accept_prob)).float()  # acceptance mask

        def update(old, prop, a=acc_mask):
            ones = [1] * (len(old.shape) - 1)
            a = a.view(-1, *ones)  # add extra dims to enable broadcasting
            return (old * (1. - a)) + (prop * a)

        s_accept = update(s, s_prop)

        # update cache of local parameters
        if self.use_cache:
            logp, g = self.cache
            logp_accept = update(logp, logp_prop)
            g_accept = update(g, g_prop)
            self.cache = (logp_accept, g_accept)

        return acc_mask.squeeze(), s_accept

    def update_precon_multiplier(self):
        """Adjust self.precon_multiplier to maximise the L1 distance jumped between successive states

         to do this, we define X = self.adaptive_update_freq and then compute
            cur_hops = average L1 distance jumped during last X iterations
            prev_hops = average L1 distance jumped between 2X and X iterations ago

         Then we update self.precon_multiplier based on the logic:
            cur_hops > prev_hops -> update multiplier in same direction as last time
            cur_hops < prev_hops? -> update multiplier in opposite direction to last time

         The magnitude of the update depends on self.adapt_stepsize, which is exponentially decayed during sampling.
         Specifically, every X iterations, we set self.adapt_stepsize *= self.adapt_stepsize_decay
         """
        window_size = self.adaptive_update_freq
        if len(self.acc_hops) < 2*window_size: return

        cur_hops = np.array(self.acc_hops[-window_size:]).mean()
        prev_hops = np.array(self.acc_hops[-2*window_size:-window_size]).mean()

        have_improved = cur_hops > prev_hops
        last_action = getattr(self, "prev_precon_mult_update", "increment")
        use_additive_adjustment = np.abs(self.precon_multiplier) < 1.0
        if (have_improved and last_action == 'decrement') or (not have_improved and last_action == 'increment'):
            if use_additive_adjustment:
                adjustment = -self.adapt_stepsize
            else:
                adjustment = -self.adapt_stepsize / (1. + self.adapt_stepsize)
            self.prev_precon_mult_update = 'decrement'
        else:
            adjustment = self.adapt_stepsize
            self.prev_precon_mult_update = 'increment'

        if use_additive_adjustment:
            self.precon_multiplier += adjustment
        else:
            self.precon_multiplier = self.precon_multiplier * (1 + adjustment)

        self.adapt_stepsize *= self.adapt_stepsize_decay
        if self.iter % 2500 == 0:
            print("itr:", self.iter,
                  "precon multiplier:", self.precon_multiplier,
                  "precon multiplier learn rate:", self.adapt_stepsize)

    def maybe_update_precon_mat(self, model, s):

        if self.allow_adaptation_of_precon_matrix:
            logp, g = self.cache

            # totally ignore the first 100 iterations, then start collecting samples for fitting M
            if self.iter > 100:
                self.data_for_fitting_precon_mat.append(
                    [s.detach().clone().cpu(), g.detach().clone().cpu(), logp.detach().clone().cpu()]
                )
            if len(self.data_for_fitting_precon_mat) == self.num_iteration_before_fitting_M:
                self.epsilon = self.postadapt_epsilon
                self.fit_preconditioning_mat(model, verbose=True)
                self.allow_adaptation_of_precon_matrix = False
                del self.data_for_fitting_precon_mat

    @torch.no_grad()
    def fit_preconditioning_mat(self, model, verbose=False):

        chains, grads, logps = zip(*self.data_for_fitting_precon_mat)
        chains = torch.stack(chains, dim=0)  # (num_steps, num_chains, num_dims)
        grads = torch.stack(grads, dim=0)  # (num_steps, num_chains, num_dims)
        logps = torch.stack(logps, dim=0)  # (num_steps, num_chains)
        if verbose:
            print("!" * 20 + f"\n SETTING PAVG COVARIANCE MATRIX WITH {np.prod(chains.shape[:2])} SAMPLES \n" + "!" * 20)

        # fit scaled covariance
        cov = torch.cov(chains.view(-1, chains.shape[-1]).T)  # (D, D)
        cov_diag = torch.diag(cov)
        is_zero_var = torch.isclose(cov_diag, torch.zeros_like(cov_diag), atol=1e-2)
        cov[is_zero_var, is_zero_var] += 0.01  # avoid instability (in eigval decomp or inversion)

        scaled_cov, cov_loss = self.modify_quadratic_approx(chains, grads, logps, cov, verbose=verbose)

        # fit scaled precision matrix
        precision = torch.inverse(cov)
        scaled_precision, prec_loss = self.modify_quadratic_approx(chains, grads, logps, precision, verbose=verbose)

        # pick best fit & cache
        P = scaled_cov if cov_loss <= prec_loss else scaled_precision
        if self.variable_type == "categorical":
            P = mask_invalid(P, self.n_dims, self.n_categorical_states)
        self.cache_precon_eigendecomp(P)

    def modify_quadratic_approx(self, s, g, logp, H, verbose=False):

        snext, logp_next = s[1:], logp[1:]
        s, logp, g = s[:-1], logp[:-1], g[:-1].view(-1, g.shape[-1])

        s_diff = (snext - s).view(-1, s.shape[-1])  # ((n_steps-1)*n_chains, n_dims)
        logp_diff = (logp_next - logp).view(-1)  # ((n_steps-1)*n_chains,)

        H, lr_loss = PAVGSampler.learn_quadratic_approximation(
            s_diff=s_diff,
            logp_diff=logp_diff,
            g=g,
            H=H,
            loss_type="lstq",
            verbose=verbose,
            return_loss=True,
        )
        return H, lr_loss

    def sample_proposal(self, s, model):
        H, Hd_sqrt, deps = self.precon_params
        logp, g = fn_and_grad(model, s)
        b = g - (s @ H.T)
        pz_given_s = self.pz_given_s(s @ Hd_sqrt.T)
        z = pz_given_s.sample()
        logits = self.get_discrete_conditional_logits(z, b, Hd_sqrt, deps)
        ps_given_z = self.ps_given_z(logits)
        s_prop = ps_given_z.sample()
        return s_prop

    @staticmethod
    @torch.no_grad()
    def learn_quadratic_approximation(s_diff,
                                      logp_diff,
                                      g,
                                      H,
                                      loss_type="lstq",
                                      feats_type="simple",
                                      verbose=False,
                                      return_loss=False,
                                      ):
        """
        Use linear least squares to fit param b for the following problem:

            (Least-squares loss) (1/k) \Sum_i^k [ f(s_i; a, b) - t_i ]^2
            (targets) t_i = logp_diff_i - <g_i, s_i>
            (Linear model)
                if feats_type == 'simple':   f(s; a, b) =  a + (1/2) b <s, H s>
                if feats_type == 'full':     f(s; a, H) =  a + (1/2) <s, H s>

        s_diff: tensor of shape (n, d)
        logp_diff: tensor of shape (n, )
        g: tensor of shape (n, d)
        H: tensor of shape (n, d, d) or (d, d)
        mode: if 'simple', learn a single scalar, if 'full', learn a symmetric matrix.
        e_min: tensor of shape (n,) or () - min eigval of H used as part of regularisation (only matters if lamb > 0).
        nonneg_constrained: if True, we enfoce a, b, c all greater than zero.
        share_params_across_data: if True, we share (a, b, c) across all regression problems
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import HuberRegressor
        from sklearn.linear_model._huber import _huber_loss_and_gradient

        n, d = s_diff.shape
        target = logp_diff - (g * s_diff).sum(-1)  # (n, )

        if feats_type == "simple":
            Hs = (s_diff @ H.T)  # (n, d)
            f1 = 0.5 * (s_diff * Hs).sum(-1)  # (n,)
            design_mat = torch.cat([torch.ones_like(f1)[:, None], f1[:, None]], dim=1)  # (n, 2)
        elif feats_type == "full":
            poly = PolynomialFeatures(degree=(2, 2), include_bias=True)  # no first-order terms
            design_mat = 0.5 * torchify(poly.fit_transform(numpify(s_diff)), device=target.device)  # (n, 1 + d(d-1)/2)
        else:
            raise ValueError(f"do not recognise feats_type {feats_type}")

        if loss_type == "huber":
            X, y = numpify(design_mat), numpify(target)
            huber = HuberRegressor().fit(X, y)
            coef = torchify(huber.coef_, design_mat.device)
            huber_params = np.concatenate([huber.coef_, np.array([huber.intercept_]), np.array([huber.scale_])])
            loss, _ = _huber_loss_and_gradient(huber_params, X, y, huber.epsilon, huber.alpha, np.ones_like(y))
        elif loss_type == "lstq":
            res = torch.linalg.lstsq(design_mat, target)
            coef = res[0]
            loss = res[1]
            if loss.numel() == 0:  # sometimes the lstq solver doesn't return the loss...
                loss = (((design_mat * coef).sum(-1) - target) ** 2).mean(-1)  # (n,)
        else:
            raise ValueError(f"do not recognise loss_type {loss_type}")

        if verbose:
            print(f"regression cost = ", loss)
            if feats_type == "simple": print("regression slope = ", coef)

        if feats_type == "simple":
            coef = coef[1]
            H_new = coef * H
        elif feats_type == "full":
            H_new = torchify(get_matrix_from_poly2_coefs(coef, d))
        else:
            raise ValueError

        if return_loss:
            return H_new, loss
        else:
            return H_new


class BGAVSampler(AbstractAuxiliarySampler):
    """Block-Gibbs auxiliary-variable Sampler as defined in Martens & Sutskever and developed by Zhang et al.
    https://proceedings.mlr.press/v9/martens10a.html
    https://proceedings.neurips.cc/paper/2012/file/c913303f392ffc643f7240b180602652-Paper.pdf
    """

    def __init__(self,
                 n_dims,
                 model_name="",
                 variable_type="binary",
                 use_cache=True,
                 n_categorical_states=None,
                 state_space=None,
                 device=torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
                 ):

        super().__init__(n_dims=n_dims,
                         epsilon=1000.0,
                         variable_type=variable_type,
                         n_categorical_states=n_categorical_states,
                         state_space=state_space,
                         use_cache=use_cache,
                         device=device)

        self.model_name = model_name

    def step(self, s, model):
        self.start_time = time.time()

        if self.use_cache and self.cache is not None:
            b, Hd_sqrt, d = self.cache
        else:
            b, Hd_sqrt, d = self.get_cachable_params(model)
            self.cache = b, Hd_sqrt, d

        pz_given_s = self.pz_given_s(s @ Hd_sqrt.T)
        z = pz_given_s.sample()

        logits = self.get_discrete_conditional_logits(z, b, Hd_sqrt, d)
        ps_given_z = self.ps_given_z(logits)
        sprop = ps_given_z.sample()

        self.track_metrics(acc_mask=torch.ones_like(s[:, 0]), s=s, s_prop=sprop, s_accept=sprop)

        return sprop

    def get_cachable_params(self, model):

        if self.model_name.lower() == "boltzmann":
            b = model.bias
            H = model.J
        elif "ising" in self.model_name.lower():
            # model is computed as f(x) = bias^T x + x^T J x, where x has domain {-1, 1}
            # after inserting s = 2x-1 (i.e. change domain to {0, 1}) we can re-write it as
            # f(s) = (2*bias - 4J.sum(0))^T x + (1/2) x^T (8J) x
            b = 2 * model.bias - 4 * (model.J.sum(0))
            H = 8 * model.J.clone()
        else:
            raise ValueError(f"Don't recognise model {self.model_name}")

        e, Q = torch.linalg.eigh(H)
        min_e = e.min(-1, keepdims=True)[0]  # (n_batch,)
        d = (1 / self.epsilon) - torch.minimum(min_e, torch.zeros_like(min_e))
        d = d.tile(H.size(-1))
        e = e + d
        Hd_sqrt = Q @ torch.diag_embed(e ** 0.5) @ Q.T

        return b, Hd_sqrt, d

    def get_discrete_conditional_logits(self, z, b, Hd_sqrt, d):

        if self.variable_type == "binary":
            zrecon = z @ Hd_sqrt.T
            logits = b + zrecon - (0.5 * d)  # (n_batch, n_dims)
        else:
            raise NotImplementedError

        return logits
