#!/usr/bin/env python

import torch
import torch.nn as nn
from mixed_diffusion.utils.cont_utils import sample_v, log_normal, sample_vp_truncated_q
import numpy as np

PI = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

class VariancePreservingSDE(torch.nn.Module):
    ###############################################
    # Implementation of the variance preserving SDE proposed by Song et al. 2021
    # See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    ###############################################

    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0, N = 256, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon
        self.dt = T/N
        self.N = N
        self.s = 0

    def cosine_schedule(self, t, s = 0.008):
        self.s = s
        return torch.cos((t+s)/(1+s) * 0.5 * np.pi)**2

    def tm1(self, t):
        return t-self.dt.to(t)

    def beta(self, t):
        return torch.clip(1 - self.alpha(t)/self.alpha(t - self.dt.to(t)), min=0, max=0.999) #torch.clip( (PI/(1+self.s)) * torch.tan((t+self.s)/(1+self.s) * 0.5 * PI), max=0.999) #torch.clip(1 - self.alpha(t)/self.alpha(self.tm1(t)), max=0.999)

    def alpha(self, t):
        if t.squeeze().ndim == 0:
            if t.squeeze() == 1:
                t = t*257/256

        return self.cosine_schedule(t)

    def mean_weight(self, t):
        return self.alpha(t)**0.5

    def std(self, t):
        return (1. - self.alpha(t))**0.5

    def f(self, t, y):
        return - 0.5 * self.beta(t) * y

    def g(self, t, y):
        beta_t = self.beta(t)
        return torch.ones_like(y) * beta_t**0.5

    def posterior_mean_coefs(self, t):
        return torch.sqrt(self.alpha(t/self.N))*self.beta((t+1)/self.N)/(1-self.alpha((t+1)/self.N)), torch.sqrt(1 - self.beta((t+1)/self.N))*(1 - self.alpha(t/self.N))/(1-self.alpha((t+1)/self.N))

    def posterior_var_coef(self, t):
        return self.beta((t+1)/self.N)*(1 - self.alpha(t/self.N))/(1-self.alpha((t+1)/self.N))

    def sample(self, t, y0, return_noise=False):
        ###############################################
        # sample yt | y0
        # if return_noise=True, also return std and g for reweighting the denoising score matching loss
        ###############################################
        mu = self.mean_weight(t) * y0
        std = self.std(t)
        epsilon = torch.randn_like(y0)
        yt = epsilon * std + mu

        if not return_noise:
            return yt
        else:
            return yt, epsilon, std

class PluginReverseSDE(torch.nn.Module):
    ###############################################
    # inverting a given base sde with drift `f` and diffusion `g`, and an inference sde's drift `a` by
    # f <- g a - f
    # g <- g
    # (time is inverted)
    ###############################################

    def __init__(self, scorefunc, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001, N = 256, vtype='rademacher'):
        super().__init__()
        self.score_fn = scorefunc
        self.T = torch.nn.Parameter(torch.FloatTensor([T]), requires_grad=False)
        self.vtype = vtype
        self.N = N
        self.base_sde = VariancePreservingSDE(beta_min, beta_max, self.T, t_epsilon, N)
        self.dsm_loss = nn.MSELoss()

    # Drift
    def mu(self, t, y, lmbd=0.):
        return (1. - 0.5 * lmbd) * self.base_sde.g(self.T-t, y) * self.score_fn(y, self.T - self.modify_t(t)) - \
               self.base_sde.f(self.T - t, y)

    # Diffusion
    def sigma(self, t, y, lmbd=0.):
        return (1. - lmbd) ** 0.5 * self.base_sde.g(self.T-t, y)

    def modify_t(self, t):
        return (self.N*t).int().squeeze()

    def x0_2_ytm1(self, t, y):
        pred_x0 = self.score_fn(y, self.modify_t(t))
        pred_x0 = torch.clip(pred_x0, -1, 1)
        ytm1 = self.base_sde.mean_weight(self.base_sde.tm1(t))*pred_x0 + self.base_sde.std(self.base_sde.tm1(t))*score

        return ytm1

    def x0_2_posteriormean(self, t, y):
        pred_x0 = self.score_fn(y, t.squeeze())
        pred_x0 = torch.clip(pred_x0, -1, 1)
        mean_coef1, mean_coef2 = self.base_sde.posterior_mean_coefs(t)

        return mean_coef1*pred_x0 + mean_coef2*y

    def posterior_variance(self, t, y):
        return self.base_sde.posterior_var_coef(t)
        
    def posterior_log_varaince(self, t, y):
        return torch.log(torch.clamp(self.posterior_variance(t, y), min=1e-20))

    def x0_2_epsilon(self, t, y, condition, grad_fn=None, replace=None, return_predx = False):
        ts = t.squeeze().unsqueeze(-1)/self.N
        pred_x0 = (1+0.1)*self.score_fn(y, ts, condition) - 0.1*self.score_fn(y, ts, None) 

        if replace is not None:
            mask, value = replace[:, :, :pred_x0.shape[-1]], replace[:, :, pred_x0.shape[-1]:]
            mask = mask.reshape(-1, mask.shape[-1])
            value = value.reshape(-1, value.shape[-1])
            pred_x0[mask.long()] = value[mask.long()]
        
        alpha_t = self.base_sde.alpha(t/self.N)
        epsilon = (y - torch.sqrt(alpha_t)*pred_x0) / torch.sqrt(1 - alpha_t)
        epsilon = torch.clip(epsilon, -1, 1)

        with torch.enable_grad():
            if grad_fn is not None:
                for _ in range(2):
                    grad_value = -grad_fn.calc_grad(pred_x0)
                    grad_value = 0.01*torch.clip(grad_value, -1, 1)
                    pred_x0 = pred_x0 - grad_value
            else:
                grad_value = torch.zeros_like(epsilon)

        next_epsilon = epsilon - torch.sqrt(1 - alpha_t)*grad_value
        next_epsilon = torch.clip(next_epsilon, -1, 1)
        next_pred_x0 = (y - torch.sqrt(1 - alpha_t)*next_epsilon) / torch.sqrt(alpha_t)
        
        if return_predx:
            return next_epsilon, next_pred_x0, alpha_t
        else:
            return next_epsilon 

    def sample(self, t, y, condition, grad_fn=None, replace=None):
        # print("ddim sampling: ", t[0], t.shape)
        epsilon, pred_x0, alpha_t = self.x0_2_epsilon(t, y, condition, grad_fn, replace=replace, return_predx=True)
        alpha_tm1 = self.base_sde.alpha((t-1)/self.N)
        
        return torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*epsilon

    @torch.enable_grad()
    def dsm(self, x, condition):
        ###############################################
        # denoising score matching loss
        ###############################################
        
        t_ = torch.rand([x.size(0), ] + [1 for _ in range(x.ndim - 1)]).to(x) * self.T
        y, target, std = self.base_sde.sample(t_, x, return_noise=True)
    
        # print(target.shape, std.shape, x.shape, y.shape)

        if np.random.rand() > 0.3:
            with_condition = self.score_fn(y, t_.squeeze(), condition)
            without_condition = self.score_fn(y, t_.squeeze())

            pred_x0 = 4*with_condition - 3*without_condition

        else:
            pred_x0 = self.score_fn(y, t_.squeeze())

        return self.dsm_loss(pred_x0, x)
