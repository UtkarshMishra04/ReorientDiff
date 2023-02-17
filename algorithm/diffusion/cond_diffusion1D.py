import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np

from algorithm.diffusion.utils.diff_utils import *
from models.diffusion.unet1D import GraspNet
from models.diffusion.pointwisenet import PointwiseNet
from algorithm.diffusion.sde_cont import PluginReverseSDE


class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffusionGrasps(Module):

    def __init__(self, net:GraspNet):
        super().__init__()
        self.net = net
        self.gensde = PluginReverseSDE(scorefunc=net)

    def get_loss(self, x_0, condition=None):
        """
        Args:
            x_0:  Input grasps, (B, N, d).
            condition:  Shape latent, (B, F).
        """

        loss = self.gensde.dsm(x_0, condition=condition).mean()

        return loss

    @torch.no_grad()
    def sample(self, condition, num_grasps=16, grasp_dim=7, num_steps=256, return_diffusion=False, grad_fn=None, replace=None):

        batch_size = condition.size(0)

        condition = condition.repeat(num_grasps, 1)

        if replace is not None:
            replace = torch.Tensor(replace).repeat(batch_size*int(num_grasps/replace.shape[0]), 1).to(condition.device)

        x_T = torch.randn([num_grasps*batch_size, grasp_dim]).to(condition.device)
        diffusion = {num_steps: x_T.cpu().numpy()}

        sde = self.gensde
        delta = sde.T / num_steps
        sde.base_sde.dt = delta
        ts = torch.linspace(1, 0, num_steps + 1).to(x_T) * sde.T
        ones = torch.ones(num_grasps*batch_size, 1).to(x_T)/num_steps

        for t in range(num_steps, 0, -1):
            xt = sde.sample(ones * t, x_T, condition, grad_fn, replace=replace)

            # if grad_fn is not None:
            #     xt = xt.cpu().numpy()
            #     xt[:, :7] = grad_fn.return_mapped_poses(xt[:, :7])
            #     xt = torch.Tensor(xt).to(condition.device)

            if replace is not None:
                h, w = replace.size(0), replace.size(1)
                xt[:h, :w] = replace

            x_T = xt

            diffusion[t - 1] = xt.cpu().numpy().reshape(batch_size, num_grasps, grasp_dim)
        
        if return_diffusion:
            return diffusion[0], diffusion
        else:
            return diffusion[0]

class DiffusionPointCloud(Module):

    def __init__(self, net:PointwiseNet):
        super().__init__()
        self.net = net
        self.gensde = PluginReverseSDE(scorefunc=net)

    def get_loss(self, x_0, condition=None):
        """
        Args:
            x_0:  Input points, (B, N, d).
            condition:  Shape latent, (B, F).
        """

        loss = self.gensde.dsm(x_0, condition=condition).mean()

        return loss

    @torch.no_grad()
    def sample(self, condition, num_points=16, point_dim=3, num_steps=256, return_diffusion=False):
        batch_size = condition.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(condition.device)
        diffusion = {num_steps: x_T.cpu().numpy()}

        sde = self.gensde
        delta = sde.T / num_steps
        sde.base_sde.dt = delta
        ts = torch.linspace(1, 0, num_steps + 1).to(x_T) * sde.T
        ones = torch.ones(batch_size, 1, 1).to(x_T)/num_steps

        for t in range(num_steps, 0, -1):
            xt = sde.sample(ones * t, x_T, condition)
            x_T = xt

            diffusion[t - 1] = xt.cpu().numpy()
        
        if return_diffusion:
            return diffusion[0], diffusion
        else:
            return diffusion[0]


class DiffusionGraspsRefactored(Module):

    def __init__(self, net:GraspNet, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, condition=None, t=None):
        """
        Args:
            x_0:  Input grasps, (B, N, d).
            condition:  Shape latent, (B, F).
        """
        batch_size, _, grasp_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, time=torch.Tensor(t).unsqueeze(-1).to(x_0.device), cond=condition)

        loss = F.mse_loss(e_theta.reshape(-1, grasp_dim), e_rand.reshape(-1, grasp_dim), reduction='mean')
        return loss

    def sample(self, condition, num_grasps=16, grasp_dim=7, flexibility=0.0, return_diffusion=False):
        batch_size = condition.size(0)
        x_T = torch.randn([batch_size, num_grasps, grasp_dim]).to(condition.device)
        diffusion = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = diffusion[t]
            time = torch.Tensor([t]*batch_size).unsqueeze(-1).to(condition.device)
            e_theta = self.net(x_t, time=time, cond=condition)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            diffusion[t-1] = x_next.detach()     # Stop gradient and save diffusionectory.
            diffusion[t] = diffusion[t].cpu()         # Move previous output to CPU memory.
            if not return_diffusion:
                del diffusion[t]
        
        if return_diffusion:
            return diffusion
        else:
            return diffusion[0]

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_batch = 10
    num_grasps = 16
    grasp_dim = 7
    condition_dim = 256

    diff_net = GraspNet(
        num_grasps=num_grasps,
        grasp_dim=grasp_dim,
        condition_dim = condition_dim
    ).to(device)

    grasp_diff = DiffusionGrasps(diff_net).to(device)

    condition = torch.randn(test_batch, condition_dim).to(device)

    x_0 = torch.randn(test_batch, num_grasps, grasp_dim).to(device)

    loss = grasp_diff.get_loss(x_0, condition=condition)

    print(loss)

if __name__ == '__main__':
    main() 