#!/usr/bin/env python

##############################################################################
# Code derived from "Planning with Diffusion for Flexible Behavior Synthesis"
# Janner et al. (2022) https://diffusion-planning.github.io/
##############################################################################

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from models.utils.helpers import (
    SinusoidalPosEmb,
    DownsampleLinear,
    UpsampleLinear,
    Linear1DBlock,
)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, num_samples):
        super().__init__()

        self.blocks = nn.ModuleList([
            Linear1DBlock(inp_channels, out_channels),
            Linear1DBlock(out_channels, out_channels),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
        )

        self.residual_net = nn.Linear(inp_channels, out_channels) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_net(x)

class ScoreNet(nn.Module):

    def __init__(
        self,
        num_samples,
        sample_dim,
        condition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [sample_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = condition_dim
        time_embed_dim = int(condition_dim/2)*2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, condition_dim * 2),
            nn.Mish(),
            nn.Linear(condition_dim * 2, condition_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim+condition_dim, num_samples=num_samples),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim+condition_dim, num_samples=num_samples),
                DownsampleLinear(dim_out, dim_out, embed_dim=time_dim+condition_dim) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim+condition_dim, num_samples=num_samples)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim+condition_dim, num_samples=num_samples)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim+condition_dim, num_samples=num_samples),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim+condition_dim, num_samples=num_samples),
                UpsampleLinear(dim_in, dim_in, embed_dim=time_dim+condition_dim) if not is_last else nn.Identity()
            ]))

        self.final_linear = nn.Linear(in_out[0][1], in_out[0][0])

    def forward(self, x, time, cond=None):
        '''
            x : [ batch x sample_dim ]
        '''

        if time.ndim > 1:
            time = time.squeeze(-1)
        elif time.ndim == 0:
            time = time.unsqueeze(0)

        t = self.time_mlp(time)

        h = []

        if cond is None:
            cond = torch.zeros_like(t)

        t = torch.cat((t, cond), dim=-1)

        num_resolutions = len(self.downs)

        for ind, (resnet, resnet2, downsample) in enumerate(self.downs):
            is_last = ind >= (num_resolutions - 1)
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x, t) if not is_last else downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        num_resolutions = len(self.ups)

        for ind, (resnet, resnet2, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x, t)

        x = self.final_linear(x)

        return x
