import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from models.diffusion.utils.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, num_grasps, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x num_grasps ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x num_grasps ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class GraspNet(nn.Module):

    def __init__(
        self,
        num_grasps,
        grasp_dim,
        condition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [grasp_dim, *map(lambda m: dim * m, dim_mults)]
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
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim+condition_dim, num_grasps=num_grasps),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim+condition_dim, num_grasps=num_grasps),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                num_grasps = num_grasps // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim+condition_dim, num_grasps=num_grasps)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim+condition_dim, num_grasps=num_grasps)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim+condition_dim, num_grasps=num_grasps),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim+condition_dim, num_grasps=num_grasps),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                num_grasps = num_grasps * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, grasp_dim, 1),
        )

    def forward(self, x, time, cond=None):
        '''
            x : [ batch x num_grasps x grasp_dim ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        # print(x.shape, time.shape)

        if time.ndim > 1:
            time = time.squeeze(-1)
        elif time.ndim == 0:
            time = time.unsqueeze(0)

        t = self.time_mlp(time)

        # print(x.shape, t.shape, time.shape)

        h = []

        if cond is None:
            cond = torch.zeros_like(t)

        t = torch.cat((t, cond), dim=-1)

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        # print(x.shape)

        return x

def main():

    batch_size = 10
    num_grasps = 16
    grasp_dim = 7
    pose_dim = 7

    diff_net = GraspNet(
        num_grasps=num_grasps,
        grasp_dim=grasp_dim,
        condition_dim = pose_dim,
    )

    random_input = torch.rand(batch_size, num_grasps, grasp_dim)
    random_condition = torch.rand(batch_size, pose_dim)
    random_time = torch.rand(batch_size, 1)

    output_none = diff_net(random_input, random_time)
    output_cond = diff_net(random_input, random_time, random_condition)

if __name__=="__main__":
    main()
