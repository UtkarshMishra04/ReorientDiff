import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

from models.diffusion.utils.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    ConcatSquashLinear
)

class PointwiseNet(nn.Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()

        condition_dim = context_dim
        time_embed_dim = int(context_dim/2)*2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, condition_dim * 2),
            nn.Mish(),
            nn.Linear(condition_dim * 2, condition_dim),
        )

        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            ConcatSquashLinear(3, 128, context_dim + context_dim),
            ConcatSquashLinear(128, 256, context_dim + context_dim),
            ConcatSquashLinear(256, 512, context_dim + context_dim),
            ConcatSquashLinear(512, 256, context_dim + context_dim),
            ConcatSquashLinear(256, 128, context_dim + context_dim),
            ConcatSquashLinear(128, 3, context_dim + context_dim),
        ])

    def forward(self, x, time, context=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)

        if time.ndim > 1:
            time = time.squeeze(-1)

        time_emb = self.time_mlp(time).view(batch_size, 1, -1)

        if context is None:
            context = torch.zeros_like(time_emb)
            
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out