#!/usr/bin/env python

##############################################################################
# Code derived from "Learning Object Reorientation for Specific-Posed Placement"
# Wada et al. (2022) https://github.com/wkentaro/reorientbot
##############################################################################

import numpy as np
import torch


class ConvEncoder(torch.nn.Module):
    def __init__(self, 
                out_channels, # M
                image_channels, # C
                mlp_channels # N
                ):
        super().__init__()

        # heightmap: 1
        in_channels = image_channels
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, 4, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, stride=4),
        )

        in_channels = mlp_channels
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        image_input, # B x C x H x W
        mlp_input, # B x A x P, A = num_samples, P = sample_dim
        constant_mlp_input, # B x Q, Q = constant_dim
    ):
        B, _, H, W = image_input.shape
        _, A, _ = mlp_input.shape

        h_obs = self.encoder(image_input) # B x M x 1 x 1
        h_obs = h_obs.reshape(B, h_obs.shape[1]) # B x M

        h_obs = torch.cat([h_obs, constant_mlp_input], dim=1) # B x M + Q

        h_obs = h_obs[:, None, :].repeat(1, A, 1) # B x A x M + Q

        h = torch.cat([h_obs, mlp_input], dim=2) # B x A x M + Q + P

        h = self.mlp(h) # B x A x M

        return h


class ScoreModel(torch.nn.Module):
    def __init__(self, out_channels=128, image_channels=1, mlp_channels=64):
        super().__init__()

        self.encoder = ConvEncoder(
            out_channels=out_channels,
            image_channels=image_channels,
            mlp_channels=mlp_channels,
        )
        self.fc_scores = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 3),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        samples, # B x A x P, A = num_samples, P = sample_dim
        image, # B x C x H x W
        constants, # B x Q, Q = constant_dim
    ):

        samples = samples.to(image.device)

        h = self.encoder(
            image,
            samples,
            constants,
        ) # B x A x M
        
        scores = self.fc_scores(h) # B x A x 3
        scores = torch.prod(scores, dim=2) # B x A

        return scores