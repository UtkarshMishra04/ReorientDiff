#!/usr/bin/env python

import os
import random
import numpy as np
import torch
import torch.nn as nn

from models.classifier import ScoreModel
from models.unet1D import ScoreNet
from mixed_diffusion.cond_diffusion1D import VarianceSchedule, Diffusion
from mixed_diffusion.grad_discriminator import GradDiscriminator

def main():

    batch_size = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_samples = 10
    sample_dim = 7
    condition_dim = 64

    diffnet = ScoreNet(
        num_samples=num_samples,
        sample_dim=sample_dim,
        condition_dim=condition_dim,
    ).to(device)

    diffusion = Diffusion(
        net=diffnet,
    ).to(device)

    out_channels = 128
    image_channels = 4
    constant_dim = 7
    mlp_channels = out_channels + sample_dim + constant_dim

    # two score models are used in the paper: Picking Grasp Feasibility and Placement Grasp Feasibility 
    score_models = [
        ScoreModel(
            out_channels=out_channels,
            image_channels=image_channels,
            mlp_channels=mlp_channels,
        ).to(device)
    ]

    # conditions for each score model
    conditions = [
        {
            "image": torch.randn(batch_size, image_channels, 224, 224).to(device), # include RGB+heightmap
            "constants": torch.randn(batch_size, constant_dim).to(device), # include other object information
        }
    ]

    # initialize the discriminator for gradient calculations
    grad_fn = GradDiscriminator(score_models, conditions, batch_size)

    # random evaluation condition (can handle batches of conditions)
    # in the paper, we use the scene representation as the condition
    # which is formulated with CLIP encodings from HuggingFace CLIP
    # and final placement information
    random_condition = torch.randn(batch_size, condition_dim).to(device)

    # mask = torch.zeros(num_samples, sample_dim) #.to(device)
    # mask[:, int(0.8*sample_dim):] = 1
    # true_samples = torch.ones(num_samples, sample_dim) #.to(device)

    sampled_solutions, all_diffusion = diffusion.sample(condition=random_condition,
                                        num_samples=num_samples, 
                                        sample_dim=sample_dim,
                                        grad_fn=grad_fn, 
                                        # replace=torch.cat([mask, true_samples], dim=-1),
                                        num_steps=200,
                                        return_diffusion=True
                                    )

    # Optional Refinement (Not used in the paper)
    selected_samples = grad_fn.last_step_discrimination(sampled_solutions, num_grad_steps=10)
    
    # Mandatory: Select the top n samples for final sequential evaluation in the environment
    selected_samples = grad_fn.order_for_model(sampled_solutions, score_models[0], conditions[0], n_top=3)

    print("Final selected samples shape.", selected_samples.shape)

if __name__ == "__main__":
    main()
