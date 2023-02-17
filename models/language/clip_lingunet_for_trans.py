import torch
import torch.nn as nn
import torch.nn.functional as F

import models.language.utils.utils as utils
from models.language.resnet import IdentityBlock, ConvBlock
from models.language.core.unet import Up
from models.language.core import fusion
from models.language.clip_lingunet_lat import CLIPLingUNetLat


class CLIPLingUNet(CLIPLingUNetLat):
    """ CLIP RN50 with U-Net skip connections """

    def __init__(self, input_shape, output_dim, pose_dim, object_dim, device, preprocess):
        super().__init__(input_shape, output_dim, pose_dim, object_dim, device, preprocess)

    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            ConvBlock(1024, [512, 256, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm)
        )

        self.final_embedding = nn.Sequential(
            nn.Linear(128*7*10, self.output_dim),
        )

    def forward(self, x, l):
        # x = utils.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        # x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text_batch(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.layer1(x)
    
        x = x.view(x.size(0), -1)

        x = self.final_embedding(x)

        return x