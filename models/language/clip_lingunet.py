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

        self.pose_dim = pose_dim
        self.object_dim = object_dim

    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 1024)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_dim, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            ConvBlock(1024 + 512, [512, 256, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm)
        )

        self.layer2 = nn.Sequential(
            ConvBlock(1024 + 512, [512, 256, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm)
        )

        self.final_embedding = nn.Sequential(
            nn.Linear(128*7*10, self.output_dim),
        )

        self.final_embedding_pose = nn.Sequential(
            nn.Linear(128*7*10, self.output_dim),
        )

        self.object_decoder_1 = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim),
            nn.ReLU(),
        )

        self.object_decoder_2 = nn.Sequential(
            nn.Linear(self.output_dim, self.object_dim),
            nn.Sigmoid()
        )

        # self.pose_vae = PoseConditionalVAE(self.pose_dim, self.output_dim, self.output_dim + self.output_dim, [128, 256])

    def forward(self, x, l):
        x = utils.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
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

    def forward_pose(self, x, l):
        x = utils.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text_batch(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv2(x)

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.layer2(x)
    
        x = x.view(x.size(0), -1)

        x = self.final_embedding_pose(x)

        return x

    def loss(self, image, lang, pose, object_onehot):
        
        # forward pass
        out = self.forward(image, lang)

        # object loss
        object_embed = self.object_decoder_1(out)
        object_out = self.object_decoder_2(object_embed)

        object_loss = F.binary_cross_entropy(object_out, object_onehot)

        # pose loss
        # pose_loss = self.pose_vae.get_loss(pose, torch.cat([out, object_embed], dim=1))

        total_loss = object_loss

        return total_loss #, pose_loss, object_loss

    def inference(self, image, lang):
        embedding = self.forward(image, lang)

        object_embed = self.object_decoder_1(embedding)
        object_onehot = self.object_decoder_2(object_embed)

        # pose = self.pose_vae.sample(torch.cat([embedding, object_embed], dim=1))

        return object_onehot

    def get_embedding(self, image, lang):
        embedding_1 = self.forward(image, lang)
        embedding_2 = self.forward_pose(image, lang)

        return torch.cat([embedding_1, embedding_2], dim=1)

    def get_conditions(self, image, lang):
        embedding = self.forward(image, lang)

        object_embed = self.object_decoder_1(embedding)

        return torch.cat([embedding, object_embed], dim=1).detach()

    def get_conditions_v2(self, image, lang):
        embedding = self.forward_pose(image, lang)

        object_embed = self.object_decoder_1(embedding).detach()

        object_out = self.object_decoder_2(object_embed).detach()

        return torch.cat([embedding, object_embed, object_out], dim=1)

# Path: models/language/clip_lingunet_lat.py