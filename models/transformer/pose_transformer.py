############################################################
# input (5, batch_size, 512)
# output (12, batch_size, 512) --> 1 + 1 + 10
############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.autoencoder import AutoEncoder, AutoEncoderwithCondition

class ObjectPoseTransformer(nn.Module):
    def __init__(self, clip_encoder, input_dim, hidden_dim, num_layers, num_heads, autoenc_config=None, train_transformer=False, dropout=0.1, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.train_transformer = train_transformer
        self.device = device

        self.clip_encoder = clip_encoder
        self.object_autoencoder = AutoEncoder(autoenc_config["object_dim"], autoenc_config["object_hdims"], self.input_dim, use_softmax=True)
        self.pose_autoencoder = AutoEncoder(autoenc_config["pose_dim"], autoenc_config["pose_hdims"], self.input_dim)
        self.grasp_autoencoder = AutoEncoderwithCondition(autoenc_config["grasp_dim"], autoenc_config["grasp_hdims"], self.input_dim, autoenc_config["object_dim"] + autoenc_config["pose_orient_dim"])

        self.input_sequence_length = autoenc_config["input_sequence_length"]
        self.output_sequence_length = autoenc_config["object_sequence_length"] + autoenc_config["pose_sequence_length"] + autoenc_config["grasp_sequence_length"]

        self.transformer = nn.Transformer(
            d_model=self.input_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            activation='relu'
        )

        self.criterion = nn.MSELoss()

    def encode_object_pose(self, object_batch, pose_batch, grasp_batch):

        object_sequence = self.object_autoencoder.encode(object_batch)
        pose_sequence = self.pose_autoencoder.encode(pose_batch)

        grasp_modified = grasp_batch.view(-1, grasp_batch.shape[-1]) # grasp batch -> (B, N, 7), modified -> (B*N, 7)
        grasp_condition = torch.cat((object_batch, pose_batch[:, 3:]), dim=-1) # (B, 7) + (B, 4) -> (B, 11)

        grasp_condition = grasp_condition.repeat(1, grasp_batch.shape[1]).view(-1, grasp_condition.shape[-1]) # (B, 11) -> (B*N, 11)

        # print("grasp_modified.shape", grasp_modified.shape)
        # print("grasp_condition.shape", grasp_condition.shape)
        # print("object_sequence.shape", object_sequence.shape)
        # print("pose_sequence.shape", pose_sequence.shape)
        
        grasp_sequence = self.grasp_autoencoder.encode(grasp_modified, grasp_condition) # grasp sequence -> (B*N, 512)

        grasp_sequence = grasp_sequence.view(grasp_batch.shape[0], grasp_batch.shape[1], -1) # (B*N, 512) -> (B, N, 512)

        return object_sequence, pose_sequence, grasp_sequence

    def decode_object_pose(self, object_sequence, pose_sequence, grasp_sequence):

        object_pred = self.object_autoencoder.decode(object_sequence)
        pose_pred = self.pose_autoencoder.decode(pose_sequence)

        batch_size = object_sequence.shape[0]
        num_grasps = grasp_sequence.shape[1]

        grasp_sequence = grasp_sequence.view(-1, grasp_sequence.shape[-1]) # (B, N, 512) -> (B*N, 512)
        grasp_condition = torch.cat((object_pred, pose_pred[:, 3:]), dim=-1) # (B, 7) + (B, 4) -> (B, 11)

        grasp_condition = grasp_condition.repeat(1, num_grasps).view(-1, grasp_condition.shape[-1]) # (B, 11) -> (B*N, 11)

        grasp_sequence = self.grasp_autoencoder.decode(grasp_sequence, grasp_condition) # (B*N, 512) -> (B*N, 7)

        grasp_sequence = grasp_sequence.view(batch_size, num_grasps, -1) # (B*N, 7) -> (B, N, 7)

        return object_sequence, pose_sequence, grasp_sequence

    def get_reconstruction_loss(self, object_batch, pose_batch, grasp_batch):

        object_loss = self.object_autoencoder.get_loss(object_batch)
        pose_loss = self.pose_autoencoder.get_loss(pose_batch)
        
        grasp_modified = grasp_batch.view(-1, grasp_batch.shape[-1]) # grasp batch -> (B, N, 7), modified -> (B*N, 7)

        grasp_condition = torch.cat((object_batch, pose_batch[:, 3:]), dim=-1) # (B, 7) + (B, 4) -> (B, 11)
        grasp_condition = grasp_condition.repeat(1, grasp_batch.shape[1]).view(-1, grasp_condition.shape[-1]) # (B, 11) -> (B*N, 11)

        grasp_loss = self.grasp_autoencoder.get_loss(grasp_modified, grasp_condition)

        return object_loss + pose_loss + grasp_loss, object_loss, pose_loss, grasp_loss

    def get_target_sequence_and_mask(self, object_batch, pose_batch, grasp_batch):
        
        object_embedding, pose_embedding, grasp_embedding = self.encode_object_pose(object_batch, pose_batch, grasp_batch) # object_embedding -> (B, 512), pose_embedding -> (B, 512), grasp_embedding -> (B, N, 512)

        target_sequence = torch.cat((object_embedding.unsqueeze(0), pose_embedding.unsqueeze(0), grasp_embedding.transpose(0, 1)), dim=0) # (1, B, 512), (1, B, 512), (N, B, 512) -> (N+2, B, 512)

        assert target_sequence.shape[0] == self.output_sequence_length

        target_sequence = target_sequence.permute(1, 0, 2) # (N+2, B, 512) -> (B, N+2, 512)

        tgt_mask = self.transformer.generate_square_subsequent_mask(self.output_sequence_length).to(self.device) # (N+2, N+2)

        return target_sequence, tgt_mask

    def get_input_sequence_and_mask(self, image_batch, text_batch):

        image_embedding = self.clip_encoder.forward_pose(image_batch, text_batch) # image embedding -> (B, 512)

        image_embedding = image_embedding.view(image_batch.shape[0], 1, -1) # (B, 512) -> (B, 1, 512)

        assert image_embedding.shape[1] == self.input_sequence_length

        src_mask = self.transformer.generate_square_subsequent_mask(self.input_sequence_length).to(self.device) # (N, N)

        return image_embedding, src_mask

    def forward(self, image_batch, text_batch, object_batch, pose_batch, grasp_batch, return_tgt_sequence=False):
            
        with torch.no_grad():
            input_sequence, src_mask = self.get_input_sequence_and_mask(image_batch, text_batch)
            target_sequence, tgt_mask = self.get_target_sequence_and_mask(object_batch, pose_batch, grasp_batch)

        x = input_sequence.permute(1, 0, 2) # (B, N, 512) -> (N, B, 512)
        y = target_sequence.permute(1, 0, 2) # (B, N+2, 512) -> (N+2, B, 512)

        src_mask = self.transformer.generate_square_subsequent_mask(self.input_sequence_length).to(self.device) # (N, N)

        output = self.transformer(x, y, src_mask=src_mask, tgt_mask=tgt_mask)

        output = output.permute(1, 0, 2) # (N+2, B, 512) -> (B, N+2, 512)

        if return_tgt_sequence:
            return output, target_sequence, tgt_mask

        return output

    def get_loss(self, image_batch, text_batch, object_batch, pose_batch, grasp_batch):

        output, target_sequence, tgt_mask = self.forward(image_batch, text_batch, object_batch, pose_batch, grasp_batch, return_tgt_sequence=True)

        transformer_loss = self.criterion(output, target_sequence)

        # reconstruction_loss = self.get_reconstruction_loss(object_batch, pose_batch, grasp_batch)

        return transformer_loss # + reconstruction_loss

    def get_prediction(self, image_batch, text_batch):
            
        input_sequence, src_mask = self.get_input_sequence_and_mask(image_batch, text_batch)

        x = input_sequence.permute(1, 0, 2)

        src_mask = self.transformer.generate_square_subsequent_mask(self.input_sequence_length).to(self.device)

        target_sequence = torch.zeros(1, x.shape[1], x.shape[-1]).to(self.device)

        output = torch.zeros(0, x.shape[1], x.shape[-1]).to(self.device)

        while output.shape[0] < self.output_sequence_length:

            output = torch.cat((output, torch.zeros(1, x.shape[1], x.shape[-1]).to(self.device)), dim=0)
            tgt_mask = self.transformer.generate_square_subsequent_mask(output.shape[0]).to(self.device)
            
            print(output.shape, x.shape, src_mask.shape, tgt_mask.shape)

            output = self.transformer(x, output, src_mask=src_mask, tgt_mask=tgt_mask)

        assert output.shape[0] == self.output_sequence_length

        output = output.permute(1, 0, 2)

        object_sequence = output[:, 0, :]
        pose_sequence = output[:, 1, :]
        grasp_sequence = output[:, 2:, :]

        object_sequence, pose_sequence, grasp_sequence = self.decode_object_pose(object_sequence, pose_sequence, grasp_sequence)

        return object_sequence, pose_sequence, grasp_sequence