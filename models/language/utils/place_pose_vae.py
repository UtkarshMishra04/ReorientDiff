import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class ObjectPredictor(nn.Module):
    def __init__(self, object_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.object_dim = object_dim
        self.embedding_dim = embedding_dim

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.embedding_dim, hidden_dim[-1])

        hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim[-1], self.object_dim),
            nn.Softmax(dim=1)
        )

        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, x):
        x = self.decoder_input(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def get_loss(self, true_object, pose_condition):
        pred_object = self.forward(pose_condition)
        loss = self.bce_loss(pred_object, true_object)
        return loss

class PoseConditionalVAE(nn.Module):
    def __init__(self, pose_dim, embedding_dim_pose, embedding_dim_cond, hidden_dim):
        super().__init__()
        self.pose_dim = pose_dim
        self.embedding_dim_pose = embedding_dim_pose
        self.embedding_dim_cond = embedding_dim_cond
        self.hidden_dim = copy.deepcopy(hidden_dim)

        self.xyz_dim = 3
        self.orientation_dim = 2

        modules = []

        # Build Encoder
        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Linear(pose_dim, h_dim),
                    nn.ReLU()
                )
            )
            pose_dim = h_dim

        self.encoder_1 = nn.Sequential(*modules)

        self.encoder_2 = nn.Sequential(
            nn.Linear(pose_dim + embedding_dim_cond, hidden_dim[-1]),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim[-1], self.embedding_dim_pose)
        self.fc_logvar = nn.Linear(hidden_dim[-1], self.embedding_dim_pose)

        # Build Decoder Pose
        modules = []

        self.decoder_input1 = nn.Linear(self.embedding_dim_pose + self.embedding_dim_cond, hidden_dim[-1])

        hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder1 = nn.Sequential(*modules)

        self.final_layer1 = nn.Sequential(
            nn.Linear(hidden_dim[-1], self.xyz_dim),
        )

        # Build Decoder Orientation
        modules = []

        self.decoder_input2 = nn.Linear(self.embedding_dim_pose + self.embedding_dim_cond, hidden_dim[-1])

        # hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder2 = nn.Sequential(*modules)

        self.final_layer2 = nn.Sequential(
            nn.Linear(hidden_dim[-1], self.orientation_dim),
        )

        self.bce_loss = nn.MSELoss(reduction='sum')

    def encode(self, x, condition=None):
        result = self.encoder_1(x)
        result = result.view(-1, self.hidden_dim[-1])
        result = torch.cat((result, condition), dim=1)
        result = self.encoder_2(result)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, condition=None, pose=True):

        if condition is None:
            condition = torch.zeros_like(z)
            z = torch.cat((z, condition), dim=1)
        else:
            z = torch.cat((z, condition), dim=1)
            
        if pose:
            result = self.decoder_input1(z)
            result = result.view(-1, self.hidden_dim[-1])
            result = self.decoder1(result)
            result = self.final_layer1(result)
        else:
            result = self.decoder_input2(z)
            result = result.view(-1, self.hidden_dim[-1])
            result = self.decoder2(result)
            result = self.final_layer2(result)
        return result

    def forward(self, x, condition=None):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        y1 = self.decode(z, condition, pose=True)
        y2 = self.decode(z, condition, pose=False)
        return y1, y2, mu, logvar

    def sample(self, condition):
        z = torch.randn(condition.size(0), self.embedding_dim_pose)
        z = z.to(condition.device)
        samples1 = self.decode(z, condition, pose=True).detach()
        samples2 = self.decode(z, condition, pose=False).detach()
        samples3 = torch.zeros_like(samples2)
        samples = torch.cat((samples1, samples3, samples2), dim=1)
        return samples

    def get_loss(self, x, condition=None):
        if torch.isnan(x).any():
            print("NAN x")
            assert False
        recon_x1, recon_x2, mu, logvar = self.forward(x, condition)
        BCE = self.bce_loss(recon_x1, x[:, :self.xyz_dim]) + self.bce_loss(recon_x2, x[:, -self.orientation_dim:])
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= x.size(0)*self.pose_dim
        # print("pose loss: ", BCE, KLD, mu[0,0], logvar[0,0], recon_x[0,0])
        if torch.isnan(BCE) or torch.isnan(KLD):
            print("NAN loss")
            assert False
        return (BCE + KLD), BCE, KLD

    def get_reconstruction(self, x, condition=None):
        recon_x1, recon_x2, _, _ = self.forward(x, condition)
        recon_x3 = torch.zeros_like(recon_x2)
        recon_x = torch.cat((recon_x1, recon_x3, recon_x2), dim=1)
        return recon_x

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)


def main():
    
    # Test
    batch_size = 256
    pose_dim = 7
    embedding_dim_pose = 256
    embedding_dim_cond = 512
    hidden_dim = [128, 256]
    model = PoseConditionalVAE(pose_dim, embedding_dim_pose, embedding_dim_cond, hidden_dim)
    
    x = torch.randn(batch_size, 7)
    condition = torch.randn(batch_size, embedding_dim_cond)
    recon_x, mu, logvar = model(x, condition)

    loss = model.get_loss(x, condition)

    print(recon_x.shape)
    print(mu.shape)
    print(logvar.shape)
    print(loss)

    sample_pose = model.sample(condition)

    print(sample_pose.shape)

if __name__ == '__main__':
    main()
