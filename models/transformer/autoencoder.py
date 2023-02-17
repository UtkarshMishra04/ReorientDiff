import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, use_softmax=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = copy.deepcopy(hidden_dims)
        self.latent_dim = latent_dim
        self.use_softmax = use_softmax

        modules = []

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.ReLU()
                )
            )
            input_dim = h_dim

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.ReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[-1]),
                nn.ReLU()
            )
        )

        # Build Decoder
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_dim),
        )

        if self.use_softmax:
            self.final_layer.add_module('softmax', nn.Softmax(dim=1))

        self.bce_loss = nn.MSELoss(reduction='sum')

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.final_layer(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_loss(self, x):
        x_recon = self.forward(x)
        loss = self.bce_loss(x_recon, x)
        return loss

    def get_latent(self, x):
        return self.encode(x)        

class AutoEncoderwithCondition(AutoEncoder):

    def __init__(self, input_dim, hidden_dims, latent_dim, condition_dim):
        super().__init__(input_dim, hidden_dims, latent_dim)
        self.condition_dim = condition_dim

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim + condition_dim, hidden_dims[0]),
                nn.ReLU()
            )
        )

        enc_dim = hidden_dims[0]


        # Build Encoder
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(enc_dim, h_dim),
                    nn.ReLU()
                )
            )
            enc_dim = h_dim

        modules.append(
            nn.Sequential(
                nn.Linear(enc_dim, latent_dim),
                nn.ReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(latent_dim + condition_dim, hidden_dims[-1]),
                nn.ReLU()
            )
        )

        # Build Decoder
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.input_dim),
        )

        self.bce_loss = nn.MSELoss(reduction='sum')


    def encode(self, x, condition):
        return self.encoder(torch.cat([x, condition], dim=1))

    def decode(self, z, condition):
        return self.final_layer(self.decoder(torch.cat([z, condition], dim=1)))

    def forward(self, x, condition):
        z = self.encode(x, condition)
        return self.decode(z, condition)

    def get_loss(self, x, condition):
        x_recon = self.forward(x, condition)
        loss = self.bce_loss(x_recon, x)
        return loss

    def get_latent(self, x, condition):
        return self.encode(x, condition)


def main():
    model = AutoEncoder(7, [64, 128, 256], 512)
    x = torch.randn(10, 7)
    
    print(model(x).shape)

    print(model.get_loss(x))

    print(model.get_latent(x).shape)

    model = AutoEncoderwithCondition(7, [64, 128, 256], 512, 7)

    print(model(x, torch.randn(10, 7)).shape)

    print(model.get_loss(x, torch.randn(10, 7)))

    print(model.get_latent(x, torch.randn(10, 7)).shape)

if __name__ == '__main__':
    main()