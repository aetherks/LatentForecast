import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import copy
#Data prep
from math import pi
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class VAE_MLP(nn.Module):
    def __init__(self):
        super(VAE_MLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(70 * 125, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64)  # Assuming latent space of size 32 (32 for mean and 32 for log variance)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 70 * 125),
            nn.Unflatten(1, (70, 125))
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class VAE_MLP2(nn.Module):
    def __init__(self, num_ocean_points):
        super(VAE_MLP, self).__init__()
        self.num_ocean_points = num_ocean_points

        self.encoder = nn.Sequential(
            nn.Linear(num_ocean_points, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64)  # Latent space size
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, num_ocean_points)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Create VAE model
num_ocean_points = ocean_data_tensor.shape[1]
vae_model = VAE_MLP(num_ocean_points)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1x1 = nn.Conv2d(10, 64, kernel_size=1)

        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_mu = nn.Linear(256*18*32, 32)
        self.fc_logvar = nn.Linear(256*18*32, 32)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.fc_decode = nn.Linear(32, 256*18*32)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 256, 18, 32)
        return self.decoder(z)

    def forward(self, sst, lat, lon):
        # Fourier features
        fourier_tensor = torch.cat([torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=1)
        fourier_maps = self.conv1x1(fourier_tensor)  # shape [batch_size, 64, 70, 125]

        # Elementwise multiplication with the original SST map
        rescaled_sst = sst * fourier_maps
        
        mu, logvar = self.encode(rescaled_sst)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar


