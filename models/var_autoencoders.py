import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)

        self.fc_mu = None
        self.fc_logvar = None
        self.flatten_size = None
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        if self.flatten_size is None:
            self.flatten_size = x.size(1) * x.size(2) * x.size(3)
            self.fc_mu = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(self.flatten_size, self.latent_dim).to(x.device)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4*4*4)
        self.conv3d1 = nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv3d2 = nn.ConvTranspose3d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.conv3d3 = nn.ConvTranspose3d(hidden_dim // 4, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), -1, 4, 4, 4)
        z = F.relu(self.conv3d1(z))
        z = F.relu(self.conv3d2(z))
        z = torch.sigmoid(self.conv3d3(z))
        return z


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_shape=(1, 32, 32, 32)):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)
        self.output_shape = output_shape

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_output, target, mu, logvar):
        if recon_output.shape != target.shape:
            print(f"Reshaping target from {target.shape} to {recon_output.shape}")
            target = F.interpolate(target, size=recon_output.shape[2:], mode='trilinear', align_corners=False)
        recon_loss = F.mse_loss(recon_output, target, reduction='mean')
        kl_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar) / mu.size(0)
        return recon_loss + kl_loss



    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_output = self.decoder(z)
        recon_output = recon_output.view(-1, *self.output_shape)

        print("Decoder output shape:", recon_output.shape)
        return recon_output, mu, logvar

