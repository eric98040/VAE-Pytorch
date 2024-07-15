import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.LeakyReLU(0.2)
            )

        self.downlayers = nn.Sequential(
            *block(input_dim, latent_dim * 2),
            *block(latent_dim * 2, latent_dim * 2),
        )

        self.mean = nn.Linear(latent_dim * 2, latent_dim)
        self.logvar = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        mean = self.mean(self.downlayers(x))
        logvar = self.logvar(self.downlayers(x))

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU(0.2),
            )

        self.uplayers = nn.Sequential(
            *block(latent_dim, latent_dim * 2),
            *block(latent_dim * 2, latent_dim * 2),
        )

        self.reconstruction = nn.Linear(latent_dim * 2, output_dim)

    def forward(self, x):
        x = self.uplayers(x)
        x = self.reconstruction(x)
        x_hat = torch.sigmoid(x)
        return x_hat


def reparameterize(mean, logvar):
    eps = torch.randn_like(mean)
    std = torch.exp(logvar / 2)
    z = mean + std * eps
    return z


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder

    def forward(self, x):
        mean, logvar = self.Encoder(x)
        z = reparameterize(mean, logvar)
        x_hat = self.Decoder(z)
        return x_hat, mean, logvar
