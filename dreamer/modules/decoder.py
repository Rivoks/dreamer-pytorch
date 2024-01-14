import torch
import torch.nn as nn
from dreamer.utils.utils import (
    initialize_weights,
)


class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = nn.ReLU()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            # Couche linéaire à partir de tailles déterministes et stochastiques
            nn.Linear(
                self.deterministic_size + self.stochastic_size, self.config.depth * 32
            ),
            # Déplie pour la convolution: 1D -> 2D
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            # Déplie davantage pour la convolution: 2D -> 3D
            nn.Unflatten(2, (1, 1)),
            # Convolution transposée pour la reconstruction
            nn.ConvTranspose2d(
                self.config.depth * 32,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            # activation ReLU
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size + 1,
                self.config.stride,
            ),
        )

        self.network.apply(initialize_weights)

    def forward(self, stochastic, deterministic):
        batch_with_horizon_shape = stochastic.shape[: -len((-1,))]

        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (1,)

        x = torch.cat((stochastic, deterministic), -1)
        input_shape = (x.shape[-1],)

        x = x.reshape(-1, *input_shape)
        x = self.network(x)
        x = x.reshape(*batch_with_horizon_shape, *self.observation_shape)

        mean = x  # prediction moyenne de l'observation
        std = 1
        dist = torch.distributions.Normal(mean, std)  # distribution normale

        # distribution indépendante pour chaque dimension spatiale: i.e pour chaque pixel
        return torch.distributions.Independent(dist, len(self.observation_shape))
