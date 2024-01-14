import torch.nn as nn
from dreamer.utils.utils import (
    initialize_weights,
)


class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder

        activation = nn.ReLU()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                self.config.depth,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth * 2,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.Conv2d(
                self.config.depth * 4,
                self.config.depth * 8,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
        )

        self.network.apply(initialize_weights)

    def forward(self, x):
        input_shape = self.observation_shape  # Récupère la forme de l'observation
        batch_with_horizon_shape = x.shape[
            : -len(input_shape)
        ]  # Conserve la forme du batch et de l'horizon

        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (
                1,
            )  # Assigne 1 si aucune forme de batch/horizon

        x = x.reshape(-1, *input_shape)  # Redimensionne pour appliquer le réseau
        x = self.network(x)  # Passe à travers le réseau de l'encodeur
        x = x.reshape(
            *batch_with_horizon_shape, *(-1,)
        )  # Restaure la forme du batch/horizon

        return x  # Renvoie le résultat
