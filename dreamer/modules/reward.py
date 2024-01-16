import torch
import torch.nn as nn
from dreamer.utils.utils import initialize_weights


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.reward
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = nn.Sequential(
            # On prend en entrée l'état stochastique et l'état déterministe qu'on projette dans un espace latent
            nn.Linear(
                self.stochastic_size + self.deterministic_size,
                self.config.hidden_size,
            ),
            nn.ELU(),
            # Renforcer la capacité du modèle à apprendre des représentations efficaces
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            # Prédiction de la récompense
            nn.Linear(self.config.hidden_size, 1),
        )

        self.network.apply(initialize_weights)

    def forward(self, stochastic, deterministic):
        input_shape = (-1,)  # Taille variable
        batch_with_horizon_shape = stochastic.shape[
            : -len(input_shape)
        ]  # On récupère la taille du batch et de l'horizon

        x = torch.cat(
            (stochastic, deterministic), -1
        )  # On concatène l'état stochastique et l'état déterministe
        input_shape = (x.shape[-1],)  # On récupère la taille de l'entrée

        # On reshape l'entrée pour le réseau de neurones
        x = x.reshape(-1, *input_shape)

        # On passe l'entrée dans le réseau de neurones
        x = self.network(x)

        # Réintégation de la taille du batch et de l'horizon
        x = x.reshape(*batch_with_horizon_shape, *(1,))

        # Distribution normale
        mean = x
        std = 1
        dist = torch.distributions.Normal(mean, std)

        # Distribution indépendante pour considérer chaque dimension indépendante
        dist = torch.distributions.Independent(dist, 1)
        return dist
