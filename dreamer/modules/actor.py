import torch
import torch.nn as nn
from torch.distributions import TanhTransform

from dreamer.utils.utils import create_normal_dist, build_network


class Actor(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = nn.Sequential(
            nn.Linear(
                self.stochastic_size + self.deterministic_size, self.config.hidden_size
            ),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            nn.Linear(self.config.hidden_size, action_size),
        )

    def forward(self, posterior, deterministic):
        # On concatène l'état stochastique et l'état déterministe
        x = torch.cat((posterior, deterministic), -1)

        # On passe l'entrée dans le réseau de neurones
        x = self.network(x)

        # Distribution catégorielle pour sélectionner une action dans un contexte discret
        # logits: score non normalisé
        dist = torch.distributions.OneHotCategorical(logits=x)

        # Pour conserver le calcul du gradient lors de la backpropagation
        dist_probs = dist.probs - dist.probs.detach()

        # Créer un gradient qui reflète la probabilité de l'action échantillonnée
        # tout en conservant la différentiabilité pour l'entraînement
        action = dist.sample() + dist_probs

        return action
