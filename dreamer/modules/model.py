import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from dreamer.utils.utils import initialize_weights


class RSSM(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm

        self.recurrent_model = RecurrentModel(
            action_size, config
        )  # Met à jour l'état récurrent (déterministe)
        self.transition_model = TransitionModel(
            config
        )  # Modélise la dynamique de l'état latent
        self.representation_model = RepresentationModel(
            config
        )  # Encode les observations en un l'état latent

    def recurrent_model_input_init(self, batch_size):
        # Initialisation de l'entrée du modèle récurrent
        return self.transition_model.init_state(
            batch_size
        ), self.recurrent_model.init_state(batch_size)


class RecurrentModel(nn.Module):
    def __init__(self, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.recurrent_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        # Activation ELU plus adaptée pour les réseaux récurrents
        self.activation = nn.ELU()

        # Création de l'état stochastique à partir de l'état déterministe et de l'action
        self.linear = nn.Linear(
            self.stochastic_size + action_size, self.config.hidden_size
        )

        # Mise à jour de l'état déterministe à partir de l'état stochastique, de l'action
        # et de l'état déterministe précédent
        self.gru = nn.GRUCell(self.config.hidden_size, self.deterministic_size)

    def forward(self, stochastic, action, deterministic):
        # Combiner l'état stochastique et l'action pour comprendre comment
        # les actions affectent les transitions d'état de l'environnement
        x = torch.cat((stochastic, action), 1)

        # Mise à jour de l'état déterministe
        return self.gru(self.activation(self.linear(x)), deterministic)

    def init_state(self, batch_size):
        # Initialisation de l'état déterministe avec des zéros
        return torch.zeros(batch_size, self.deterministic_size).to(self.device)


class TransitionModel(nn.Module):
    # Prédit la distribution de l'état stochastique à partir de l'état déterministe
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.transition_model
        self.device = config.operation.device
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = nn.Sequential(
            # On prend en entrée l'état déterministe et on le passe dans un réseau de neurones
            nn.Linear(self.deterministic_size, self.config.hidden_size),
            # ELU ajoute de la non-linéarité
            nn.ELU(),
            # Renforcer la capacité du modèle à apprendre des représentations efficaces
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            # On prédit la distribution de l'état stochastique pour la moyenne et l'écart-type (x2)
            nn.Linear(self.config.hidden_size, self.stochastic_size * 2),
        )

        self.network.apply(initialize_weights)

    def forward(self, x):
        x = self.network(x)

        # On sépare la sortie en deux parties : la moyenne et l'écart-type
        mean, std = torch.chunk(x, 2, -1)

        # On applique une fonction d'activation pour avoir des valeurs positives
        # et un écart-type minimum pour éviter les valeurs nulles
        std = F.softplus(std) + self.config.min_std

        # Distribution normale
        prior_dist = torch.distributions.Normal(mean, std)

        # On échantillonne la distribution pour obtenir l'état stochastique
        stochastic = prior_dist.rsample()

        # On retourne la distribution et l'état stochastique
        return prior_dist, stochastic

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).to(self.device)


class RepresentationModel(nn.Module):
    # Ce modèle transforme les observations et états déterministes en une distribution latente stochastique
    # d'où de nouveaux états stochastiques peuvent être échantillonnés.
    def __init__(self, config):
        super().__init__()
        self.config = config.parameters.dreamer.rssm.representation_model
        self.stochastic_state_size = config.parameters.dreamer.stochastic_state_size
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = nn.Sequential(
            # On prend en entrée l'état stochastique et l'état déterministe qu'on projette dans un espace latent
            nn.Linear(
                self.stochastic_state_size + self.deterministic_size,
                self.config.hidden_size,
            ),
            # ELU ajoute de la non-linéarité
            nn.ELU(),
            # Renforcer la capacité du modèle à apprendre des représentations efficaces
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ELU(),
            # On prédit la distribution de l'état stochastique pour la moyenne et l'écart-type (x2)
            nn.Linear(self.config.hidden_size, self.stochastic_size * 2),
        )

        self.network.apply(initialize_weights)

    def forward(self, stochastic, deterministic):
        # On combine l'état stochastique et l'état déterministe
        x = torch.cat((stochastic, deterministic), 1)
        x = self.network(x)

        # On sépare la sortie en deux parties : la moyenne et l'écart-type
        mean, std = torch.chunk(x, 2, -1)

        # On applique une fonction d'activation pour avoir des valeurs positives
        # et un écart-type minimum pour éviter les valeurs nulles
        std = F.softplus(std) + self.config.min_std

        # Distribution normale
        new_stochastic_dist = torch.distributions.Normal(mean, std)

        # On échantillonne la distribution pour obtenir le nouvel état stochastique
        new_stochastic = new_stochastic_dist.rsample()

        # On retourne la distribution et l'état stochastique
        return new_stochastic_dist, new_stochastic


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
