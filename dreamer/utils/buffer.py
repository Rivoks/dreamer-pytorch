from dreamer.utils.utils import attrdict_monkeypatch_fix

attrdict_monkeypatch_fix()

from attrdict import AttrDict
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, observation_shape, action_size, device, config):
        self.config = config.parameters.dreamer.buffer
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observation = np.empty(
            (self.capacity, *observation_shape), dtype=np.float32
        )
        self.next_observation = np.empty(
            (self.capacity, *observation_shape), dtype=np.float32
        )
        self.action = np.empty((self.capacity, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(self, observation, action, reward, next_observation, done):
        self.observation[self.buffer_index] = observation
        self.action[self.buffer_index] = action
        self.reward[self.buffer_index] = reward
        self.next_observation[self.buffer_index] = next_observation
        self.done[self.buffer_index] = done

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size):
        # Calcul de l'index du dernier élément rempli dans le buffer
        last_filled_index = self.buffer_index - chunk_size + 1

        # Génération aléatoire des indices de départ pour chaque échantillon
        sample_index = np.random.randint(0, last_filled_index, batch_size).reshape(
            -1, 1
        )

        # Création d'un tableau pour représenter la longueur de chaque chunk
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        # Calcul des indices réels à échantillonner dans le buffer
        sample_index = (sample_index + chunk_length) % self.capacity

        # Extraction des échantillons du buffer et conversion en tenseurs Torch
        observation = torch.as_tensor(
            self.observation[sample_index], device=self.device
        ).float()

        next_observation = torch.as_tensor(
            self.next_observation[sample_index], device=self.device
        ).float()

        action = torch.as_tensor(self.action[sample_index], device=self.device)
        reward = torch.as_tensor(self.reward[sample_index], device=self.device)
        done = torch.as_tensor(self.done[sample_index], device=self.device)

        # Création d'un dictionnaire avec les échantillons pour un accès facile
        sample = AttrDict(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done,
            }
        )
        return sample
