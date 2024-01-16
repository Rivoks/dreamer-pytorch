import torch
import torch.nn as nn
import numpy as np
import time
from dreamer.modules.rssm import RSSM
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.modules.reward import RewardModel

from dreamer.utils.utils import (
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


class Dreamer:
    def __init__(
        self,
        observation_shape,
        action_size,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        self.actor = Actor(action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer
        self.operation_config = config.operation

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.num_total_episode = 0

    def train(self, env):
        total_start_time = time.time()

        # Interagir avec l'environnement pour initialiser le buffer
        self.environment_interaction(env, self.config.seed_episodes)

        for _ in range(self.config.train_iterations):
            start_time = time.time()
            collection_loss = 0.0

            for _ in range(self.config.collect_interval):
                # Extraire des échantillons du buffer
                samples = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )

                # Apprentissage dynamique avec les données échantillonnées
                stochastic, deterministics, model_loss = self.dynamic_learning(samples)
                collection_loss += model_loss

                # Apprentissage comportemental
                self.behavior_learning(stochastic, deterministics)

            # Nouvelle interaction avec l'environnement pour collecter plus de données
            self.environment_interaction(env, self.config.num_interaction_episodes)

            # Évaluation du modèle
            self.evaluate(env)

            # Affichage de la perte accumulée durant l'itération
            print(
                f"[{(time.time() - start_time).__round__(1)}s] Collection loss : {collection_loss / self.config.collect_interval}"
            )

        # Affichage du temps total d'entraînement
        print(f"training time : {(time.time() - total_start_time).__round__(1)}s")

        # Sauvegarde du modèle
        self.save()

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        # Initialisation des états stochastique et déterministe
        stochastic, deterministic = self.rssm.recurrent_model_input_init(
            len(data.action)
        )

        # Encodage des observations
        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            # Mise à jour de l'état déterministe à partir de l'état stochastique précédent et de l'action
            deterministic = self.rssm.recurrent_model(
                stochastic, data.action[:, t - 1], deterministic
            )

            # Modèle de transition pour prédire le prochain état stochastique
            prior_dist, stochastic = self.rssm.transition_model(deterministic)

            # Modèle de représentation pour prédire l'état stochastique à partir de l'observation encodée
            stochastic_dist, stochastic = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            # Enregistrement des informations pour l'apprentissage
            self.dynamic_learning_infos.append(
                priors=stochastic,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                stochastics=stochastic,
                stochastic_dist_means=stochastic_dist.mean,
                stochastic_dist_stds=stochastic_dist.scale,
                deterministics=deterministic,
            )

            stochastic = stochastic

        # Compilation des informations et mise à jour du modèle
        infos = self.dynamic_learning_infos.get_stacked()
        model_loss = self._model_update(data, infos)
        return infos.stochastics.detach(), infos.deterministics.detach(), model_loss

    def _model_update(self, data, stochastic_info):
        reconstructed_observation_dist = self.decoder(
            stochastic_info.stochastics, stochastic_info.deterministics
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )

        reward_dist = self.reward_predictor(
            stochastic_info.stochastics, stochastic_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        stochastic_dist = create_normal_dist(
            stochastic_info.stochastic_dist_means,
            stochastic_info.stochastic_dist_stds,
            event_shape=1,
        )
        stochastic_dist = create_normal_dist(
            stochastic_info.stochastic_dist_means,
            stochastic_info.stochastic_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(stochastic_dist, stochastic_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

        return model_loss.item()

    def behavior_learning(self, states, deterministics):
        # Redimensionnement des états stochastiques et déterministes pour les traiter
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # Boucle sur l'horizon de temps défini pour l'apprentissage comportemental
        for t in range(self.config.horizon_length):
            # Utilisation de l'acteur pour déterminer l'action basée sur l'état actuel
            action = self.actor(state, deterministic)

            # Mise à jour de l'état déterministe avec l'état stochastique actuel et l'action
            deterministic = self.rssm.recurrent_model(state, action, deterministic)

            # Mise à jour de l'état stochastique pour le prochain pas de temps
            _, state = self.rssm.transition_model(deterministic)

            # Enregistrement des informations d'état pour la mise à jour ultérieure de l'agent
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        # Mise à jour des politiques et valeurs de l'agent basée sur les informations enregistrées
        self._agent_update(self.behavior_learning_infos.get_stacked())

    def compute_lambda_values(
        self, rewards, values, continues, horizon_length, device, lambda_
    ):
        # Supprimer la dernière étape pour avoir les bonnes dimensions
        rewards = rewards[:, :-1]
        continues = continues[:, :-1]
        next_values = values[:, 1:]
        last = next_values[:, -1]

        # Calcul des entrées pour le calcul des valeurs lambda
        inputs = rewards + continues * next_values * (1 - lambda_)

        outputs = []
        # Calcul des valeurs lambda étape par étape (de la fin au début)
        for index in reversed(range(horizon_length - 1)):
            last = inputs[:, index] + continues[:, index] * lambda_ * last
            outputs.append(last)

        # Réorganiser les valeurs calculées dans l'ordre correct
        returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
        return returns

    def _agent_update(self, behavior_learning_infos):
        # Calcul des récompenses prédites
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        # Calcul des valeurs prédites
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        # Facteur de réduction qui atténue la valeur des récompenses futures.
        continues = self.config.discount * torch.ones_like(values)

        # lambda_values calcule une version ajustée des valeurs basée sur le facteur de réduction et les récompenses prédites.
        lambda_values = self.compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        # Calcul de l'erreur de valeur
        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Évaluation du modèle critique pour estimer les valeurs attendues.
        # On utilise les informations sur les actions prises (priors) et les informations déterministes, détachées du graphe de calcul.
        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )

        # Calcul de la perte (loss) en comparant les valeurs prédites (value_dist) avec une autre distribution (lambda_values).
        # La perte est calculée en utilisant la log-probabilité négative et en prenant la moyenne.
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def save(self):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "rssm": self.rssm.state_dict(),
                "reward_predictor": self.reward_predictor.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "model_optimizer": self.model_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "num_total_episode": self.num_total_episode,
            },
            self.config.save_path,
        )

    def load(
        self,
    ):
        checkpoint = torch.load(self.config.save_path)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.rssm.load_state_dict(checkpoint["rssm"])
        self.reward_predictor.load_state_dict(checkpoint["reward_predictor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.num_total_episode = checkpoint["num_total_episode"]

    def play(self, env):
        stochastic, deterministic = self.rssm.recurrent_model_input_init(1)
        action = torch.zeros(1, self.action_size).to(self.device)

        observation = env.reset()
        embedded_observation = self.encoder(
            torch.from_numpy(observation.copy()).float().to(self.device)
        )

        score = 0
        done = False

        while not done:
            env.render()

            deterministic = self.rssm.recurrent_model(stochastic, action, deterministic)
            embedded_observation = embedded_observation.reshape(1, -1)

            _, stochastic = self.rssm.representation_model(
                embedded_observation, deterministic
            )
            action = self.actor(stochastic, deterministic).detach()

            buffer_action = action.cpu().numpy()
            env_action = buffer_action.argmax()

            next_observation, reward, done, info = env.step(env_action)

            print("next_observation :", next_observation.shape)

            score += reward
            embedded_observation = self.encoder(
                torch.from_numpy(next_observation).float().to(self.device)
            )

            done = done or info["flag_get"] or info["time"] == 0 or info["life"] == 0
            if done:
                break

        # print("score :", score)

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        # Itère sur le nombre d'épisodes d'interaction avec l'environnement
        for _ in range(num_interaction_episodes):
            # Initialise les états stochastique et déterministe du modèle
            stochastic, deterministic = self.rssm.recurrent_model_input_init(1)
            # Initialise une action vide
            action = torch.zeros(1, self.action_size).to(self.device)

            # Réinitialise l'environnement et obtient la première observation
            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation.copy()).float().to(self.device)
            )

            # Initialise le score et la liste de scores
            score = 0
            score_lst = np.array([])
            done = False

            # Boucle tant que l'épisode n'est pas terminé
            while not done:
                # Mise à jour du modèle stochastique et déterministe
                deterministic = self.rssm.recurrent_model(
                    stochastic, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)

                # Mise à jour de l'état stochastique avec le modèle de représentation
                _, stochastic = self.rssm.representation_model(
                    embedded_observation, deterministic
                )

                # Détermine l'action à prendre
                action = self.actor(stochastic, deterministic).detach()
                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()

                # Applique l'action à l'environnement et reçoit le prochain état et la récompense
                next_observation, reward, done, info = env.step(env_action)

                # Ajoute l'expérience au buffer si en mode entraînement
                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )

                # Met à jour le score et l'observation
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation

                # Vérifie si l'épisode est terminé
                done = (
                    done or info["flag_get"] or info["time"] == 0 or info["life"] == 0
                )

                # Si l'épisode est terminé, met à jour le score total ou la liste des scores
                if done:
                    if train:
                        self.num_total_episode += 1
                    else:
                        score_lst = np.append(score_lst, score)
                    break

        # Affiche le score si en mode évaluation
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score :", evaluate_score)
