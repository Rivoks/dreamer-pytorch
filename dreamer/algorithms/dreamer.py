import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from dreamer.modules.model import RSSM, RewardModel
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic

from dreamer.utils.utils import (
    compute_lambda_values,
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
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.num_total_episode = 0

    def train(self, env):
        total_start_time = time.time()

        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        for iteration in range(self.config.train_iterations):
            # print("start training...")
            start_time = time.time()

            collection_loss = 0.0

            for collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )

                posteriors, deterministics, model_loss = self.dynamic_learning(data)
                collection_loss += model_loss

                self.behavior_learning(posteriors, deterministics)

            self.environment_interaction(env, self.config.num_interaction_episodes)
            self.evaluate(env)

            print(
                f"[{(time.time() - start_time).__round__(1)}s] Collection loss : {collection_loss / self.config.collect_interval}"
            )

        print(f"training time : {(time.time() - total_start_time).__round__(1)}s")
        self.save()

    def evaluate(self, env):
        # print("start evaluation...")
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()

        model_loss = self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach(), model_loss

    def _model_update(self, data, posterior_info):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

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
        """
        #TODO : last posterior truncation(last can be last step)
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
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
        posterior, deterministic = self.rssm.recurrent_model_input_init(1)
        action = torch.zeros(1, self.action_size).to(self.device)

        observation = env.reset()
        embedded_observation = self.encoder(
            torch.from_numpy(observation.copy()).float().to(self.device)
        )

        score = 0
        done = False

        while not done:
            env.render()

            deterministic = self.rssm.recurrent_model(posterior, action, deterministic)
            embedded_observation = embedded_observation.reshape(1, -1)

            _, posterior = self.rssm.representation_model(
                embedded_observation, deterministic
            )
            action = self.actor(posterior, deterministic).detach()

            buffer_action = action.cpu().numpy()
            env_action = buffer_action.argmax()

            next_observation, reward, done, info = env.step(env_action)

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
        for _ in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation = env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation.copy()).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)

                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                buffer_action = action.cpu().numpy()
                env_action = buffer_action.argmax()

                next_observation, reward, done, info = env.step(env_action)

                if train:
                    self.buffer.add(
                        observation, buffer_action, reward, next_observation, done
                    )

                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_observation).float().to(self.device)
                )
                observation = next_observation

                done = (
                    done or info["flag_get"] or info["time"] == 0 or info["life"] == 0
                )

                if done:
                    if train:
                        self.num_total_episode += 1
                    else:
                        score_lst = np.append(score_lst, score)
                    break

        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score :", evaluate_score)
