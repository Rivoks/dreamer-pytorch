import torch
import torch.nn as nn
import torch.optim as optim

from encoder import Encoder
from decoder import Decoder
from rssm import RSSM
from reward import RewardModel
from actor import Actor
from critic import Critic
from buffer import ReplayBuffer


class Dreamer(nn.Module):
    def __init__(self, observation_shape, action_size, config):
        super(Dreamer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        self.actor = Actor(action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config

        # Initialize optimizers for each component if needed
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config["actor_lr"]
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config["critic_lr"]
        )
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=config["encoder_lr"]
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=config["decoder_lr"]
        )
        self.rssm_optimizer = optim.Adam(self.rssm.parameters(), lr=config["rssm_lr"])

    # If there are other components or specific initialization, add
    def forward(self, observations, actions, rewards):
        # This method would define the forward pass of the model.
        # Implement the model forward pass here.
        pass

    def train_model(self, dataset):
        # This method would be responsible for the training process.
        # You would call other methods from here to perform:
        # - Dynamics learning
        # - Behavior learning
        # - Environment interaction
        pass

    def dynamics_learning(self, data_batch):
        # Unpack data_batch
        observations, actions, rewards, non_terminals = data_batch

        # Initialize the recurrent and transition model states for the batch
        batch_size = observations.size(0)
        transition_state, recurrent_state = self.rssm.recurrent_model_input_init(
            batch_size
        )

        # We will accumulate losses here
        total_state_transition_loss = 0
        total_reward_prediction_loss = 0

        # Loop through each time step in the sequence
        for t in range(
            observations.size(1) - 1
        ):  # Assuming observations is [batch, time, channels, height, width]
            current_observation = observations[:, t]
            next_observation = observations[:, t + 1]
            action = actions[:, t]
            reward = rewards[:, t]
            non_terminal = non_terminals[:, t]

            # Encode the current and next observations
            latent_state = self.encoder(current_observation)
            next_latent_state = self.encoder(next_observation)

            # Transition and recurrent model predictions
            _, next_transition_state = self.rssm.transition_model(recurrent_state)
            recurrent_state = self.rssm.recurrent_model(
                next_transition_state, action, recurrent_state
            )

            # Predict and calculate reward loss
            predicted_reward_dist = self.reward_predictor(
                next_transition_state, recurrent_state
            )
            reward_prediction_loss = -predicted_reward_dist.log_prob(reward).mean()
            total_reward_prediction_loss += reward_prediction_loss

            # Predict and calculate state transition loss
            # Assuming the latent state is the mean of a distribution for next state
            state_transition_loss = nn.MSELoss()(
                next_transition_state, next_latent_state.detach()
            )
            total_state_transition_loss += state_transition_loss

            # Update transition state only if not the end of an episode
            if non_terminal:
                transition_state = next_transition_state

        # Combine losses
        loss = total_state_transition_loss + total_reward_prediction_loss

        # Zero the gradients before backward pass
        self.encoder_optimizer.zero_grad()
        self.rssm_optimizer.zero_grad()
        self.reward_predictor_optimizer.zero_grad()  # Assurez-vous d'avoir initialisé cet optimiseur

        # Backward pass to compute the gradient of loss w.r.t. parameters
        loss.backward()

        # Clip gradients if necessary to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.rssm.parameters(), self.config["grad_clip_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(), self.config["grad_clip_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.reward_predictor.parameters(), self.config["grad_clip_norm"]
        )

        # Update the parameters
        self.rssm_optimizer.step()
        self.encoder_optimizer.step()
        self.reward_predictor_optimizer.step()

        # Return the loss for logging purposes
        return {
            "total_state_transition_loss": total_state_transition_loss.item(),
            "total_reward_prediction_loss": total_reward_prediction_loss.item(),
            "total_loss": loss.item(),
        }

    def behavior_learning(self, imagined_trajectories):
        # Unpack the imagined trajectories
        states, actions = imagined_trajectories

        # Calculate the predicted rewards and values for each state-action pair
        predicted_rewards = []
        values = []
        for state, action in zip(states, actions):
            # Predict rewards for the current state and action
            reward_dist = self.reward_predictor(state, action)
            predicted_rewards.append(reward_dist.mean)

            # Predict the value for the current state
            value_dist = self.critic(state, action)
            values.append(value_dist.mean)

        # Convert lists to tensors
        predicted_rewards = torch.stack(predicted_rewards)
        values = torch.stack(values)

        # Compute the target value using rewards and values
        # This can use n-step returns, lambda returns, or other value bootstrapping methods
        target_values = self.compute_target_values(predicted_rewards, values)

        # Compute the value loss as the difference between predicted values and target values
        value_loss = nn.MSELoss()(values, target_values.detach())

        # Compute the actor loss using policy gradient methods, such as PPO, SAC, etc.
        # This typically involves computing the advantage as the difference between
        # target values and predicted values, then using it to weigh the log probabilities
        # of the actions taken by the actor.
        actor_loss = self.compute_actor_loss(states, actions, values)

        # Combine the losses
        loss = value_loss + actor_loss

        # Zero the gradients before backward pass for all optimizers
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.reward_predictor_optimizer.zero_grad()

        # Backward pass to compute the gradient of loss w.r.t. parameters
        loss.backward()

        # Clip gradients if necessary to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config["grad_clip_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config["grad_clip_norm"]
        )
        torch.nn.utils.clip_grad_norm_(
            self.reward_predictor.parameters(), self.config["grad_clip_norm"]
        )

        # Update the parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.reward_predictor_optimizer.step()

        # Return the loss for logging purposes
        return {
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "total_loss": loss.item(),
        }

    def interact_with_environment(self, env, initial_state):
        # Implement environment interaction:
        # - Reset environment
        # - Compute states and predict action
        # - Add exploration noise to action
        # - Step through the environment with the action
        # - Add experience to the dataset
        pass

    def imagine_trajectories(self, initial_states):
        # Given an initial state, imagine trajectories by rolling out the actor, critic, and rssm model.
        pass

    def update_parameters(self):
        # Implement parameter update logic for all the neural network components.
        pass

    def save_model(self, file_path):
        # Implement model saving functionality.
        pass

    def load_model(self, file_path):
        # Implement model loading functionality.
        pass
