import jax
import numpy as np
import jax.numpy as jnp
import optax
import distrax
import chex
import gymnasium as gym

from typing import *
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from jax import random
from flax import linen as nn

from torch.utils.tensorboard import SummaryWriter


HIDDEN_DIM = 256
REPRESENTATION_SIZE = 256
RNN_SIZE = 128
STATE_DIM = 128

BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
CHUNK_LENGTH = 50
ENV_NAME = "HalfCheetah-v3"
IMAGINATION_HORIZON = 50
LEARNING_RATE = 3e-4
N_INTERACTION_STEPS = 1
N_UPDATE_STEPS = 1
SEED_EPISODES = 5e3
SEQUENCE_LENGTH = 50
TOTAL_TIMESTEPS = 1_000_000
FREE_NATS = 3


# Encoder, Recurrent state space, Observation and Reward models adapted from
# https://github.com/cross32768/PlaNet_PyTorch/blob/master/model.py


class Encoder(nn.Module):
    """A standard Convolution Network for image observations"""

    @nn.compact
    def __call__(obs: chex.Array):
        hidden = nn.relu(
            nn.Conv(features=32, kernel_size=4, strides=2, name="cv1")(obs)
        )
        hidden = nn.relu(
            nn.Conv(features=64, kernel_size=4, strides=2, name="cv2")(obs)
        )
        hidden = nn.relu(
            nn.Conv(features=128, kernel_size=4, strides=2, name="cv3")(obs)
        )
        embedded_obs = nn.relu(
            nn.Conv(features=256, kernel_size=4, strides=2, name="cv3")(obs)
        ).reshape((hidden[0], -1))
        return embedded_obs


class RSSM:
    """Joint Prior / Posterion network mimicking the Kalman filter"""

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        rnn_hidden: chex.Array,
        embedded_next_obs: chex.Array,
    ) -> Tuple[distrax.Normal, chex.Array]:

        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden, min_stddev=0.1):
        hidden = nn.relu(
            nn.Dense(features=HIDDEN_DIM, name="fc_state_action")(
                jnp.cat([state, action], dim=1)
            )
        )
        rnn_hidden = nn.GRUCell()(hidden, rnn_hidden)
        hidden = nn.relu(
            nn.Dense(features=HIDDEN_DIM, name="fc_rnn_hidden")(rnn_hidden)
        )

        mean = nn.Dense(features=STATE_DIM, name="fc_state_mean_prior")(hidden)
        stddev = (
            nn.activation.softplus(
                nn.Dense(features=STATE_DIM, name="fc_state_stddev_prior")(hidden)
            )
            + min_stddev
        )
        return distrax.Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs, min_stddev=0.1):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = self.act(
            nn.Dense(features=STATE_DIM, name="fc_rnn_hidden_embedded_obs")(
                jnp.cat([rnn_hidden, embedded_obs], dim=1)
            )
        )
        mean = nn.Dense(features=STATE_DIM, name="fc_state_mean_posterior")(hidden)
        stddev = (
            nn.activation.softplus(
                nn.Dense(features=STATE_DIM, name="fc_state_stddev_posterior")(hidden)
            )
            + min_stddev
        )
        return distrax.Normal(mean, stddev)


class Observation(nn.Module):
    """Image reconstruction model"""

    @nn.compact
    def __call__(obs: chex.Array, rnn_state: chex.Array):
        hidden = nn.Dense(features=1024, name="fc")(jnp.cat([obs, rnn_state], dim=1))
        hidden = hidden.reshape((hidden.shape[0], 1024, 1, 1))
        hidden = nn.relu(
            nn.ConvTranspose(128, kernel_size=5, strides=2, name="dc1")(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(64, kernel_size=5, strides=2, name="dc2")(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(32, kernel_size=6, strides=2, name="dc3")(hidden)
        )
        obs = nn.ConvTranspose(3, kernel_size=6, strides=2, name="dc4")(hidden)
        return obs


class Reward(nn.Module):
    @nn.compact
    def __call__(obs: chex.Array, rnn_state: chex.Array):
        hidden = nn.relu(
            nn.Dense(features=HIDDEN_DIM, name="fc1")(jnp.cat([obs, rnn_state], dim=1))
        )
        hidden = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc2")(hidden))
        hidden = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc3")(hidden))
        reward = nn.Dense(features=1, name="fc4")(hidden)
        return reward


class Action(nn.Module):
    """Action model"""

    action_size: int

    @nn.compact
    def __call__(self, s):
        dtype = jnp.float32

        x = s

        x = jnp.concatenate([s], 1)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc2", dtype=dtype)(x))
        x = nn.Dense(features=self.action_size, name="fc_out", dtype=dtype)(x)

        return x


class Value(nn.Module):
    """Value model"""

    @nn.compact
    def __call__(self, s):
        dtype = jnp.float32

        x = s

        x = jnp.concatenate([s], 1)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc2", dtype=dtype)(x))
        x = nn.Dense(features=1, name="fc_out", dtype=dtype)(x)

        return x


def dreamer():
    writer = SummaryWriter()
    env = gym.make(ENV_NAME)
    random_key = random.PRNGKey(0)

    # Optimizers
    optimizer = optax.adam(learning_rate=LEARNING_RATE)

    # Neural Networks
    initializer = jax.nn.initializers.xavier_uniform()

    encoder_nn = Encoder()
    random_key, subkey = random.split(random_key)
    encoder_params = encoder_nn.init(
        subkey,
        initializer(subkey, env.observation_space.shape, jnp.float32),
    )

    rssm_nn = RSSM()
    random_key, subkey = random.split(random_key)
    rssm_params = rssm_nn.init(
        subkey,
        initializer(subkey, (1, REPRESENTATION_SIZE), jnp.float32),
        initializer(subkey, (1, env.action_space.shape[0]), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
        initializer(subkey, (1, RNN_SIZE + 1024), jnp.float32),
    )
    rssm_opt_state = optimizer.init(rssm_params)

    obs_nn = Observation
    random_key, subkey = random.split(random_key)
    obs_params = obs_nn.init(
        subkey,
        initializer(subkey, (1, REPRESENTATION_SIZE), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
    )
    obs_opt_state = optimizer.init(obs_params)

    reward_nn = Reward()
    random_key, subkey = random.split(random_key)
    reward_params = reward_nn.init(
        subkey,
        initializer(subkey, (1, REPRESENTATION_SIZE), jnp.float32),
    )
    reward_opt_state = optimizer.init(reward_params)

    action_nn = Action(env.action_space.shape[0])
    random_key, subkey = random.split(random_key)
    action_params = action_nn.init(
        subkey,
        initializer(subkey, (1, REPRESENTATION_SIZE), jnp.float32),
    )
    action_opt_state = optimizer.init(action_params)

    value_nn = Value()
    random_key, subkey = random.split(random_key)
    value_params = value_nn.init(
        subkey,
        initializer(subkey, (1, REPRESENTATION_SIZE), jnp.float32),
    )
    value_opt_state = optimizer.init(value_params)

    rb = ReplayBuffer(
        BUFFER_SIZE, env.observation_space.shape, env.action_space.shape[0]
    )

    @jax.jit
    @jax.value_and_grad
    def reward_loss_and_grad(params, states, rnn_hidden_states, rewards):
        predicted_rewards = reward_nn.apply(reward_params, states, rnn_hidden_states)
        reward_loss = (
            0.5 * (predicted_rewards[1:] - rewards[1:]).pow(2).sum(dim=2).mean()
        )
        return reward_loss

    @jax.jit
    @jax.value_and_grad
    def reconstruction_loss_and_grad(params, states, rnn_hidden_states, observations):
        reconstructed_obs = obs_nn.apply(obs_params, states, rnn_hidden_states)
        obs_loss = (
            0.5 * (reconstructed_obs[1:] - observations[1:]).pow(2).sum(dim=2).mean()
        )
        return obs_loss

    @jax.jit
    def rssm_loss(rssm_state, state, actions, rnn_hidden_state, embedded_obs):
        # from https://github.com/cross32768/PlaNet_PyTorch/blob/master/train.py
        kl_loss = 0
        states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, REPRESENTATION_SIZE))
        rnn_hidden_states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, RNN_SIZE))
        for l in range(CHUNK_LENGTH - 1):
            (next_state_prior, next_state_posterior, rnn_hidden_state,) = rssm_nn.apply(
                rssm_state, state, actions[l], rnn_hidden_state, embedded_obs[l + 1]
            )
            state = next_state_posterior.sample()
            states[l + 1] = state
            rnn_hidden_states[l + 1] = rnn_hidden_state
            kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
            kl_loss += jax.lax.clamp(kl, min=FREE_NATS).mean()

        kl_loss /= CHUNK_LENGTH - 1
        return kl_loss, (states, rnn_hidden_states)

    rssm_loss_and_grad = jax.value_and_grad(rssm_loss, has_aux=True)

    def update_action_model(action_params, action_opt_state, value_targets):
        return action_params, action_opt_state

    def update_value_model(value_params, value_opt_state, value_targets):
        return value_params, value_opt_state

    def imagine_trajectories(trans_params, action_params, representations):

        trajectrories = []

        for _ in IMAGINATION_HORIZON:
            actions = action_nn.apply(action_params, representations)
            trajectrories.append((representations, actions))
            representations = trans_nn.apply(trans_params, representations, actions)

        return trajectrories

    prev_reps, _ = env.reset()
    prev_actions = np.zeros_like(env.action_space.shape[0])
    current_observations = np.zeros_like(env.observation_space.shape)

    for global_step in tqdm(range(TOTAL_TIMESTEPS)):

        if global_step < SEED_EPISODES:
            actions = np.array([env.action_space.sample()])
        else:
            actions = action_nn.apply(action_params, prev_reps)

        next_observations, current_rewards, dones, _, infos = env.step(actions)
        rb.push(current_observations, actions, rewards, dones)

        current_observations = next_observations

        if global_step < SEED_EPISODES:
            continue

        representations = jnp.zeros((BATCH_SIZE, REPRESENTATION_SIZE))
        for _ in range(N_UPDATE_STEPS):
            observations, actions, rewards, _ = rb.sample(BATCH_SIZE, CHUNK_LENGTH)

            # Dynamics Learning
            # In original paper these parameters are shared with the RSSM
            # Here for simplicity they are separate networks
            embedded_obs = encoder_nn.apply(encoder_params, observations)
            state = jnp.zeros((BATCH_SIZE, REPRESENTATION_SIZE))
            rnn_hidden_state = jnp.zeros((BATCH_SIZE, RNN_SIZE))

            (kl_loss, (states, rnn_hidden_states)), rssm_grad = rssm_loss_and_grad(
                rssm_params, state, actions, rnn_hidden_state, embedded_obs
            )
            writer.add_scalar("loss/rssm_kl", np.array(kl_loss), global_step)
            updates, rssm_opt_state = optimizer.update(rssm_grad, rssm_opt_state)
            rssm_params = optax.apply_updates(rssm_params, updates)

            reward_loss, reward_grad = reward_loss_and_grad(
                reward_params, states, rnn_hidden_states, rewards
            )
            writer.add_scalar("loss/reward", np.array(reward_loss), global_step)
            updates, reward_opt_state = optimizer.update(reward_grad, reward_opt_state)
            reward_params = optax.apply_updates(reward_params, updates)

            obs_loss, obs_grad = reconstruction_loss_and_grad(
                obs_params, states, rnn_hidden_states, observations
            )
            writer.add_scalar("loss/observation", np.array(obs_loss), global_step)
            updates, obs_opt_state = optimizer.update(obs_grad, obs_opt_state)
            obs_params = optax.apply_updates(obs_params, updates)

            # ~~~~~~~~~~~~~~~~~~~

            # Behavior Learning
            trajectories = imagine_trajectories(
                trans_params, action_params, representations
            )
            rewards = jax.lax.vmap(reward_nn.apply)(
                reward_params, [t[0] for t in trajectories]
            )
            value_targets = jax.lax.vmap(reward_nn.apply)(
                reward_params, [t[0] for t in trajectories]
            )

            action_params, action_opt_state = update_action_model(
                action_params, action_opt_state, value_targets
            )

            value_params, value_opt_state = update_value_model(
                value_params, value_opt_state, value_targets
            )

        for _ in range(N_INTERACTION_STEPS):

            prev_reps = rep_nn.apply(rep_params, prev_reps)
            cur_rep = rep_nn.apply(prev_reps, prev_actions, cur_rep)
            cur_actions = action_nn.apply(action_params, cur_rep)
            next_obs, rewards, dones, _, infos = envs.step(actions)
            rb.add(
                prev_reps,
                next_obs.copy(),
                cur_actions.copy(),
                rewards,
                dones,
                [infos],
            )
            prev_reps = next_obs
            prev_actions = cur_actions


def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))


def preprocess_obs(obs, bit_depth=5):
    """
    Reduces the bit depth of image for the ease of training
    and convert to [-0.5, 0.5]
    In addition, add uniform random noise same as original implementation
    """
    obs = obs.astype(np.float32)
    reduced_obs = np.floor(obs / 2 ** (8 - bit_depth))
    normalized_obs = reduced_obs / 2**bit_depth - 0.5
    normalized_obs += np.random.uniform(0.0, 1.0 / 2**bit_depth, normalized_obs.shape)
    return normalized_obs


# https://github.com/cross32768/PlaNet_PyTorch/blob/022baf724b52bf79e610f9d7a31e4195d6be6455/utils.py#L4
class ReplayBuffer(object):
    """
    Replay buffer for training with RNN
    """

    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.bool)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """
        Add experience to replay buffer
        NOTE: observation should be transformed to np.uint8 before push
        """
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        """
        Sample experiences from replay buffer (almost) uniformly
        The resulting array will be of the form (batch_size, chunk_length)
        and each batch is consecutive sequence
        NOTE: too large chunk_length for the length of episode will cause problems
        """
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(
                    initial_index <= episode_borders, episode_borders < final_index
                ).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:]
        )
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1]
        )
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_indexes].reshape(batch_size, chunk_length, 1)
        return (
            jnp.array(sampled_observations),
            jnp.array(sampled_actions),
            jnp.array(sampled_rewards),
            jnp.array(sampled_done),
        )

    def __len__(self):
        return self.capacity if self.is_filled else self.index
