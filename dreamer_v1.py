import jax
import unittest
import numpy as np
import jax.numpy as jnp
import optax
import distrax
import chex
import gymnasium as gym

from jax import random
from flax import linen as nn

from PIL import Image
from typing import *
from tqdm import tqdm
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter


HIDDEN_DIM = 256
RNN_SIZE = 128
STATE_DIM = 128
PIXEL_OBS_SHAPE = (1, 64, 64, 3)

BATCH_SIZE = 64
BUFFER_SIZE = 10_000
CHUNK_LENGTH = 16
ENV_NAME = "BipedalWalker-v3"
IMAGINATION_HORIZON = 50
LEARNING_RATE = 3e-4
N_INTERACTION_STEPS = 1
N_UPDATE_STEPS = 2
SEED_EPISODES = 5e3
SEQUENCE_LENGTH = 50
TOTAL_TIMESTEPS = 1_000_000
FREE_NATS = 3.0
DISCOUNT = 0.2


# Encoder, Recurrent state space, Observation and Reward models adapted from
# https://github.com/cross32768/PlaNet_PyTorch/blob/master/model.py


class Encoder(nn.Module):
    """A standard Convolution Network for image observations"""

    @nn.compact
    def __call__(self, obs: chex.Array):
        hidden = nn.relu(
            nn.Conv(features=32, kernel_size=[4, 4], strides=[2, 2], name="cv1")(obs)
        )
        hidden = nn.relu(
            nn.Conv(features=64, kernel_size=[4, 4], strides=[2, 2], name="cv2")(hidden)
        )
        hidden = nn.relu(
            nn.Conv(features=128, kernel_size=[4, 4], strides=[2, 2], name="cv3")(
                hidden
            )
        )
        return nn.relu(
            nn.Conv(features=256, kernel_size=[4, 4], strides=[2, 2], name="cv4")(
                hidden
            )
        )


class RSSM(nn.Module):
    """Joint Prior / Posterion network mimicking the Kalman filter"""

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        rnn_hidden: chex.Array,
        next_state: chex.Array,
    ) -> Tuple[distrax.Normal, chex.Array]:
        """
        h_t+1 = f(s_t, a_t, h_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """
        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, next_state)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden, min_stddev=0.1):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = nn.relu(
            nn.Dense(features=HIDDEN_DIM, name="fc_state_action")(
                jnp.concatenate([state, action], -1)
            )
        )
        rnn_hidden, hidden = nn.GRUCell()(rnn_hidden, hidden)
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

        hidden = nn.relu(
            nn.Dense(features=STATE_DIM, name="fc_rnn_hidden_embedded_obs")(
                jnp.concatenate([rnn_hidden, embedded_obs], -1)
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
    def __call__(self, obs: chex.Array, rnn_state: chex.Array):
        hidden = nn.Dense(features=1024, name="fc")(
            jnp.concatenate([obs, rnn_state], -1)
        )
        hidden = hidden.reshape((-1, 1, 1, 1024))
        hidden = nn.relu(
            nn.ConvTranspose(
                128, padding="VALID", kernel_size=[5, 5], strides=[2, 2], name="dc1"
            )(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(
                64, padding="VALID", kernel_size=[5, 5], strides=[2, 2], name="dc2"
            )(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(
                32, padding="VALID", kernel_size=[6, 6], strides=[2, 2], name="dc3"
            )(hidden)
        )
        obs = nn.ConvTranspose(
            3, padding="VALID", kernel_size=[6, 6], strides=[2, 2], name="dc4"
        )(hidden)

        return obs


class Reward(nn.Module):
    @nn.compact
    def __call__(self, obs: chex.Array, rnn_state: chex.Array):
        x = jnp.concatenate([obs, rnn_state], -1)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc1")(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc2")(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc3")(x))
        return nn.Dense(features=1, name="fc4")(x)


def dreamer():
    writer = SummaryWriter()
    env = gym.wrappers.PixelObservationWrapper(
        gym.make(ENV_NAME, render_mode="rgb_array")
    )

    random_key = random.PRNGKey(0)

    # Optimizers
    optimizer = optax.adam(learning_rate=LEARNING_RATE)

    # Neural Networks
    initializer = jax.nn.initializers.xavier_uniform()

    encoder_nn = Encoder()
    random_key, subkey = random.split(random_key)
    encoder_params = encoder_nn.init(
        subkey,
        initializer(subkey, PIXEL_OBS_SHAPE, jnp.float32),
    )

    sample_encoded = encoder_nn.apply(
        encoder_params, jnp.zeros(PIXEL_OBS_SHAPE)
    ).reshape(1, -1)

    rssm_nn = RSSM()
    random_key, subkey = random.split(random_key)
    rssm_params = rssm_nn.init(
        subkey,
        initializer(subkey, (1, STATE_DIM), jnp.float32),
        initializer(subkey, (1, env.action_space.shape[0]), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
        sample_encoded,
    )
    rssm_opt_state = optimizer.init(rssm_params)

    obs_nn = Observation()
    random_key, subkey = random.split(random_key)
    obs_params = obs_nn.init(
        subkey,
        initializer(subkey, (1, STATE_DIM), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
    )
    obs_opt_state = optimizer.init(obs_params)

    reward_nn = Reward()
    random_key, subkey = random.split(random_key)
    reward_params = reward_nn.init(
        subkey,
        initializer(subkey, (1, STATE_DIM), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
    )
    reward_opt_state = optimizer.init(reward_params)

    rb = ReplayBuffer(BUFFER_SIZE, PIXEL_OBS_SHAPE[1:], env.action_space.shape[0])

    @jax.jit
    @jax.value_and_grad
    def reward_loss_and_grad(params, states, rnn_hidden_states, rewards):
        predicted_rewards = reward_nn.apply(params, states, rnn_hidden_states)

        reward_loss = (
            0.5 * jnp.power(predicted_rewards[1:] - rewards[1:], 2).mean([0, 1]).sum()
        )
        return reward_loss

    @jax.jit
    def reconstruction_loss(params, states, rnn_hidden_states, observations):
        reconstructed_obs = obs_nn.apply(params, states, rnn_hidden_states).reshape(
            (CHUNK_LENGTH, BATCH_SIZE, PIXEL_OBS_SHAPE[1], PIXEL_OBS_SHAPE[2], 3)
        )

        obs_loss = (
            0.5
            * jnp.power(reconstructed_obs[1:] - observations[1:], 2).mean([0, 1]).sum()
        )

        return obs_loss, (reconstructed_obs)

    reconstruction_loss_and_grad = jax.value_and_grad(reconstruction_loss, has_aux=True)

    @jax.jit
    def rssm_loss(params, embedded_obs: chex.Array, actions: chex.Array, random_key):
        # from https://github.com/cross32768/PlaNet_PyTorch/blob/master/train.py
        state = jnp.zeros((BATCH_SIZE, STATE_DIM))
        states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, STATE_DIM))
        rnn_hidden_state = jnp.zeros((BATCH_SIZE, RNN_SIZE))
        rnn_hidden_states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, RNN_SIZE))

        kl_loss = 0
        for l in range(CHUNK_LENGTH - 1):
            random_key, subkey = random.split(random_key)

            (next_state_prior, next_state_posterior, rnn_hidden_state,) = rssm_nn.apply(
                params, state, actions[l], rnn_hidden_state, embedded_obs[l + 1]
            )
            state = next_state_posterior.sample(seed=subkey)
            states = states.at[l + 1].set(state)
            rnn_hidden_states = rnn_hidden_states.at[l + 1].set(rnn_hidden_state)
            kl = next_state_prior.kl_divergence(next_state_posterior).sum(1)
            kl_loss += jax.lax.clamp(x=kl, min=3.0, max=1e8).mean()

        kl_loss /= CHUNK_LENGTH - 1
        return kl_loss, (states, rnn_hidden_states)

    rssm_loss_and_grad = jax.value_and_grad(rssm_loss, has_aux=True)

    env.reset()
    current_observations = np.zeros(PIXEL_OBS_SHAPE[1:])

    for global_step in tqdm(range(TOTAL_TIMESTEPS)):
        if global_step < SEED_EPISODES:
            actions = np.array(env.action_space.sample())
        else:
            pass
            # TODO: Implement
            # actions = action_nn.apply(action_params, prev_reps["pixels"])

        next_observations, rewards, dones, _, _ = env.step(actions)
        rb.push(current_observations, actions, rewards, dones)
        current_observations = next_observations["pixels"]

        if len(rb) < BATCH_SIZE:
            continue

        for _ in range(N_UPDATE_STEPS):
            observations, actions, rewards, _ = rb.sample(BATCH_SIZE, CHUNK_LENGTH)

            # Adjust dimensions to match (chuck_size, batch_size, ...) pattern
            observations = jnp.transpose(observations, (1, 0, 2, 3, 4))
            actions = jnp.transpose(actions, (1, 0, 2))
            rewards = jnp.transpose(rewards, (1, 0, 2))

            # Dynamics Learning
            # ~~~~~~~~~~~~~~~~~~~
            # In original paper these parameters are shared with the RSSM
            # Here for simplicity they are separate networks
            embedded_obs = encoder_nn.apply(encoder_params, observations).reshape(
                CHUNK_LENGTH, BATCH_SIZE, -1
            )

            random_key, subkey = random.split(random_key)
            (kl_loss, (states, rnn_hidden_states)), rssm_grad = rssm_loss_and_grad(
                rssm_params, embedded_obs, actions, subkey
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

            (obs_loss, (reconstructed_obs),), obs_grad = reconstruction_loss_and_grad(
                obs_params, states, rnn_hidden_states, observations
            )

            writer.add_image(
                "image/original",
                np.transpose(
                    np.array(observations[1][0] * 255).astype(np.uint8), (2, 0, 1)
                ),
                global_step,
            )

            writer.add_image(
                "image/reconstructed",
                np.transpose(
                    np.array(reconstructed_obs[1][0] * 255).astype(np.uint8), (2, 0, 1)
                ),
                global_step,
            )

            writer.add_scalar("loss/observation", np.array(obs_loss), global_step)
            updates, obs_opt_state = optimizer.update(obs_grad, obs_opt_state)
            obs_params = optax.apply_updates(obs_params, updates)


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
        self.done = np.zeros((capacity, 1), dtype=np.bool_)

        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        """
        Add experience to replay buffer
        NOTE: observation should be transformed to np.uint8 before push
        """

        observation = resize(observation, (PIXEL_OBS_SHAPE[1], PIXEL_OBS_SHAPE[2]))
        observation = (observation * 255).astype(np.uint8)

        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done

        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

        # print(observation.shape)
        # im = Image.fromarray((observation * 255).astype(np.uint8))
        # im.save("images/{index}.jpeg".format(index=self.index))

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


class TestReplayBuffer(unittest.TestCase):
    def test_push(self):
        env = gym.wrappers.PixelObservationWrapper(
            gym.make(ENV_NAME, render_mode="rgb_array")
        )
        rb = ReplayBuffer(BUFFER_SIZE, PIXEL_OBS_SHAPE[1:], env.action_space.shape[0])
        actions = np.array(env.action_space.sample())
        next_observations, rewards, dones, _, _ = env.step(actions)
        rb.push(next_observations["pixels"], actions, rewards, dones)


if __name__ == "__main__":
    dreamer()
