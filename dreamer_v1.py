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


HIDDEN_DIM = 100  # 200
RNN_SIZE = 100  # 200
STATE_DIM = 32  # 64 wirks better
PIXEL_OBS_SHAPE = (1, 64, 64, 3)

ACTOR_CRITIC_DIM = 128  # 256
LOG_STD_MAX = 4
LOG_STD_MIN = -20

BATCH_SIZE = 25  # 50
BUFFER_SIZE = 500_000  # 1_000_000
CHUNK_LENGTH = 5  # 10
ENV_NAME = "HalfCheetah-v4"
IMAGINATION_HORIZON = 6  # 12
LEARNING_RATE = 1e-3
N_INTERACTION_STEPS = 1
N_UPDATE_STEPS = 1
SEED_EPISODES = 1_000_000_000
TOTAL_TIMESTEPS = 1_000_000_000
DISCOUNT = 0.2
MAX_EPISODE_LENGTH = 100


# Encoder, Recurrent state space, Observation and Reward models adapted from
# https://github.com/cross32768/PlaNet_PyTorch/blob/master/model.py


class Encoder(nn.Module):
    """A standard Convolution Network for image observations"""

    @nn.compact
    def __call__(self, observation: chex.Array):
        x = nn.relu(
            nn.Conv(features=32, kernel_size=[4, 4], strides=[2, 2], name="cv1")(
                observation
            )
        )
        x = nn.relu(
            nn.Conv(features=64, kernel_size=[4, 4], strides=[2, 2], name="cv2")(x)
        )
        x = nn.relu(
            nn.Conv(features=128, kernel_size=[4, 4], strides=[2, 2], name="cv3")(x)
        )
        return nn.relu(
            nn.Conv(features=256, kernel_size=[4, 4], strides=[2, 2], name="cv4")(x)
        )


class Decoder(nn.Module):
    """Image reconstruction model"""

    @nn.compact
    def __call__(self, obs: chex.Array, rnn_state: chex.Array):
        hidden = nn.Dense(features=1024, name="fc")(
            jnp.concatenate([obs, rnn_state], -1)
        )
        hidden = hidden.reshape((-1, 1, 1, 1024))
        hidden = nn.relu(
            nn.ConvTranspose(
                128, padding="VALID", kernel_size=[5, 5], strides=[2, 2], name="cv1"
            )(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(
                64, padding="VALID", kernel_size=[5, 5], strides=[2, 2], name="cv2"
            )(hidden)
        )
        hidden = nn.relu(
            nn.ConvTranspose(
                32, padding="VALID", kernel_size=[6, 6], strides=[2, 2], name="cv3"
            )(hidden)
        )
        obs = nn.ConvTranspose(
            3, padding="VALID", kernel_size=[6, 6], strides=[2, 2], name="cv4"
        )(hidden)

        return obs


class RSSM(nn.Module):
    """Joint Prior / Posterion network mimicking the Kalman filter"""

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        rnn_hidden: chex.Array,
        next_state_encoded: chex.Array,
    ) -> Tuple[distrax.Normal, chex.Array]:
        """
        h_t+1 = f(s_t, a_t, h_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """

        next_state_prior, rnn_hidden = self.prior(state, action, rnn_hidden)
        next_state_posterior = self.posterior(rnn_hidden, next_state_encoded)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden, min_stddev=0.1):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        x = jnp.concatenate([state, action], -1)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc_state_action")(x))
        rnn_hidden, x = nn.GRUCell()(rnn_hidden, x)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc_rnn_hidden")(rnn_hidden))

        mean = nn.Dense(features=STATE_DIM, name="fc_state_mean_prior")(x)
        stddev = (
            nn.activation.softplus(
                nn.Dense(features=STATE_DIM, name="fc_state_stddev_prior")(x)
            )
            + min_stddev
        )

        return distrax.Normal(mean, stddev), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs, min_stddev=0.1):
        """
        Compute posterior q(s_t | h_t, o_t)
        """

        x = jnp.concatenate([rnn_hidden, embedded_obs], -1)
        x = nn.relu(nn.Dense(features=STATE_DIM, name="fc_rnn_hidden_embedded_obs")(x))
        mean = nn.Dense(features=STATE_DIM, name="fc_state_mean_posterior")(x)
        stddev = (
            nn.activation.softplus(
                nn.Dense(features=STATE_DIM, name="fc_state_stddev_posterior")(x)
            )
            + min_stddev
        )
        return distrax.Normal(mean, stddev)


class Reward(nn.Module):
    @nn.compact
    def __call__(self, state: chex.Array, rnn_hidden: chex.Array):
        x = jnp.concatenate([state, rnn_hidden], -1)
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc1")(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc2")(x))
        x = nn.relu(nn.Dense(features=HIDDEN_DIM, name="fc3")(x))
        return nn.Dense(features=1, name="fc4")(x)


class JointModel(nn.Module):
    @nn.compact
    def __call__(
        self,
        observations: chex.Array,
        states: chex.Array,
        actions: chex.Array,
        rnn_hidden: chex.Array,
        random_key,
    ):
        batch_size = observations.shape[0]
        next_state_encoded = Encoder()(observations).reshape(batch_size, -1)

        next_state_prior, next_state_posterior, rnn_hidden = RSSM()(
            states, actions, rnn_hidden, next_state_encoded
        )
        next_state_sample = next_state_posterior.sample(seed=random_key)

        reconstructed_reward = Reward()(next_state_sample, rnn_hidden)
        reconstructed_observation = Decoder()(next_state_sample, rnn_hidden)

        return (
            next_state_prior,
            next_state_posterior,
            next_state_sample,
            rnn_hidden,
            reconstructed_reward,
            reconstructed_observation,
        )


class NextStatePriorAndReward(nn.Module):
    @nn.compact
    def __call__(self, state, action, rnn_hidden):
        next_state_prior, _, rnn_hidden = RSSM()(
            state, action, rnn_hidden, jnp.zeros((1, 4096))
        )
        reward = Reward()(state, rnn_hidden)
        return next_state_prior, rnn_hidden, reward


class Value(nn.Module):

    action_size: int
    state_size: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32

        x = nn.relu(nn.Dense(features=ACTOR_CRITIC_DIM, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=ACTOR_CRITIC_DIM, name="fc2", dtype=dtype)(x))
        x = nn.Dense(features=1, name="fc_out", dtype=dtype)(x)

        return x


class Actor(nn.Module):
    action_size: int
    state_size: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32

        x = nn.relu(nn.Dense(features=ACTOR_CRITIC_DIM, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=ACTOR_CRITIC_DIM, name="fc2", dtype=dtype)(x))
        mean = nn.Dense(features=self.action_size, name="fc_mean", dtype=dtype)(x)
        log_std = nn.tanh(
            nn.Dense(features=self.action_size, name="fc_logstd", dtype=dtype)(x)
        )

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


def dreamer():
    writer = SummaryWriter()
    orig_env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.PixelObservationWrapper(orig_env)

    random_key = random.PRNGKey(0)

    # Optimizers
    joint_optimizer = optax.adam(learning_rate=LEARNING_RATE)
    actor_critic_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=LEARNING_RATE)
    )

    # NNs
    initializer = jax.nn.initializers.xavier_uniform()
    # Joint RSSM NN
    joint_nn = JointModel()
    random_key, subkey = random.split(random_key)
    joint_params = joint_nn.init(
        subkey,
        initializer(subkey, PIXEL_OBS_SHAPE, jnp.float32),
        initializer(subkey, (1, STATE_DIM), jnp.float32),
        initializer(subkey, (1, env.action_space.shape[0]), jnp.float32),
        initializer(subkey, (1, RNN_SIZE), jnp.float32),
        subkey,
    )
    joint_opt_state = joint_optimizer.init(joint_params)

    # Actor (Policy) NN
    action_dim = orig_env.action_space.shape[0]
    actor = Actor(state_size=(1, STATE_DIM), action_size=action_dim)
    random_key, subkey = random.split(random_key)
    actor_params = actor.init(
        subkey,
        initializer(subkey, (1, STATE_DIM), jnp.float32),
    )
    actor_opt_state = actor_critic_optimizer.init(actor_params)

    action_scale = (orig_env.action_space.high - orig_env.action_space.low) / 2.0
    action_bias = (orig_env.action_space.high + orig_env.action_space.low) / 2.0

    # Next State Prior NN
    next_state_prior_and_reward_nn = NextStatePriorAndReward()

    # Critic (Value) NN
    value_nn = Value(state_size=(1, STATE_DIM), action_size=action_dim)
    random_key, subkey = random.split(random_key)
    value_params = value_nn.init(
        subkey,
        initializer(subkey, (1, STATE_DIM), jnp.float32),
    )
    value_opt_state = actor_critic_optimizer.init(value_params)

    rb = ReplayBuffer(BUFFER_SIZE, PIXEL_OBS_SHAPE[1:], env.action_space.shape[0])

    # Sampling
    @jax.jit
    def get_action(params, x, random_key, sigma=1e-6):
        mean, log_std = actor.apply(params, x)
        std = jnp.exp(log_std)
        normal = distrax.Normal(mean, std)
        x_t, log_prob = normal.sample_and_log_prob(seed=random_key)
        y_t = jnp.tanh(x_t)
        action = y_t * action_scale + action_bias

        log_prob -= jnp.log(action_scale * (1 - y_t**2) + sigma)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob

    @jax.jit
    def joint_loss(
        params,
        observations: chex.Array,
        actions: chex.Array,
        rewards: chex.Array,
        random_key,
    ):
        # from https://github.com/cross32768/PlaNet_PyTorch/blob/master/train.py
        state = jnp.zeros((BATCH_SIZE, STATE_DIM))
        rnn_hidden_state = jnp.zeros((BATCH_SIZE, RNN_SIZE))

        states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, STATE_DIM))
        reconstructed_observations = jnp.zeros(
            (CHUNK_LENGTH, BATCH_SIZE, *PIXEL_OBS_SHAPE[1:])
        )
        reconstructed_rewards = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, 1))
        rnn_hidden_states = jnp.zeros((CHUNK_LENGTH, BATCH_SIZE, RNN_SIZE))

        kl_loss = 0.0
        reconstruction_loss = 0.0
        reward_loss = 0.0

        for l in range(CHUNK_LENGTH - 1):
            random_key, subkey = random.split(random_key)

            (
                next_state_prior,
                next_state_posterior,
                next_state_sample,
                rnn_hidden_state,
                reconstructed_reward,
                reconstructed_observation,
            ) = joint_nn.apply(
                params, observations[l + 1], state, actions[l], rnn_hidden_state, subkey
            )

            states = states.at[l + 1].set(next_state_sample)
            rnn_hidden_states = rnn_hidden_states.at[l + 1].set(rnn_hidden_state)

            reconstructed_observations = reconstructed_observations.at[l + 1].set(
                reconstructed_observation
            )
            reconstructed_rewards = reconstructed_rewards.at[l + 1].set(
                reconstructed_reward
            )

            kl = next_state_posterior.kl_divergence(next_state_prior)
            kl_loss += jax.lax.clamp(x=kl, min=3.0, max=1e6).mean()

            reconstruction_loss += (
                0.5
                * jnp.power(reconstructed_observation - observations[l + 1], 2)
                .mean([0, 1])
                .sum()
            )

            reward_loss += (
                0.5
                * jnp.power(reconstructed_rewards - rewards[l + 1], 2)
                .mean([0, 1])
                .sum()
            )

        kl_loss /= CHUNK_LENGTH - 1
        reconstruction_loss /= CHUNK_LENGTH - 1
        reward_loss /= CHUNK_LENGTH - 1

        joint_loss = kl_loss + reconstruction_loss + reward_loss

        return joint_loss, (
            (kl_loss, reconstruction_loss, reward_loss),
            states,
            reconstructed_observations,
            reconstructed_rewards,
            rnn_hidden_states,
        )

    joint_loss_and_grad = jax.value_and_grad(joint_loss, has_aux=True)

    @jax.jit
    def update_joint_nn(
        joint_params, joint_opt_state, observations, actions, rewards, random_key
    ):
        (
            (
                cummulative_loss,
                (
                    (kl_loss, reconstruction_loss, reward_loss),
                    states,
                    reconstructed_observations,
                    reconstructed_rewards,
                    rnn_hidden_states,
                ),
            ),
            joint_grad,
        ) = joint_loss_and_grad(
            joint_params,
            observations,
            actions,
            rewards,
            random_key,
        )

        updates, joint_opt_state = joint_optimizer.update(joint_grad, joint_opt_state)
        joint_params = optax.apply_updates(joint_params, updates)

        return (
            (cummulative_loss, kl_loss, reconstruction_loss, reward_loss),
            joint_params,
            joint_opt_state,
            states,
            rnn_hidden_states,
            reconstructed_observations,
            reconstructed_rewards,
        )

    @jax.jit
    def imagine_trajectories(states, rnn_hidden_states, random_key):
        trajectories = []
        for i in (0, states.shape[0]):
            trajectory = []
            for _ in range(0, IMAGINATION_HORIZON):
                random_key, subkey1, subkey2 = random.split(random_key, 3)
                # Get next best action from policy
                action, _ = get_action(actor_params, states[i].reshape(1, -1), subkey1)

                (
                    next_state_prior,
                    rnn_hidden_state,
                    reward,
                ) = next_state_prior_and_reward_nn.apply(
                    joint_params,
                    states[i].reshape(1, -1),
                    action.reshape(1, -1),
                    rnn_hidden_states[i].reshape(1, -1),
                )

                value = value_nn.apply(value_params, states[i].reshape(1, -1))
                value_target = 0.0

                trajectory.append(
                    (
                        states[i].reshape(-1),
                        action.reshape(-1),
                        reward.reshape(-1),
                        value.reshape(-1),
                        value_target,
                    )
                )

                rnn_hidden_states = rnn_hidden_states.at[i].set(
                    rnn_hidden_state.reshape(-1)
                )
                state_sample = next_state_prior.sample(seed=subkey2)
                states = states.at[i].set(state_sample.reshape(-1))

            trajectories.append(trajectory)

        # def estimate_value(trajectory, offset) -> np.array:
        #     """Estimates the value of the given state."""
        #     # Uses Equation (5) from paper https://arxiv.org/pdf/1912.01603.pdf
        #     # Should be switched to Equation (7) once rest of implementation is verified

        #     print(len(trajectory))

        #     for t in trajectory[offset:]:
        #         print(len(t))
        #         print(t[2])
        #         print("----")

        #     rewards = jnp.array([t[2] for t in trajectory[offset:]])
        #     return jnp.sum(rewards)

        # for j, trajectory in enumerate(trajectories):
        #     for h in range(IMAGINATION_HORIZON):
        #         value_estimate = estimate_value(trajectory, h)
        #         trajectories[j][4] = value_estimate

        return trajectories

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

        if dones or global_step % MAX_EPISODE_LENGTH == 0:
            env.reset()

        if len(rb) < BATCH_SIZE:
            continue

        for _ in range(N_UPDATE_STEPS):
            observations, actions, rewards, _ = rb.sample(BATCH_SIZE, CHUNK_LENGTH)

            observations = preprocess_obs(observations)
            # Adjust dimensions to match (chuck_size, batch_size, ...) pattern
            observations = jnp.transpose(observations, (1, 0, 2, 3, 4))
            actions = jnp.transpose(actions, (1, 0, 2))
            rewards = jnp.transpose(rewards, (1, 0, 2))

            # We train a model jointly
            random_key, subkey = random.split(random_key)
            (
                (cummulative_loss, kl_loss, reconstruction_loss, reward_loss),
                joint_params,
                joint_opt_state,
                states,
                rnn_hidden_states,
                reconstructed_observations,
                _reconstructed_rewards,
            ) = update_joint_nn(
                joint_params,
                joint_opt_state,
                observations,
                actions,
                rewards,
                random_key,
            )

            writer.add_scalar(
                "loss/cummulative", np.array(cummulative_loss), global_step
            )
            writer.add_scalar("loss/rssm_kl", np.array(kl_loss), global_step)
            writer.add_scalar("loss/reward", np.array(reward_loss), global_step)
            writer.add_scalar(
                "loss/reconstruction", np.array(reconstruction_loss), global_step
            )

            writer.add_image(
                "image/original",
                np.transpose(
                    np.array(observations[1][10] * 255).astype(np.uint8),
                    (2, 0, 1),
                ),
                global_step,
            )

            writer.add_image(
                "image/reconstructed",
                np.transpose(
                    np.array(reconstructed_observations[1][10] * 255).astype(np.uint8),
                    (2, 0, 1),
                ),
                global_step,
            )

            # Imagine trajectories (s, a, r, v) from each state
            random_key, subkey = random.split(random_key)
            trajectories = imagine_trajectories(
                states.reshape((-1, STATE_DIM)),
                rnn_hidden_states.reshape((-1, RNN_SIZE)),
                subkey,
            )

            # exit()


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
