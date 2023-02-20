import jax
import numpy as np
import jax.numpy as jnp
import optax
import distrax
import gymnasium as gym

from typing import *
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from jax import random
from flax import linen as nn


from torch.utils.tensorboard import SummaryWriter


ALPHA = 0.2
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
ENV_NAME = "BipedalWalker-v3"
# ENV_NAME = "Ant-v4"
GAMMA = 0.99
LEARNING_STARTS = 5e3
LOG_STD_MAX = 4
LOG_STD_MIN = -20
NET_SIZE = 256
POLICY_FREQUENCY = 2
POLICY_LEARNING_RATE = 3e-4
Q_LEARNING_RATE = 1e-3
TARGET_NETWORK_FREQUENCY = 1
TAU = 0.005
TOTAL_TIMESTEPS = 1_000_000


class Q(nn.Module):

    action_size: int
    state_size: int

    @nn.compact
    def __call__(self, x, a):
        dtype = jnp.float32

        x = jnp.concatenate([x, a], 1)
        x = nn.relu(nn.Dense(features=NET_SIZE, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=NET_SIZE, name="fc2", dtype=dtype)(x))
        # x = nn.relu(nn.Dense(features=NET_SIZE, name="fc3", dtype=dtype)(x))
        x = nn.Dense(features=1, name="fc_out", dtype=dtype)(x)

        return x


class Actor(nn.Module):

    action_size: int
    state_size: int

    @nn.compact
    def __call__(self, x):
        dtype = jnp.float32

        x = nn.relu(nn.Dense(features=NET_SIZE, name="fc1", dtype=dtype)(x))
        x = nn.relu(nn.Dense(features=NET_SIZE, name="fc2", dtype=dtype)(x))
        mean = nn.Dense(features=self.action_size, name="fc_mean", dtype=dtype)(x)
        log_std = nn.tanh(
            nn.Dense(features=self.action_size, name="fc_logstd", dtype=dtype)(x)
        )

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class Alpha(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def main():
    writer = SummaryWriter()
    envs = gym.vector.SyncVectorEnv([make_env(ENV_NAME, 0, 0, False, "run_1")])
    random_key = random.PRNGKey(0)
    initializer = jax.nn.initializers.xavier_uniform()
    # Q Network
    q = Q(
        action_size=envs.single_action_space.shape[0],
        state_size=envs.single_observation_space.shape[0],
    )
    random_key, subkey = random.split(random_key)
    q1_params = q.init(
        subkey,
        initializer(subkey, (1, envs.single_observation_space.shape[0]), jnp.float32),
        initializer(subkey, (1, envs.single_action_space.shape[0]), jnp.float32),
    )
    random_key, subkey = random.split(random_key)
    q2_params = q.init(
        subkey,
        initializer(subkey, (1, envs.single_observation_space.shape[0]), jnp.float32),
        initializer(subkey, (1, envs.single_action_space.shape[0]), jnp.float32),
    )
    q1_target_params = q1_params.copy({})
    q2_target_params = q2_params.copy({})
    # Policy Network
    actor = Actor(
        state_size=envs.single_observation_space.shape[0],
        action_size=envs.single_action_space.shape[0],
    )
    random_key, subkey = random.split(random_key)
    actor_params = actor.init(
        subkey,
        initializer(subkey, (1, envs.single_observation_space.shape[0]), jnp.float32),
    )
    # Optimizers
    q_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=Q_LEARNING_RATE)
    )
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=Q_LEARNING_RATE)
    )
    actor_opt_state = actor_optimizer.init(actor_params)
    q1_opt_state = q_optimizer.init(q1_params)
    q2_opt_state = q_optimizer.init(q2_params)

    rb = ReplayBuffer(
        BUFFER_SIZE,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=True,
    )

    # Algorithm
    obs, _ = envs.reset()
    cum_reward = []
    n_runs = 0

    # Scaling
    action_scale = (envs.single_action_space.high - envs.single_action_space.low) / 2.0
    action_bias = (envs.single_action_space.high + envs.single_action_space.low) / 2.0

    # Autotune alpha
    alpha_ = Alpha(ent_coef_init=1.0)
    random_key, subkey = random.split(random_key)
    alpha_params = alpha_.init(subkey)
    target_entropy = -np.prod(envs.action_space.shape).astype(np.float32)
    alpha_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=POLICY_LEARNING_RATE)
    )
    alpha_opt_state = alpha_optimizer.init(alpha_params)

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

    # Losses
    @jax.jit
    def p_loss_(params, q1_params, q2_params, state: jnp.ndarray, alpha, random_key):
        """Policy Loss"""
        a, a_logprob = get_action(params, state, random_key)
        q_value = jnp.minimum(
            q.apply(q1_params, state, a), q.apply(q2_params, state, a)
        ).reshape(-1)
        return -(q_value - alpha * a_logprob).mean(), -a_logprob.mean()

    p_loss_and_grad = jax.value_and_grad(p_loss_, has_aux=True)

    @jax.jit
    @jax.value_and_grad
    def q_loss_and_grad(params, state, action, y_target):
        """Q Loss"""
        loss = q.apply(params, state, action).reshape(-1) - y_target
        loss = jnp.power(loss, 2).mean()
        return 0.5 * loss

    @jax.jit
    @jax.value_and_grad
    def alpha_loss_and_grad(params, entropy):
        ent_coef_value = alpha_.apply(params)
        ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
        return ent_coef_loss

    @jax.jit
    def update_alpha(alpha_params, entropy: float, alpha_opt_state):
        alpha_loss, grads = alpha_loss_and_grad(alpha_params, entropy)
        updates, alpha_opt_state = alpha_optimizer.update(grads, alpha_opt_state)
        alpha_params = optax.apply_updates(alpha_params, updates)
        return (alpha_params, alpha_opt_state, alpha_loss)

    @jax.jit
    def update_critic(
        actor_params,
        q1_target_params,
        q2_target_params,
        q1_params,
        q2_params,
        q1_opt_state,
        q2_opt_state,
        observations,
        actions,
        next_observations,
        rewards,
        alpha,
        dones,
        subkey,
    ):

        next_state_actions, next_state_log_probs = get_action(
            actor_params, next_observations, subkey
        )
        qf1_next_target = q.apply(
            q1_target_params, next_observations, next_state_actions
        )
        qf2_next_target = q.apply(
            q2_target_params, next_observations, next_state_actions
        )
        min_qf_next_target = (
            jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_state_log_probs
        ).reshape(-1)
        next_q_value = rewards.reshape(-1) + (1 - dones.reshape(-1)) * GAMMA * (
            min_qf_next_target
        )

        qf1_loss, gf1_grad = q_loss_and_grad(
            q1_params, observations, actions, next_q_value
        )
        updates, q1_opt_state = q_optimizer.update(gf1_grad, q1_opt_state)
        q1_params = optax.apply_updates(q1_params, updates)

        qf2_loss, gf2_grad = q_loss_and_grad(
            q2_params, observations, actions, next_q_value
        )
        updates, q2_opt_state = q_optimizer.update(gf2_grad, q2_opt_state)
        q2_params = optax.apply_updates(q2_params, updates)

        return q1_params, q2_params, qf1_loss, qf2_loss

    @jax.jit
    def update_actor(
        actor_params,
        q1_params,
        q2_params,
        actor_opt_state,
        alpha,
        observations,
        subkey,
    ):
        (p_loss, p_entropy), p_grad = p_loss_and_grad(
            actor_params, q1_params, q2_params, observations, alpha, subkey
        )
        updates, actor_opt_state = actor_optimizer.update(p_grad, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, updates)

        return (
            actor_params,
            actor_opt_state,
            p_loss,
            p_entropy,
        )

    for global_step in tqdm(range(TOTAL_TIMESTEPS)):
        if global_step < LEARNING_STARTS:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            random_key, subkey = random.split(random_key)
            actions, _ = get_action(actor_params, obs, subkey)

        next_obs, rewards, dones, _, infos = envs.step(actions)

        cum_reward.append(np.mean(rewards))
        if dones:
            writer.add_scalar("reward/cummulative", np.sum(cum_reward), n_runs)
            cum_reward = []
            n_runs += 1

        rb.add(obs, next_obs.copy(), actions, rewards, dones, [infos])
        obs = next_obs

        if global_step < LEARNING_STARTS:
            continue

        data = rb.sample(BATCH_SIZE)
        next_observations, observations, dones, rewards, actions = (
            jnp.array(data.next_observations.numpy()),
            jnp.array(data.observations.numpy()),
            jnp.array(data.dones.numpy()),
            jnp.array(data.rewards.numpy()),
            jnp.array(data.actions.numpy()),
        )

        random_key, subkey = random.split(random_key)
        q1_params, q2_params, qf1_loss, qf2_loss = update_critic(
            actor_params,
            q1_target_params,
            q2_target_params,
            q1_params,
            q2_params,
            q1_opt_state,
            q2_opt_state,
            observations,
            actions,
            next_observations,
            rewards,
            alpha_.apply(alpha_params),
            dones,
            subkey,
        )

        writer.add_scalar("loss/q1", np.array(qf1_loss), global_step)
        writer.add_scalar("loss/q2", np.array(qf2_loss), global_step)

        if global_step % POLICY_FREQUENCY == 0:

            for _ in range(POLICY_FREQUENCY):

                alpha = alpha_.apply(alpha_params)
                random_key, subkey = random.split(random_key)
                (actor_params, actor_opt_state, p_loss, p_entropy) = update_actor(
                    actor_params,
                    q1_params,
                    q2_params,
                    actor_opt_state,
                    alpha,
                    observations,
                    subkey,
                )
                random_key, subkey = random.split(random_key)

                (alpha_params, alpha_opt_state, alpha_loss) = update_alpha(
                    alpha_params, p_entropy, alpha_opt_state
                )

                writer.add_scalar("loss/actor", np.array(p_loss), global_step)
                writer.add_scalar("loss/alpha", np.array(alpha_loss), global_step)
                writer.add_scalar("params/alpha", np.array(alpha), global_step)

        if global_step % TARGET_NETWORK_FREQUENCY == 0:
            q1_target_params = optax.incremental_update(
                q1_params, q1_target_params, TAU
            )
            q2_target_params = optax.incremental_update(
                q2_params, q2_target_params, TAU
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
