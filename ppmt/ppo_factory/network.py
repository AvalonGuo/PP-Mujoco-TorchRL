from ml_collections import config_dict
from torch import nn

def ppo_config(env_config)->config_dict.ConfigDict:
    rl_config = config_dict.create(
        episode_length=env_config.episode_length,
        normalize_observations=True,
        action_repeat=env_config.action_repeat,
        reward_scaling=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(32, 32, 32, 32),
            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
            policy_obs_key="state",
            value_obs_key="state",
        ),
    )
    rl_config.num_timesteps = 20_000_000
    rl_config.num_evals = 4
    rl_config.unroll_length = 10
    rl_config.num_minibatches = 32
    rl_config.num_updates_per_batch = 8
    rl_config.discounting = 0.97
    rl_config.learning_rate = 1e-3
    rl_config.entropy_cost = 2e-2
    rl_config.num_envs = 2048
    rl_config.batch_size = 512
    rl_config.network_factory.policy_hidden_layer_sizes = (32, 32, 32, 32)

    return rl_config

def create_network(rl_config):
    


