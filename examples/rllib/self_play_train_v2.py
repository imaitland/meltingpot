# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs an example of a self-play training experiment."""


from collections.abc import Collection, Mapping, Sequence, Set
from ml_collections import config_dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env
from meltingpot import substrate

from ray import air
from ray import tune

from . import utils
from .utils import PLAYER_STR_FORMAT, MeltingPotEnv

# params
VF_SHARE_LAYERS = True
NUM_ROLLOUT_WORKERS = 1
ROLLOUT_FRAGMENT_LENGTH = 10


def get_config(
        player_roles: Sequence[str],
        env_name: str = "meltingpot",
        substrate_name: str = "bach_or_stravinsky_in_the_matrix__repeated",
        num_rollout_workers: int = 1,
        rollout_fragment_length: int = 100,
        train_batch_size: int = 6400,
        fcnet_hiddens=(64, 64),
        fcnet_activations="relu",
        conv_activation="relu",
        post_fcnet_hiddens=(384,),
        lstm_cell_size: int = 256,
        sgd_minibatch_size: int = 128,
        sprite_x: int = 5,
        sprite_y: int = 5,
):
    """Get the configuration for running an agent on a substrate using RLLib.

    We need the following 2 pieces to run the training:

    Args:
      player_roles (Sequence[str]): Role names of the players.
      env_name (str, optional): Environment name. Defaults to "meltingpot", must be registered with ray
      substrate_name: The name of the MeltingPot substrate, coming from
        `substrate.AVAILABLE_SUBSTRATES`.
      num_rollout_workers: The number of workers for playing games. Defaults to 1. Use 0 if debugging in the IDE
      rollout_fragment_length: Unroll time for learning.
      train_batch_size: Batch size (batch * rollout_fragment_length)
      fcnet_hiddens: Fully connected layers.
      post_fcnet_hiddens: Layer sizes after the fully connected torso.
      lstm_cell_size: Size of the LSTM.
      sgd_minibatch_size: Size of the mini-batch for learning.
      sprite_x: The number of x dimensions of the player's observation_space sprite. [WARNING] This only works if all players have the same observation space dimensions.
      sprite_y: The number of y dimensions of the player's observation_space sprite sprite. [WARNING] This only works if all players have the same observation space dimensions.

    Returns:
      The configuration for running the experiment.
    """

    # utils
    def policy_mapping_fn(agent_id, *args, **kwargs):
        index = int(agent_id.split("_")[-1])
        return PLAYER_STR_FORMAT.format(index=index)

    config = (PPOConfig()
              # env name here must match an env registered with the ray.tune.registry
              .environment(env_name, env_config={"substrate": substrate_name, "roles": player_roles})
              # enable Rllib latest api features
              .api_stack(enable_env_runner_and_connector_v2=True, enable_rl_module_and_learner=True)
              # set to 0 to get IDE debugger to work.
              .env_runners(num_env_runners=num_rollout_workers)
              # Flatten the Melting Pot observation_spaces dict to a RGB vector, must be matched by RLLib default, or policy custom conv_filters attribute
              .env_runners(env_to_module_connector=lambda env: FlattenObservations(multi_agent=True))
              .rl_module(
                  model_config=DefaultModelConfig(
                      fcnet_hiddens=fcnet_hiddens,
                      fcnet_activation=fcnet_activations,
                      vf_share_layers=VF_SHARE_LAYERS,
                      conv_filters=[
                          (16, (8, 8), 8),  # First layer aligns with 8×8 sprites of the MeltingPot playspace.
                          (128, (sprite_x, sprite_y), 1),  # Second layer processes the downsampled observation, using the player sprite.
                      ],
                      conv_activation=conv_activation,
                  )
              )
              .multi_agent(
                  # ref: https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/different_spaces_for_agents.py#L98
                  # Use a simple set of policy IDs. Spaces for the individual policies
                  # are inferred automatically using reverse lookup via the
                  # `policy_mapping_fn` and the env provided spaces for the different
                  # agents. Alternatively, you could use:
                  # policies: {main0: PolicySpec(...), main1: PolicySpec}
                  policies={
                      PLAYER_STR_FORMAT.format(index=i): (None, None, None, {})
                      for i in range(len(player_roles))
                  },
                  policy_mapping_fn=policy_mapping_fn,  # Using the separated function here
              ))
    return config

# Ray env register requires a callable param.
def curried_env_creator(substrate_name, player_roles):
    def env_creator(config):
        """Outputs an environment for registering."""
        env = substrate.build(name=substrate_name, roles=player_roles)
        env = MeltingPotEnv(env)
        return env
    return env_creator

def main():
  env_name="meltingpot"
  substrate_name="bach_or_stravinsky_in_the_matrix__repeated"
  player_roles = substrate.get_config(substrate_name).default_player_roles

  # Optimization: We define a custom 2 layer conv_filter that matches the MeltingPot Observation space, to do this we need to know sprite dimensions
  # [WARNING] here we assume substrate's players have the same observation_space dimensions... consider MultiRLModuleSpec (https://docs.ray.io/en/latest/rllib/rl-modules.html#construction-through-rlmodulespecs) in get_config
  config = get_config(
      player_roles=player_roles,
      substrate_name=substrate_name,
      env_name=env_name,
      sprite_x=5,
      sprite_y=5
  )

  register_env(env_name, curried_env_creator(substrate_name,player_roles))

  ray.init()
  stop = {
      "training_iteration": 1,
  }
  results = tune.Tuner(
      "PPO",
      param_space=config.to_dict(),
      run_config=air.RunConfig(stop=stop, verbose=1),
  ).fit()

  print(results)
  assert results.num_errors == 0


if __name__ == "__main__":
  main()
