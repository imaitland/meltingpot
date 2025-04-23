import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind, wrap_atari_for_new_api_stack
from ray.rllib.utils.test_utils import run_rllib_example_script_experiment
from ray.tune.registry import register_env
from meltingpot import substrate



from ray import air
from ray import tune

from . import utils
from .utils import PLAYER_STR_FORMAT

# params
SUBSTRATE_NAME = "bach_or_stravinsky_in_the_matrix__repeated"
FCNET_HIDDENS = [64,64]
FCNET_ACTIVATIONS = "relu"
CONV_ACTIVATION= "relu"
VF_SHARE_LAYERS = True

# bootstrap
ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}})

def _env_creator(cfg):
    # Util function in case
    return utils.env_creator(cfg)

# register the env before init the config
register_env("meltingpot", _env_creator)

def policy_mapping_fn(agent_id, *args, **kwargs):
    index = int(agent_id.split("_")[-1])
    return PLAYER_STR_FORMAT.format(index=index)

# init config
config = PPOConfig().api_stack(enable_env_runner_and_connector_v2=True, enable_rl_module_and_learner=True)

config.env_runners(num_env_runners=0)
config.env_runners(env_to_module_connector=lambda env: FlattenObservations(multi_agent=True))

# Player roles
# "bach_fan", "stravinsky_fan"
player_roles = substrate.get_config(SUBSTRATE_NAME).default_player_roles
config.env_config= {"substrate": SUBSTRATE_NAME, "roles": player_roles}

# use env registered earlier with associated _env_creator
config.env="meltingpot"

# Warning: here we assume players have the same space... iterate over players if that's not true.
# If iterating you will need to define a multiagentspec on the config, one policy per player, rather than default config.rl_module
test_env = utils.env_creator(config.env_config)
rgb_shape = test_env.observation_spaces[f"player_{0}"]["RGB"].shape
sprite_x = rgb_shape[0] // 8
sprite_y = rgb_shape[1] // 8

# default agent config, the respective agents' observation and action spaces are defined in the env...
config.rl_module(
    model_config=DefaultModelConfig(
        fcnet_hiddens=FCNET_HIDDENS,
        fcnet_activation=FCNET_ACTIVATIONS,
        vf_share_layers=VF_SHARE_LAYERS,
        conv_filters=[
            (16, (8, 8), 8),  # First layer aligns with 8×8 sprites of the MeltingPot playspace.
            (128, (sprite_x, sprite_y), 1),  # Second layer processes the downsampled observation, using the player sprite.
        ],
        conv_activation=CONV_ACTIVATION,
    )
)

# multi agent policy
config.multi_agent(
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
)

#algo = config.build_algo()
# algo.train()

# RuntimeError: Running the example script resulted in one or more errors! [ValueError('No default configuration for obs shape [40, 40, 3], you must specify `conv_filters` manually as a model option. Default configurations are only available for inputs of the following shapes: [42, 42, K], [84, 84, K], [64, 64, K], [10, 10, K]. You may alternatively want to use a custom model or preprocessor.')]
# run_rllib_example_script_experiment(config)

stop = {
    # num iterations
    "training_iteration": 1,
}
results = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=1),
).fit()

print(results)
