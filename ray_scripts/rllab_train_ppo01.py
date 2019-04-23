from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import inspect
import time
import random
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.agent import with_common_config, COMMON_CONFIG
from ray.rllib.models import MODEL_DEFAULTS

from ray.rllib import train
from ray.rllib import rollout

'''
python3 -m ddpg.rllab_train_ppo01 train

python3 -m ddpg.rllab_train_ppo01 rollout checkpoint(pathdir) #--run PPO

'''

def env_input_config(train_or_rollout):
    envInputs = {
        'urdfRoot': parentdir,
        'renders': False,
        'isDiscrete': False,
        'action_dim': 9,
        'rewardtype': 'rdense',
        'randomInitial': False,
        'actionRepeat': 1,
        'isEnableSelfCollision': True,
        'maxSteps': 1e3,
        'wsboundary': 1,
    }
    if not train_or_rollout:
        envInputs['renders'] = True
    return envInputs


def env_creator(env_config):
    env = NeobotixSchunkGymEnv(**env_config)
    return env  # return an env instance

def train_agent(config):
    parser = argparse.ArgumentParser(
        description="Train or Run an RLlib Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommand_group = parser.add_subparsers(
        help="Commands to train or run an RLlib agent.", dest="command")
    train_parser = train.create_parser(
        lambda **kwargs: subcommand_group.add_parser("train", **kwargs))
    rollout_parser = rollout.create_parser(
        lambda **kwargs: subcommand_group.add_parser("rollout", **kwargs))
    options = parser.parse_args()

    register_env('env_mp500lwa4dpg70', env_creator)
    configFromYaml = {
        'train-ppo':{
            'env': 'env_mp500lwa4dpg70',
            'run': 'PPO',
            'config': config,
            'checkpoint_freq': 1000,
            #'local_dir': "~/train_results",
            'stop':{
                "timesteps_total": 1e10,
            },
        }         
    }

    ray.init()

    if options.command == "train":
        configFromYaml['train-ppo']['config']['env_config'] = env_input_config(True)
        run_experiments(configFromYaml)
        #train.run(options, train_parser)
    elif options.command == "rollout":
        options.run = configFromYaml['train-ppo']['run']
        options.env = configFromYaml['train-ppo']['env']
        options.no_render = True
        options.steps = 1000
        options.out = None
        configFromYaml['train-ppo']['config']['env_config'] = env_input_config(False)
        configFromYaml['train-ppo']['config']['monitor'] = True
        options.config = configFromYaml['train-ppo']['config']
        rollout.run(options, rollout_parser)
    else:
        parser.print_help()


if __name__ == '__main__':
    print('default common', COMMON_CONFIG)
    print('default model', MODEL_DEFAULTS)

    #config_to_be_used = COMMON_CONFIG
    #config_to_be_used['env_config'] = env_input_config()
    #config_to_be_used['num_workers'] = 15
    config_to_use = {
        #"lambda": 0.995,
        "num_workers": 3,
        "num_gpus": 0,
        #"num_cpus_per_worker": 1,
        #"num_gpus_per_worker": 0,
        "monitor": False,
        "lambda": lambda: random.uniform(0.9, 1.0),
        #"clip_param": lambda: random.uniform(0.01, 0.5),#
        #"num_sgd_iter": lambda: random.randint(1, 30),
        "kl_coeff": 0.9,
        "sample_batch_size": 2000,
        "train_batch_size": 6000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "lr": 3e-4,
        "vf_loss_coeff": 1.0,
        "vf_share_layers": False,
        "clip_param": 0.3,
        "vf_clip_param": 1e3,
        "simple_optimizer": False,
        # Whether to rollout "complete_episodes" or "truncate_episodes"
        "batch_mode": "truncate_episodes",
        "synchronize_filters": True,
        "model":
            {
                "fcnet_activation": "tanh",
                "fcnet_hiddens": [256, 256],
            },
    }
    train_agent(config_to_use)



'''

DEFAULT_CONFIG = with_common_config({
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Size of batches collected from each worker
    "sample_batch_size": 200,
    # Number of timesteps collected for each SGD round
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "lr": 5e-5,
    # Learning rate schedule
    "lr_schedule": None,
    # Share layers for value function
    "vf_share_layers": False,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount
    "grad_clip": None,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This does
    # not support minibatches.
    "simple_optimizer": False,
    # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
    # sampling tasks for performance but can waste a large portion of samples.
    "straggler_mitigation": False,
})



COMMON_CONFIG = {
    # === Debugging ===
    # Whether to write episode stats and videos to the agent log dir
    "monitor": False,
    # Set the ray.rllib.* log level for the agent process and its evaluators.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level).
    "log_level": "INFO",
    # Callbacks that will be run during various phases of training. These all
    # take a single "info" dict as an argument. For episode callbacks, custom
    # metrics can be attached to the episode by updating the episode object's
    # custom metrics dict (see examples/custom_metrics_and_callbacks.py).
    "callbacks": {
        "on_episode_start": None,     # arg: {"env": .., "episode": ...}
        "on_episode_step": None,      # arg: {"env": .., "episode": ...}
        "on_episode_end": None,       # arg: {"env": .., "episode": ...}
        "on_sample_end": None,        # arg: {"samples": .., "evaluator": ...}
        "on_train_result": None,      # arg: {"trainer": ..., "result": ...}
        "on_postprocess_traj": None,  # arg: {"batch": ..., "episode": ...}
    },
    # Whether to attempt to continue training if a worker crashes.
    "ignore_worker_failures": False,

    # === Policy ===
    # Arguments to pass to model. See models/catalog.py for a full list of the
    # available model options.
    "model": MODEL_DEFAULTS,
    # Arguments to pass to the policy optimizer. These vary by optimizer.
    "optimizer": {},

    # === Environment ===
    # Discount factor of the MDP
    "gamma": 0.99,
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": None,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": False,
    # Arguments to pass to the env creator
    "env_config": {},
    # Environment name can also be passed via config
    "env": None,
    # Whether to clip rewards prior to experience postprocessing. Setting to
    # None means clip for Atari only.
    "clip_rewards": None,
    # Whether to np.clip() actions to the action space low/high range spec.
    "clip_actions": True,
    # Whether to use rllib or deepmind preprocessors by default
    "preprocessor_pref": "deepmind",

    # === Evaluation ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,
    # Extra arguments to pass to evaluation workers.
    # Typical usage is to pass extra args to evaluation env creator
    # and to disable exploration by computing deterministic actions
    # TODO(kismuz): implement determ. actions and include relevant keys hints
    "evaluation_config": {},

    # === Resources ===
    # Number of actors used for parallelism
    "num_workers": 2,
    # Number of GPUs to allocate to the driver. Note that not all algorithms
    # can take advantage of driver GPUs. This can be fraction (e.g., 0.3 GPUs).
    "num_gpus": 0,
    # Number of CPUs to allocate per worker.
    "num_cpus_per_worker": 1,
    # Number of GPUs to allocate per worker. This can be fractional.
    "num_gpus_per_worker": 0,
    # Any custom resources to allocate per worker.
    "custom_resources_per_worker": {},
    # Number of CPUs to allocate for the driver. Note: this only takes effect
    # when running in Tune.
    "num_cpus_for_driver": 1,

    # === Execution ===
    # Number of environments to evaluate vectorwise per worker.
    "num_envs_per_worker": 1,
    # Default sample batch size (unroll length). Batches of this size are
    # collected from workers until train_batch_size is met. When using
    # multiple envs per worker, this is multiplied by num_envs_per_worker.
    "sample_batch_size": 200,
    # Training batch size, if applicable. Should be >= sample_batch_size.
    # Samples batches will be concatenated together to this size for training.
    "train_batch_size": 200,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    # (Deprecated) Use a background thread for sampling (slightly off-policy)
    "sample_async": False,
    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter"
    "observation_filter": "NoFilter",
    # Whether to synchronize the statistics of remote filters.
    "synchronize_filters": True,
    # Configure TF for single-process operation by default
    "tf_session_args": {
        # note: overriden by `local_evaluator_tf_session_args`
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        "allow_soft_placement": True,  # required by PPO multi-gpu
    },
    # Override the following tf session args on the local evaluator
    "local_evaluator_tf_session_args": {
        # Allow a higher level of parallelism by default, but not unlimited
        # since that can cause crashes with many concurrent drivers.
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    # Whether to LZ4 compress individual observations
    "compress_observations": False,
    # Drop metric batches from unresponsive workers after this many seconds
    "collect_metrics_timeout": 180,
    # Smooth metrics over this many episodes.
    "metrics_smoothing_episodes": 100,
    # If using num_envs_per_worker > 1, whether to create those new envs in
    # remote processes instead of in the same worker. This adds overheads, but
    # can make sense if your envs can take much time to step / reset
    # (e.g., for StarCraft). Use this cautiously; overheads are significant.
    "remote_worker_envs": False,
    # Timeout that remote workers are waiting when polling environments.
    # 0 (continue when at least one env is ready) is a reasonable default,
    # but optimal value could be obtained by measuring your environment
    # step / reset and model inference perf.
    "remote_env_batch_wait_ms": 0,

    # === Offline Datasets ===
    # Specify how to generate experiences:
    #  - "sampler": generate experiences via online simulation (default)
    #  - a local directory or file glob expression (e.g., "/tmp/*.json")
    #  - a list of individual file paths/URIs (e.g., ["/tmp/1.json",
    #    "s3://bucket/2.json"])
    #  - a dict with string keys and sampling probabilities as values (e.g.,
    #    {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2}).
    #  - a function that returns a rllib.offline.InputReader
    "input": "sampler",
    # Specify how to evaluate the current policy. This only has an effect when
    # reading offline experiences. Available options:
    #  - "wis": the weighted step-wise importance sampling estimator.
    #  - "is": the step-wise importance sampling estimator.
    #  - "simulation": run the environment in the background, but use
    #    this data for evaluation only and not for learning.
    "input_evaluation": ["is", "wis"],
    # Whether to run postprocess_trajectory() on the trajectory fragments from
    # offline inputs. Note that postprocessing will be done using the *current*
    # policy, not the *behaviour* policy, which is typically undesirable for
    # on-policy algorithms.
    "postprocess_inputs": False,
    # If positive, input batches will be shuffled via a sliding window buffer
    # of this number of batches. Use this if the input data is not in random
    # enough order. Input is delayed until the shuffle buffer is filled.
    "shuffle_buffer_size": 0,
    # Specify where experiences should be saved:
    #  - None: don't save any experiences
    #  - "logdir" to save to the agent log dir
    #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
    #  - a function that returns a rllib.offline.OutputWriter
    "output": None,
    # What sample batch columns to LZ4 compress in the output data.
    "output_compress_columns": ["obs", "new_obs"],
    # Max output file size before rolling over to a new file.
    "output_max_file_size": 64 * 1024 * 1024,

    # === Multiagent ===
    "multiagent": {
        # Map from policy ids to tuples of (policy_graph_cls, obs_space,
        # act_space, config). See policy_evaluator.py for more info.
        "policy_graphs": {},
        # Function mapping agent ids to policy ids.
        "policy_mapping_fn": None,
        # Optional whitelist of policies to train, or None for all policies.
        "policies_to_train": None,
    },
}



MODEL_DEFAULTS = {
    # === Built-in options ===
    # Filter config. List of [out_channels, kernel, stride] for each filter
    "conv_filters": None,
    # Nonlinearity for built-in convnet
    "conv_activation": "relu",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "tanh",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [256, 256],
    # For control envs, documented in ray.rllib.models.Model
    "free_log_std": False,
    # (deprecated) Whether to use sigmoid to squash actions to space range
    "squash_to_range": False,

    # == LSTM ==
    # Whether to wrap the model with a LSTM
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20
    "max_seq_len": 20,
    # Size of the LSTM cell
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1}, r_{t-1} to LSTM
    "lstm_use_prev_action_reward": False,

    # == Atari ==
    # Whether to enable framestack for Atari envs
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom preprocessor to use
    "custom_preprocessor": None,
    # Name of a custom model to use
    "custom_model": None,
    # Extra options to pass to the custom classes
    "custom_options": {},
}

'''