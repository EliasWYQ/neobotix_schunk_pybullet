'''
developed by Z. Zheng, @KIT-IPR

python3 -m ray_scripts.rllib_ppo_neoschunk train

python3 -m ray_scripts.rllib_ppo_neoschunk rollout checkpoint(pathdir) #--run PPO
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import inspect
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
    config_to_use = {
        "lambda": 0.995,
        "num_workers": 3,
        "num_gpus": 0,
        "monitor": False,
        #"lambda": lambda: random.uniform(0.9, 1.0),
        "kl_coeff": 0.9,
        "sample_batch_size": 200,
        "train_batch_size": 800,
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