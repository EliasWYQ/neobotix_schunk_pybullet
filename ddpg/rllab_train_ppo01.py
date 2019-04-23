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
        'wsrange': 1,
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
            'checkpoint_freq': 100,
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
    config_to_use = {
        "lambda": 0.995,
        "num_workers": 11,
        "num_gpus": 1,
        "monitor": False,
        #"lambda": lambda: random.uniform(0.9, 1.0),
        "kl_coeff": 0.9,
        "sample_batch_size": 200,
        "train_batch_size": 2200,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 30,
        "lr": 3e-4,
        "vf_loss_coeff": 1.0,
        "vf_share_layers": False,
        "clip_param": 0.3,
        "vf_clip_param": 9e8,
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

    ''''
    (pid=6350) 2019-04-23 19:53:58,948      INFO ppo.py:104 -- Important! Since 0.7.0, observation normalization is no longer enabled by default. To enable running-mean normalization, set 'observation_filter': 'MeanStdFilter'. You can ignore this message if your environment doesn't require observation normalization.
2019-04-23 19:53:59,064 WARNING util.py:62 -- The `experiment_checkpoint` operation took 0.11681318283081055 seconds to complete, which may be a performance bottleneck.
(pid=6355) success rate :  2 87 0.022988505747126436
(pid=6356) success rate :  1 87 0.011494252873563218
Result for PPO_env_mp500lwa4dpg70_0:
  custom_metrics: {}
  date: 2019-04-23_19-54-00
  done: false
  episode_len_mean: 978.19
  episode_reward_max: 534.5586318670825
  episode_reward_mean: -1021.6559748439066
  episode_reward_min: -1837.2675541654241
  episodes_this_iter: 2
  episodes_total: 954
  experiment_id: 354c8bd4337a494cbbae8483f1782415
  hostname: IAR-IPR-C1081
  info:
    grad_time_ms: 841.653
    learner:
      default_policy:
        cur_kl_coeff: 2.5628905296325684
        cur_lr: 0.0003000000142492354
        entropy: 13.3450345993042
        kl: 0.015339342877268791
        policy_loss: -0.10350044816732407
        total_loss: 2.2976021766662598
        vf_explained_var: 0.9980353116989136
        vf_loss: 2.3617894649505615
    load_time_ms: 0.596
    num_steps_sampled: 935000
    num_steps_trained: 924800
    sample_time_ms: 515.566
    update_time_ms: 2.908
  iterations_since_restore: 425
  node_ip: 141.3.81.81
  num_healthy_workers: 11
  num_metric_batches_dropped: 0
  off_policy_estimator: {}
  pid: 6350
  policy_reward_mean: {}
  sampler_perf:
    mean_env_wait_ms: 1.080450664356542
    mean_inference_ms: 0.8721615178319209
    mean_processing_ms: 0.14680029361900465
  time_since_restore: 519.17840051651
  time_this_iter_s: 1.1778597831726074
  time_total_s: 519.17840051651
  timestamp: 1556042040
  timesteps_since_restore: 935000
  timesteps_this_iter: 2200
  timesteps_total: 935000
  training_iteration: 425
  
== Status ==
Using FIFO scheduling algorithm.
Resources requested: 12/12 CPUs, 1/1 GPUs
Unknown memory usage. Please run `pip install psutil` (or ray[debug]) to resolve)
Result logdir: /home/zheng/ray_results/train-ppo
Number of trials: 1 ({'RUNNING': 1})
RUNNING trials:
 - PPO_env_mp500lwa4dpg70_0:    RUNNING, [12 CPUs, 1 GPUs], [pid=6350], 519 s, 425 iter, 935000 ts, -1.02e+03 rew

(pid=6350) 2019-04-23 19:54:00,131      INFO ppo.py:104 -- Important! Since 0.7.0, observation normalization is no longer enabled by default. To enable running-mean normalization, set 'observation_filter': 'MeanStdFilter'. You can ignore this message if your environment doesn't require observation normalization.

    '''