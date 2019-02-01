from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv
import argparse, ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, Model
import tensorflow as tf
from env import CarlaEnv, ENV_CONFIG
from models import register_carla_model
from scenarios import TOWN2_STRAIGHT

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
}


ENV_CONFIG = {
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
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
    # Target value for KL divergence
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "MeanStdFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This does
    # not support minibatches.
    "simple_optimizer": False,
    # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
    # sampling tasks for performance but can waste a large portion of samples.
    "straggler_mitigation": False,
}

class NeoSchunkModelClass(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Define the layers of a custom model.
        init_w = tf.contrib.layers.xavier_initializer()
        init_b = tf.constant_initializer(0.001)
        layer1 = tf.layers.dense(input_dict["obs"], 200, activation=tf.nn.tanh,
                                  kernel_initializer=init_w, bias_initializer=init_b)
        layer2 = tf.layers.dense(layer1, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b)
        layerout = tf.layers.dense(layer2, num_outputs, activation=tf.nn.relu)

        return layerout, layer2

    def value_function(self):
        """Builds the value function output.

        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).

        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        return True

    def loss(self):
        """Builds any built-in (self-supervised) loss for the model.

        For example, this can be used to incorporate auto-encoder style losses.
        Note that this loss has to be included in the policy graph loss to have
        an effect (done for built-in algorithms).

        Returns:
            Scalar tensor for the self-supervised loss.
        """
        return tf.constant(0.0)

ModelCatalog.register_custom_model("model_env_mp500lwa4dpg70", NeoSchunkModelClass)


def env_creator(env_config):
    env = NeobotixSchunkGymEnv(renders=False, isDiscrete=False, action_dim = 9, rewardtype='rdense', randomInitial=False)
    return env  # return an env instance

register_env('env_mp500lwa4dpg70', env_creator())

def train():

    env_config = ENV_CONFIG.copy()
    env_config.update({
        "verbose": False,
        "x_res": 80,
        "y_res": 80,
        "use_depth_camera": False,
        "discrete_actions": False,
        "server_map": "/Game/Maps/Town02",
        "scenarios": TOWN2_STRAIGHT,
    })
    register_carla_model()

    ray.init(redirect_output=True)
    run_experiments({
        "carla": {
            "run": "PPO",
            "env": CarlaEnv,
            "config": {
                "env_config": env_config,
                "model": {
                    "custom_model": "carla",
                    "custom_options": {
                        "image_shape": [
                            env_config["x_res"], env_config["y_res"], 6
                        ],
                    },
                    "conv_filters": [
                        [16, [8, 8], 4],
                        [32, [4, 4], 2],
                        [512, [10, 10], 1],
                    ],
                },
                "num_workers": 1,
                "train_batch_size": 2000,
                "sample_batch_size": 100,
                "lambda": 0.95,
                "clip_param": 0.2,
                "num_sgd_iter": 20,
                "lr": 0.0001,
                "sgd_minibatch_size": 32,
                "num_gpus": 1,
            },
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()