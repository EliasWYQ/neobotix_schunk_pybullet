#!/usr/bin/env python
# main.py
# 导入环境和学习方法
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv
from ddpg.ddpg_nn import DDPG  #_dropout
import numpy as np
import time
import tensorflow as tf
from env.neobotixGymEnv import NeobotixGymEnv

# 设置全局变量
MAX_EPISODES = 5000
MAX_EP_STEPS = 1000
ON_TRAIN = 1  # True or False
LEARN_START = 10000
ALPHA = LEARN_START / MAX_EP_STEPS
BELTA = MAX_EPISODES - LEARN_START / MAX_EP_STEPS
VAR = 4  # control exploration
ACTION_NOISE = True

# 设置环境
env = NeobotixSchunkGymEnv(renders=0, isDiscrete=False, maxSteps=1000, action_dim=9)
# env = NeobotixGymEnv(renders=0, isDiscrete=False, maxSteps=1000, action_dim=2)
s_dim = env.observation_dim
a_dim = env._action_dim
a_bound = env.action_bound[1]

# 设置学习方法 (这里使用 DDPG)
tf.reset_default_graph()
rl = DDPG(a_dim, s_dim, a_bound, ACTION_NOISE)

t1 = time.time()

# 开始训练
def train():
    var = VAR

    for i in range(MAX_EPISODES):

        s = env.reset()                # 初始化回合设置
        ep_r = 0.

        STU_FLAG = 0
        for j in range(MAX_EP_STEPS):
            var = var*0.99

            # env.render()                # 环境的渲染
            a = rl.choose_action(s)     # RL 选择动作
            a = np.clip(np.random.normal(a, var), -1, 1)
            if np.sum(np.absolute(a)) <= 1e-4:
                STU_FLAG += 1
                if STU_FLAG >= 500:
                    print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                    break
            else:
                STU_FLAG = 0

            s_, r, done, info = env.step(a)   # 在环境中施加动作

            # DDPG 这种强化学习需要存放记忆库
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.pointer > LEARN_START:
                rl.learn()              # 记忆库满了, 开始学习

            s = s_                      # 变为下一回合
            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                break

    rl.save()


def eval():
    var = VAR
    rl.restore()
    i = 0.0
    j = 0
    T = 0.0
    f = open('results.txt', 'a')
    while True:
        s = env.reset()
        ep_r = 0.
        i += 1
        for j in range(MAX_EP_STEPS):
            var = var*0.9999
            a = rl.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -1, 1)
            s, r, done, info = env.step(a)
            ep_r += r
            if done or j == MAX_EP_STEPS-1:
                if done:
                    T += 1
                # else:
                #     for t in range(len(s)):
                #         f.write(str(s[t]))
                #         f.write(' ')
                #     f.write(str(env.goal['y']))
                #     f.write('\n')
                print('Ep: %i | %s | ep_r: %.1f | steps: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
        if i >= 10000:
            accuracy = T / i
            print("Steps accuracy: %f " % (accuracy))
            f.close()
            break

if ON_TRAIN:
    train()
else:
    eval()

print('running time: ', time.time()-t1)