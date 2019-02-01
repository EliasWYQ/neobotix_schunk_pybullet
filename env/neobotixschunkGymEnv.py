'''
original built by X. Wang, @KIT-IPR
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
'''
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import time

from pkg_resources import parse_version
import pyglet

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from env import neobotixschunk

pyglet.clock.set_fps_limit(10000)

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class NeobotixSchunkGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=parentdir,
                 actionRepeat=50,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 maxSteps=1000,
                 rewardtype='rdense',
                 action_dim=9,
                 randomInitial=False):
        # print("init")
        self._p = p
        self._action_dim = action_dim
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._renders = renders
        self._rewardtype = rewardtype
        self._maxSteps = maxSteps
        self._isEnableRandInit = randomInitial
        self._isDiscrete = isDiscrete
        self._textID = None
        self._observation = []
        self._envStepCounter = 0
        self._timeStep = 0.01
        self._terminated = 0
        self._cam_dist = 4
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dis_vor = 100
        self._count = 0

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self._seed()
        self.reset()
        observation_dim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * observation_dim)

        da = 1
        if self._isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            self.action_bound = np.ones(self._action_dim) * da
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self._terminated = 0
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(numSolverIterations=150)

        p.loadURDF(os.path.join(self._urdfRoot, "neobotix_schunk_pybullet/data/plane.urdf"), [0, 0, 0])

        d_space_scale = len(str(abs(self._count))) * 0.5
        # self._maxSteps = 1000 + 500 * len(str(abs(self._count)))
        print('scale here: ', self._count, d_space_scale, self._maxSteps)
        # d_space_scale = 1
        xpos = np.random.uniform(-d_space_scale, d_space_scale) + 0.20
        ypos = np.random.uniform(-d_space_scale, d_space_scale)
        zpos = np.random.uniform(0.4, 1.3)
        self.goal = np.array([xpos, ypos, zpos])

        self.goalUid = p.loadURDF(os.path.join(self._urdfRoot, "neobotix_schunk_pybullet/data/spheregoal.urdf"), self.goal)
        self._neobotixschunk = neobotixschunk.NeobotixSchunk(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, randomInitial=self._isEnableRandInit)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()

        eedisvec = np.subtract(self._observation[0:3], self.goal)
        self.dis_init = np.linalg.norm(eedisvec)

        return np.array(self._observation)

    def getActionDimension(self):
        return self._action_dim

    def getExtendedObservation(self):
        observation = self._neobotixschunk.getObservation()
        observation.extend(self.goal)
        self._observation = observation
        # print('observation: ', self._observation)
        return self._observation

    def __del__(self):
        p.disconnect()

    def _step(self, action):
        p_scale = 0.01
        action_scaled = np.multiply(action, self.action_bound*p_scale)
        for i in range(self._actionRepeat):
            self._neobotixschunk.applyAction(action_scaled)
            p.stepSimulation()
            done = self._termination()
            if done:
                break
            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._actions = action
        reward = self._reward()

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        self._observation = self.getExtendedObservation()
        if self._terminated or (self._envStepCounter > self._maxSteps):
            return True

        eedisvec = np.subtract(self._observation[0:3], self.goal)
        self.ee_dis = np.linalg.norm(eedisvec)
        bdisvec = np.subtract(self._observation[6:8], self.goal[0:2])
        self.base_dis = np.linalg.norm(bdisvec)

        if self.ee_dis < 0.05:
            self._terminated = 1
            self._count += 1
            print('terminate:', self._observation, self.ee_dis, self.goal)
            # terminate:
            # [0.31351114847553907, -0.7641432674513139, 1.1630955439527204, 0.0017817470162401388, 0.8550287805124699, -1.0684642243321618,
            # -0.00832275367860193, -0.1757742594867312, -0.006658387818796845, 0.0010982953800254935, -0.014004384097023394, -1.0770997673085894,
            # 0.3056298875778705, -0.7944274743618929, 1.1985681959576757]
            # 0.04730281705992296
            # [0.3056298875778705, -0.7944274743618929, 1.1985681959576757]
            return True
        return False

    def _reward(self):
        # rewards is accuracy of target position
        # closestPoints = self._p.getClosestPoints(self._neobotixschunk.neobotixschunkUid, self.goalUid, 1000,self._neobotixschunk.neobotixschunkEndEffectorIndex,-1)
        #
        #     numPt = len(closestPoints)
        #     reward = -1000
        #     if (numPt > 0):
        #         reward = -closestPoints[0][8]  # contact distance
        #     return reward
        delta_dis = self.ee_dis - self._dis_vor
        self._dis_vor = self.ee_dis

        tau = (self.ee_dis/self.dis_init)**2
        if tau > 1:
            penalty = (1-tau)*self.ee_dis + self._envStepCounter/self._maxSteps/2
        else:
            penalty = self._envStepCounter/self._maxSteps/2

        if self._rewardtype == 'rdense':
            reward = (1-tau)*self.ee_dis + tau*self.base_dis - penalty
            # reward = self.ee_dis
            reward = -reward
        elif self._rewardtype == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1
        return reward

    def _render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._neobotixschunk.neobotixschunkUid)
        text = 'goal position : ' + str(self.goal) + '. ee position : ' + str(self._observation[0:3])
        self._textID = p.addUserDebugText(text, [0, -2, 2])
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _sample_action(self):
        if not self._isDiscrete:
            if self._action_dim == 9:
                d = 0.01
                action = np.array([np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d),
                                   np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d),
                                   np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d)])
        return action

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step

