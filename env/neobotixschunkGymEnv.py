'''
original built by X. Wang & Z. Zheng, @KIT-IPR
developed by Z. Zheng
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
'''
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import time
from pkg_resources import parse_version
import pyglet

from env import neobotixschunk
from env import fieldDirection

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
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 maxSteps=1e3,
                 rewardtype='rdense',
                 action_dim=9,
                 randomInitial=True,
                 wsboundary=1,
                 colliObj=True):
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._isDiscrete = isDiscrete
        self._renders = renders
        self._maxSteps = maxSteps
        self._rewardtype = rewardtype
        self._action_dim = action_dim
        self._isEnableRandInit = randomInitial
        self.d_ws = wsboundary
        self.objsOrNot = colliObj
        self._observation = []
        self._envStepCounter = 0
        self._timeStep = 1. / 240.
        self.r_penalty = 0
        self._terminated = 0
        self._cam_dist = 4
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dis_vor = 100
        self._count = 0
        self._count_ep = 0
        self.dis_init = 1e5
        self.dis_collision = 1
        self.ee_dis = 1e5
        self.base_dis = 1e5
        self.goal = []
        self.obs_pose = []
        self.flag_collide = 0
        self._p = p

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
            # p.setRealTimeSimulation(1)
        self.seed()

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)  # , enableFileCaching=0)
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -9.8)

        p.loadURDF(os.path.join(self._urdfRoot, "neobotix_schunk_pybullet/data/plane.urdf"))
        self.goal = np.zeros(3)
        self.origoal = np.array([0, 0, 0, 1])

        self.goalUid = p.loadURDF(os.path.join(self._urdfRoot, "neobotix_schunk_pybullet/data/spheregoal.urdf"), basePosition=self.goal)

        self.obs_pose = np.ones(3)
        if self.objsOrNot:
            self.obsUid = p.loadURDF(os.path.join(self._urdfRoot, "neobotix_schunk_pybullet/data/cube.urdf"), basePosition=self.obs_pose)
        self._neobotixschunk = neobotixschunk.NeobotixSchunk(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, randomInitial=self._isEnableRandInit, wsboundary=self.d_ws)

        self.reset()

        self.roughDirection = np.zeros(3)
        self.observation_dim = len(self.getExtendedObservation())
        observation_high = np.array([largeValObservation] * self.observation_dim)
        # print('ob',observation_high)
        daction = 1
        if self._isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            self.action_bound = np.ones(self._action_dim) * daction
            self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound, dtype=np.float32)

        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.viewer = None
        self.calculateField = fieldDirection.FieldDirection()
        # help(neobotixschunk)

    def reset(self):
        self.reset_again()
        while 1:
            flag1 = self.check_collision_obs()
            flag2 = self.check_collision_self()
            if ((flag1) or (flag2)):
                self.reset_again()
            else:
                #print('no collision')
                break
        #self._count+=1
        #print(self._count)
        return np.array(self._observation)

    def reset_again(self):
        self.r_penalty = 0
        self._terminated = 0
        # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "TEST_GUI1.mp4")

        # d_space_scale = len(str(abs(self._count))) * 0.5
        # self._maxSteps = 1000 + 500 * len(str(abs(self._count)))
        d_space_scale = self.d_ws
        if self._count_ep:
            print('Scale here: ', self._count, self._count_ep, self._count/self._count_ep, d_space_scale, self._maxSteps)
        xpos = np.random.uniform(-d_space_scale, d_space_scale) + 0.20
        ypos = np.random.uniform(-d_space_scale, d_space_scale)
        zpos = np.random.uniform(0.5, 1.5)
        self.goal = np.array([xpos, ypos, zpos])
        p.resetBasePositionAndOrientation(self.goalUid, self.goal, self.origoal)


        if self.objsOrNot:
            xopos = np.random.uniform(-d_space_scale, d_space_scale) + 0.20
            yopos = np.random.uniform(-d_space_scale, d_space_scale)
            zopos = np.random.uniform(0, 1.5)
            self.obs_pose = np.array([xopos, yopos, zopos])
            p.resetBasePositionAndOrientation(self.obsUid, self.obs_pose, self.origoal)

        self._neobotixschunk.reset()

        self._envStepCounter = 0
        p.stepSimulation()
        #time.sleep(0.1)
        # time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        #eedisvec = np.subtract(self._observation[0:3], self.goal)
        self.dis_init = np.linalg.norm(self._observation[0:3])
        self.ee_dis = self.dis_init
        #bdisvec = np.subtract(self._observation[6:8], self.goal[0:2])
        self.base_dis = np.linalg.norm(self._observation[6:8])

        #return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getActionDimension(self):
        return self._action_dim

    def getExtendedObservation(self):
        observation = self._neobotixschunk.getObservation()
        #observation.extend(self.goal)
        relative_pos = self.goal - observation[0:3]
        observation[0:3] = relative_pos
        relative_pos_base = self.goal[0:2] - observation[6:8]
        observation[6:8] = relative_pos_base
        if self.objsOrNot:
            self.roughDirection = relative_pos/np.linalg.norm(relative_pos)
            # observation.append(self.flag_collide)
            observation.extend(self.roughDirection)
        self._observation = observation
        return self._observation

    def step(self, input_action):
        '''
        if self.ee_dis < 0.5:
            p_scale = 0.01
            # input_action[0:2] = input_action[0:2]*np.ones(2)*p_scale
        else:
            p_scale = 1
            # input_action[2:9] = input_action[2:9]*np.ones(7)*p_scale
        scaled_action = np.multiply(input_action, self.action_bound*p_scale)
        '''
        scaled_action = input_action
        return self.step_shaped(scaled_action)

    def step_shaped(self, action_scaled):
        self.r_penalty = 0
        for i in range(self._actionRepeat):
            self._neobotixschunk.applyAction(action_scaled)
            p.stepSimulation()
            done = self._termination()
            if done:
                self._count_ep += 1
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)

        self._actions = action_scaled
        reward = self._reward()
        return np.array(self._observation), reward, done, {}

    def render(self, mode='rgb_array', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._neobotixschunk.neobotixschunkUid)
        text = 'goal position : ' + str(self.goal) + '. ee position : ' + str(self._observation[0:3])
        self._textID = self._p.addUserDebugText(text, [0, -1, 1])
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
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def check_collision_obs(self):
        dcontact = []
        if self.objsOrNot:
            dcontact = p.getContactPoints(self.obsUid)
        self.flag_collide = len(dcontact)
        if self.flag_collide:
            self.roughDirection = dcontact[0][7]
            self.dis_collision = dcontact[0][8]
            return True
        return False

    def check_collision_self(self):
        for i in self._neobotixschunk.checkCollisonIndex:
            dcontact = p.getContactPoints(self._neobotixschunk.neobotixschunkUid, self._neobotixschunk.neobotixschunkUid, i)
            if len(dcontact):
                #print('2 collison')
                return True
        return False

    def _termination(self):
        self._observation = self.getExtendedObservation()

        if self._terminated or (self._envStepCounter >= self._maxSteps-1):
            return True
        #eedisvec = np.subtract(self._observation[0:3], self.goal)
        self.ee_dis = np.linalg.norm(self._observation[0:3])
        #bdisvec = np.subtract(self._observation[6:8], self.goal[0:2])
        self.base_dis = np.linalg.norm(self._observation[6:8])

        if self.check_collision_obs():
            self._terminated = 0
            #force, d_force = self.calculateField.compute_sum_force()
            self._observation[-3:] = self.roughDirection
            if self.dis_collision > 0.01:
                self.r_penalty = -1/(self.dis_collision**2)
            else:
                self.r_penalty = -1e4
            #print('ACHTUNG : collision with obs!')
            return False

        if self.check_collision_self():
            self._terminated = -1
            self.r_penalty = -1e5
            #print('ACHTUNG : self-collision!')
            return True

        if self.ee_dis < 0.1:
            self._terminated = 1
            self.r_penalty = 1e4
            self._count += 1
            print('Terminate:', self._observation, self.ee_dis, self.goal)
            # terminate:
            # [0.31351114847553907, -0.7641432674513139, 1.1630955439527204, 0.0017817470162401388, 0.8550287805124699, -1.0684642243321618,
            # -0.00832275367860193, -0.1757742594867312, -0.006658387818796845, 0.0010982953800254935, -0.014004384097023394, -1.0770997673085894,
            # 0.3056298875778705, -0.7944274743618929, 1.1985681959576757]
            # 0.04730281705992296
            # [0.3056298875778705, -0.7944274743618929, 1.1985681959576757]
            return True
        return False

    def _reward(self):
        if self.objsOrNot:
            closestPoints = p.getClosestPoints(self._neobotixschunk.neobotixschunkUid, self.obsUid, 0.2)
            if len(closestPoints):
                rewardgghghg = -closestPoints[0][8]  # contact distance
        #
        #     numPt = len(closestPoints)
        #     reward = -1000
        #     if (numPt > 0):
        #
        #     return reward
        delta_dis = self.ee_dis - self._dis_vor
        self._dis_vor = self.ee_dis

        tau = (self.ee_dis/self.dis_init)**2
        if tau > 1:
            penalty = (1-tau)*self.ee_dis - self.dis_init #+ self._envStepCounter/self._maxSteps/2
        else:
            penalty = 0 #self._envStepCounter/self._maxSteps/2

        if self._rewardtype == 'rdense':
            # reward = -(1-tau)*self.ee_dis - tau*self.base_dis + self.r_penalty -np.linalg.norm(self._actions)#+ penalty
            reward = 1/(self.ee_dis**2) - self.ee_dis**2 + self.r_penalty
            # reward = -reward
        elif self._rewardtype == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1
        return reward

    def _sample_action(self):
        if not self._isDiscrete:
            if self._action_dim == 9:
                d = 5
                action = np.array([np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d),
                                   np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d),
                                   np.random.uniform(-d, d), np.random.uniform(-d, d), np.random.uniform(-d, d)])
        return action

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
