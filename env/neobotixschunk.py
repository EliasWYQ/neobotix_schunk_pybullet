'''
original built by X. Wang, @KIT-IPR
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
nebotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
'''
import os
import inspect

import pybullet as p
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

URDF_USE_SELF_COLLISION = 1

class NeobotixSchunk:

    def __init__(self,
                 urdfRootPath=parentdir,
                 timeStep=0.01,
                 randomInitial=False):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.5
        self.maxForce = 100
        self.useSimulation = 1
        self.useNullSpace = 0
        self.useOrientation = 1
        self.read_sim = 0
        self.randInitial =randomInitial
        self.j1_limit = np.pi
        self.j2_limit = 123/180*np.pi
        self.j3_limit = 125/180*np.pi
        self.j4_limit = 170/180*np.pi
        self.jointPosition = []
        self.baseVelocity = []
        self.wheelIndex = [1, 2]
        self.armIndex = [6, 7, 8, 9, 10, 11, 12]
        self.neobotixschunkEndEffectorIndex = 17
        self.reset()

    def reset(self):
        #self.neobotixschunkUid = p.loadURDF(os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/NeobotixSchunk.urdf"))
        self.neobotixschunkUid = p.loadURDF(os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/mp500lwa4d.urdf"), flags=p.URDF_USE_SELF_COLLISION)

        for i in range(p.getNumJoints(self.neobotixschunkUid)):
            print(p.getJointInfo(self.neobotixschunkUid, i))

        # reset arm joint positions and controllers
        if self.randInitial:
            j1 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j2 = np.random.uniform(-self.j2_limit, self.j2_limit)
            j3 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j4 = np.random.uniform(-self.j3_limit, self.j3_limit)
            j5 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j6 = np.random.uniform(-self.j4_limit, self.j4_limit)
            j7 = np.random.uniform(-self.j4_limit, self.j4_limit)
            initial_joint_positions = np.array([j1, j2, j3, j4, j5, j6, j7])

            bpos, born = p.getBasePositionAndOrientation(self.neobotixschunkUid)
            initial_basep = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), bpos[2]])
            initial_basea = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            initial_baseo = p.getQuaternionFromEuler(initial_basea)
            p.resetBasePositionAndOrientation(self.neobotixschunkUid, initial_basep, initial_baseo)
        else:
            initial_joint_positions = np.zeros(len(self.armIndex))

        initial_wheel_vel = np.zeros(len(self.wheelIndex))
        self.baseVelocity = np.zeros(len(self.wheelIndex))
        self.jointPosition = initial_joint_positions

        for i in self.wheelIndex:
            p.resetJointState(self.neobotixschunkUid, i, initial_wheel_vel[i-1])
            p.setJointMotorControl2(self.neobotixschunkUid, i, p.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheel_vel[i-1], force=self.maxForce)

        for j in self.armIndex:
            p.resetJointState(self.neobotixschunkUid, j, initial_joint_positions[j-6])
            p.setJointMotorControl2(self.neobotixschunkUid, j, p.POSITION_CONTROL,
                                    targetPosition=initial_joint_positions[j-6], force=self.maxForce)


    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        # get ee pose
        state = p.getLinkState(self.neobotixschunkUid, self.neobotixschunkEndEffectorIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
        # get base pose
        basepos, baseorn = p.getBasePositionAndOrientation(self.neobotixschunkUid)
        baseeul = p.getEulerFromQuaternion(baseorn)
        observation.extend(list(basepos))
        observation.extend(list(baseeul))

        return observation

    def check_base_velocity(self, base_vel, delta_bv):
        if np.abs(base_vel[0]) > 1.5:
            base_vel[0] = base_vel[0] - delta_bv[0]
        if np.abs(base_vel[1]) > 2:
            base_vel[1] = base_vel[1] - delta_bv[1]
        return base_vel

    def check_joint_states(self, joint_state, delta_j):
        # joint limits from lwa4d data sheet
        if np.abs(joint_state[0]) > self.j1_limit:
            joint_state[0] = joint_state[0] - delta_j[0]

        if np.abs(joint_state[1]) > self.j2_limit:
            joint_state[1] = joint_state[1] - delta_j[1]

        if np.abs(joint_state[2]) > self.j1_limit:
            joint_state[2] = joint_state[2] - delta_j[2]

        if np.abs(joint_state[3]) > self.j3_limit:
            joint_state[3] = joint_state[3] - delta_j[3]

        if np.abs(joint_state[4]) > self.j1_limit:
            joint_state[4] = joint_state[4] - delta_j[4]

        if np.abs(joint_state[5]) > self.j4_limit:
            joint_state[5] = joint_state[5] - delta_j[5]

        if np.abs(joint_state[6]) > self.j4_limit:
            joint_state[6] = joint_state[6] - delta_j[6]

        return joint_state

    def applyAction(self, action):
        dbasevelocity = np.array(action[0:2])
        djoint = np.array(action[2:9])

        # read from simulation, robot state reading from pybullet function
        if self.read_sim:
            basevl, baseva = p.getBaseVelocity(self.neobotixschunkUid)
            self.baseVelocity = np.array([basevl[0], baseva[2]])
            dae = []
            for i in self.armIndex:
                jointstate = p.getJointState(self.neobotixschunkUid, i)
                dae.append(jointstate[0])
            self.jointPosition = dae

        self.baseVelocity += dbasevelocity
        self.baseVelocity = self.check_base_velocity(self.baseVelocity, dbasevelocity)
        # vel relations from data sheet and
        # https://github.com/neobotix/neo_kinematics_differential/blob/master/common/src/DiffDrive2WKinematics.cpp
        # scale 0.01 from urdf
        wheelVelR = (self.baseVelocity[0] + 0.2535 * self.baseVelocity[1]) / 0.13 * 0.01
        wheelVelL = (self.baseVelocity[0] - 0.2535 * self.baseVelocity[1]) / 0.13 * 0.01
        wheelVel = np.array([wheelVelL, wheelVelR])

        self.jointPosition += djoint
        self.jointPosition = self.check_joint_states(self.jointPosition, djoint)

        for motor_wheel in self.wheelIndex:
            p.setJointMotorControl2(self.neobotixschunkUid, motor_wheel, p.VELOCITY_CONTROL,
                                    targetVelocity=wheelVel[motor_wheel-1], force=self.maxForce)
        for motor_arm in self.armIndex:
            p.setJointMotorControl2(self.neobotixschunkUid, motor_arm, p.POSITION_CONTROL,
                                    targetPosition=self.jointPosition[motor_arm-6], force=self.maxForce)