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
URDF_USE_SELF_COLLISION_EXCLUDE_PARENT = 1  # unused
URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS = 1  # unused


class Neobotix:

    def __init__(self,
                 urdfRootPath=parentdir,
                 timeStep=0.01,
                 randomInitial=False):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.5  # unused yet
        self.maxForce = 1000
        self.useSimulation = 1  # unused yet
        self.useNullSpace = 0  # unused yet
        self.useOrientation = 1  # unused yet
        self.read_sim = 0
        self.randInitial =randomInitial
        self.j1_limit = np.pi  # limits for arm link 1, 3, 5
        self.j4_limit = 120/180*np.pi  # limits for arm link 4
        self.j6_limit = 115/180*np.pi  # limits for arm link 2, 6
        self.j7_limit = 170/180*np.pi  # limits for arm link 7
        self.jointPosition = []
        self.baseVelocity = []
        self.wheelIndex = []
        self.armIndex = []
        self.endEffectorIndex = []
        self.checkCollisonIndex = []
        # self.neobotixschunkUid = None
        # load robot model
        self.neobotixUid = p.loadURDF(
            os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/base_mp500.urdf"))
        # print('uid', self.neobotixUid)
        jointNameToId = {}
        for i in range(p.getNumJoints(self.neobotixUid)):
            jointInfo = p.getJointInfo(self.neobotixUid, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
            # print(jointInfo)
        Id_wheel_left_joint = jointNameToId['wheel_left_joint']
        Id_wheel_right_joint = jointNameToId['wheel_right_joint']
        Id_base_joint = jointNameToId['base_footprint_joint']

        self.wheelIndex = [Id_wheel_left_joint, Id_wheel_right_joint]
        self.reset()

    def reset(self):

        # disable collision between link 10 and 12 : arm link 5 and 7
        # p.setCollisionFilterPair(self.neobotixUid, self.neobotixUid, self.armIndex[4], self.armIndex[-1], enableCollision=0)
        # p.createConstraint(self.neobotixUid, -1, self.neobotixUid, 5, p.JOINT_FIXED, [0, 0, 0], [0.19, 0, 0.5], [0., 0., 0])

        # reset arm joint positions and controllers
        if self.randInitial:

            bpos, born = p.getBasePositionAndOrientation(self.neobotixUid)
            initial_basep = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), bpos[2]])
            initial_basea = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            initial_baseo = p.getQuaternionFromEuler(initial_basea)
        else:
            initial_basep = np.zeros(3)
            initial_baseo = np.array([0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.neobotixUid, initial_basep, initial_baseo)

        initial_wheel_vel = np.zeros(len(self.wheelIndex))
        self.baseVelocity = np.zeros(len(self.wheelIndex))

        for i in range(len(self.wheelIndex)):
            p.resetJointState(bodyUniqueId=self.neobotixUid, jointIndex=self.wheelIndex[i], targetValue=initial_wheel_vel[i], targetVelocity=initial_wheel_vel[i])
            p.setJointMotorControl2(bodyUniqueId=self.neobotixUid, jointIndex=self.wheelIndex[i], controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheel_vel[i], force=self.maxForce)

    def getActionDimension(self):
        return len(self.armIndex)+len(self.wheelIndex)

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        # get base pose
        basepos, baseorn = p.getBasePositionAndOrientation(self.neobotixUid)
        baseeul = p.getEulerFromQuaternion(baseorn)
        observation.extend(list(basepos))
        observation.extend(list(baseeul))

        # get base linear and angular vel
        # basev = p.getBaseVelocity(self.neobotixUid)
        # observation.extend(list(basev[0][0:2]))
        # observation.append(basev[1][2])

        return observation

    def check_base_velocity(self, base_vel, delta_bv):
        if np.abs(base_vel[0]) > 1.5:
            base_vel[0] = base_vel[0] - delta_bv[0]
        if np.abs(base_vel[1]) > 2:
            base_vel[1] = base_vel[1] - delta_bv[1]
        return base_vel

    def applyAction(self, action):
        '''
        gripper_l_state = p.getLinkState(bodyUniqueId=self.neobotixUid, linkIndex=self.checkCollisonIndex[1])
        gripper_l_pos = gripper_l_state
        gripper_r_state = p.getLinkState(bodyUniqueId=self.neobotixUid, linkIndex=self.checkCollisonIndex[2])
        gripper_r_pos = gripper_r_state
        '''
        dbasevelocity = np.array(action[0:2])

        # read from simulation, robot state reading from pybullet function
        if self.read_sim:
            basevl, baseva = p.getBaseVelocity(self.neobotixUid)
            self.baseVelocity = np.array([basevl[0], baseva[2]])

        self.baseVelocity += dbasevelocity
        self.baseVelocity = self.check_base_velocity(self.baseVelocity, dbasevelocity)
        # vel relations from data sheet and
        # https://github.com/neobotix/neo_kinematics_differential/blob/master/common/src/DiffDrive2WKinematics.cpp
        # scale 0.01 from urdf
        wheelVelR = (self.baseVelocity[0] + 0.2535 * self.baseVelocity[1]) / 0.13 #* 0.01
        wheelVelL = (self.baseVelocity[0] - 0.2535 * self.baseVelocity[1]) / 0.13 #* 0.01
        wheelVel = np.array([wheelVelL, wheelVelR])

        for motor_wheel in self.wheelIndex:
            p.setJointMotorControl2(bodyUniqueId=self.neobotixUid, jointIndex=motor_wheel, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=wheelVel[motor_wheel-self.wheelIndex[0]], force=self.maxForce)
