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


class NeobotixSchunk:

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
        self.neobotixschunkUid = p.loadURDF(
            os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/mp500lwa4d.urdf"),
            useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
        # print('uid', self.neobotixschunkUid)
        jointNameToId = {}
        for i in range(p.getNumJoints(self.neobotixschunkUid)):
            jointInfo = p.getJointInfo(self.neobotixschunkUid, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
            # print(jointInfo)
        Id_wheel_left_joint = jointNameToId['wheel_left_joint']
        Id_wheel_right_joint = jointNameToId['wheel_right_joint']
        Id_arm_1_joint = jointNameToId['arm_1_joint']
        Id_arm_2_joint = jointNameToId['arm_2_joint']
        Id_arm_3_joint = jointNameToId['arm_3_joint']
        Id_arm_4_joint = jointNameToId['arm_4_joint']
        Id_arm_5_joint = jointNameToId['arm_5_joint']
        Id_arm_6_joint = jointNameToId['arm_6_joint']
        Id_arm_7_joint = jointNameToId['arm_7_joint']
        Id_grasping_frame_joint = jointNameToId['grasping_frame_joint']
        Id_gripper_l_joint = jointNameToId['gripper_finger_left_joint']
        Id_gripper_r_joint = jointNameToId['gripper_finger_right_joint']
        Id_gripper_joint = jointNameToId['gripper_palm_joint']
        Id_base_joint = jointNameToId['base_footprint_joint']
        Id_laser_joint = jointNameToId['laserscanner_front_joint']
        Id_armpodest_joint = jointNameToId['arm_podest_joint']

        self.wheelIndex = [Id_wheel_left_joint, Id_wheel_right_joint]
        self.armIndex = [Id_arm_1_joint, Id_arm_2_joint, Id_arm_3_joint, Id_arm_4_joint, Id_arm_5_joint, Id_arm_6_joint,
                         Id_arm_7_joint]
        self.endEffectorIndex = Id_grasping_frame_joint
        self.checkCollisonIndex = [Id_gripper_joint, Id_gripper_l_joint, Id_gripper_r_joint, Id_arm_3_joint,
                                   Id_arm_5_joint, Id_arm_6_joint, Id_base_joint, Id_laser_joint]
        self.reset()

    def reset(self):

        # disable collision between link 10 and 12 : arm link 5 and 7
        # p.setCollisionFilterPair(self.neobotixschunkUid, self.neobotixschunkUid, self.armIndex[4], self.armIndex[-1], enableCollision=0)
        # p.createConstraint(self.neobotixschunkUid, -1, self.neobotixschunkUid, 5, p.JOINT_FIXED, [0, 0, 0], [0.19, 0, 0.5], [0., 0., 0])

        # reset arm joint positions and controllers
        if self.randInitial:
            j1 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j2 = np.random.uniform(-self.j6_limit, self.j6_limit)
            j3 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j4 = np.random.uniform(-self.j4_limit, self.j4_limit)
            j5 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j6 = np.random.uniform(-self.j6_limit, self.j6_limit)
            j7 = np.random.uniform(-self.j7_limit, self.j7_limit)
            initial_joint_positions = np.array([j1, j2, j3, j4, j5, j6, j7])

            bpos, born = p.getBasePositionAndOrientation(self.neobotixschunkUid)
            initial_basep = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), bpos[2]])
            initial_basea = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
            initial_baseo = p.getQuaternionFromEuler(initial_basea)
        else:
            initial_joint_positions = np.zeros(len(self.armIndex))
            initial_basep = np.zeros(3)
            initial_baseo = np.array([0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.neobotixschunkUid, initial_basep, initial_baseo)

        initial_wheel_vel = np.zeros(len(self.wheelIndex))
        self.baseVelocity = np.zeros(len(self.wheelIndex))
        self.jointPosition = initial_joint_positions
        initial_joint_vel = np.zeros(len(self.armIndex))

        for i in range(len(self.wheelIndex)):
            p.resetJointState(bodyUniqueId=self.neobotixschunkUid, jointIndex=self.wheelIndex[i], targetValue=initial_wheel_vel[i], targetVelocity=initial_wheel_vel[i])
            p.setJointMotorControl2(bodyUniqueId=self.neobotixschunkUid, jointIndex=self.wheelIndex[i], controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheel_vel[i], force=self.maxForce)

        for j in range(len(self.armIndex)):
            p.resetJointState(bodyUniqueId=self.neobotixschunkUid, jointIndex=self.armIndex[j], targetValue=initial_joint_positions[j], targetVelocity=initial_joint_vel[j])
            p.setJointMotorControl2(bodyUniqueId=self.neobotixschunkUid, jointIndex=self.armIndex[j], controlMode=p.POSITION_CONTROL,
                                    targetPosition=initial_joint_positions[j], force=self.maxForce)

    def getActionDimension(self):
        return len(self.armIndex)+len(self.wheelIndex)

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        # get ee pose
        state = p.getLinkState(bodyUniqueId=self.neobotixschunkUid,  linkIndex=self.endEffectorIndex,
                               computeLinkVelocity=1,  computeForwardKinematics=1)
        pos = state[0]
        orn = state[1]
        if len(state) > 6:
            vel = state[6]
        euler = p.getEulerFromQuaternion(orn)
        observation.extend(list(pos))
        observation.extend(list(euler))
        # observation.extend(list(vel))
        # get base pose
        basepos, baseorn = p.getBasePositionAndOrientation(self.neobotixschunkUid)
        baseeul = p.getEulerFromQuaternion(baseorn)
        observation.extend(list(basepos))
        observation.extend(list(baseeul))

        # get joint positions
        #joint_s = p.getJointStates(bodyUniqueId=self.neobotixschunkUid,  jointIndices=list(self.armIndex))
        for i in self.armIndex:
            joints = p.getJointState(bodyUniqueId=self.neobotixschunkUid, jointIndex=i)
            observation.append((joints[0]))

        # get base linear and angular vel
        basev = p.getBaseVelocity(self.neobotixschunkUid)
        observation.extend(list(basev[0][0:2]))
        observation.append(basev[1][2])

        return observation

    def check_base_velocity(self, base_vel, delta_bv):
        if np.abs(base_vel[0]) > 1.5:
            base_vel[0] = base_vel[0] - delta_bv[0]
        if np.abs(base_vel[1]) > 2:
            base_vel[1] = base_vel[1] - delta_bv[1]
        return base_vel

    def check_joint_states(self, joint_state, delta_j):
        # joint limits from lwa4d data sheet and modified based on rviz visual
        if np.abs(joint_state[0]) > self.j1_limit:
            joint_state[0] = joint_state[0] - delta_j[0]
        if np.abs(joint_state[1]) > self.j6_limit:
            joint_state[1] = joint_state[1] - delta_j[1]
        if np.abs(joint_state[2]) > self.j1_limit:
            joint_state[2] = joint_state[2] - delta_j[2]
        if np.abs(joint_state[3]) > self.j4_limit:
            joint_state[3] = joint_state[3] - delta_j[3]
        if np.abs(joint_state[4]) > self.j1_limit:
            joint_state[4] = joint_state[4] - delta_j[4]
        if np.abs(joint_state[5]) > self.j6_limit:
            joint_state[5] = joint_state[5] - delta_j[5]
        if np.abs(joint_state[6]) > self.j7_limit:
            joint_state[6] = joint_state[6] - delta_j[6]
        return joint_state

    def applyAction(self, action):
        '''
        gripper_l_state = p.getLinkState(bodyUniqueId=self.neobotixschunkUid, linkIndex=self.checkCollisonIndex[1])
        gripper_l_pos = gripper_l_state
        gripper_r_state = p.getLinkState(bodyUniqueId=self.neobotixschunkUid, linkIndex=self.checkCollisonIndex[2])
        gripper_r_pos = gripper_r_state
        '''
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
        wheelVelR = (self.baseVelocity[0] + 0.2535 * self.baseVelocity[1]) / 0.13 #* 0.01
        wheelVelL = (self.baseVelocity[0] - 0.2535 * self.baseVelocity[1]) / 0.13 #* 0.01
        wheelVel = np.array([wheelVelL, wheelVelR])

        self.jointPosition += djoint
        self.jointPosition = self.check_joint_states(self.jointPosition, djoint)

        for motor_wheel in self.wheelIndex:
            p.setJointMotorControl2(bodyUniqueId=self.neobotixschunkUid, jointIndex=motor_wheel, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=wheelVel[motor_wheel-self.wheelIndex[0]], force=self.maxForce)
        for motor_arm in self.armIndex:
            p.setJointMotorControl2(bodyUniqueId=self.neobotixschunkUid, jointIndex=motor_arm, controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.jointPosition[motor_arm-self.armIndex[0]], force=self.maxForce)

        '''
        (0, b'base_footprint_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
        (1, b'wheel_left_joint', 0, 7, 6, 1, 0.0, 0.0, 0.0, -1.0, 500.0, 5.0, b'wheel_left_link', (0.0, 1.0, 0.0),
         (-0.1550000011920929, 0.2549999952316284, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (2, b'wheel_right_joint', 0, 8, 7, 1, 0.0, 0.0, 0.0, -1.0, 500.0, 5.0, b'wheel_right_link', (0.0, 1.0, 0.0),
         (-0.1550000011920929, -0.2549999952316284, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (3, b'hanger_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'hanger', (0.0, 0.0, 0.0),
         (0.2549999952316284, 0.0, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (4, b'arm_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_podest_link', (0.0, 0.0, 0.0),
         (0.1899999976158142, 0.0, 0.36000001430511475), (0.0, 0.0, 0.0, 1.0), 0)
        (5, b'arm_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_base_link', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.14000000059604645), (0.0, 0.0, 0.0, 1.0), 4)
        (6, b'arm_1_joint', 0, 9, 8, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 216.0, 0.43633, b'arm_1_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 5)
        (7, b'arm_2_joint', 0, 10, 9, 1, 0.9, 0.08, -2.12, 2.12, 216.0, 0.43633, b'arm_2_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.23999999463558197),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 6)
        (8, b'arm_3_joint', 0, 11, 10, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 81.5, 0.4189, b'arm_3_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
         (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 7)
        (9, b'arm_4_joint', 0, 12, 11, 1, 0.9, 0.08, -2.16, 2.16, 81.5, 0.4189, b'arm_4_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.328000009059906),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 8)
        (10, b'arm_5_joint', 0, 13, 12, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 20.7, 0.43633, b'arm_5_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
         (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 9)
        (11, b'arm_6_joint', 0, 14, 13, 1, 0.0, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_6_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.3230000138282776),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 10)
        (12, b'arm_7_joint', 0, 15, 14, 1, 0.0, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_7_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.0), (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 11)
        (13, b'gripper_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_podest_link',
         (0.0, 0.0, 0.0), (0.0, 0.0, 0.1324000060558319), (0.0, 0.0, 0.0, 1.0), 12)
        (14, b'gripper_palm_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_palm_link', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 13)
        (15, b'gripper_finger_left_joint', 4, -1, -1, 0, 0.08, 0.08, -0.0301, -0.01, 10.0, 0.041,
         b'gripper_finger_left_link', (0.0, 0.0, 0.0), (0.0, -0.004999999888241291, -0.026100002229213715),
         (0.0, 0.0, 0.0, 1.0), 14)
        (16, b'gripper_finger_right_joint', 4, -1, -1, 0, 0.08, 0.08, 0.01, 0.0301, 10.0, 0.041,
         b'gripper_finger_right_link', (0.0, 0.0, 0.0), (0.0, 0.004999999888241291, -0.026100002229213715),
         (0.0, 0.0, 0.0, 1.0), 14)
        (17, b'grasping_frame_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'grasping_frame', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.24629999697208405), (0.0, 0.0, 0.0, 1.0), 12)
        (18, b'laserscanner_front_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'laserscanner_front_link',
         (0.0, 0.0, 0.0), (-0.38999998569488525, 0.0, 0.2809999883174896),
         (0.7073880263692225, -0.7068249755751145, -0.0005628267118611763, -0.0005632750548569806), 0)
         '''