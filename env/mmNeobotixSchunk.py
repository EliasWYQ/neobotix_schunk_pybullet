import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import pybullet as p
import math
import random
import numpy as np

URDF_USE_SELF_COLLISION=1

class MMNeobotixSchunk:

    def __init__(self, urdfRootPath=parentdir, timeStep=0.01, randomInitial = False):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.randInitial = randomInitial
        self.maxVelocity = .35
        self.maxForce = 1000.
        self.useSimulation = 1
        self.useOrientation = 1
        self.schunkEndEffectorIndex = 8
        self.useNullSpace = 1
        self.j1_limit = 3.14  # limits for arm link 1, 3, 5
        self.j4_limit = 2.11  # 121 / 180 * np.pi   limits for arm link 4
        self.j6_limit = 2.00  # 115 / 180 * np.pi  # limits for arm link 2, 6
        self.j7_limit = 2.96  # 170 / 180 * np.pi  # limits for arm link 7
        # lower limits for null space
        self.ll = [-self.j1_limit, -self.j6_limit, -self.j1_limit, -self.j4_limit,
                   -self.j1_limit, -self.j6_limit, -self.j7_limit]
        # upper limits for null space
        self.ul = [self.j1_limit, self.j6_limit, self.j1_limit, self.j4_limit,
                   self.j1_limit, self.j6_limit, self.j7_limit]
        # joint ranges for null space
        self.jr = [6.283, 4.0142, 6.283, 4.2236, 6.283, 4.0142, 5.9344]
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # restposes for null space
        self.rp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.reset()

    def reset(self):
        #p.setGravity(0, 0, -9.8)
        self.neoUID = p.loadURDF(os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/base_mp500.urdf"),
                                 [0.0, 0, 0], flags=p.URDF_USE_SELF_COLLISION)
        self.schunkUID = p.loadURDF(os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/neobotixschunk/arm_lwa4d.urdf"),
                                    [0.2, 0, 0.36], flags=p.URDF_USE_SELF_COLLISION)

        for i in range(p.getNumJoints(self.neoUID)):
            print(p.getJointInfo(self.neoUID, i))
        for i in range(p.getNumJoints(self.schunkUID)):
            print(p.getJointInfo(self.schunkUID, i))
        '''
        (0, b'base_footprint_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base_link', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
        (1, b'wheel_left_joint', 0, 7, 6, 1, 0.0, 0.0, 0.0, -1.0, 500.0, 5.0, b'wheel_left_link', (0.0, 1.0, 0.0),
         (-0.1550000011920929, 0.2549999952316284, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (2, b'wheel_right_joint', 0, 8, 7, 1, 0.0, 0.0, 0.0, -1.0, 500.0, 5.0, b'wheel_right_link', (0.0, 1.0, 0.0),
         (-0.1550000011920929, -0.2549999952316284, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (3, b'hanger_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'hanger', (0.0, 0.0, 0.0),
         (0.2549999952316284, 0.0, 0.125), (0.0, 0.0, 0.0, 1.0), 0)
        (0, b'arm_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'arm_base_link', (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.14000000059604645), (0.0, 0.0, 0.0, 1.0), -1)
        (1, b'arm_1_joint', 0, 7, 6, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 216.0, 0.43633, b'arm_1_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
        (2, b'arm_2_joint', 0, 8, 7, 1, 0.9, 0.08, -2.12, 2.12, 216.0, 0.43633, b'arm_2_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.23999999463558197),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 1)
        (3, b'arm_3_joint', 0, 9, 8, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 81.5, 0.4189, b'arm_3_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
         (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 2)
        (4, b'arm_4_joint', 0, 10, 9, 1, 0.9, 0.08, -2.16, 2.16, 81.5, 0.4189, b'arm_4_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.328000009059906),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 3)
        (5, b'arm_5_joint', 0, 11, 10, 1, 0.0, 0.0, -3.12159265359, 3.12159265359, 20.7, 0.43633, b'arm_5_link',
         (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
         (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 4)
        (6, b'arm_6_joint', 0, 12, 11, 1, 0.0, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_6_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.3230000138282776),
         (0.49999999999999944, 0.49999999999999944, 0.500000021855695, 0.4999999781443049), 5)
        (7, b'arm_7_joint', 0, 13, 12, 1, 0.0, 0.0, -2.94, 2.94, 15.0, 1.2566, b'arm_7_link', (0.0, 0.0, 1.0),
         (0.0, 0.0, 0.0), (0.49999999999999944, 0.500000021855695, 0.49999999999999944, -0.4999999781443049), 6)
        (8, b'gripper_podest_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_podest_link',
         (0.0, 0.0, 0.0), (0.0, 0.0, 0.08500000089406967), (0.0, 0.0, 0.0, 1.0), 7)
        '''
        p.createConstraint(self.neoUID, -1, self.schunkUID, -1, p.JOINT_FIXED, [0, 0, 0], [0.2, 0, 0.36], [0, 0., 0])

        initial_wheelVel = [0, 0]
        self.wheels = [1, 2]
        #p.resetBaseVelocity(self.neoUID, initial_wheelVel, initial_wheelVel)
        for wheelIndex in range(len(self.wheels)):
            # reset no-zero base velocities
            # not necessary
            # p.resetJointState(self.neoUID, wheelIndex, self.wheelVel[wheelIndex], self.wheelVel[wheelIndex])
            p.setJointMotorControl2(self.neoUID, self.wheels[wheelIndex], controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=initial_wheelVel[wheelIndex], force=self.maxForce)
        huskyPos, huskyOrn = p.getBasePositionAndOrientation(self.neoUID)
        huskyEul = p.getEulerFromQuaternion(huskyOrn)

        # reset arm joint positions and controllers
        if self.randInitial:
            j1 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j2 = np.random.uniform(-self.j6_limit, self.j6_limit)
            j3 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j4 = np.random.uniform(-self.j4_limit, self.j4_limit)
            j5 = np.random.uniform(-self.j1_limit, self.j1_limit)
            j6 = np.random.uniform(-self.j6_limit, self.j6_limit)
            j7 = np.random.uniform(-self.j7_limit, self.j7_limit)
            initial_jointPositions = [j1, j2, j3, j4, j5, j6, j7]
        else:
            initial_jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.jointstates = initial_jointPositions
        self.numJoints = p.getNumJoints(self.schunkUID)
        for jointIndex in range(self.numJoints-2):
            # p.resetJointState(self.schunkUID, jointIndex+1, initial_jointPositions[jointIndex])
            p.setJointMotorControl2(self.schunkUID, jointIndex+1, controlMode=p.POSITION_CONTROL,
                                    targetPosition=initial_jointPositions[jointIndex], force=self.maxForce)

        self.wheelDeltasTurn = [1, -1, 1, -1]
        self.wheelDeltasFwd = [1, 1, 1, 1]

        # initial_kukastate = p.getLinkState(self.schunkUID, self.schunkEndEffectorIndex)
        # print('kuka',initial_kukastate)
        # self.kukastate = [initial_kukastate[0][0],initial_kukastate[0][1], initial_kukastate[0][2]]

        initial_base_vel = p.getBaseVelocity(self.neoUID)
        self.baseVel = 0
        self.baseAng = 0
        '''
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.schunkUID, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        for i in range (p.getNumJoints(self.neoUID)):
            wheelInfo = p.getJointInfo(self.neoUID, i)
            qIndex = wheelInfo[3]
            if qIndex > -1:
                self.wheelNames.append(str(wheelInfo[1]))
                self.wheelIndices.append(i)
        '''

    def getActionDimension(self):
        return len(self.motorIndices)+len(self.wheelIndices)
        # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        #huskystate = p.getLinkState(self.neoUID, 0)
        kukastate = p.getLinkState(bodyUniqueId=self.schunkUID,  linkIndex=self.schunkEndEffectorIndex,
                                   computeLinkVelocity=0,  computeForwardKinematics=0)
        state = kukastate
        #print('state: ', state)
        pos = state[0]
        orn = state[1]
        if (len(state)>6):
            vel = state[6]
        else:
            vel = [0,0,0]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        # observation.extend(list(vel))
        observation.extend(list(euler))

        huskyPos, huskyOrn = p.getBasePositionAndOrientation(self.neoUID)
        huskyEul = p.getEulerFromQuaternion(huskyOrn)
        observation.extend(list(huskyPos))
        # observation.extend(list(vel))
        observation.extend(list(huskyEul))
        # print('o', huskyPos)

        return observation

    def accurateCalculateInverseKinematics(self, kukaId, endEffectorId, targetPos, threshold, maxIter):
        closeEnough = False
        iter = 0
        while (not closeEnough and iter < maxIter):
            jointPoses = p.calculateInverseKinematics(kukaId, endEffectorId, targetPos)
            ls = p.getLinkState(kukaId, endEffectorId)
            newPos = ls[4]
            diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1
        return jointPoses

    def check_jointstates(self, joint_state, delta_j):
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

    def check_baseV(self, base_vel, delta_bv):
        if (abs(base_vel) > 1.5):
            base_vel =base_vel - delta_bv
        return base_vel

    def check_baseA(self, base_ang, delta_ba):
        if (abs(base_ang) > 2):
            base_ang =base_ang - delta_ba
        return base_ang

    def applyAction(self, motorCommands):
        # action of arm joint states changes

        if (len(motorCommands)==5):
            dp = motorCommands[0:3]
            kukastates = p.getLinkState(self.schunkUID, self.schunkEndEffectorIndex)
            pos = kukastates[4]
            eeposx = pos[0] + dp[0]
            eeposy = pos[1] + dp[1]
            eeposz = pos[2] + dp[2]

            eepos = [eeposx, eeposy, eeposz]
            print(eepos)
            # self.kukastate = eepos
            # baseVel is the translational speed of husky
            self.baseVel = self.baseVel + motorCommands[3]
            # baseAng is the rotational speed of husky
            self.baseAng = self.baseAng + motorCommands[4]
            self.baseVel = self.check_baseV(self.baseVel, motorCommands[3])
            self.baseAng = self.check_baseA(self.baseAng, motorCommands[4])

            if (self.useNullSpace == 1):
                jointPoses = p.calculateInverseKinematics(self.schunkUID, self.schunkEndEffectorIndex, eepos, lowerLimits=self.ll,
                                                              upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            else:
                threshold = 0.001
                maxIter = 100
                #jointPoses = self.accurateCalculateInverseKinematics(self.schunkUID, self.schunkEndEffectorIndex, eepos, threshold, maxIter)
                jointPoses = p.calculateInverseKinematics(self.schunkUID, self.schunkEndEffectorIndex, eepos)
            print('jp', jointPoses)
        # action of arm ee position changes
        elif (len(motorCommands) == 9):
            dp = motorCommands[0:7]
            # baseVel is the translational speed of husky
            self.baseVel = self.baseVel + motorCommands[7]
            # baseAng is the rotational speed of husky
            self.baseAng = self.baseAng + motorCommands[8]
            self.baseVel = self.check_baseV(self.baseVel, motorCommands[7])
            self.baseAng = self.check_baseA(self.baseAng, motorCommands[8])

            self.jointstates = [x+y for x, y in zip(self.jointstates, dp)]
            self.jointstates = self.check_jointstates(self.jointstates, dp)
            # print('jointstate : ', self.jointstates, abs(-9))
            jointPoses = self.jointstates

        else:
            print('motor command wrong!')

        self.wheelVelR = (self.baseVel + 0.2535 * self.baseAng) / 0.13
        self.wheelVelL = (self.baseVel - 0.2535 * self.baseAng) / 0.13
        wheelVel = [self.wheelVelL, self.wheelVelR]

        for i in range(self.numJoints-2):
            p.setJointMotorControl2(bodyUniqueId=self.schunkUID, jointIndex=i+1, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], force=1000)
        for i in range(len(self.wheels)):
            p.setJointMotorControl2(self.neoUID, self.wheels[i], p.VELOCITY_CONTROL, targetVelocity=wheelVel[i], force=1000)

