import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
import pybullet as p
import math
import random
import numpy as np


class MKuka:

    def __init__(self, urdfRootPath=parentdir, timeStep=0.01, randomInitial = False):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.randInitial = randomInitial
        self.maxVelocity = .35
        self.maxForce = 200.
        self.useSimulation = 1
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.reset()
        self.useNullSpace = 0
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def reset(self):
        #p.setGravity(0, 0, -9.8)
        self.kukaUid = p.loadURDF(os.path.join(self.urdfRootPath, "neobotix_schunk_pybullet/data/kuka_iiwa/model_free_base.urdf"),[0,0,1])

        # for i in range (p.getNumJoints(self.kukaUid)):
            # print(p.getJointInfo(self.kukaUid,i))

        p.createConstraint(self.kukaUid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, -1], [0., 0., 0], [0, 0, 0, 1])

        # reset arm joint positions and controllers
        if self.randInitial:
            j1 = random.uniform(-2.967, 2.967)
            j2 = random.uniform(-2.094, 2.094)
            j3 = random.uniform(-2.967, 2.967)
            j4 = random.uniform(-2.094, 2.094)
            j5 = random.uniform(-2.967, 2.967)
            j6 = random.uniform(-2.094, 2.094)
            j7 = random.uniform(-3.054, 3.054)
            initial_jointPositions = [j1, j2, j3, j4, j5, j6, j7]
        else:
            initial_jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.jointstates = initial_jointPositions
        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.kukaUid, jointIndex, initial_jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid, jointIndex, controlMode=p.POSITION_CONTROL,
                                    targetPosition=initial_jointPositions[jointIndex], force=self.maxForce)

        initial_kukastate = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
        # print('kuka',initial_kukastate)
        self.kukastate = [initial_kukastate[0][0],initial_kukastate[0][1], initial_kukastate[0][2]]


    def getActionDimension(self):
        return len(self.numJoints)
        # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        #huskystate = p.getLinkState(self.huskyUid, 0)
        kukastate = p.getLinkState( bodyUniqueId=self.kukaUid,  linkIndex=self.kukaEndEffectorIndex,
                                    computeLinkVelocity=1,  computeForwardKinematics=1)
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
        if (abs(joint_state[0]) > 2.967):
            joint_state[0] = joint_state[0] - delta_j[0]

        if abs(joint_state[1]) > 2.094:
            joint_state[1] = joint_state[1] - delta_j[1]

        if abs(joint_state[2]) > 2.967:
            joint_state[2] = joint_state[2] - delta_j[2]

        if abs(joint_state[3]) > 2.094:
            joint_state[3] = joint_state[3] - delta_j[3]

        if abs(joint_state[4]) > 2.967:
            joint_state[4] = joint_state[4] - delta_j[4]

        if abs(joint_state[5]) > 2.094:
            joint_state[5] = joint_state[5] - delta_j[5]

        if abs(joint_state[6]) > 3.054:
            joint_state[6] = joint_state[6] - delta_j[6]

        return joint_state

    def applyAction(self, motorCommands):
        # action of arm joint states changes
        jointPoses =[]
        if (len(motorCommands)==3):
            dp = motorCommands[0:3]
            kukastates = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
            pos = kukastates[0]
            eeposx = pos[0] + dp[0]
            eeposy = pos[1] + dp[1]
            eeposz = pos[2] + dp[2]

            eepos = [eeposx, eeposy, eeposz]


            if (self.useNullSpace == 1):
                jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, eepos, lowerLimits=self.ll,
                                                              upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
            else:
                threshold = 0.001
                maxIter = 100
                #jointPoses = self.accurateCalculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, eepos, threshold, maxIter)
                jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, eepos)
        # action of arm ee position changes
        elif (len(motorCommands) == 7):
            dp = motorCommands[0:7]

            self.jointstates = [x+y for x, y in zip(self.jointstates, dp)]
            self.jointstates = self.check_jointstates(self.jointstates, dp)
            # print('jointstate : ', self.jointstates, abs(-9))
            # jointPoses = self.jointstates

        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyUniqueId=self.kukaUid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.jointstates[i], force=1000)
