'''
developed by Z. Zheng, @KIT-IPR
compute summery direction in an environment with obstacles
artificial field method
used for the observation
'''
import os
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

class FieldDirection:
    '''
    def __init__(self,
                 robot_pos,
                 goal_pos,
                 obs_pos):
        self.PositionRobot = robot_pos[0:2]
        self.PositionGoal = goal_pos[0:2]
        self.PositionObstacle = obs_pos[0:2]
    '''
    def compute_angles(self, pos_a, pos_b):
        # calculate the angles of two points
        deltaXY = np.subtract(pos_a, pos_b)
        dis_XY = np.linalg.norm(deltaXY)
        thetaX = np.arccos(deltaXY[0]/dis_XY)
        return thetaX

    def compute_attract(self, pos_goal, pos_r):
        # calculate attract force
        deltaXY = np.subtract(pos_goal, pos_r)
        dis_XY = np.linalg.norm(deltaXY)
        return dis_XY, deltaXY

    def compute_repulse(self, pos_r, pos_obs):
        # calculate repulse force
        deltaXY = np.subtract(pos_r, pos_obs)
        dis_XY = np.linalg.norm(deltaXY)
        deltaXY = (1/dis_XY - 1/0.2)/(dis_XY**2)*np.gradient(deltaXY)
        dis_XY = np.linalg.norm(deltaXY)
        return dis_XY, deltaXY

    def compute_sum_force(self, pos_r, pos_goal, pos_obs):
        # calculte the sum force
        fattr, dattr = self.compute_attract(pos_goal, pos_r)
        frep, drep = self.compute_repulse(pos_r, pos_obs)
        force_direction = 20*dattr + 20*drep
        force = np.linalg.norm(force_direction)
        force_dunit = force_direction/force
        return force, force_direction