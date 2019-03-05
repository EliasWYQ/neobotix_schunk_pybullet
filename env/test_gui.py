import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

import pybullet as p
import time
import pybullet_data

URDF_USE_SELF_COLLISION=1

physicsClient = p.connect(p.GUI)
p.resetDebugVisualizerCamera(4, 180, -40, [0.52, -0.2, -0.33])
p.resetSimulation()
p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
p.setTimeStep(1./240.)
p.setGravity(0, 0, -10)
p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "TEST_GUI5.mp4")
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

planeId = p.loadURDF(os.path.join(parentdir, "neobotix_schunk_pybullet/data/plane.urdf"))
neobotixschunkUid = p.loadURDF(os.path.join(parentdir, "neobotix_schunk_pybullet/data/neobotixschunk/mp500lwa4d_devs.urdf"), flags=p.URDF_USE_SELF_COLLISION)
# p.loadURDF("quadruped/minitaur_v1.urdf")
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
