'''
original built by X. Wang & Z. Zheng, @KIT-IPR
developed by Z. Zheng
schunk model meshes source : https://github.com/ipa320/schunk_modular_robotics
neobotix model meshed source : https://github.com/neobotix/neo_mp_500
model modified by Y. Zhang and J. Su.
'''
import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from env.neobotixschunkGymEnv import NeobotixSchunkGymEnv

def main():
    environment = NeobotixSchunkGymEnv(renders=True, isDiscrete=False, maxSteps=1e5, action_dim=9, colliObj=1, wsboundary=1)
    # environment._p.startStateLogging(environment._p.STATE_LOGGING_VIDEO_MP4, "TEST_GUI.mp4")
    dv = 1
    actionIds = []
    actionIds.append(environment._p.addUserDebugParameter("basevelocity", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("baseangularvelocity", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_1_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_2_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_3_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_4_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_5_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_6_joint", -dv, dv, 0))
    actionIds.append(environment._p.addUserDebugParameter("arm_7_joint", -dv, dv, 0))

    done = 0

    t=0
    disc_total_rew=0
    #
    while not done:
        time.sleep(1)
        environment.reset()
        action = []
        for actionId in actionIds:
            action.append(environment._p.readUserDebugParameter(actionId))
        state, reward, done, info = environment.step(action)
        #state, reward, done, info = environment.step(environment._sample_action())
        # print('r', reward)
        #print('step', state, reward, done, info)
        obs = environment.getExtendedObservation()
        # print(environment._p.getPhysicsEngineParameters)
        # environment.render()
        disc_total_rew += 1 * 0.99 ** t
        t += 1
    print(disc_total_rew, t)

if __name__=="__main__":
    main()
