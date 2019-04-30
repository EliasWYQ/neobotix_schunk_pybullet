import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import fieldDirection

D_WS = 1
HIGH_R = 1e3

def fun_field(x, y, goal, obs):
    computefield = fieldDirection.FieldDirection()
    d = np.ones(np.size(x))
    d1 = np.ones(np.size(x))
    d2 = np.ones(np.size(x))
    deltax = np.ones(np.size(x))*goal[0] - x
    deltay = np.ones(np.size(y))*goal[1] - y

    odeltax = np.ones(np.size(x))*obs[0] - x
    odeltay = np.ones(np.size(y))*obs[1] - y

    for i in range(np.size(x)):
        p_robot = np.array([x[i], y[i]])
        di_g = np.sqrt(deltax[i]**2+deltay[i]**2)
        di_o = np.sqrt(odeltax[i] ** 2 + odeltay[i] ** 2)
        force, d_force = computefield.compute_sum_force(p_robot, goal, obs)
        rforce, d_rforce = computefield.compute_repulse(p_robot, obs)
        aforce, d_aforce = computefield.compute_attract(goal, p_robot)
        # print('force ', d_force)
        if di_o < 0.0:
            d1[i] = -1e1-7
        elif di_o < 0.2:
            d1[i] = -force
        else:
            d1[i] = -20*aforce

        if di_g < 0.0:
            d2[i] = 1e2
        elif di_g < 0.2:
            d2[i] = 20*aforce
        else:
            d2[i] = 20*aforce

        d[i] = d1[i] #+ d2[i]
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    return d, maxgoal


def fun_normal_dis(x, y, goal, obs):
    computefield = fieldDirection.FieldDirection()
    d = np.ones(np.size(x))
    d1 = np.ones(np.size(x))
    d2 = np.ones(np.size(x))
    deltax = np.ones(np.size(x))*goal[0] - x
    deltay = np.ones(np.size(y))*goal[1] - y

    odeltax = np.ones(np.size(x))*obs[0] - x
    odeltay = np.ones(np.size(y))*obs[1] - y

    for i in range(np.size(x)):
        p_robot = np.array([x[i], y[i]])
        di_g = np.sqrt(deltax[i]**2+deltay[i]**2)
        di_o = np.sqrt(odeltax[i] ** 2 + odeltay[i] ** 2)

        if di_o < 0.05:
            d1[i] = -1e1-10
        elif di_o < 0.2:
            d1[i] = -1/di_o
        else:
            d1[i] = 0

        if di_g < 0.05:
            d2[i] = 1e1+10
        elif di_g < 0.2:
            d2[i] = 1/(di_g**1)-di_g**2
        else:
            d2[i] = 1/(di_g**1)-di_g**2

        d[i] = d1[i] + d2[i]

    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    return d, maxgoal

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    x = y = np.arange(-D_WS, D_WS, 0.01)
    X, Y = np.meshgrid(x, y)
    lengh = np.size(X)
    goal = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    obs = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    zs, znew = np.array(fun_normal_dis(np.ravel(X), np.ravel(Y), goal, obs))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.plot(np.array([goal[0]]), np.array([goal[1]]), np.array([2]), 'ro', markersize=10, label='goal')
    ax.plot(np.array([obs[0]]), np.array([obs[1]]), np.array([2]), 'y*', markersize=10, label='obs')
    ax.plot(np.array([znew[0]]), np.array([znew[1]]), np.array([znew[2]]), 'g.', markersize=10, label='max point')
    ax.legend()


    plt.show()