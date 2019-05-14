import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import fieldDirection
from matplotlib import cm

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
        elif di_o <= 0.2:
            d1[i] = -force
        else:
            d1[i] = -force

        if di_g < 0.00:
            d2[i] = 1e2
        elif di_g < 0.2:
            d2[i] = 20*aforce
        else:
            d2[i] = 20*aforce

        d[i] = -force#np.log(force)#np.exp(force)#d1[i] #+ d2[i]
        if d[i] < -10:
            d[i] = -10
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs


def fun_extended_dis(x, y, goal, obs):
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

        if di_o < 0.:
            d1[i] = -1e1-10
        elif di_o < 0.2:
            d1[i] = -1/di_o
        else:
            d1[i] = 0

        if di_g < 0.0:
            d2[i] = 1e1+10
        elif di_g < 0.2:
            d2[i] = 1/(di_g**1)-di_g**2
        else:
            d2[i] = 1/(di_g**1)-di_g**2

        d[i] = d1[i] + d2[i]

        if d[i] < -10:
            d[i] = -10

        if d[i] > 10:
            d[i] = 10
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs

def penalty(dee):
    d0 = 1
    de = dee
    dm = de + np.random.uniform(-0.5, 0.5)
    if de/d0>1:
        p = (1-de/d0)*de
    else:
        p = 0
    return d0, de, dm, p

def fun_prioritized_dis(x, y, goal, obs):
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

        d0, de, dm, p = penalty(di_g)

        tau = de / d0
        tau = tau #** 2
        r = (1 - tau) * de + tau * dm - p
        d[i] = -r
        if d[i] < -10:
            d[i] = -10

        if d[i] > 10:
            d[i] = 10
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    minobs = np.array([x[np.argmin(d)], y[np.argmin(d)], np.amin(d)])
    return d, maxgoal, minobs


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    x = y = np.arange(-D_WS, D_WS, 0.005)
    X, Y = np.meshgrid(x, y)
    lengh = np.size(X)
    goal = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    obs = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    #goal =[0,0]#test without obs
    #obs=[10,10]#test without obs
    zs, zmax, zmin = np.array(fun_extended_dis(np.ravel(X), np.ravel(Y), goal, obs))
    #zs, znew = np.array(fun_field(np.ravel(X), np.ravel(Y), goal, obs))
    Z = zs.reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0.0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-D_WS, D_WS);
    ax.set_ylim(-D_WS, D_WS);
    ax.plot(np.array([goal[0]]), np.array([goal[1]]), np.array([2]), 'r.', markersize=10, label='goal')
    ax.plot(np.array([obs[0]]), np.array([obs[1]]), np.array([2]), 'b.', markersize=10, label='obstacle')
    ax.plot(np.array([zmax[0]]), np.array([zmax[1]]), np.array([zmax[2]]), 'r*', markersize=10, label='maximum')
    ax.plot(np.array([zmin[0]]), np.array([zmin[1]]), np.array([zmin[2]]), 'b*', markersize=10, label='minimum')
    ax.legend(numpoints=1, fontsize=20)

    plt.show()