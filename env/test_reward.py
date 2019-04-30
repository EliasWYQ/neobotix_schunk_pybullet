import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from env import fieldDirection

D_WS = 1

def fun(x, y, goal, obs):
    d = np.ones(np.size(x))
    deltax = np.ones(np.size(x))*goal[0] - x
    deltay = np.ones(np.size(y))*goal[1] - y

    odeltax = np.ones(np.size(x))*obs[0] - x
    odeltay = np.ones(np.size(y))*obs[1] - y

    for i in range(np.size(x)):
        di = np.linalg.norm(deltax[i]**2+deltay[i]**2)
        di_o = np.linalg.norm(odeltax[i] ** 2 + odeltay[i] ** 2)
        if di_o < 0.2:
            d[i] = -1
        else:
            d[i] = -di ** 3
    #print('max ', x[np.argmax(d)], y[np.argmax(d)], np.amax(d))
    maxgoal = np.array([x[np.argmax(d)], y[np.argmax(d)], np.amax(d)])
    return d, maxgoal

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    x = y = np.arange(-D_WS, D_WS, 0.005)
    X, Y = np.meshgrid(x, y)
    lengh = np.size(X)
    goal = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    obs = np.array([np.random.uniform(-D_WS, D_WS), np.random.uniform(-D_WS, D_WS)])
    zs, znew = np.array(fun(np.ravel(X), np.ravel(Y), goal, obs))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.plot(np.array([goal[0]]), np.array([goal[1]]), np.array([2]), 'ro', markersize=10)
    ax.plot(np.array([obs[0]]), np.array([obs[1]]), np.array([2]), 'y*', markersize=10)
    ax.plot(np.array([znew[0]]), np.array([znew[1]]), np.array([znew[2]]), 'g.', markersize=10)

    #print('goal ', goal)

    plt.show()