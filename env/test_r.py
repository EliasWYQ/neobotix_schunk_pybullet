import matplotlib.pyplot as plt
import numpy as np

ri = 0
rj = 0
rk = 0
returns = 0
tau = 0.995

r = []
# every step with collision
g = []
# every step without collision but not reach
g_normal = []
# every step without collision and reach
g_reach = []

edge = 1
r_collison = -1e3
r_reach = 1e3
N = 1000

r_scale = 1


plt.figure(1)
plt.hold()
for i in range(N):
    returns = 0
    returns_no_end = 0
    returns_normal = 0
    returns_reach = 0
    for j in range(i):
        if j == i-1:
            ri = r_collison
            rj = r_reach
        else:
            ri = np.random.uniform(-edge*np.sqrt(edge), 0)
            rj = np.random.uniform(-edge*np.sqrt(edge), 0)
        ri = ri**r_scale
        rj = rj**r_scale
        returns += ri*tau**j
        returns_reach += rj*tau**j
    for k in range(N):
        rk = np.random.uniform(-edge*np.sqrt(edge), 0)
        rk = rk**r_scale
        returns_normal += rk * tau ** k
    g.append(returns)
    g_normal.append(returns_normal)
    g_reach.append(returns_reach)

    plt.plot(g, 'o-', g_normal, '.-', g_reach, '+-')
plt.show()
