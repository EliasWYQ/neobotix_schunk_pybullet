import matplotlib.pyplot as plt
import numpy as np
r1 = 0
r2 = 0
r = 0
returns = []
ree=[]
rbase=[]
ddee=[]
ddbase=[]
ts = []
discount = 0.995
d0 = 1
N=200
return1=[]
return2=[]
def intd(t):
    if t<500:
        d0 = np.random.uniform(0.5, 1)
    elif t < 1000:
        d0 = np.random.uniform(1, 1.5)
    elif t < 1500:
        d0 = np.random.uniform(1.5, 2)
    else:
        d0 = np.random.uniform(2, 2.5)
    return d0

def penalty(t, de):
    #d0 = intd(t)
    # de = np.random.uniform(0, (1-t/1500)*2+0.1)
    dm = de + np.random.uniform(-0.1, 0.1)

    if de/d0>1:
        p = -(1-de/d0)*de
    else:
        p = 0

    return d0, de, dm, p

def dee(t):
    de = (1-t/N)*d0 + np.random.uniform(-0.1, 0.1)
    return de

for t in range(N):
    de = dee(t)
    d0, de, dm, p = penalty(t, de)


    # print(de,dm,p)
    tau = de/d0
    tau=tau**2
    rt1= (1-tau)*de+tau*dm
    #print(p)
    rt2 = (1-tau)*de + dm*tau - p - t/N/2
    rt = -rt1
    de = -de
    dm = -dm
    r1 += rt1 * discount ** t
    r2 += rt2*discount**t
    ts.append(t)
    return1.append(r1)
    return2.append(r2)
    returns.append(rt)
    ree.append(de)
    rbase.append(dm)
    ddee.append(de-rt)
    ddbase.append(dm-rt)
plt.plot(ree, 'bo-', rbase, 'g*-' ,returns, 'r.-')
plt.figure(2)
plt.plot(ddee, 'bo-', ddbase, 'g*-')
#plt.plot(return1, 'bo-', return2, 'g*-')
plt.show()
print(returns)
#plt.plot(returns)