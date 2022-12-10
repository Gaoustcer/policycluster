import numpy as np

import matplotlib.pyplot as plt

x = []
y = []

from random import random
N = 3
paramx = []
paramy = []
for _ in range(N):
    paramx.append(random())
for _ in range(N):
    paramy.append(random())
Num_sample = 1024
for index in range(N):
    x.append(np.random.normal(paramx[index],1,Num_sample))
    y.append(np.random.normal(paramy[index],1,Num_sample))
# x = np.concatenate(x,axis=-1)
# y = np.concatenate(y,axis=-1)
# plt.scatter(x,y,s = 0.1)
colorlist = ['r','g','b']
for index in range(N):
    plt.scatter(x[index],y[index],s=0.1,c=colorlist[index])
plt.savefig("./distribution/data.png")
# x.append(np.random.normal(0,1,10))
# y.append