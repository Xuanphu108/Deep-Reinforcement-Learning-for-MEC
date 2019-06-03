import numpy as np
import os
import itertools
import time
import copy
from Optimization import Optimize

N = 4
M = 6
path = 'D:/DL_MEC/Data_4x6/Input2Run/'

# Arrays = [np.reshape(np.array(i), (N, M)) for i in itertools.product([0, 1], repeat=N*M)]
def GenActions(state, ListArrays, col):
    if ((col == 0)):
        return ListArrays
    else:
        state = copy.deepcopy(state)
        for k in state:
            for i in range(k.shape[0]):
                k = copy.deepcopy(k)
                k[:, col-1] = 0
                k[i, col-1] = 1
                ListArrays.append(k)
        return GenActions(ListArrays, ListArrays, col-1)
state = np.zeros([N, M])
col = state.shape[1]
ListArrays = GenActions([state], [state], col)

'''
#for i in ListArrays:
#    print(i)
'''

print('The number of files: %d' % len(os.listdir(path)))
start_time = time.time()
MaxRewards = []
for i in range(1000):  # len(os.listdir(path))
    channel_idx = np.genfromtxt(path + os.listdir(path)[i], delimiter=',')
    # print('channel_idx:')
    # print(channel_idx)
    channel_idx_mat = channel_idx.reshape(M, N).T
    # print('channel_idx_mat:')
    # print(channel_idx_mat)
    Rewards = []
    TimeFractions = []
    for j in ListArrays:
        # print(j)
        TimeFraction, Reward = Optimize(channel_idx_mat, j, N, M).GetReward()
        Rewards.append(Reward)
        TimeFractions.append(TimeFraction)
    print('Rewards at channel %d' % i)
    print(Rewards)
    print('Time fractions at channel %d' % i)
    print(TimeFractions)
    MaxReward = np.amax(Rewards)
    print('Max reward: %f' % MaxReward)
    MaxRewards.append(MaxReward)

total_time = time.time() - start_time
print('Optimization time: %f' % total_time)
print('Max rewards')
print(MaxRewards)
MaxRewards = np.asarray(MaxRewards)
print(MaxRewards)
np.savetxt("MaxCompRate/MaxCompRate.csv", MaxRewards, delimiter=",")


'''
import pandas as pd
import csv

txtfile = 'D:/DL_MEC/Data_4x6/MaxCompRate/y.txt'
csvfile = 'D:/DL_MEC/Data_4x6/MaxCompRate/MaxCompRate.csv'
array = np.loadtxt(txtfile, delimiter=',')
print(array)
np.savetxt("MaxCompRate/MaxCompRate.csv", array, delimiter=",")
'''