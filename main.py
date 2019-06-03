#  #################################################################
#  version 1.0 -- April 2019. Written by Phu X. Nguyen (nxphu.1994@gmail.com)
#  #################################################################

import scipy.io as sio
import numpy as np
from memory import MemoryDNN
from Optimization import Optimize
import os
import csv
import time

def plot_gain(gain_his):
    #display data
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    # colors = "bgrcmykw"
    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    rolling_intv = 200
    df_roll = df.rolling(rolling_intv, min_periods=0).mean()

    plt.plot(np.arange(len(gain_array))+1, df_roll, 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color='b', alpha=0.2)

    # plt.plot(np.arange(len(gain_array)), gain_array, colors[np.random.randint(0, 8)])
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
    '''
    n = 50000               # number of time frames
    net_num = 3             # number of DNNs
    N = 4                   # number of users
    M = 6                   # number of frequency blocks
    Memory = 1024           # capacity of memory structure

    # Load data
    PathInput = "D:/DL_MEC/Data_4x6/Channels/"  # "D:/DL_MEC/Data_4x6/Input/"
    PathMaxCompRate = "D:/DL_MEC/Data_4x6/MaxCompRate/MaxCompRate.csv"

    MaxCompRate = np.genfromtxt(PathMaxCompRate, delimiter=',')

    split_idx = int(.8*len(os.listdir(PathInput)))
    num_test = min(len(os.listdir(PathInput)) - split_idx, n - int(.8*n))  # training data size

    mem = MemoryDNN(net=[N*M, 120, 80, N*M],
                    net_num=net_num,
                    learning_rate=0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=1024)
    start_time = time.time()

    gain_his = []
    gain_his_ratio = []
    knm_idx_his = []

    for i in range(n):

        if i % (n//100) == 0:
           print("----------------------------------------------rate of progress:%0.2f" % (i/n))
        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        channel_idx = np.genfromtxt(PathInput + os.listdir(PathInput)[i_idx], delimiter=',')

        # pretreatment，for better train
        channel_idx = channel_idx*1000000
        channel_idx = np.reshape(channel_idx, (1, len(channel_idx)))

        # produce offloading decision
        m_list = mem.decode(channel_idx, N)
        r_list = []
        channel_idx_mat = channel_idx.reshape(M, N).T  

        for m in m_list:
            r_list.append(Optimize(channel_idx_mat/1000000, m, N, M).GetReward()[1])

        # memorize the largest reward and train DNN
        # the train process is included in mem.encode()
        BestAction = m_list[np.argmax(r_list)]

        mem.encode(channel_idx, BestAction.T.flatten().reshape(1, N*M))

        # memorize the largest reward
        gain_his.append(np.max(r_list))
        gain_his_ratio.append(gain_his[-1]/MaxCompRate[i_idx])  

        # record the index of largest reward
        knm_idx_his.append(np.argmax(r_list))

    total_time = time.time() - start_time
    print('time_cost:%s' % total_time)
    print("gain/max ratio of test: ", sum(gain_his_ratio[-num_test: -1])/num_test)
    print("The number of net: ", net_num)

    mem.plot_cost()
    # cost of DNN
    plot_gain(gain_his_ratio)
