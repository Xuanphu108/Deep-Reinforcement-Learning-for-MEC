#  #################################################################
#  version 1.0 -- April 2019. Written by Phu X. Nguyen (nxphu.1994@gmail.com)
#  #################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas
import matplotlib as mpl
import time

class Optimize(object):
    '''
    :param h: matrix channel (N x M)
    :param b: matrix decision (N x M)
    :param M: the number of frequency blocks
    :param N: the number of users
    '''
    def __init__(self, h, b, N, M):
        self.h = h
        self.b = b
        self.N = N
        self.M = M
        # parameters and equations
        self.phi = 100
        self.p = 3
        self.u = 0.7
        self.ki = 10**(-26)
        self.eta1 = ((self.u*self.p)**(1.0/3))/self.phi
        self.noise = 10**(-10)
        self.T = 1
        self.eta2 = self.u*self.p/self.noise
        self.B = 2*(10**6)
        self.epsilon = 0.18*(10**6)/(np.log(2))

        self.b0 = []
        self.b1 = []

        for j in range(self.b.shape[0]):
            self.b0.append(np.where(b[j, :] == 0)[0])
            self.b1.append(np.where(b[j, :] == 1)[0])

        self.h0 = []
        self.h1 = []
        for j in range(self.h.shape[0]):
            self.h0.append(self.h[j, self.b0[j]])
            self.h1.append(self.h[j, self.b1[j]])
    '''
        for j in range(self.M):
            self.b0.append(np.where(self.b[:, j] == 0)[0])
            self.b1.append(np.where(b[:, j] == 1)[0])

        self.h0 = []
        self.h1 = []
        for j in range(self.h.shape[1]):
            self.h0.append(self.h[self.b0[j], j])
            self.h1.append(self.h[self.b1[j], j])
    
    def LocalComputationRate(self, a):
        LocalComputationRate = 0 
        for i in range(len(self.h0)):
            LocalComputaionRate += sum(self.eta1*(self.h0[i]/self.ki)**(1.0/3)*a**(1.0/3))
        return LocalComputaionRate
    
    def OffloadComputationRate(self, a):
        OffloadComputationRate = 0
        for i in range(len(self.h1)):
            OffloadComputationRate += sum((1 - a)*self.epsilon*np.log(1 + self.eta2*self.h1[i]**2*a/(1 - a)) + \
                                      self.eta1 * (self.h1[i] / self.ki) ** (1.0/3) * a ** (4.0/3))
        return OffloadComputationRate
        
    def Derivative(self, a):
        Derivative_1 = 0
        for i in range(len(self.h0)):
            Derivative_1 += sum((1.0/3)*self.eta1*(self.h0[i]/self.ki)**(1.0/3)*a**(-2.0/3))
        Derivative_2 = 0
        for i in range(len(self.h1)):
            Derivative_2 += sum(self.epsilon*(self.eta2*self.h1[i]**2 - 1)*(1 - a)/(1 - a + (self.eta2*self.h1[i]**2)*a) + \
                            self.epsilon + self.epsilon*np.log((1 - a)/(1 - a + (self.eta2*self.h1[i]**2)*a)) + \
                            (4.0/3) * self.eta1 * (self.h1[i] / self.ki) ** (1.0/3) * a ** (1.0/3))
        Derivative = Derivative_1 + Derivative_2
        return Derivative
    '''

    def LocalComputationRate(self, a):
        LocalComputationRate = 0
        for j in range(self.b.shape[0]):
            if (np.count_nonzero(self.b[j, :]) == 0):
                LocalComputationRate += self.eta1*(np.sum(self.h[j, :])/self.ki)**(1.0/3)*a**(1.0/3)
        return LocalComputationRate

    def OffloadComputationRate(self, a):
        OffloadComputationRate = 0
        for j in range(len(self.b1)):
            if (np.count_nonzero(self.b[j, :]) != 0):
                for i in range(len(self.h1[j])):
                    OffloadComputationRate += self.epsilon*(1 - a)*np.log(1 + self.eta2*sum(self.h[j, :])*self.h1[j][i]*a/(1 - a))
                OffloadComputationRate += self.eta1*(np.sum(self.h[j, :])/self.ki)**(1.0/3)*a**(4.0/3)
        return OffloadComputationRate

    def Derivative(self, a):
        Derivative_1 = 0
        for j in range(self.b.shape[0]):
            if (np.count_nonzero(self.b[j, :]) == 0):
                Derivative_1 += (1.0/3)*self.eta1*(np.sum(self.h[j, :])/self.ki)**(1.0/3)*a**(-2.0/3)
        Derivative_2 = 0
        for j in range(len(self.b1)):
            if (np.count_nonzero(self.b[j, :]) != 0):
                for i in range(len(self.h1[j])):
                    Derivative_2 += self.epsilon*(self.eta2*np.sum(self.h[j, :])*self.h1[j][i] - 1)*(1 - a)/(1 - a + self.eta2*np.sum(self.h[j, :])*self.h1[j][i]*a) + \
                                    self.epsilon + self.epsilon*np.log((1 - a)/(1 - a + self.eta2*np.sum(self.h[j, :])*self.h1[j][i]*a))
                Derivative_2 += (4.0/3)*self.eta1*(np.sum(self.h[j, :])/self.ki)**(1.0/3)*a**(1.0/3)
        Derivative = Derivative_1 + Derivative_2
        return Derivative

    # Bisection search
    def Bisection(self):
        delta = 0.00001
        UpperBound = 1  # - 0.001
        LowerBound = 0
        while UpperBound - LowerBound > delta:
            v = (UpperBound + LowerBound)/2
            if self.Derivative(v) > 0:
                LowerBound = v
            else:
                UpperBound = v
        MaximizeComputationRate = self.LocalComputationRate(v) + self.OffloadComputationRate(v)
        return v, MaximizeComputationRate

    def GetReward(self):
        if (np.count_nonzero(self.b) == 0):
            a = 1
            Reward = self.LocalComputationRate(a)
        else:
            a, Reward = self.Bisection()
        return a, Reward

if __name__ == "__main__":
    h = np.random.rand(5, 5)*(10**(-5))
    print(h.shape[1])
    print('Channel gain:')
    print(h)
    b0 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    b1 = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    b2 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    start_time = time.time()
    a0, Reward0 = Optimize(h, b0, 5, 5).GetReward()
    a1, Reward1 = Optimize(h, b1, 5, 5).GetReward()
    a2, Reward2 = Optimize(h, b2, 5, 5).GetReward()
    total_time = time.time() - start_time
    print('Fraction 0: %f' %a0)
    print('Fraction 1: %f' %a1)
    print('Fraction 2: %f' %a2)
    print('Reward 0: %f' %Reward0)
    print('Reward 1: %f' %Reward1)
    print('Reward 2: %f' %Reward2)
    print('Optimization time: %f' %total_time)













































