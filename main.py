import math
from gameOfThrone import gameOfThrone
import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Manager


def go(threadID, c1, c2, c3, N, K, T, rho, delta, c, epsilon, dic = None):
    game = gameOfThrone(c1, c2, c3, N, K, T, rho, delta, c, epsilon)
    game.Run()
    if dic is not None:
        dic[threadID] = game.getTotalUtility()


if __name__ == '__main__':
    N = 6  # number of players
    K = 6  # number of arms  K>=N
    T = int(5 * math.pow(10, 6))  # time horizon
    rho = 0.5
    delta = 0
    c = N
    epsilon = 0.01
    c1 = 1000
    c2 = 6000
    c3 = 6000
    experimentTime = 100
    colorAndStyle = ['b-', 'r-', 'y-', 'k--', 'g--', 'pink--']
    epsilonList = [0.0001, 0.001, 0.01]
    numOfLine = 3
    # colorAndStyle = ['b-']
    # epsilonList = [0.001]
    # numOfLine = 1

    for l in range(numOfLine):
        pool = Pool(12)
        starttime = time.time()
        dic = Manager().dict()
        for i in range(experimentTime):
            pool.apply_async(go, args=(i, c1, c2, c3, N, K, T, rho, delta, c, epsilonList[l], dic, ))
        pool.close()
        pool.join()
        endtime = time.time()
        print('cost time {}'.format(endtime-starttime))
        percent = np.zeros(experimentTime)
        '''90/50'''
        for i in range(experimentTime):
            percent[i] = dic[i][T]
        rank_index = np.argsort(percent)
        ninety_index = rank_index[:int(experimentTime*0.9)]
        half_index = rank_index[:int(experimentTime * 0.5)]
        sum = np.zeros(T + 1)
        for i in ninety_index:
            sum += dic[i]
        sum /= len(ninety_index)
        x = np.linspace(0, T+1, T+1)
        plt.plot(x, sum, colorAndStyle[l], label=r"$\epsilon$={},best 90%".format(epsilonList[l]), linewidth=2, linestyle='--')
        sum = np.zeros(T + 1)
        for i in half_index:
            sum += dic[i]
        sum /= len(half_index)
        x = np.linspace(0, T+1, T+1)
        plt.plot(x, sum, colorAndStyle[l], label=r"$\epsilon$={},median".format(epsilonList[l]))
        # print('rank index is', rank_index)
        # print('ninety index is ', ninety_index)
        # print('half index is', half_index)
        '''90/50'''
        # sum = np.zeros(T + 1)
        # for i in range(experimentTime):
        #     sum += dic[i]
        # sum /= experimentTime
        # x = np.linspace(0, T + 1, T + 1)
        # plt.plot(x, sum, colorAndStyle[l], label=r"$\epsilon$={}".format(epsilonList[l]))  # 0.0001的时候可以画出来
    plt.xlim(0, T)
    #plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.show()




