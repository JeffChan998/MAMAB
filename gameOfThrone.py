import numpy as np
import numpy.random as nprdn
import math
import matplotlib.pyplot as plt
import time

class gameOfThrone():

    def __init__(self, c1, c2, c3, N, K, T, rho, delta, c, epsilon):
        # 时间
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.T = T
        self.phaseNum = 0
        # 真实均值以及估计
        self.correct_reward = nprdn.uniform(0.05, 0.95, size=(N, K))
        self.o = np.zeros((N, K))
        self.s = np.zeros((N, K))
        self.u = np.zeros((N, K))
        # 画图用
        self.totalUtility = np.zeros(T+1)
        self.time = 0
        # 用于判断
        self.collision = list()
        self.content = list()
        self.discontent = list()
        # 参数设置
        self.rho = rho
        self.delta = delta
        self.c = c
        self.epsilon = epsilon
        # 其他
        self.fixedArm = np.zeros(N)
        self.lastArm = np.zeros(N)
        self.pulledArm = np.zeros((N, K))
        self.N = N
        self.K = K
        #print('mean reward is\n', self.correct_reward)
        #print('argmax is \n', self.correct_reward.argmax(axis=1))

    def resetList(self):
        self.collision = list()

    def resetContentList(self):
        self.content = list()
        self.discontent = list()

    def allreset(self, N, K):
        self.fixedArm = np.zeros(N)
        self.lastArm = np.zeros(N)
        self.pulledArm = np.zeros((N, K))
        self.N = N
        self.K = K
        self.collision = list()
        self.content = list()
        self.discontent = list()
        self.o = np.zeros((N, K))
        self.s = np.zeros((N, K))
        self.u = np.zeros((N, K))
        self.phaseNum = 0
        self.totalUtility = np.zeros(self.T+1)
        self.time = 0


    def pullArm(self, K):
        return nprdn.randint(0,K)

    def getReward(self, idOfPlayer, idOfArm):
        return self.correct_reward[idOfPlayer][idOfArm] + nprdn.uniform(-0.05, 0.05)

    def phase1(self, N, K):
        c1 = self.c1
        for t in range(c1):
            if self.time == self.T:
                break
            self.time += 1
            self.resetList()
            sumReward = np.zeros(K)
            for idOfPlayers in range(N):
                idOfArm= self.pullArm(K)
                self.lastArm[idOfPlayers] = idOfArm
                self.collision.append(idOfArm)
            for idOfPlayers in range(N):
                idOfArm = int(self.lastArm[idOfPlayers])
                if self.collision.count(idOfArm) == 1:
                    sumReward[idOfArm] = self.getReward(idOfPlayers, idOfArm)
                    self.s[idOfPlayers][idOfArm] += sumReward[idOfArm]
                    self.o[idOfPlayers][idOfArm] += 1
                self.totalUtility[self.time] = ((self.totalUtility[self.time - 1])*(self.time -1) + sumReward.sum())/self.time
        for idOfPlayers in range(N):
            for idOfArm in range(K):
                if self.o[idOfPlayers][idOfArm] != 0:
                    self.u[idOfPlayers][idOfArm] = self.s[idOfPlayers][idOfArm] / self.o[idOfPlayers][idOfArm]
            lastarm = self.lastArm[idOfPlayers]
            if self.collision.count(lastarm) == 1:
                self.content.append(idOfPlayers)
            else:
                self.discontent.append(idOfPlayers)

    def phase2(self, N, K):
        phasetime = int(self.c2 * math.pow(self.phaseNum, 1 + self.delta))
        counttime = int(np.around(self.rho * phasetime))
        self.pulledArm = np.zeros((N, K))
        self.resetContentList()
        for idOfPlayers in range(N):  # 判断最后一次玩家的满意情况
            choseArm = self.lastArm[idOfPlayers]
            if self.collision.count(choseArm) == 1:
                self.content.append(idOfPlayers)
            else:
                self.discontent.append(idOfPlayers)
        #print('LAST RESULT IN PHASE\n', self.collision)
        for t in range(phasetime):

            if self.time == self.T:
                break
            self.time += 1
            thisarm = np.zeros(N)
            sumReward = np.zeros(K)
            self.resetList()
            #print('CHECK\n', self.collision)
            for idOfPlayers in range(N):
                if self.content.count(idOfPlayers) == 1:  # 满意的player
                    pro = math.pow(self.epsilon, self.c)
                    situation = [0, 1]
                    proba = [pro, 1 - pro]
                    choice = nprdn.choice(situation, p=proba)
                    if choice == 1:
                        thisarm[idOfPlayers] = self.lastArm[idOfPlayers]
                        self.collision.append(thisarm[idOfPlayers])
                    else:
                        thisarm[idOfPlayers] = self.pullArm(K)
                        self.collision.append(thisarm[idOfPlayers])
                # 不满意的player
                else:
                    thisarm[idOfPlayers] = self.pullArm(K)
                    self.collision.append(thisarm[idOfPlayers])
            #print('THIS TURN\n', self.collision)
            # 状态转移
            #print('--------')
            for idOfPlayers in range(N):
                idOfArm = int(thisarm[idOfPlayers])
                # 满足的player继续满足
                if idOfArm == self.lastArm[idOfPlayers] \
                        and self.collision.count(idOfArm) == 1 \
                        and self.content.count(idOfPlayers) == 1:
                    sumReward[idOfArm] = self.getReward(idOfPlayers, idOfArm)
                    self.lastArm[idOfPlayers] = idOfArm
                    if t >= counttime:
                        self.pulledArm[idOfPlayers][idOfArm] += 1
                    #print('player {} is content'.format(idOfPlayers))
                # 其他情况
                else:
                    # eta = np.zeros((K, K))
                    # for index in range(K):
                    #     if self.collision.count(index) == 1:
                    #         eta[index][index] = 1
                    #     # 这里的eta有问题
                    # corU = self.u[idOfPlayers].dot(eta)
                    # # print('U\n',self.u[idOfPlayers])
                    # # print('corU\n', eta) 有问题eta
                    # uMax = corU.max()
                    uMax = self.u[idOfPlayers].max()  # 这里得改进
                    if self.collision.count(idOfArm) == 1:
                        uN = self.u[idOfPlayers][idOfArm]
                        # sumReward[idOfArm] = self.getReward(idOfPlayers, idOfArm)
                        sumReward[idOfArm] = uN
                        #print('player id:{},pulling unique arm:{}'.format(idOfPlayers, idOfArm))
                    else:
                        uN = 0
                        #print('player id:{},pulling arm:{} which is collision'.format(idOfPlayers, idOfArm))   #冲突了
                    if uMax == 0:
                        pro = 0
                    elif uN == uMax:
                        pro = 1
                    else:
                        pro = uN / uMax * math.pow(self.epsilon, uMax - uN)
                    situation = [0, 1]
                    proba = [pro, 1 - pro]
                    choice = nprdn.choice(situation, p=proba)

                    if choice == 1:
                        if self.content.count(idOfPlayers) == 1:  # 满足的变成不满足
                            self.content.remove(idOfPlayers)
                            self.discontent.append(idOfPlayers)
                            # print('players {} become discontent'.format(idOfPlayers))
                        self.lastArm[idOfPlayers] = idOfArm
                    else:
                        if self.discontent.count(idOfPlayers) == 1:  # 不满足的变成满足
                            self.discontent.remove(idOfPlayers)
                            self.content.append(idOfPlayers)
                            # print('players {} become content'.format(idOfPlayers))
                            if t >= counttime:
                                self.pulledArm[idOfPlayers][idOfArm] += 1
                        self.lastArm[idOfPlayers] = idOfArm
            #print('--------')
            self.totalUtility[self.time] = ((self.totalUtility[self.time - 1])*(self.time - 1) + sumReward.sum())/self.time

    def phase3(self, N, K):
        #print('the F matrix is \n', self.pulledArm)
        for idOfPlayers in range(N):
            idOfArm = np.argmax(self.pulledArm[idOfPlayers])
            self.fixedArm[idOfPlayers] = idOfArm
        #print('In phase 3 players use strategy', self.fixedArm)
        phasetime = int(self.c3 * math.pow(2, self.phaseNum))
        for t in range(phasetime):
            if self.time == self.T:
                break
            self.time += 1
            self.resetList()
            sumReward = np.zeros(K)
            for idOfPlayers in range(N):
                idOfArm = int(self.fixedArm[idOfPlayers])
                self.collision.append(idOfArm)
            for idOfPlayers in range(N):
                idOfArm = int(self.fixedArm[idOfPlayers])
                if self.collision.count(idOfArm) == 1:
                    sumReward[idOfArm] = self.u[idOfPlayers][idOfArm]
            self.totalUtility[self.time] = ((self.totalUtility[self.time - 1])*(self.time - 1) + sumReward.sum())/self.time

    def Run(self):
        self.allreset(self.N, self.K)
        while self.time < self.T:
            self.phaseNum += 1
            #startTime = time.time()
            self.phase1(self.N, self.K)
            self.phase2(self.N, self.K)
            self.phase3(self.N, self.K)
            #endTime = time.time()
            #print('Epoch {} is finished, cost {} s'.format(self.phaseNum, endTime - startTime))
        #print('One game have done')

    def plotUtilities(self):
        x = np.linspace(0, self.T+1, self.T+1)
        plt.plot(x, self.totalUtility, 'b')
        plt.show()

    def getTotalUtility(self):
        self.totalUtility /= np.max(self.totalUtility)
        return self.totalUtility