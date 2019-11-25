# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:52:46 2016

@author: toviah.moldwin
"""
import numpy as np
#import matplotlib.image as mpimg
import numpy.linalg as lnal
import scipy.stats as stats
import random
import matplotlib
from matplotlib import pyplot as plt
from distutils.dist import Distribution
from axisConfigure import axisConfigure
from pylab import rcParams
#rcParams['figure.figsize'] = 15, 15
rcParams['figure.autolayout'] = True
rcParams['figure.figsize'] = 22, 4
# matplotlib.rc('xtick', labelsize=18) 
# matplotlib.rc('ytick', labelsize=18)
# matplotlib.rc('axes', labelsize=20)
# matplotlib.rc('axes', titlesize=20)


def generateBinaryPatterns(patternSize, activePixelNum, numPatterns):
    allPatterns = np.zeros((numPatterns, patternSize))
    for i in range(numPatterns):
        posInds = np.random.choice(list(range(patternSize)), activePixelNum, replace=False)
        allPatterns[i, posInds] = 1
    return allPatterns

def generateBinary(numPatterns, patternSize, bias=True):
     X = np.round(np.random.rand(numPatterns, patternSize))
     if bias:
        binVec = np.ones(numPatterns)
        X[:, -1] = binVec
     return X

# def generateRand(numPatterns, patternSize, bias=True):
#     X = 10*(np.random.rand(numPatterns, patternSize)-0.5)
#     if bias:
#        binVec = np.ones(numPatterns)
#        X[:, -1] = binVec
#     return X

def generateNonOverlapping(numPatterns, patternSize):
    numOnes = patternSize/numPatterns
    X = np.zeros((numPatterns, patternSize))
    for i in range(numPatterns):
       X[i,i*numOnes:(i+1)*numOnes] = np.ones(numOnes)
    return X

def generateLabels(numPatterns):
    y = np.squeeze(np.round(np.random.rand(1, numPatterns)))
    y[y==0] = -1
    return y

def generateXor(patternSize):
    binSize = patternSize/2
    ones = np.ones(binSize)
    zeros = np.zeros(binSize)
    np.array(list(zeros) +list(zeros))
    X = np.asarray([np.append(zeros,zeros),np.append(zeros, ones),np.append(ones, zeros), np.append(ones, ones)])
    y = np.asarray([-1,1,1,-1])
    print(X,y)
    return X, y

def generateFixedActivation(P,N, inhibRatio, activeRatio = None):
    inhibInd = int(np.round((1-inhibRatio) * N))
    allPatterns = np.zeros((P, N))
    for i in range(P):
        if activeRatio == None:
            activeRatio = np.random.rand(1)
        posInds = np.random.choice(list(range(inhibInd)), int(round(activeRatio * (inhibInd))), replace=False)
        allPatterns[i, posInds] = 1
        if not inhibRatio == 0:
            negInds = np.random.choice(list(range(inhibInd, N)), int(round(activeRatio * (N-inhibInd))), replace=False)
            allPatterns[i, negInds] = 1
    return allPatterns, inhibInd

def generateRand(P,N, inhibRatio, activeNum):
    activeRatio = float(activeNum)/N
    inhibInd = int(np.round((1-inhibRatio) * N))
    X = np.zeros((P, N))
    for i in range(P):
        posInds = np.random.choice(list(range(inhibInd)), int(round(activeRatio * (inhibInd))), replace=False)
        X[i, posInds] = 1
        if not inhibRatio == 0:
            negInds = np.random.choice(list(range(inhibInd, N)), int(round(activeRatio * (N-inhibInd))), replace=False)
            X[i, negInds] = 1
    y = np.ones(P)
    y[0:P/2] = -1
    return Dataset(X,y,inhibInd)


def generateAllPos(P,N, inhibRatio, activeNum):
    activeRatio = float(activeNum)/N
    inhibInd = int(np.round((1-inhibRatio) * N))
    X = np.zeros((P, N))
    for i in range(P):
        posInds = np.random.choice(list(range(inhibInd)), int(round(activeRatio * (inhibInd))), replace=False)
        X[i, posInds] = 1
        if not inhibRatio == 0:
            negInds = np.random.choice(list(range(inhibInd, N)), int(round(activeRatio * (N-inhibInd))), replace=False)
            X[i, negInds] = 1
    y = np.ones(P)
    return Dataset(X,y,inhibInd)

def generateAllPosRatio(P,N, inhibRatio, activeRatio):
    inhibInd = int(np.round((1-inhibRatio) * N))
    X = np.zeros((P, N))
    for i in range(P):
        posInds = np.random.choice(list(range(inhibInd)), int(round(activeRatio * (inhibInd))), replace=False)
        X[i, posInds] = 1
        if not inhibRatio == 0:
            negInds = np.random.choice(list(range(inhibInd, N)), int(round(activeRatio * (N-inhibInd))), replace=False)
            X[i, negInds] = 1
    y = np.ones(P)
    return Dataset(X,y,inhibInd)

def generateNonBinary(P,N, inhibRatio):
    inhibInd = int(np.round((1-inhibRatio) * N))
    y = np.ones(P)
    y[0:P/2] = -1
    X = np.random.rand(P,N)
    print X
    return Dataset(X,y,inhibInd)

def generatelinear(X, inhibInd):
    P = X.shape[0]
    N = X.shape[1]
    weights = np.random.rand(N)
    print(inhibInd)
    weights[inhibInd:N] = -weights[inhibInd:N]
    lin = np.dot(X, weights)
    bias = np.median(lin)
    y = np.sign(lin-bias)
    return weights, bias, y

def hamMat(X):
    P = X.shape[0]
    hamMat = np.empty((P,P))
    for i in range(P):
        for j in range(P):
            hamMat[i,j] = np.count_nonzero(X[i]!=X[j])
    print(hamMat)
    plt.ion()
    plt.figure()
    plt.imshow(hamMat)
    plt.colorbar()
    plt.show()
    return hamMat

def generateDistributionsExc(P, N, activeNum):
    X = np.zeros((P, N))
    y = []
    meansPos = np.random.rand(N)
    meansPos = activeNum*(meansPos/sum(meansPos))
    print meansPos
    posVecs = [[np.random.binomial(1, meansPos[i]) for i in range(N) ] for j in range(P/2)]
    print posVecs
    meansNeg = np.random.rand(N)
    meansNeg = activeNum*(meansNeg/sum(meansNeg))
    print meansNeg
    negVecs = [[np.random.binomial(1, meansNeg[i]) for i in range(N) ] for j in range(P/2)]
    print negVecs
    posVecs.append(negVecs)
    print posVecs
    y = np.ones(P/2).tolist()
    y.append(-1 * np.ones(P/2).tolist())
    print y
    return Dataset(X,y,-1), meansPos, meansNeg




class Dataset():
    #structure
    #Posbias---unbiasedStart---inhibInd----unbiasedEnd----(end)
    def  __init__(self, X, y, inhibInd, weights = None, bias = None):
        self.X = X
        self.y = y
        self.P = X.shape[0]
        self.N_orig = X.shape[1]
        self.N = self.N_orig
        self.unbiasedStart = 0
        self.unbiasedEnd = self.N
        self.inhibInd = inhibInd
        self.w0 = weights
        self.b = bias

    def addExciteBias(self, cnt):
        bias = np.ones((self.P, cnt))
        self.X = np.concatenate((bias, self.X), axis = 1)
        self.inhibInd = self.inhibInd + cnt
        self.N = self.N + cnt
        self.unbiasedStart += cnt
        self.unbiasedEnd = self.unbiasedStart + self.N_orig

    def addInhibBias(self, cnt):
        bias = np.ones((self.P, cnt))
        self.X = np.concatenate((self.X, bias), axis = 1)
        self.N = self.N + cnt

    def sort(self):
        posInds = np.where(self.y > 0)
        negInds = np.where(self.y < 0)
        self.y = np.concatenate((self.y[posInds], self.y[negInds]))
        self.X = np.concatenate((self.X[posInds], self.X[negInds]))

class DataSetBitFlip():
    #Creates "context" vectors - i.e. probability distributions - (of the length of labels) which randomly generate binary vectors drawn from the relevant probability distribution
    def __init__(self, N, activeNum, labels, numFlips, inhibRatio = 0):
        self.N = N
        self.labels = labels
        self.activeNum = activeNum
        self.activeRatio = float(activeNum)/N

        self.numContexts = len(labels)
        self.contexts = np.zeros((self.numContexts, N))
        self.unbiasedStart = 0
        self.unbiasedEnd = self.N
        self.P = numFlips 
        self.inhibInd = int(np.round((1-inhibRatio) * N))



        for i in range(self.numContexts):
            posInds = np.random.choice(list(range(self.inhibInd)), int(round(self.activeRatio * (self.inhibInd))), replace=False)
            self.contexts[i, posInds] = 1
            if not inhibRatio == 0:
                negInds = np.random.choice(list(range(self.inhibInd, N)), int(round(self.activeRatio * (N-self.inhibInd))), replace=False)
                self.contexts[i, negInds] = 1  
            
    
    def genBinaryVec(self, contextNum):
        contextVec = np.squeeze(self.contexts[contextNum])
        #print 'Expected value', sum(self.contextDist)
        #binVec = [np.random.binomial(1, contextVec[i]) for i in range(self.N)]
        #Trick to get our vector
        binVec = contextVec
        activeInds = np.where(binVec > 0)[0]
        passiveInds = np.where(binVec == 0)[0]
        activeFlipInds = np.random.choice(list(activeInds), int(round(self.P/2)), replace=False)
        passiveFlipInds = np.random.choice(list(passiveInds), int(round(self.P/2)), replace=False)
        binVec[activeFlipInds] = 0
        binVec[passiveFlipInds] = 1
        label = self.labels[contextNum]
        return binVec, label
        
        #fig.canvas.flush_events() 
class DataSetGen():
    #Creates "context" vectors - i.e. probability distributions - (of the length of labels) which randomly generate binary vectors drawn from the relevant probability distribution
    def __init__(self, N, activeNum, labels, distribution = 'beta', params = []):
        self.N = N
        self.labels = labels
        self.activeRatio = activeNum
        self.numContexts = len(labels)
        self.contexts = np.zeros((self.numContexts, N))
        self.unbiasedStart = 0
        self.unbiasedEnd = self.N
        self.inhibInd = self.N
        self.distribution = distribution
        self.params = params
        self.P = self.params[0]
        
        #independently draw each synapse probability from a uniform distribution then normalize to get desired number of active synapses
        if distribution == 'uniform':
            for contextNum in range(self.numContexts):
                means = np.random.rand(N)
                self.contextDist = activeNum*(means/sum(means))
            
        #draw synapse probabilities from gaussian distribution
        if distribution == 'gauss':
            for contextNum in range(self.numContexts):
                lower, upper = 0, 1
                mu = params[0] 
                sigma = params[1]
                a = (lower - mu) / sigma
                b = (upper - mu) / sigma
                X = stats.truncnorm(a, b, loc=mu, scale=sigma)
                self.contextDist = X.rvs(N)
        
        #draw synapse probabilities from gaussian distribution       
        if distribution == 'beta':
            #beta is uniform when mean is 0.5 and variance is 1/12
                mu = float(activeNum)/N #The mean is determined by activeRatio 
                param = params[0] #Note: This is the *variance* of the Beta distribution must be beteen 0 and 0.25
                print param
                alpha = ((1-mu)/param - (1.0/mu))*(mu**2)
                beta = alpha * (1.0/mu-1)
                print alpha
                print beta
                X = stats.beta(alpha, beta)
                self.contextDist = X.rvs(N)
        
        #print 'Expected value', sum(self.contextDist)
        #self.probDistDisplay()
        for contextNum in range(self.numContexts):
            self.contexts[contextNum] = random.sample(self.contextDist, len(self.contextDist)) 
          #  print self.contexts[contextNum]
            
               
#         if distribution == 'gauss':
#             for contextNum in range(self.numContexts):
#                 lower, upper = 0, 1
#                 expectedValue = float(activeRatio)/N
#                 sigma = 0.1
#                 alpha = (lower - mu) / sigma
#                 beta = (upper - mu) / sigma
#                 X = stats.truncnorm(alpha, beta, loc=mu, scale=sigma)
#                 context = X.rvs(N)
#                 print sum(context)
#                 self.contexts[contextNum] = context
#                 self.probDistDisplay(contextNum)
        
            
    def genBinaryVec(self, contextNum):
        contextVec = np.squeeze(self.contexts[contextNum])
        #print 'Expected value', sum(self.contextDist)
        #binVec = [np.random.binomial(1, contextVec[i]) for i in range(self.N)]
        #Trick to get our vector
        binVec = np.random.uniform(size = self.N)
        binVec = self.contexts[contextNum] - binVec
        binVec = (binVec > 0)
        #print sum(binVec)
        label = self.labels[contextNum]
        return binVec, label
    
    
    def genRandomVec(self):
            means = np.random.rand(self.N)
            means = self.activeRatio*(means/sum(means))
            contextVec = means
            binVec = [np.random.binomial(1, contextVec[i]) for i in range(self.N)]
            label = -1
            return binVec, label
        
    def probDistDisplay(self):
        context = self.contextDist
        fig = plt.figure()
#         plt.subplot(1,2,1)
#         plt.bar(range(len(context)), context)
#         plt.subplot(1,2,2)
        ax = plt.subplot(1,1,1)
        weights = np.ones(len(context))/len(context)
        plt.hist(context, weights = weights)
        plt.xlabel('Probability of synaptic activation')
        plt.ylabel('Proportion of synapses')
        plt.xlim(0,1)
        plt.title(self.distribution + " param = " +  str(self.params[0]))
        axisConfigure(ax)
        fig.canvas.draw()
        plt.savefig(self.distribution + str(self.params) + '.png', bbox_inches=None, pad_inches=0.1)
        
class DataSetTrainTest():
    #Creates "context" vectors - i.e. probability distributions - (of the length of labels) which randomly generate binary vectors drawn from the relevant probability distribution
    def __init__(self, N, activeNum, labels, trainSize, testSize, distribution = 'beta', params = []):
        self.N = N
        self.labels = labels
        self.activeRatio = activeNum
        self.numContexts = len(labels)
        self.contexts = np.zeros((self.numContexts, N))
        self.unbiasedStart = 0
        self.unbiasedEnd = self.N
        self.inhibInd = self.N
        self.distribution = distribution
        self.params = params
        self.trainX = []
        self.trainY = [labels[i/(trainSize/2)] for i in range(trainSize)]
        self.testX = []
        self.testY = [labels[i/(testSize/2)] for i in range(testSize)]
        print self.testY
        
        #draw synapse probabilities from beta distribution       
        if distribution == 'beta':
            #beta is uniform when mean is 0.5 and variance is 1/12
                mu = float(activeNum)/N #The mean is determined by activeRatio 
                param = params[0] #Note: This is the *variance* of the Beta distribution must be beteen 0 and 0.25
                alpha = ((1-mu)/param - (1.0/mu))*(mu**2)
                beta = alpha * (1.0/mu-1)
                X = stats.beta(alpha, beta)
                self.contextDist = X.rvs(N)
        
        #print 'Expected value', sum(self.contextDist)
        #self.probDistDisplay()
        for contextNum in range(self.numContexts):
            self.contexts[contextNum] = random.sample(self.contextDist, len(self.contextDist)) 
        for i in range(trainSize):
            self.trainX.append(self.genBinaryVec(self.trainY[i])[0])
        for i in range(testSize):
            self.testX.append(self.genBinaryVec(self.testY[i])[0])
            
    def genBinaryVec(self, contextNum):
        contextVec = np.squeeze(self.contexts[contextNum])
        #print 'Expected value', sum(self.contextDist)
        #binVec = [np.random.binomial(1, contextVec[i]) for i in range(self.N)]
        #Trick to get our vector
        binVec = np.random.uniform(size = self.N)
        binVec = self.contexts[contextNum] - binVec
        binVec = (binVec > 0)
        #print sum(binVec)
        label = self.labels[contextNum]
        return np.asarray(binVec, dtype = int), label
    
    def genRandomVec(self):
            means = np.random.rand(self.N)
            means = self.activeRatio*(means/sum(means))
            contextVec = means
            binVec = [np.random.binomial(1, contextVec[i]) for i in range(self.N)]
            label = -1
            return binVec, label
        
    def probDistDisplay(self):
        context = self.contextDist
        fig = plt.figure()
#         plt.subplot(1,2,1)
#         plt.bar(range(len(context)), context)
#         plt.subplot(1,2,2)
        ax = plt.subplot(1,1,1)
        weights = np.ones(len(context))/len(context)
        plt.hist(context, weights = weights)
        plt.xlabel('Probability of synaptic activation')
        plt.ylabel('Proportion of synapses')
        plt.xlim(0,1)
        plt.title(self.distribution + " param = " +  str(self.params[0]))
        axisConfigure(ax)
        fig.canvas.draw()
        plt.savefig(self.distribution + str(self.params) + '.png', bbox_inches=None, pad_inches=0.1)
                
# if __name__ == "__main__":
# #     a = 1
# #     N = 1000
# #     activeRatio = N/2
# #     rcParams['figure.figsize'] = 10, 20
# #     paramsList = [0.246, 0.08] #Note: This is the *variance* of the Beta distribution must be beteen 0 and 0.25, the mean is determined by activeRatio
# #     paramsList = [1e-4, 1e-3, 1e-2, 1e-1]
# #     nContextList = [1]
# #     cnt = 1
# #     for params in paramsList:
# #         for nContexts in nContextList:
# #             data = DataSetGen(N,activeRatio, np.ones(nContexts),  distribution = 'beta', params = [params])
# #         #     for i in range(1):
# #         #         binVec,label = data.genBinaryVec(0)
# #         #         print (binVec, label)
# #         #         print 'sm', sum(binVec)
# #             #data.probDistDisplay()
# #             mn = np.mean(data.contexts,0)
# #         #     #print 'averagedist', mn
# #             plt.figure('Hists')
# #             plt.subplot(len(paramsList), len(nContextList), cnt)
# #             plt.hist(mn)
# #             plt.xlim(0,1)
# #             plt.xlabel('Probability of synaptic activation')
# #             plt.ylabel('Proportion of synapses')
# #             plt.title(r'$\sigma^2$ = '   + str(params), fontsize = 14)
# #             cnt = cnt + 1
# #     plt.show()
#     
#     
# #     data = DataSetTrainTest(20, 10, [-1,1], 6, 10, distribution = 'beta', params = [0.1])
# #     print 'hi', data.trainX[5]
# #     print data.trainY
# #     print data.testX
# #     print data.testY
# 
#     rcParams['figure.figsize'] = 8, 2
#     plt.figure()
#     variances = [1e-50, 1.0/12, 0.185]
#     folder = '/Figure4/'
#     #variances = [1e-07, 1e-05,  1e-3]
#     #folder = '/Figure5/'
#     N = 10000
#     activeRatio = N/2
#     for k in range(len(variances)):
#         variance = variances[k]
#         data = DataSetGen(N,activeRatio, np.ones(1),  distribution = 'beta', params = [variance])
#         context = data.contextDist
#         weights = np.ones(len(context))/len(context)
#         ax = plt.subplot(1, len(variances),k+1)
#         bar = plt.hist(context, weights = weights, bins = 20)
# #         plt.xlim(0,1)
# #         plt.ylim(0,1)
#         
#         if k == 0:
#             plt.xlabel('Probability of synaptic activation')
#             plt.ylabel('Proportion of synapses')
#         if folder == '/Figure4/':
#             plt.title(r'$\sigma^2$ = '   + str(np.round(variance,2)))
#         else: 
#             plt.title(r'$\sigma^2$ = '   + str(variance))
# 
#         axisConfigure(ax)
#     plt.savefig('Figures8_8_18' + folder + 'dists.pdf', bbox_inches=None, pad_inches=0.1)
#     plt.show()
#          
    
    #print generateNonOverlapping(4,40)
    
    # allpatterns = np.asarray(generateBinaryPatterns(10, 3, 1))
    # plt.subplot(1,8,1)
    # plt.pcolor(allpatterns.transpose())
    # plt.xticks([])
    # plt.yticks(np.arange(10)+0.5, np.arange(10)+1 )
    # allpatterns = np.asarray(generateBinaryPatterns(10, 3, 1))
    #
    # ax = plt.subplot(1,8,8)
    # ax.yaxis.tick_right()
    # plt.xticks([])
    # plt.yticks(np.arange(10)+0.5, np.arange(10)+1 )
    # plt.pcolor(allpatterns.transpose())
    # plt.savefig('PatternExamples')
    
    # X = generateLSBinary(10,10)
    # y = generateLabels(10)
    # print y
    
    # X, inhibInd = generateFixedActivation(40, 100, 0.4,0.1)
    # [weights, bias, y] = generatelinear(X, 0.1)
    # data = Dataset(X,y, inhibInd, weights, bias)
    # print data.X.shape
    # print np.dot(data.X, data.w0)+data.binVec
    # data.sort()
    # data.addExciteBias(3)
    # data.addInhibBias(4)
    # hamMat(data.X)
    # print data.X.shape
    # print data.unbiasedStart
    # print data.unbiasedEnd
    
    
    #generateDistributions(10,20)
    
    # print data.y
    # print(sum(y))
    
