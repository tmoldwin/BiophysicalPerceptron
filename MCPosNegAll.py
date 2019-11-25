import numpy as np
import PatternGenerator as pg
import sys
import os
import matplotlib
if "SSH_CONNECTION" in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.spatial.distance as sdist
from matplotlib.ticker import MaxNLocator
from pylab import rcParams
import pickle as pkl
import json
import cProfile
import pstats
import string
import datetime


def zeroNan(vec):
    if 0 in vec:
        ind = list(vec).index(0)
        print ind    
        if ind > -1:
            vec[ind:] = 0
        vec = list(vec)
        return vec
    else: return vec
    
class Perceptron(object):
    def __init__(self, dataset, problem = 'mem', MESW = None, passive = False, bias = False):
        self.dataset = dataset
        if bias:
            self.dataset.addExciteBias(1)
        self.N = self.dataset.N
        self.MESW = MESW
        self.problem = problem
        if passive: 
            pickleName = 'maxMapL6_0[0, 1, 1]VTrue.pkl'
        else:
            pickleName = 'maxMap/maxMapL6_0VTrue.pkl'
        print 'pr', problem
        if not self.MESW == None:
            self.maxEffectiveWeights = createDie(self.N, MESW, pickleName)
#             plt.figure('MESW dist' + str(self.MESW), figsize = (5,5))
#             print self.maxEffectiveWeights
#             plt.hist(self.maxEffectiveWeights, bins = np.arange(0,1,0.01))
            #plt.show()
        
    def learnMem(self, maxEpochs, eta, Adam = True, run = 0, cond = 0, lambd = 0):
        X = self.dataset.X
        y = self.dataset.y
        w_0 = self.dataset.w0
        self.P = X.shape[0]
        self.N = X.shape[1]
        m = 0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        v = 0
        dw = np.zeros(self.N)
        self.weights = np.random.rand(self.N)
        falsePosList = []
        falseNegList = []
        errorList = []
        weightList = []
        flipVec = np.ones(self.N)
        flipVec[self.dataset.inhibInd:self.N] = -flipVec[self.dataset.inhibInd:self.N]
        reg = np.zeros(self.N)
        for epoch in range(maxEpochs):
            #print(epoch, 'epoch')
            inds = np.random.permutation(range(self.P))
            falsePos = 0
            falseNeg = 0
            sys.stdout.flush()
            for i in range(self.P):
                reg[1:self.dataset.N] = 0.5*np.sqrt(self.weights[1:self.dataset.N]**2)
                patternNum = inds[i]
                pattern = X[patternNum]
                y_0 = y[patternNum]
                pOut = self.predict(pattern, cond = cond)
                if pOut != y_0:
                    if pOut == -1:
                        falseNeg += 1 
                    else:
                        falsePos += 1
                    if Adam:  
                        dw = y_0 * pattern - lambd * reg
                        dw = dw * flipVec
                        m = beta1*m + (1-beta1)*dw
                        v = beta2*v + (1-beta2)*(dw**2)
                        self.weights += eta * m / (np.sqrt(v) + eps)
                    else:  
                        dw = eta * (y_0 * pattern - lambd * reg)
                        dw = dw * flipVec
                        self.weights = self.weights + dw         
                    self.weights = np.maximum(self.weights, np.zeros(self.N))
                    if not self.MESW == None:
                        self.weights[self.dataset.unbiasedStart:self.N] = np.minimum(self.weights[self.dataset.unbiasedStart:self.N], self.maxEffectiveWeights[self.dataset.unbiasedStart:self.N])                                     
            error = falsePos + falseNeg
            falsePosList.append(falsePos)
            falseNegList.append(falseNeg)
            errorList.append(error)
            ubconductances = self.weights[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]
            weightList = self.weights
            outputs = {'falseNegList': falseNegList, 'falsePosList': falsePosList, 'errorList': errorList,  'weightList':weightList.tolist()}
            if errorList[epoch] == 0:
                print 'converged'
#                 self.plotErrors(outputs)               
#                 self.jsonOutput(outputs, run, None)
                return errorList, falsePosList, falseNegList
#         self.jsonOutput(outputs, run, None)
#         self.plotErrors(outputs)
        print('bias', weightList[0])
        print('mean', np.mean(weightList[1:self.N]))
#         plt.figure('Weights' +  str(self.MESW), figsize = (5,5))
#         plt.hist(self.weights[self.dataset.unbiasedStart:self.N], bins = 50)
        #plt.show()                          
        return errorList, falsePosList, falseNegList
    
    def learnGen(self, maxEpochs, maxIterations, eta, Adam = True, run = 0, cond = 0):
        m = 0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        v = 0    
        self.weights = np.random.rand(self.N)
        falsePosList = []
        falseNegList = []
        errorList = []
        weightList = []
        flipVec = np.ones(self.N)
        flipVec[self.dataset.inhibInd:self.N] = -flipVec[self.dataset.inhibInd:self.N]
        for epoch in range(maxEpochs):
            dw = np.zeros(self.N)
            falsePos = 0
            falseNeg = 0
            sys.stdout.flush()
            for iter in range(maxIterations):
                pattern, y_0 = self.dataset.genBinaryVec(np.random.randint(self.dataset.numContexts))
                pOut = self.predict(pattern, cond = cond)
                if pOut != y_0:
                    if pOut == -1:
                        #print 'fn'
                        falseNeg += 1 
                    else:
                        #print 'fp'
                        falsePos += 1
                    if Adam:  
                        dw = y_0 * pattern
                        dw = dw * flipVec
                        m = beta1*m + (1-beta1)*dw
                        v = beta2*v + (1-beta2)*(dw**2)
                        self.weights += eta * m / (np.sqrt(v) + eps)
                    else:  
                        dw = eta * y_0 * pattern
                        dw = dw * flipVec 
                        self.weights = self.weights + dw        
                    self.weights = np.maximum(self.weights, np.zeros(self.N))
                    if not self.MESW == None: 
                        self.weights = np.minimum(self.weights, self.maxEffectiveWeights)                                      
            error = falsePos + falseNeg
            falsePosList.append(falsePos)
            falseNegList.append(falseNeg)
            errorList.append(error)
            ubconductances = self.weights[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]
            weightList = self.weights
            #print('weights', weightList)
#             plt.hist('weights')
            #plt.show()
            outputs = {'falseNegList': falseNegList, 'falsePosList': falsePosList, 'errorList': errorList,  'weightList':weightList.tolist()}
            #self.jsonOutput(outputs, run, None)
            if errorList[epoch] == 0:
                print 'converged'
                #self.plotErrors(outputs)               
                #return errorList, falsePosList, falseNegList
            #self.plotErrors(outputs)                                         
        return errorList, falsePosList, falseNegList

    def plotErrors(self, outputs):
        plt.ion()
        fp = outputs['falsePosList']
        fn = outputs['falseNegList']
        errors = outputs['errorList']
#         meanWeights = np.sum(weightList, axis = 1)
#         print meanWeights      
        fig = plt.figure(1)
        inds = np.arange(len(errors))
#         print inds
#         print fp
        plt.plot(inds, errors, color = 'blue', linewidth = 2)
        plt.plot(inds, fp, color = 'green', ls = 'dashed')
        plt.plot(inds, fn, color = 'red', ls = 'dotted')
        plt.legend(['Errors', 'False Positives', 'False Negatives'])
        plt.xlabel('Epoch')
        plt.ylabel('Errors')
        plt.title('Errors')
        plt.subplot(1,4,2)
        plt.show()
        
    def outputResults(self, N,P, errors, run):
        alpha = float(P)/float(N);
        epoch = str(len(errors))
        saveString = 'MCP/Capacity/' + 'N_' + str(N) + '_A_' + str(alpha) + '_R_' + str(run) + '.txt'
        with open(saveString, 'w+') as file:
            file.write(epoch +'\n')
            file.write(str(errors))

    def predict(self, example, cond = 0):
        rev = 0
        thresh = -22.8
        rest = -77.13
        inhibInd = self.dataset.inhibInd
        flipVec = np.ones(self.N)
        flipVec[self.dataset.inhibInd:self.N] = -flipVec[self.dataset.inhibInd:self.N]
        weights = self.weights * flipVec
        dot = np.dot(example, weights)
        #print 'orig', dot
        if cond: 
            dot = (rev-rest) * np.tanh(dot/50)
            print 'cond', dot
        return int(np.sign(dot + rest - thresh))
    
    def jsonOutput(self, outputs, run, epoch):
#         descStr = self.problem      
#         directory = 'MCPOutputs' +'/' + descStr + '/' + 'N_' + str(self.dataset.N) + '_P_' + str(self.dataset.P) + '_R_' + str(run)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         #print outputs
#         with open(directory + '/Out.json', 'w+') as outfile:  
#             json.dump(outputs, outfile)
        return

#     def kernelPredict(self, example, kernel = 'Gauss'):
#         
#     def kernelGauss(X)
def createDie(N, MESW, pickleName):
    #pickleName = 'maxMap/maxMapL5PC41e+99.pkl'
    with open(pickleName, 'r') as f:
        maxMap = pkl.load(f)
    allLocs = maxMap.keys()
    validLocations = maxMap.keys()
    print MESW
    if MESW == 'soma':
        maxes = [76.9]
    else:
        if MESW == 'aTuft':
            validLocations = [loc for loc in validLocations if loc[0] == 0 and loc[1] > 36 and loc[1] < 79 ]
        elif MESW == 'notATuft':
            validLocations = [loc for loc in validLocations if not (loc[0] == 0 and loc[1] > 36 and loc[1] < 79)]
        elif MESW == 'bDist':
            validLocations = [loc for loc in validLocations if loc [0] == 1 and loc[2] > 0.8]
        elif MESW == 'bProx':
            validLocations = [loc for loc in validLocations if loc [0] == 1 and loc[2] < 0.8]
        elif MESW == 'full':    
            validLocations = [loc for loc in validLocations if loc [0] == 0 or loc[0] == 1]
        elif MESW == 'basal':    
            validLocations = [loc for loc in validLocations if loc[0] == 1]
        elif MESW == 'soma':
             validLocations = [loc for loc in validLocations if loc[0] == 2]
        maxes = [maxMap[loc] for loc in validLocations]
    #print validLocations
    hist, bins = np.histogram(maxes, 20)
    density = [float(x) for x in hist]/sum(hist)
    return np.random.choice(bins[0:-1], p = density, size = N)
            
def axisConfigure(ax, numTicks = 2):      
        plt.tick_params(axis='both',which='both',left = 'off' ,right = 'off', bottom='off',top='off')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False) 
        ax.xaxis.set_major_locator( MaxNLocator(numTicks) )
        ax.xaxis.set_minor_locator( MaxNLocator(numTicks) )
        ax.yaxis.set_major_locator( MaxNLocator(numTicks) )
        ax.yaxis.set_minor_locator( MaxNLocator(numTicks) )
        ax.tick_params(axis='both', which='major', pad=15)
        
def capacityCalc(order, problem, MESWs = None, passive = False, weight_decay = False, save = False):
    plt.ion()
    activeNum = 200
    activeRatio = 0.2
    maxEpochs = 200
    numRuns = 5
    if problem == 'gen2':
        maxEpochs = 5
        numRuns = 25
        maxIterations = 1000
        lr = 0.9
    elif problem == 'mem':
        lr = 0.001 * 0.8 #Note: save unbiased values!
       # lr = 0.00004
#         lr = 0.015
        lr = 0.001
        maxEpochs = 1000 
        numRuns = 3
    inhibRatio = 0
    name = 'M&P'
    plt.ion()
    dateTime = datetime.datetime.now().isoformat().replace("-", "_").replace(':',"_").replace(".","_")
    folderName = 'Outputs/M&PReview/' + str(problem)
    if passive:
        folderName = folderName + '/Passive/'
    jsonOutputs = {"lr": lr, 
               "NPs":[list(x) for x in order], 
               "activeRatio": activeRatio, "numRuns": numRuns, 
               "maxEpochs": maxEpochs, 
               "inhibRatio":inhibRatio,
               "order": order}
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    with open(folderName + '/Meta.json', 'w+') as f:
        json.dump(jsonOutputs, f)

    for MESW in MESWs:
        NPs = order
        NPMap= {NPs[ind]:np.nan * np.ones([numRuns, maxEpochs]) for ind in range(len(NPs))}
        FPMap= {NPs[ind]:np.nan * np.ones([numRuns, maxEpochs]) for ind in range(len(NPs))}
        FNMap= {NPs[ind]:np.nan * np.ones([numRuns, maxEpochs]) for ind in range(len(NPs))}
        runMap = {NPs[ind]:0 for ind in range(len(NPs))}
        plotNum = MESWs.index(MESW)+1
        for NP in order:
            N = NP[0]
            P = NP[1]
            cumTrace = np.zeros(maxEpochs)
            cumPosTrace = np.zeros(maxEpochs)
            cumNegTrace = np.zeros(maxEpochs)
            distParams = [NP[1]] #This is the variance of the beta distribution
            #if problem == 'mem':
                #lr = 0.1/P #Note: This is for review!
            for run in range(numRuns):
                print str(NP) + ' ' + str(MESW) + 'run' + str(run)
                if problem == 'mem':
                    newlr = lr
                    newlr = lr
                    if P <= 1000:
                        newlr = lr/2
                    if P == 2000:
                        newlr = lr/50
                    data = pg.generateRand(P,N,inhibRatio, activeNum)
                    perc = Perceptron(data, problem, MESW = MESW, bias = 0)
                    errors, fp, fn = perc.learnMem(maxEpochs, newlr, Adam = True, lambd = 0)
                elif problem == 'mem2':
                    activeRatio = activeRatio
                    data = pg.generateRand(P,N,inhibRatio, N*activeRatio)
                    perc = Perceptron(data, problem, MESW = MESW, passive = passive)
                    errors, fp, fn = perc.learnMem(maxEpochs, lr, Adam = True)
                elif problem == 'gen':
                    maxIterations = 100
                    numContexts = 2
                    labels = np.ones(numContexts)
                    labels[numContexts/2:numContexts] = -1
                    data = pg.DataSetGen(N, activeNum, labels, distribution = 'beta', params = distParams)
                    perc = Perceptron(data, problem, MESW = MESW)
                    errors, fp, fn = perc.learnGen(maxEpochs, maxIterations, 0.0005,  Adam = True)
                elif problem == 'gen2':
                    numContexts = 2
                    labels = np.ones(numContexts)
                    labels[numContexts/2:numContexts] = -1
                    data = pg.DataSetBitFlip(N, activeNum, [-1,1], P, inhibRatio = 0)
                    perc = Perceptron(data, problem, MESW = MESW)
                    errors, fp, fn = perc.learnGen(maxEpochs, maxIterations, lr,  Adam = 0)
                #data = pg.generateNonBinary(P,N,inhibRatio)
                #data.sort()
                NPMap[NP][run][0:len(errors)] = errors
                FPMap[NP][run][0:len(errors)] = fp
                FNMap[NP][run][0:len(errors)] = fn
                NPMap[NP][run] = zeroNan(NPMap[NP][run])
                FPMap[NP][run] = zeroNan(FPMap[NP][run]) 
                FNMap[NP][run] = zeroNan(FNMap[NP][run])              
                if problem == 'mem' or problem == 'mem2':
                    errors = np.array(errors)/float(P)
                elif problem == 'gen' or problem == 'gen2':
                    errors = np.array(errors)/float(maxIterations)               
                runMap[NP] = runMap[NP] + 1
            #print errors
                trace = np.zeros(maxEpochs)
                trace[0:len(errors)] = errors
                cumTrace = cumTrace + trace
                
                negTrace = np.zeros(maxEpochs)
                negTrace[0:len(fn)] = fn
                cumNegTrace = cumNegTrace + negTrace
                
                posTrace = np.zeros(maxEpochs)
                posTrace[0:len(fp)] = fp
                cumPosTrace = cumPosTrace + posTrace
    
            cumTrace = cumTrace/numRuns
            #print(cumTrace)
            plt.figure('Capacity' +str(problem))
            plt.subplot(1, len(MESWs), plotNum)
            plt.plot(range(len(cumTrace)), 1 - cumTrace, label = str(NP))
            plt.ylim([0.5,1])
            plt.legend(loc = 'best')
            plt.title(str(MESW))
            plt.show()
            plt.pause(0.001)
    #         
    #         cumPosTrace = cumPosTrace/numRuns
    #         print cumPosTrace
    #         plt.figure('FP')
    #         plt.subplot(1, len(MESWs), plotNum)
    #         plt.plot(range(len(cumPosTrace)), cumPosTrace, label = str(NP))
    #         plt.legend(loc = 'best')
    #         plt.title(str(MESW))
    #         plt.show()
    #         
    #         cumNegTrace = cumNegTrace/numRuns
    #         print cumNegTrace
    #         plt.figure('FN')
    #         plt.subplot(1, len(MESWs), plotNum)
    #         plt.plot(range(len(cumNegTrace)), cumNegTrace, label = str(NP))
    #         plt.legend(loc = 'best')
    #         plt.title(str(MESW))
    #         plt.show()
        plt.savefig(folderName + str(problem) + '.png')
        if save:
            pickleName = folderName + '/M&P' + stringCleaner(MESW)  + '.pkl'
            print pickleName
            print runMap
            print NPMap
            with open(pickleName, 'w+') as f:
                pkl.dump({'NPMap':NPMap, 'FPMap':FPMap, 'FNMap':FNMap, 'runMap':runMap}, f)
        #plt.close('all')
        #finals = np.empty((len(Ns), len(Ps)))      
    #     for i in range(len(Ns)):
    #         for j in range(len(Ps)):
    #                 avgTrace = np.nanmean(NPArray[i][j],0)/Ps[j]
    #                 finals[i][j] = 1 - np.nanmin(avgTrace)
    #     print finals
                                
                #plt.title(str(P) + ' patterns')
    #plt.show()


def stringCleaner(my_str):
    if my_str == None:
        return ''
    else: 
        my_str = string.replace(my_str, '.pkl', '')
        my_str = string.replace(my_str, 'PassCurrent', ' current')
        my_str = string.replace(my_str, 'Pass', 'xyz')
        my_str = string.replace(my_str, "xyz", ' passive')
        my_str = string.replace(my_str, 'notATuft', 'No tuft ')
        my_str = string.replace(my_str, 'aTuft', 'Apical')
        my_str = list(my_str)
        my_str[0] = my_str[0].upper()
        my_str = "".join(my_str)
        my_str = string.replace(my_str, 'Apical', ' Apical')
        my_str = string.replace(my_str, 'Basal', ' Basal')
        my_str = string.replace(my_str, 'Soma', ' Soma')
        my_str = string.replace(my_str, 'Full', ' Full')
    print my_str
    return my_str           

# X,y,inhibInd = pg.generateNonBinary(P,N,inhibRatio)
# weights = None
# bias = None
# data = pg.dataset(X,y, inhibInd, weights, bias)
# print X
# print y 
#perc = Perceptron(data)

# errors1, fp1, fn1 = perc.learn(1000, 0.1, Adam = False)
# P = 500
# N = 500
# # activeRatio = 0.6
# inhibRatio = 0.2 #Percentage of inhibitory synapses
#  #starting index of inhibitory synapses
             
# X, inhibInd = pg.generateFixedActivation(P, N, inhibRatio, 0.1)
# [weights, bias, y] = pg.generatelinear(X, inhibInd)
# print y[1:10]
# np.random.shuffle(y)
# print y[1:10]
# 
# data = pg.Dataset(X,y, inhibInd, weights, bias)
# data.addExciteBias(1)
# data.addInhibBias(1)
# data.sort()
# 
# # X,y,inhibInd = pg.generateNonBinary(P,N,inhibRatio)
# # weights = None
# # bias = None
# # data = pg.dataset(X,y, inhibInd, weights, bias)
# # print X
# # print y
#   
# perc = Perceptron(data)
# errors1, fp1, fn1 = perc.learn(1000, 0.1, Adam = False)
# print 'errors', errors1
# cProfile.run('capacityCalc(cond = 0)', 'restats')
# p = pstats.Stats('restats')
# p.sort_stats('cumtime')
# p.print_stats(20)

 

MESWs =  ['soma', 'full', 'basal', 'aTuft', None]
#MESWs = ['None']
#MESWs = ['aTuft']
# order = [(10000,1000), (2000,1000), (1000,1000)]
# order = [(1000,100), (1000,1000), (1000,2000)]
# capacityCalc(order, problem = 'mem', MESWs = MESWs)
 
order = [(10000, 0.0001), (1000, 0.0001), (1000, 0.001), (1000, 0.01)]
order = [(10000, 1e-05), (10000, 0.0001), (10000, 0.001), (10000, 0.01)]
order = [(10000, 1e-07), (10000, 1e-05), (10000, 1e-3)]
order = [(10000, 1000), (1000, 1000), (500, 1000)]
order = [(1000, 2), (1000, 70), (1000,130)]
order = [(1000, 2), (1000, 100), (1000, 200)]
order = [(1000, 0), (1000, 100), (1000, 200)]
order = [(1000, 100), (1000, 1000), (1000, 2000)]
# print order
save = 0
# order = [(1000, 0), (1000, 100), (1000, 200)]
# capacityCalc(order, problem = 'gen2', MESWs = MESWs, passive = False, save = save)
order = [(1000, 100), (1000, 1000), (1000, 2000)]
capacityCalc(order, problem = 'mem', MESWs = MESWs, passive = False, save = save)
