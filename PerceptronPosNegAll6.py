# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:40:40 2016

@author: toviah.moldwin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:29:15 2016

@author: toviah.moldwin
"""

#import NEURONFix
import sys
# sys.path.append("D:/nrn/lib/python")
# sys.path.append('/Applications/NEURON-7.3/nrn/lib/python')
#sys.path.append('/Applications/NEURON-7.5/nrn/lib/python')
# sys.path.append("/home/ls/users/toviah.moldwin/Dropbox/IdanLab/PythonFiles/NEURONPython")
import matplotlib
#matplotlib.use('Qt5Agg')
import numpy as np
from neuron import h, gui
from L5PC6 import L5PC6
import os
if not "SSH_CONNECTION" in os.environ:
    from matplotlib import pyplot as plt
import spikeAnalysis as spk
import cPickle as pkl
import hocHelper as hH
import copy
import subprocess  
import random
import PatternGenerator as pg
import scipy.spatial.distance as sdist
import time
import iterationData as iD
import json
import socket
import datetime
from pylab import rcParams
from collections import Counter
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('axes', titlesize=20)  
rcParams['figure.figsize'] = 5, 5

#ys.path.append("/home/ls/users/toviah.moldwin/Dropbox/IdanLab/PythonFiles/NEURONPython")

""" This class extends an L5PC neuron to perform perceptron learn and predict behaviors """
class PerceptronPosNegAll6():
    #Local
    def __init__(self, dataset, problem = 'mem', inputWeightMap = None, inputLocationMap = None, NMDABlock = False, DendBlock = False, Current = False, dispersion = [0,1], DistanceScale = False, saveFolder = "PPNTest", saveIterations = False, Traces = False):
        self.runStop = 50 ##These are the original parameters!
        self.runStop = 80
        self.runStop = 100
        self.DistanceScale = DistanceScale
        self.saveFolder = saveFolder
        self.cell = L5PC6(NMDABlock = NMDABlock, dispersion = dispersion, Current = Current, DendBlock = DendBlock)
        print self.cell.validLocations
        self.dataset = dataset
        self.problem = problem
        self.iterations = []
        self.falsePosList = []
        self.falseNegList = []
        self.errorList = []
        self.saveIterations = saveIterations
        self.Traces = Traces
        
        if inputWeightMap is None:
            self.inputWeightMap = np.empty(self.dataset.N, dtype = 'object')
            self.inputLocationMap = np.empty(self.dataset.N, dtype = 'object')
            self.generateInitialInputMap()
        else:
            self.inputWeightMap = inputWeightMap
            self.inputLocationMap = inputLocationMap        
        if DendBlock:
            self.cell.blockDendriticChannels()
        
    def str2Tup(self,locStr):
        s = locStr[1:-1]
        splitS = str.split(s, ',')
        return tuple([int(x) for x in splitS])
    
    def generateInitialInputMap(self, numArbors = 1):
        validLocations = [str(loc) for loc in self.cell.validLocations]
        for i in range(self.dataset.N):
            t = np.random.choice(validLocations, p = self.cell.uniformDistDie)
            loc = self.str2Tup(t)
            self.inputLocationMap[i] = loc
            #self.inputWeightMap[i] =  (0.02*np.random.rand())
            self.inputWeightMap[i] = 0
            if self.cell.Current:
                self.inputWeightMap[i] = 15.5
            self.initialInputWeightMap = self.inputWeightMap
#         print self.inputLocationMap
#         cnt = Counter([str(x) for x in self.inputLocationMap])
#         print cnt
#         plt.hist([str(x) for x in self.inputLocationMap])
#         plt.xticks([])
#         plt.show()
            
    def activate(self, input, weightMap, numStims, jitter):
        weightMapPos = [(self.inputLocationMap[i], self.inputWeightMap[i] * input[i]) for i in range(self.dataset.inhibInd)]
        weightMapNeg = [(self.inputLocationMap[i], self.inputWeightMap[i] * input[i]) for i in range(self.dataset.inhibInd, self.dataset.N)]
        #print weightMapPos
        if self.cell.Current:
            scale = 1
            weightMapPos = [(self.inputLocationMap[i], scale * self.inputWeightMap[i] * input[i]) for i in range(self.dataset.inhibInd)]
            weightMapNeg = [(self.inputLocationMap[i], scale * self.inputWeightMap[i] * input[i]) for i in range(self.dataset.inhibInd, self.dataset.N)]
        soma_v_vec, dend_rec_map, t_vec = self.cell.set_recording_vectors(self.cell.validLocations);
        self.cell.addExciteStim(weightMapPos, numStims, jitter)
        self.cell.addGABA(weightMapNeg, numStims, jitter)
        self.simulate()
        return soma_v_vec, dend_rec_map, t_vec
    
    def perceptronLearnMem(self, eta, numEpochs, plot = False, run = 0):
        self.run = run
        X = self.dataset.X
        y = self.dataset.y 
        w_0 = self.dataset.w0, 
        
        m = 0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        v = 0
        dw = np.zeros(self.dataset.N)
        flipVec = np.ones(self.dataset.N)
        flipVec[self.dataset.inhibInd:self.dataset.N] = -flipVec[self.dataset.inhibInd:self.dataset.N]
        X = np.array(X)
        for epoch in range(numEpochs):
            iterations = []
            inds = np.random.permutation(list(range(self.dataset.P)))
            falsePos = 0
            falseNeg = 0
            print 'Epoch', epoch
            sys.stdout.flush()
            dws = np.zeros(self.dataset.N) #changes in the iteration (iterative) or epoch(batch)
            for i in range(self.dataset.P):
                printStr = 'Iter '+ str(i)
                patternNum = inds[i]
                y_0 = y[patternNum];
                pattern = X[patternNum]
                iteration = self.predict(pattern, y_0, epoch, patternNum,  plot = plot)
                iterations.append(iteration)
                spikeCount = iteration.nSpikes
                printStr += ' spks ' + str(spikeCount)
                if bool(spikeCount) != bool(y_0+1):
                    if spikeCount == 0:
                        falseNeg += 1 
                        printStr += ' fn'
                    else:
                        falsePos += 1
                        printStr += ' fp'              
                    dw = np.squeeze(pattern) * y_0
                    if self.DistanceScale:
                        dw = np.asarray([dw[j] / self.cell.TRMap[self.inputLocationMap[j]] for j in range(self.dataset.N)])
                    dw[self.dataset.inhibInd:self.dataset.N] = -dw[self.dataset.inhibInd:self.dataset.N] #if it's an inhibitory synapse, change in the opposite direction
                    m = beta1*m + (1-beta1)*dw
                    v = beta2*v + (1-beta2)*(dw**2)
                    self.inputWeightMap += eta * m / (np.sqrt(v) + eps)
                    #self.inputWeightMap += eta * dw
                self.inputWeightMap = np.maximum(self.inputWeightMap, np.zeros(self.dataset.N)) # ensures that all weights are positive
                #print 'wgt', [round(x, 1) for x in self.inputWeightMap]
                print printStr
                
            self.falsePosList.append(falsePos)
            self.falseNegList.append(falseNeg)
            self.errorList.append(falsePos + falseNeg)
            ubconductances = self.inputWeightMap[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]*flipVec[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]
#             outputs = {'falseNegList': self.falseNegList, 'falsePosList': self.falsePosList, 'errorList': self.errorList,
#                        'dataSet': self.dataset, 'locMap': self.inputLocationMap, 'weightList': self.inputWeightMap, 'intialWeightList' : self.initialInputWeightMap}
#             self.pickleOutput(outputs, run, iterations, epoch)
            outputs = {'falseNegList': self.falseNegList, 'falsePosList': self.falsePosList, 'errorList': self.errorList, 'locMap': self.inputLocationMap.tolist(), 'weightList': self.inputWeightMap.tolist(), 'intialWeightList' : self.initialInputWeightMap.tolist()}
            self.jsonOutput(outputs, run, iterations, epoch)
            print 'errEp', self.errorList
            if self.errorList[epoch] == 0:
                print('converged')
                return self.errorList, self.falsePosList, self.falseNegList   
        return self.errorList, self.falsePosList, self.falseNegList
    
    def perceptronLearnGen(self, eta, numEpochs, iterationsPerEpoch, plot = False, run = 0, batch = False, momentum = False):        
        self.run = run
        dw = np.zeros(self.dataset.N)
        flipVec = np.ones(self.dataset.N)
        flipVec[self.dataset.inhibInd:self.dataset.N] = -flipVec[self.dataset.inhibInd:self.dataset.N]
        iterations = []
        for epoch in range(numEpochs):
            dwBatch = np.zeros(self.dataset.N)
            falsePos = 0
            falseNeg = 0
            sys.stdout.flush()
            for i in range(iterationsPerEpoch):
                pattern, y_0 = self.dataset.genBinaryVec(int(np.random.rand() > 0.5))
                print y_0
                #pattern, y_0 = self.dataset.genBinaryVec(i%2)
                printStr = 'Epoch ' + str(epoch) + ' iter ' + str(i) + ' y_0' + str(y_0)
                iteration = self.predict(pattern, y_0, epoch, 0,  plot = plot)
                iterations.append(iteration)
                spikeCount = iteration.nSpikes
                printStr += ' spks ' + str(spikeCount)
                if bool(spikeCount) != bool(y_0+1):
                    if spikeCount == 0:
                        falseNeg += 1 
                        printStr += ' fn'
                    else:
                        falsePos += 1
                        printStr += ' fp'              
                    dw = np.squeeze(pattern) * y_0
                    if self.DistanceScale:
                        dw = np.asarray([dw[j] / self.cell.TRMap[self.inputLocationMap[j]] for j in range(self.dataset.N)])
                    dw[self.dataset.inhibInd:self.dataset.N] = -dw[self.dataset.inhibInd:self.dataset.N] #if it's an inhibitory synapse, change in the opposite direction
#                     dwBatch = dwBatch + dw
#                     if not batch:
                    if momentum:
                        m = beta1*m + (1-beta1)*dw
                        v = beta2*v + (1-beta2)*(dw**2)
                        self.inputWeightMap += eta * m / (np.sqrt(v) + eps)
                    else:
                        self.inputWeightMap += eta * dw
                    #self.inputWeightMap += eta * dw
                    self.inputWeightMap = np.maximum(self.inputWeightMap, np.zeros(self.dataset.N)) # ensures that all weights are positive
                    #self.inputWeightMap = np.minimum(self.inputWeightMap, 1.5*np.ones(self.dataset.N)) # ensures that all weights are positive
                    #print 'wgt', [round(x, 1) for x in self.inputWeightMap]
#             if batch:
#                 dwBatch = dwBatch/iterationsPerEpoch
#                 m = beta1*m + (1-beta1)*dwBatch
#                 v = beta2*v + (1-beta2)*(dwBatch**2)
#                 self.inputWeightMap += eta * m / (np.sqrt(v) + eps)
#                 self.inputWeightMap = np.maximum(self.inputWeightMap, np.zeros(self.dataset.N))
            self.falsePosList.append(falsePos)
            self.falseNegList.append(falseNeg)
            self.errorList.append(falsePos + falseNeg)
            ubconductances = self.inputWeightMap[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]*flipVec[self.dataset.unbiasedStart:self.dataset.unbiasedEnd]
#             outputs = {'falseNegList': self.falseNegList, 'falsePosList': self.falsePosList, 'errorList': self.errorList,
#                        'dataSet': self.dataset, 'locMap': self.inputLocationMap, 'weightList': self.inputWeightMap, 'intialWeightList' : self.initialInputWeightMap}
#             self.pickleOutput(outputs, run, iterations, epoch)
            outputs = {'falseNegList': self.falseNegList, 'falsePosList': self.falsePosList, 'errorList': self.errorList, 'locMap': self.inputLocationMap.tolist(), 'weightList': self.inputWeightMap.tolist(), 'intialWeightList' : self.initialInputWeightMap.tolist()}
            self.jsonOutput(outputs, run, iterations, epoch)
            print 'errEp', self.errorList
            if self.errorList[epoch] == 0:
                print('converged')
                #return self.errorList, self.falsePosList, self.falseNegList   
        return self.errorList, self.falsePosList, self.falseNegList

    def loadTrs(self, fn = 'DistsAndTRs.pkl'):
        with open(fn) as file:
            self.trs = pkl.load(file)
            print(self.trs)
            
    def predict(self, pattern, label, epoch, index,  jitter = False, numStims = 1 , plot = False, traces = False ):
        spikeCounts = []
        soma_v_vec, dend_rec_map, t_vec = self.activate(pattern,  self.inputWeightMap, numStims, jitter)
        times = np.asarray(t_vec.to_python())
        somaTrace = np.asarray(soma_v_vec.to_python())
        dendRecMap = hH.hocToPython(dend_rec_map)
        spikeInds = spk.findSpikes(somaTrace)
        numSpikes = len(spikeInds)
        if self.Traces:
            iteration = iD.iterationData(index, epoch, None, somaTrace, times, numSpikes, self.inputWeightMap)
        else:
            iteration = iD.iterationData(index, epoch, None, None, None, numSpikes, self.inputWeightMap)
        #print 'Spikes', (numSpikes)
        if plot == True:
            self.plotSomaTraces(times, somaTrace, label, epoch, index)
            #locsToPlot = self.inputBackMap.keys()[:]
            #self.plotDendTraces(times, dendRecMap, locsToPlot, title) )
        self.cell.clearStimuli()
        return iteration
    
    def plotSomaTraces(self, times, trace, label, epoch, index, xor = False, simple = True):
        if label > 0:
            labelstr = '+'
        else: 
            labelstr = '-'            
        plt.ion()
        #rcParams['figure.figsize'] = 15, 10
        
        peakInd = spk.findSpikes(trace)
        fig = plt.figure('SomaTraces')
        if self.problem == 'mem':
            ax = plt.subplot(2,self.dataset.P/2,index+1)
            plt.setp([line for line in ax.lines], linestyle = 'dashed', linewidth = 1) #All the lines except for the last one are dashed
        elif self.problem in ['gen', 'gen2']:
            if label > 0:
                plotNum = 2
            else: 
                plotNum = 1
                ax = plt.subplot(1, 2, plotNum)
        if simple == True:
            ax = plt.subplot(1, 1, 1)
        #plt.setp([line for line in ax.lines], linestyle = 'dashed', linewidth = 1) #All the lines except for the last one are dashed        
        plt.cla()
        lines = plt.plot(times, trace, linewidth = 2)
        plt.xticks(list(range(0, self.runStop, 20)))
        if len(peakInd) > 0:
            spikeLocs = np.nan * np.ones([len(trace)])
            spikeLocs[peakInd] = trace[peakInd]
            #plt.scatter(times, spikeLocs)
        if xor == True:
            xorList = [[0,0],[0,1],[1,0],[1,1]]
            plt.title('Pattern: ' + str(xorList[index]) + ' Class: ' + labelstr)
        else: 
            plt.title('Pattern: ' + str(index) + ' Class: ' + labelstr)
        #if index == 0:
            #plt.legend(['Epoch' + str(x) for x in range(epoch + 1)], loc = 'best')
        plt.ylim(-90,50)
        #plt.subplot(2,self.dataset.P/2,self.dataset.P/2 +1)
        plt.xlabel('Time (ms)')
        plt.ylabel(('Voltage (mV)'))
        plt.show()
        plt.pause(0.00005)
        saveString = 'Figures/SomaTraces' + 'P' + str(self.dataset.P) + 'N' + str(self.dataset.N) + str(self.problem)
        #plt.savefig(saveString)
        return fig
                    
    def pickleOutput(self, outputs, run, iterations, epoch):
        descStr = str(int(sum(self.dataset.X[0]))) + str(self.cell.dispersion)
        if self.cell.NMDABlock:
            descStr += 'NoNMDA'
        if self.cell.Current:
            descStr += 'Current'
        if self.cell.DendBlock:
            descStr += 'Pass'
        if self.DistanceScale:
            descStr += 'DistScale'
        
        directory = self.saveFolder +'/' + descStr + '/' + 'N_' + str(self.dataset.N) + '_P_' + str(self.dataset.P) + '_R_' + str(run)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + '/Out.pkl', 'w+') as pkl_file:
            pkl.dump(outputs, pkl_file)
        if self.Traces:
            itDirectory = 'Iters/' + directory 
            if not os.path.exists(itDirectory):
                os.makedirs(itDirectory)
            outputfn =  itDirectory + '/E'+ str(epoch) + '.pkl'
            with open(outputfn, 'w+') as pkl_file:
                pkl.dump(iterations, pkl_file, protocol = 2)
                
    def jsonOutput(self, outputs, run, iterations, epoch):
        descStr = str(self.cell.dispersion)
#         if self.cell.NMDABlock:
#             descStr += 'NoNMDA'
        if self.cell.NMDABlock:
            descStr += 'NoNMDA'
        if self.cell.Current:
            descStr += 'Current'
        if self.cell.DendBlock:
            descStr += 'Pass'
        if self.DistanceScale:
            descStr += 'DistScale'
        
        directory = self.saveFolder +'/' + descStr + '/' + 'N_' + str(self.dataset.N) + '_P_' + str(self.dataset.P) + '_R_' + str(run)
        if not os.path.exists(directory):
            os.makedirs(directory)
        #print outputs
        with open(directory + '/Out.json', 'w+') as outfile:  
            json.dump(outputs, outfile)
        if self.saveIterations and self.run == 0:
            itDirectory = 'Iters/' + directory
            if not os.path.exists(itDirectory):
                os.makedirs(itDirectory)
            if epoch == 0 or epoch == 50:
                outputfn =  itDirectory + '/E'+ str(epoch)
            else:
                outputfn =  itDirectory + '/Erecent'
            with open(outputfn + '.pkl', 'wb+') as pkl_file:
                pkl.dump(iterations, pkl_file)
#             with open(outputfn + '.json', 'w+') as json_file:
#                 json.dump(iterations, json_file, default = obj_dict)
                

          
  
    
    def plotErrors(self, errorList, falsePosList, falseNegList,N,P):
        inds = np.arange(len(errorList))
        plt.ion()
        fig = plt.figure('Errors')
        plt.clf()
        plt.axis([0, len(errorList), 0, max(errorList)+1])
        plt.plot(inds, errorList, color = 'blue', linewidth = 2)
        plt.plot(inds, falsePosList, color = 'green', ls = 'dashed')
        plt.plot(inds, falseNegList, color = 'red', ls = 'dotted')
        plt.xlabel('Epoch')
        
        plt.ylabel('Errors')
        plt.xticks(inds)
        plt.title('Errors')
        plt.legend(['Errors', 'False Positives', 'False Negatives'])
        #plt.show()
        #plt.pause(0.05)
        saveString = 'Figures/Error' + 'P' + str(P) + 'N' + str(N) + '.jpg'
        plt.savefig(saveString)
        
    def simulate(self):
        h.tstop = self.runStop
        h.dt= 0.1
        h.steps_per_ms=int(1/h.dt)
        h.run()

def obj_dict(obj):
    return obj.__dict__         

print socket.gethostname()
if not "SSH_CONNECTION" in os.environ:
#         'Gen2'
         numFlips = 30
         N = 1000
         inhibRatio = 0
         activeNum = 200
         #data = pg.generateRand(P, N, inhibRatio, 0.1)
         #data = pg.generateAllPos(P, N, inhibRatio, activeNum)
#          print data.X
#          print data.y
#          print data.inhibInd
         data = pg.DataSetBitFlip(N, activeNum, [-1,1], numFlips, inhibRatio = 0)
         print data.contexts        
         pb = PerceptronPosNegAll6(data, problem = 'gen2', NMDABlock = 0, Current = 0, DendBlock = 0, DistanceScale = 0, dispersion = 'full', saveFolder = 'Outputs/Tests/', saveIterations = True, Traces = True)
         errors, fp, fn = pb.perceptronLearnGen(1e-2, numEpochs = 10, iterationsPerEpoch = 20, plot = False, batch = False) 
         print('errors', errors)
         fig = plt.figure('Errors')
         plt.plot(range(len(errors)), errors)
         plt.xlabel('Epoch')
         plt.ylabel('Error')
         plt.show()
         plt.savefig('Gen2.png')
         
         