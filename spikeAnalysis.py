# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:59:33 2016

@author: toviah.moldwin
"""
import numpy as np
from scipy import signal 
from matplotlib import pyplot as plt

def findSpikes(trace):
    threshold = 0;
    peakInd = np.asarray(signal.find_peaks_cwt(trace, np.arange(1,2))) #Find peaks
    if len(peakInd) > 0:
        peakVals = trace[peakInd] #Values at those peaks
        peakInd = peakInd[np.where(peakVals > threshold)]
    return peakInd

def findCa2Spike(caTrace):
    threshold = -0.0005
    if len(np.squeeze(np.where(caTrace < threshold))) > 1:
       return 1
    else: 
        return 0

def plotTrace(times, trace, title = ''):
    peakInd = findSpikes(trace)
    print(peakInd)
    if title == '-':
             ls = ':'
    else:
             ls = 'solid'
    plt.plot(times, trace, ls = ls)     
    if len(peakInd) > 0:
        spikeLocs = np.nan*np.ones([len(trace)])
        spikeLocs[peakInd] = trace[peakInd]
        plt.scatter(times, spikeLocs)
    plt.title('Somatic Voltage Trace ' + title)
    plt.xlabel('Time (ms)')
    plt.ylabel(('Voltage (mV)'))

def tracesToPython(traces):
    pyTraces = []
    for n in range(len(traces)):
        pyTraces.append(traces[n].to_python())
    return np.matrix(pyTraces)
    
def integrate(traces):
    return np.sum(traces, axis = 1)
    
#xs = np.arange(0, 10*np.pi, 0.05)
#trace = np.sin(xs)
#peakind = findSpikes(trace)
#peakind, xs[peakind], trace[peakind]
#plotTrace(range(len(trace)), trace)