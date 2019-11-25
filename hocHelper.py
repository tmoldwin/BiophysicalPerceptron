# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:45:32 2016

@author: toviah.moldwin
"""
import numpy as np

def hocToPython(traces):
    """Takes a map of hoc vectors and turn it into a map of python vectors"""
    if(type(traces)) is dict:
        newMap = {}
        for key in traces.keys():
            newMap[key] = np.asarray(traces[key].to_python())
        return newMap
    else:
        pyTraces = []
        for n in range(len(traces)):
            pyTraces.append(traces[n].to_python())
        return np.matrix(pyTraces)