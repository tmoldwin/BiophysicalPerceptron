import pickle
import numpy as np
import json
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt
import os

#Create dies for each synapse dispersion, do only once to create files
def generateAllProbMaps():
    for disp in ['soma', 'full', 'basal', 'aTuft']:
        cell = L5PC6(dispersion = disp)
        with open('UniformDispersionProbabilities/'+disp +'.json', 'w+') as f:
            dieMap = {str(key):cell.dieMap[key] for key in cell.dieMap.keys()}
            json.dump(dieMap, f)
    
        
def str2Tup(locStr):
    s = locStr[1:-1]
    splitS = str.split(s, ',')
    return tuple([int(x) for x in splitS])

#For a given dispersion, gets the probability map from a file and converts the strings to tuples
def loadUniformDispMap(disp):
    fn = 'UniformDispersionProbabilities/' + disp + '.json'
    with open(fn) as f:
        data = json.load(f)
    #print(data)
    return data

def loadMESWs(fn):
    MESWfn = fn
    with open(MESWfn, 'r') as f:
        data = json.load(f) 
    MESWs = data['MESWs']
    AA_MESWs = data['AA_MESWs']
    maxVoltage = data['maxVoltages']
    return MESWs, AA_MESWs, maxVoltage
#Generates N locations randomly drawn according the length distribution
def generateRandomLocations(disp, N):
    
    orderedDist = OrderedDict(disp)
#     print(orderedDist.keys())
#     print(orderedDist.values())
    sampled_locs_list = np.random.choice(list(orderedDist.keys()), N, replace = True, p = list(orderedDist.values())) #returns list of locations, sampled by uniform distance
    #sampled_locs_list = [str2Tup(str(x)) for x in sampled_locs_list]
    #print(sampled_locs_list)
    return sampled_locs_list

#Returns the MESWs for a given location list, mainly useful for histograms or weight caps   
def MESWUniformDist(randomLocationList, MESWMap):
    return [MESWMap[loc] for loc in randomLocationList]

def getMESWDistFromFile(MESWfn, disp,N = 1000):
        MESWs, AA_MESWs, maxVoltage = loadMESWs(MESWfn)
        dieMap = loadUniformDispMap(disp)
        locations = generateRandomLocations(dieMap, N)
        MESWUni = MESWUniformDist(locations, MESWs)
        AA_MESWUni = MESWUniformDist(locations, AA_MESWs)
        maxVoltagesUni = MESWUniformDist(locations, maxVoltage)
        print('MESW', np.mean(MESWUni))
        print('AA', np.mean(AA_MESWUni))
        return MESWUni, AA_MESWUni, maxVoltagesUni


#Shows MESW histograms for different levels of background activity
def compareHists(disp = 'aTuft'):
    #plt.hold('on')
    names = ['MESWs/' + x + '/MESWs.json' for x in os.listdir('MESWs/')  ]
    for name in names:
        MESWUni, AA_MESWUni, maxVoltage = getMESWDistFromFile(name, disp)
        nbins = 50
        sns.distplot(MESWUni, bins = nbins)
        sns.distplot(AA_MESWUni, bins = nbins)
    plt.legend(['MESW', 'AA'])
    plt.show()
    
def plotMeans(disp = 'aTuft'):
    names = ['MESWs/' + x + '/MESWs.json' for x in os.listdir('MESWs/')  ]
    meanMESW = []
    meanAA = []
    maxVoltages = []
    weight = []
    for name in names:
        MESWUni, AA_MESWUni, maxVoltage = getMESWDistFromFile(name, disp)
        nbins = 50
        meanMESW.append(np.mean(MESWUni))
        meanAA.append(np.mean(AA_MESWUni))
        maxVoltages.append(np.mean(maxVoltage))
    print maxVoltages
    print meanMESW
    plt.scatter(maxVoltages, meanMESW)
    plt.scatter(maxVoltages, meanAA)
    plt.legend(['MESW', 'AAMESW'])
    plt.show()

def createCaps(N, dispersion, MESW_name, AA = 1):
    #Creeate a distribution of weight caps proportional to the length
        MESWs, AA_MESWs, maxVoltage = loadMESWs(MESW_name)
        dieMap = loadUniformDispMap(dispersion)
        locations = generateRandomLocations(dieMap, N)
        if AA:
            weightsUni = MESWUniformDist(locations, AA_MESWs)
        else:
            weightsUni = MESWUniformDist(locations, MESWs)
        return weightsUni

# print dispMap
# generateRandomLocations(dispMap, 200)
if __name__ == "__main__":
    plotMeans()
  

   