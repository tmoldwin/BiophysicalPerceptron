import numpy as np
from neuron import h, gui



def random_synapse(cell,rd,total_L, total_basal_L):
    """
    returns a random location in cell - a neuron model
    rd -- NEURON random object
    total_L -- total dendritic length
    total_basal_L -- total basal length
    they are used here to choose a synaptic location out of the uniform distribution of dendritic locations
    that give the same probability to any point on the dendritic tree 
    note that just choosing segments randomly would ignore the segments physical length and would bias
    more synapses on shorter segments
    """

    synaptic_loc = rd.uniform(0,total_L)
    if synaptic_loc<total_basal_L:
        return basal_random_synapse(cell,synaptic_loc)
    else:
        return apical_random_synapse(cell,synaptic_loc-total_basal_L)


def basal_random_synapse(cell,synaptic_loc):
    ''' returns a random location in the basal tree of this cell'''
    len0 = 0
    len1 = 0
    for i in range(len(cell.dend)):
        sec = cell.dend[i]
        len1 += sec.L
        if len1 >= synaptic_loc:
            x = (synaptic_loc-len0)/sec.L
            return (1,i,x)
        h.pop_section()
        len0 = len1


def apical_random_synapse(cell,synaptic_loc):
    ''' returns a random location in the apical tree of this cell'''
    len0 = 0
    len1 = 0
    for i in range(len(cell.apic)):
        sec = cell.apic[i]
        len1 += sec.L
        if len1 >= synaptic_loc:
            x = (synaptic_loc-len0)/sec.L
            return (0,i,x)
        h.pop_section()
        len0 = len1
        
def getLengths(cell):
    seed = 1
    total_basal_L = sum([sec.L for sec in cell.dend])
    print 'basal_L', total_basal_L
    total_L = sum([sec.L for sec in cell.dend]) + sum([sec.L for sec in cell.apic]) 
    print 'total_L', total_L
    return total_basal_L, total_L

def generateLocations(cell, number_of_synapses):
    locs = []
    seed = 1
    rd = h.Random(seed)
    total_basal_L, total_L = getLengths(cell)
    print number_of_synapses
    for i in range(number_of_synapses):
        loc = random_synapse(cell,rd,total_L,total_basal_L)
        locs.append(loc)
    return locs
