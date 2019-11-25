# //Author: Toviah Moldwin, 2016
# Neuron parameters based on:
# Etay Hay, 2011
# //  Models of Neocortical Layer 5b Pyramidal Cells Capturing a Wide Range of
# //  Dendritic and Perisomatic Active Properties
# //  (Hay et al., PLoS Computational Biology, 2011) 
# //
# // A simulation of L5 Pyramidal Cell BAC firing.

import sys
#from gentoolkit.pprinter import die
sys.path.append("/ems/elsc-labs/segev-i/toviah.moldwin/Documents/Liclipse Workspace New/MinimalBP/x86_64")
sys.path.append("C://nrn/lib/python/")
sys.path.append("C://nrn")
sys.path.append("C://nrn/lib/python/")
import neuron
from neuron import h, gui
import platform
print platform.node()
# if platform.node()=='sushi':
#     h.nrn_load_dll("x86_64_sushi/.libs/libnrnmech.so.0")
# elif platform.node() == 'bs-cluster' or platform.node == 'ocean' or "cluster" in platform.node() or "brain" in platform.node():
#     h.nrn_load_dll("x86_64_cluster/.libs/libnrnmech.so.0")
# elif platform.node() == 'Sriracha':
#     h.nrn_load_dll("D:\Dropbox\IdanLab\PythonFiles\NEURONPython\x86_64_Sriracha\nrnmech.dll")
h.nrn_load_dll("x86_64_cluster/.libs/libnrnmech.so.0")
import socket
import matplotlib
import os
if "SSH_CONNECTION" in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
import numpy as np
import pickle
import spikeAnalysis as spk
from pylab import rcParams
import os
import subprocess
import UniformSynPlacement as SP

plt.ion()

class L5PC6(object):
    
    def __init__(self, NMDABlock = 0, Current = 0, SomaBlock = 0, DendBlock = 0, channelBlock = None, dispersion = 'full'):
        self.NMDABlock = NMDABlock
        self.dispersion = dispersion
        self.Current = Current
        self.SomaBlock = SomaBlock
        self.DendBlock = DendBlock
        self.channelBlock = channelBlock
        self.stimStart = 1
        self.stimStop = 20
        self.mechLocMap = None
        self.blocks = [NMDABlock, SomaBlock, DendBlock]
        self.loadTemplate()
        self.fih = neuron.h.FInitializeHandler(1, self.loadState)
        self.trees = []
        self.trees.append(self.cell.apic)
        self.trees.append(self.cell.dend)
        self.trees.append(self.cell.soma)
        self.trees.append(self.cell.axon)
        self.setValidLocations()
        self.setLocationDie()  # #Find all valid location treeples on the dendrite
        self.soma = self.cell.soma
        self.netStims = {location:[] for location in self.validLocations}
        self.netCons = {location:[] for location in self.validLocations}
        self.synapses = {location:[] for location in self.validLocations}
        self.SEClamps = {location:[] for location in self.validLocations}
        self.inputMap = {}
        self.DistMap, self.TRMap = self.loadPkl('DistsAndTRs/DistsAndTRs.pkl')
        if self.SomaBlock:
                self.blockSomaChannels()
        if self.DendBlock:
                self.blockDendriticChannels()
        if self.channelBlock:
            self.blockChannel()
        
    def loadTemplate(self):
        """Loads the cell from the hoc file"""
        h.load_file("nrngui.hoc")
        h("objref cvode")        
        h("cvode = new CVode()")
        h("cvode.active(0)")
        h.load_file("import3d.hoc")     
        morphology_file = "morphologies/cell1.asc"        
        h.load_file("models/L5PCbiophys3.hoc")
        h.load_file("models/L5PCtemplate.hoc")
        self.cell = h.L5PCtemplate(morphology_file)
        #h("cvode.active(0)")
        
    def loadPkl(self, fn):
        with open(fn, 'r') as f:
            return pickle.load(f)
        
    def setValidLocations(self):
        self.allLocations = []
        for i in range(4):
            treeNum = i
            tree = self.trees[treeNum]
            for j in range(len(tree)):
                sec = tree[j]
                for k in range(sec.nseg):
                    self.allLocations.append((treeNum,j,k))
        if self.dispersion == 'aTuft':
            self.validLocations = [loc for loc in self.allLocations if loc[0] == 0 and loc[1] > 36 and loc[1] < 79 ]
        elif self.dispersion == 'notATuft':
            self.validLocations = [loc for loc in self.allLocations if not (loc[0] == 0 and loc[1] > 36 and loc[1] < 79)]
        elif self.dispersion == 'soma':
            self.validLocations = [(2,0,0)]
        elif self.dispersion == 'full':    
            self.validLocations = [loc for loc in self.allLocations if loc [0] == 0 or loc[0] == 1]
        elif self.dispersion == 'basal':    
            self.validLocations = [loc for loc in self.allLocations if loc[0] == 1]     
               
    def setLocationDie(self):
        denominator = 0
        self.dispersionSegs = [self.loc2Seg(loc) for loc in self.validLocations]
        die = np.array([seg.sec.L/seg.sec.nseg for seg in self.dispersionSegs])
        self.uniformDistDie = die/sum(die)
        self.dieMap = {self.validLocations[i]:self.uniformDistDie[i] for i in range(len(self.validLocations))}
        #print self.dieMap
                

    def addVoltageClamp(self, weightMap, dur):
        # WeightMap is (treeNum, Sec, Seg):weight
        # treeNum specifies apical or basal, 0 is apical, 1 is basal
        # inds is an array of tuples of (sectionNum, segNum)

        # Window for stimulus inputs
        tstart = self.stimStart
        tstop = self.stimStop
        for tuple in weightMap:
            print tuple
            loc = tuple[0]
            synapseWeight = tuple[1]
            treeNum = loc[0]
            tree = self.trees[treeNum]
            section = tree[loc[1]]
            segLoc = self.getSegLoc(loc)            
            clamp = h.SEClamp(segLoc, sec=section)
            clamp.dur1=dur
            clamp.amp1=synapseWeight
            clamp.rs=0.00000001
            self.SEClamps[loc] = clamp

                                    
    def addExciteStim(self, weightMap, numStims = 1, jitter = False):
        # WeightMap is (treeNum, Sec, Seg):weight
        # treeNum specifies apical or basal, 0 is apical, 1 is basal
        # inds is an array of tuples of (sectionNum, segNum)

        # Window for stimulus inputs
        tstart = self.stimStart
        tstop = self.stimStop
        for tuple in weightMap:
            loc = tuple[0]
            synapseWeight = tuple[1]
            treeNum = loc[0]
            tree = self.trees[treeNum]
            section = tree[loc[1]]
            segLoc = self.getSegLoc(loc)   
            syn = h.ProbAMPANMDA_EMS(segLoc, sec=section)
            if self.Current:
                syn = h.ProbAMPANMDA_EMS_Current(segLoc, sec=section)
                
            #syn.NMDA_ratio = 1.6
            #syn.gateConstant = 0.08
            syn.Dep = 0
            syn.Fac = 0
            
            
            syn.NMDA_ratio = 1.6
            if self.NMDABlock or self.Current:
                syn.NMDA_ratio = 0
                syn.g_NMDA = 0            

            self.synapses[loc].append(syn)
            if jitter == True:
                stimTimes = np.random.randint(tstart, tstop, size=numStims)  # 2 in 50 ms = 40Hz
            else: 
                stimTimes = range(tstart, tstop, (tstop - tstart) / numStims)
            for time in stimTimes:
                stim = h.NetStim()                   
                stim.number = 1
                stim.start = time
                ncstim = h.NetCon(stim, syn)
                self.netStims[loc].append(stim)
                #ncstim.delay = 0
                ncstim.weight[0] = synapseWeight
                self.netCons[loc].append(ncstim)
                        
    def addGABA(self, weightMap, numStims, jitter = False):
         tstart = self.stimStart
         tstop = self.stimStop
         for tuple in weightMap:
                    loc = tuple[0]
                    synapseWeight = tuple[1]
                    treeNum = loc[0]
                    tree = self.trees[treeNum]
                    section = tree[loc[1]]
                    segLoc = self.getSegLoc(loc)   
                    # syn = h.ExpSyn(self.getSegLoc(i),sec=self.dend)
                    syn = h.ProbGABAAB_EMS(segLoc, sec=section)
                    if self.Current:
                        syn = h.ProbGABAAB_EMS_Current(segLoc, sec=section)
                    #syn.GABAB_ratio = 0.33
                    self.synapses[loc].append(syn)
                    if jitter == True:
                        stimTimes = np.random.randint(tstart, tstop, size=numStims)  # 2 in 50 ms = 40Hz
                    else: 
                        stimTimes = range(tstart, tstop, (tstop - tstart) / numStims)    
                    for time in stimTimes:
                        stim = h.NetStim()                   
                        stim.number = 1
                        stim.start = time
                        ncstim = h.NetCon(stim, syn)
                        self.netStims[loc].append(stim)
                        #ncstim.delay = 0.25
                        ncstim.weight[0] = synapseWeight
                        self.netCons[loc].append(ncstim)

                
    def clearStimuli(self):
        self.netStims = {location:[] for location in self.validLocations}
        self.synapses = {location:[] for location in self.validLocations}
        self.netCons = {location:[] for location in self.validLocations}
        self.SEClamps = {location:[] for location in self.validLocations}
        
    def blockDendriticChannels(self):
        mech_to_delete = ['SKv3_1',  'SK_E2', 'Ca_LVAst','Ca_HVA','Im', 'Ih', 'K_Pst',
                          'K_Tst','Nap_Et2','NaTa_t', 'CaDynamics_E2']
        for sec in self.cell.apic:
            for mech in mech_to_delete:
                sec.uninsert(mech)
#             sec.gSK_E2bar_SK_E2 = 0.0 
#             sec.gSKv3_1bar_SKv3_1 = 0.0  
#             sec.gNaTa_tbar_NaTa_t = 0.0 
#             sec.gImbar_Im = 0.0 
        for sec in self.cell.basal:
            for mech in mech_to_delete:
                sec.uninsert(mech)
#             sec.gIhbar_Ih = 0.0
    def blockChannel(self):
        mech_to_delete = self.channelBlock
        print mech_to_delete
        for sec in self.cell.apic:
            for mech in mech_to_delete:
                sec.uninsert(mech)
#             sec.gSK_E2bar_SK_E2 = 0.0 
#             sec.gSKv3_1bar_SKv3_1 = 0.0  
#             sec.gNaTa_tbar_NaTa_t = 0.0 
#             sec.gImbar_Im = 0.0 
        for sec in self.cell.basal:
            for mech in mech_to_delete:
                sec.uninsert(mech)

    def blockSomaChannels(self):
        mech_to_delete = ['SKv3_1',  'SK_E2', 'Ca_LVAst','Ca_HVA','Im', 'Ih', 'K_Pst',
                          'K_Tst','Nap_Et2','NaTa_t', 'CaDynamics_E2']
        for sec in self.cell.soma:
            for mech in mech_to_delete:
                sec.uninsert(mech)
#             sec.gCa_LVAstbar_Ca_LVAst = 0.0 
#             sec.gCa_HVAbar_Ca_HVA = 0.0 
#             sec.gSKv3_1bar_SKv3_1 = 0.0 
#             sec.gSK_E2bar_SK_E2 = 0.0 
#             sec.gK_Tstbar_K_Tst = 0.0 
#             sec.gK_Pstbar_K_Pst = 0.0 
#             sec.gNap_Et2bar_Nap_Et2 = 0.0 
#             sec.gNaTa_tbar_NaTa_t = 0.0
    
    def setCalcCurrentVector(self):
         CaCurrectVec = h.Vector()
         #print CaCurrectVec
         CaCurrectVec.record(self.trees[0][36](0.5)._ref_ica)
         #print CaCurrectVec
         return CaCurrectVec 
        
    def distToSegNum(self, loc): 
            treeNum = loc[0]
            tree = self.trees[treeNum]
            section = tree[loc[1]]
            seg = section(loc[2])
            segNum = list(section).index(seg)
            return((loc[0],loc[1], segNum))
                   
    def getSegLoc(self, key):
        tree = self.trees[key[0]]        
        section = tree[key[1]]
        nsegs = float(section.nseg)
        segNum = key[2]
        segLoc = (segNum / nsegs + 0.5 / nsegs)
        return segLoc
    
    def set_t_vec(self):
        t_vec = h.Vector()
        t_vec.record(h._ref_t)
        return t_vec
    
    def set_soma_rec(self, places):
        soma_v_vec = h.Vector()
        soma_v_vec.record(self.soma[0](0)._ref_v)
        return soma_v_vec
    
    def set_dend_rec(self, places):
        dend_rec_map = {}
        for place in places:
            tree = self.trees[place[0]]
            dend_rec_trace = h.Vector()
            dend_rec_trace.record(tree[place[1]](self.getSegLoc(place))._ref_v)
            dend_rec_map[place] = dend_rec_trace
        return dend_rec_map
        
    
    def set_cond_vectors(self, places):
        dend_g_map = {}
        dend_gAMPA_map = {}
        dend_gNMDA_map = {}
        for place in places:
            synapse = self.synapses[place][0]
            dend_g_trace = h.Vector()
            dend_gAMPA_trace = h.Vector()
            dend_gNMDA_trace = h.Vector()
            dend_g_trace.record(synapse._ref_g)
            dend_gAMPA_trace.record(synapse._ref_g_AMPA)
            dend_gNMDA_trace.record(synapse._ref_g_NMDA)
            dend_g_map[place] = dend_g_trace
            dend_gAMPA_map[place] = dend_gAMPA_trace
            dend_gNMDA_map[place] = dend_gNMDA_trace
        return dend_g_map, dend_gAMPA_map, dend_gNMDA_map
    
    def set_recording_vectors(self, places):
        soma_v_vec = h.Vector()
        dend_rec_map = {}
        # soma_i_vec = h.Vector()
        t_vec = h.Vector()
        soma_v_vec.record(self.soma[0](0)._ref_v)
        t_vec.record(h._ref_t)
        dend_rec_map = self.set_dend_rec(places)
        # soma_i_vec.record(cell.soma(0.5)._ref_ina)
        return soma_v_vec, dend_rec_map, t_vec
    
    def plotDendTraces(self, times, dendRecMap, locsToPlot, title=''):
        rcParams['figure.figsize'] = 15, 10
        fig = plt.figure('DendTraces')
        numSubplots = len(locsToPlot)
        if title == '-':
            ls = ':'
        else:
            ls = 'solid'
        for i in range(len(locsToPlot)):
            loc = locsToPlot[i]
            plt.subplot(math.ceil(numSubplots ** 0.5), math.ceil(numSubplots ** 0.5), i + 1)
            plt.plot(times, dendRecMap[loc], ls=ls)
            plt.title(str(loc) + " " + str(round(self.getDist(loc))))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        # plt.show()
        plt.savefig('Figures/DendTraces.jpg')
        return fig
        
    def colorPlot(self, map, LOG_SCALE = 0, filename = 'colorPlot'):
        for sec in self.cell.all:
            sec.insert("var")
        h.cvode_active(0)

        mbottom = 0.01
        mapVals = [map[x] for x in map.keys()]    
        mx = np.amax(mapVals)
        #mx = 70
        mn = np.amin(mapVals)
        if LOG_SCALE == 0:
            MIN_Z_in = mn
            MAX_Z_in = mx
        else: 
            MIN_Z_in = math.log(mn+mbottom)
            MAX_Z_in = math.log(mx+mbottom)

        for loc in map.keys():
                treeNum = loc[0]
                tree = self.trees[treeNum]
                section = tree[loc[1]]
                seg = section(self.getSegLoc(loc))
                if LOG_SCALE:
                    print map[loc]
                    val = map[loc]
                    val = max(val, mbottom)
                    seg.zin_var = math.log(val)
                else:
                    seg.zin_var = np.round(map[loc],1)
        mleft = -600
        mright = 1000
        mbottom = -500
        mtop = 2000
        windowWidth = 400
        windowHeight = 400
        mheight = 2000
        h('load_file("TColorMap.hoc")')     
        h.newshapeplot()
        ps_i=h.fast_flush_list.object(h.fast_flush_list.count()-1)

        ps_i.variable('zin_var')
        ps_i.exec_menu('Shape Plot')
        ps_i.exec_menu("View Box")
        ps_i.scale(0,1)
        cm1 = h.TColorMap("cm/jet.cm")
        cm1.set_color_map(ps_i,MIN_Z_in,MAX_Z_in)
        print cm1
        #ps_i.exec_menu("Variable Scale")
        ps_i.size(mleft,mright,mbottom,mtop)
        ps_i.view(mleft, mbottom, mright-mleft, mheight, 0, 0, windowWidth, windowHeight)


        
        #ps_i.plot_colors(cm1)

        
#         ps_i.exec_menu("View = plot")
#         ps_i.exec_menu("Shape Plot")
#         ps_i.exec_menu("View Box")
#         ps_i.variable("zin_var")
#         cm1 = h.TColorMap("cm/jet.cm")
#         cm1.set_color_map(ps_i,MIN_Z_in,MAX_Z_in)
#         ps_i.exec_menu("10% mright out")
#         ps_i.exec_menu("10% mright out")
#         h.fast_flush_list.append(ps_i)
#         ps_i.exec_menu("Variable Scale")
#         ps_i.plot_colors(cm1)
        epsfn = os.path.abspath("Figures2_12_18/Figure3/neuron" + ".ps")
        pdffn = os.path.abspath("Figures2_12_18/Figure3/neuron" + ".pdf")
        ps_i.printfile(epsfn)
#         callStr = "convert -density 100 " + epsfn + " -quality 100 " + pdffn
#         subprocess.check_call(callStr, shell=True)
        return epsfn, ps_i
         
    def getDist(self, key, key2 = (2,0,0.5)):
        tree = self.trees[key[0]]        
        section = tree[key[1]]
        segLoc = self.getSegLoc(key)
        h.distance(sec = self.loc2Seg(key2).sec)
        return h.distance(segLoc, sec=section)
        
    def ParallelSet(self,optt=0, nthreads = 8):
        '''#Set Neuron to run on threads and multisplit
        # optt =0 - start 1- off 2 - on
        '''
        if optt==0:
            h.load_file("parcom.hoc", "ParallelComputeTool")
            self.obP = h.ParallelComputeTool[0]
            self.obP.change_nthread(nthreads, 1)
            self.obP.cacheeffic(1)
            self.obP.multisplit(1)
            PSval=0
        if optt==1:
            self.obP.cacheeffic(0)
            self.obP.multisplit(0)
        if optt==2:
            self.obP.cacheeffic(1)
            self.obP.multisplit(1)

            
    def distCompare(self, loc1, loc2):
        dist1 = self.getDist(loc1)
        dist2 = self.getDist(loc2)
        return(int(np.sign(dist2 - dist1)))  
    
    def saveState(self):
        mechLocMap = {}
        for loc in self.allLocations:
            print loc
            mechLocMap[loc] = {}
            mechMap = mechLocMap[loc]
            seg = self.loc2Seg(loc)
            mechMap['v'] = seg.v
            print dir(seg)
            print [mech for mech in seg.sec]
            print seg.v
            for mech in seg:
                if '__' not in mech.name():
                    for var in dir(mech):
                        if '__' not in var and 'name' not in var and 'next' not in var:
                            try:
                                exec('tm = seg.' + var)
                                print '1', mech, var
                                mechMap[var] = (None, tm)
                            except:
                                exec('tm = seg.' + str(mech) + '.'+ var)
                                print '2', mech, var
                                mechMap[var] = (str(mech), tm)
        if self.channelBlock == None:
            fn = 'mechInits6/mechInits' + str(self.blocks) + '.pkl'
        else:
            fn = 'mechInits6/mechInits' + str(self.blocks) + str(self.channelBlock) + '.pkl'                        
        with open(fn, 'w+') as f:
            pickle.dump(mechLocMap, f)
            
            
    def loc2Seg(self, loc):
            treeNum = loc[0]
            tree = self.trees[treeNum]
            section = tree[loc[1]]
            seg = section(self.getSegLoc(loc))
            return seg
            
                
    def loadState(self):
        #print 'mechInits' + str(self.blocks) + '.pkl'
        if self.channelBlock == None:
            fn = 'mechInits6/mechInits' + str(self.blocks) + '.pkl'
        else:
            fn = 'mechInits6/mechInits' + str(self.blocks) + str(self.channelBlock) + '.pkl'
        if os.path.isfile(fn):
            mechLocMap = self.loadPkl(fn)
            arr = [mechLocMap[loc]['v'] for loc in self.allLocations if loc[0] == 2]
            #print arr
            for loc in mechLocMap.keys():
                seg = self.loc2Seg(loc)
                mechMap = mechLocMap[loc]
                #print 'before', seg.v
                seg.sec.v = mechMap['v']
                seg.v = mechMap['v']
                #print 'after', seg.v
                for mech in seg:
                    if '__' not in mech.name():
                        for var in dir(mech):
                            if '__' not in var and 'name' not in var and 'next' not in var:
                                tup = mechMap[var]
                                if tup[0] == None:
                                    exec('seg.' + str(var) + ' = ' + str(tup[1]))
                                else:
                                    exec('seg.' + str(tup[0]) + '.' + str(var) + ' = ' + str(tup[1]))
#         if self.DendBlock:
#             self.blockDendriticChannels()

                
    def simulateRelax(self, tstop = 1600):
        print('relax')
        h.tstop = tstop
        soma_v_vec, dend_rec_map, t_vec = self.set_recording_vectors(self.allLocations)
        #self.ParallelSet(2)
        h.run()
        #self.ParallelSet(1)  
        self.saveState()
        plt.figure('Train'+str(self.blocks))
        plt.plot(t_vec, soma_v_vec)
        plt.show()
        
        
    def simulateTest(self, tstop = 100):
        print 'hello'
        h.tstop = tstop
        soma_v_vec, dend_rec_map, t_vec = self.set_recording_vectors(self.allLocations)
        self.fih = neuron.h.FInitializeHandler(1, self.loadState)
        #self.v_init = -78
        #self.ParallelSet(2)
        h.run()
        #self.ParallelSet(1)
        plt.figure('Test' + str(self.blocks))
        plt.plot(t_vec, soma_v_vec)
        plt.ylim(-95,-60)
        plt.show()         
    

        
# import socket
# print socket.gethostname()
# if socket.gethostname() == 'sushi':        
#      allblocks = [[0,0,0],[0,0,1],[0,1,0],[1,0,0], [1,1,1], [0,1,1], [1,0,1], [1,1,0]]
#      for blocks in allblocks:
#          cell = L5PC6(NMDABlock = blocks[0], SomaBlock = blocks[1], DendBlock = blocks[2])
#          # print cell.getDist((0,10,0), key2 = (0,50,0))
#          # print cell.getDist((0,10,0))
#          cell.simulateRelax()
#          cell.simulateTest()
# #          
# import socket
# print socket.gethostname()
# if socket.gethostname() == 'sushi':        
#      cell = L5PC6(NMDABlock = 0, SomaBlock = 1, DendBlock = 0, channelBlock = ['Ih'])
#      # print cell.getDist((0,10,0), key2 = (0,50,0))
#      # print cell.getDist((0,10,0))
#      cell.simulateRelax()
#      cell.simulateTest()

# if __name__ == "__main__":
#     cell = L5PC4(NMDABlock = 0, SomaBlock = 0, DendBlock = 0, dispersion = 'full' )
#     pickleName = pickleName = 'maxMap/maxMap1e+99.pkl'
#     with open(pickleName, 'r') as f:
#             maxMap = pickle.load(f)  
#     epsfn, ps_i = cell.colorPlot(maxMap)

#cell = L5PC6(NMDABlock = 0, SomaBlock = 0, DendBlock = 0, dispersion = 'aTuft' )
# if __name__ == "__main__":
#      cell = L5PC6(NMDABlock = 0, SomaBlock = 0, DendBlock = 0, dispersion = 'aTuft' )
#      dieMap = cell.dieMap
#      dieMap = {loc:5 for loc in dieMap.keys()}
#      epsfn, ps_i = cell.colorPlot(dieMap)

cell = L5PC6(NMDABlock = 0, SomaBlock = 0, DendBlock = 0, dispersion = 'full' )
print cell.validLocations
