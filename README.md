# BiophysicalPerceptron

Instructions

1. Compile the hoc files in neuron by navigating to the top level folder in the terminal and writing "nrnivmodl mod". This will create the neuron model (only needs to be done once).
2. For the biophysical perceptron, go to the PerceptronPosNegAll6 file. At the bottom of the file there are two sections, one for the generalization task and one for the classification task. Change the parameters in these sections to perform different experiments.
3. For the M&P perceptron see MCPostNegAll. Because it's faster we calculate the capacity for all MESW conditions and number of patterns/synapses/bit flips at the same time. For the classification task, if we have order = [(1000,200), (2000,400)] that means we will test for cases of N = 1000 synapses, P = 200 patterns as well as N = 2000 synapses, P = 400 patterns. For the generalization condition, the second number is the number of bit flips, so the above order would mean
N = 1000 synapses, noise = 200 bit flips as well as N = 2000 synapses, noise = 400 bit flips. Other parameters (such as the active ratio) can be configured in the CapacityCalc function
