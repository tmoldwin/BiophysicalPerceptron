

class iterationData:
    def __init__(self, index, epoch, dendTrace, somaTrace, times, nSpikes, weights): 
        self.index = index
        self.nSpikes = nSpikes
        self.epoch = epoch
        if not somaTrace is None:
            self.somaTrace = somaTrace.tolist()
        self.dendTrace = dendTrace
        self.weights = weights.tolist()
        if not times is None:
            self.timeFinal = times[-1]
        
    def toJSON(self):
        return json.dump(self, fn, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)