import numpy as np


class Instance():
    def __init__(self):
        self.classLabel = None
        self.id = -1
        self.featureVector = None
       
        

class ClusteringAlgorithm(object):
    def __init__(self, distanceMeasure):
        self.distanceMeasure = distanceMeasure
    
    def parseData(self, path):
        dataSet = []
        r = open(path, 'r')
        for line in r:
            parts = line.strip().split(',')
            inst = Instance()
            inst.id = parts[0]
            inst.classLabel = parts[1]
            inst.featureVector = np.array( [float(i) for i in parts[2:]] )
            dataSet.append(inst)
        r.close()
        return dataSet
    
    
    def getDistance(self, v1, v2):
        if(self.distanceMeasure == 'euclidean'):
            dist = np.linalg.norm(v1-v2)
        return dist
    
    def getPairWiseDistances(self, dataSet):
        pairWiseDistances = {}
        for i1, inst1 in enumerate(dataSet):
            #print i1,
            for i2, inst2 in enumerate(dataSet):
                if(i1 == i2):
                    continue
                key = str(i1)+','+str(i2)
                revkey = str(i2)+','+str(i1)
                if(key not in pairWiseDistances and revkey not in pairWiseDistances):
                    dist = self.getDistance(inst1.featureVector, inst2.featureVector)
                    pairWiseDistances[key] = dist
        
        return pairWiseDistances
    
    
    def getDataMatrix(self, dataSet):
        matrix = np.array([dataSet[0].featureVector])
        for i in range(1, len(dataSet)):
            matrix = np.concatenate((matrix, [np.array(dataSet[i].featureVector)]), axis = 0)
        
        return matrix

    
    def evaluate(self, metricType, dataSet):
        pass
    
    def doClustering(self, dataSet):
        pass
        
            
        
            