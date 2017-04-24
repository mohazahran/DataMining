import numpy as np
import math

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
        rows = len(dataSet)
        cols = len(dataSet[0].featureVector)
        dim = (rows,cols)
        matrix = np.zeros(dim)
        #matrix = np.array([dataSet[0].featureVector])
        for i in range(0, len(dataSet)):
            matrix[i] = np.array(dataSet[i].featureVector)
            #matrix = np.concatenate((matrix, [np.array(dataSet[i].featureVector)]), axis = 0)
        
        return matrix
    
    
    def calAvgDistances(self, i, cluster, pairWiseDistances):
        dist = 0
        #flag = 0
        for m in cluster.members:
            if(i == m):
                #flag = 1
                continue
            d = pairWiseDistances[i][m]
            dist += d
        #dist = dist / float(len(cluster.members)-flag)
        dist = dist / float(len(cluster.members))
        return dist
         
    def evaluate(self, metricType, dataSet, pairWiseDistances):
        if metricType == 'Squared_Distances':
            '''
            dist = 0.0
            for c in self.clusters:
                for m1 in c.members:
                    for m2 in c.members:
                        dist += pairWiseDistances[m1][m2]                  
            return dist
            '''
            dist = 0.0
            for c in self.clusters:
                if(c.headVector == None or len(c.headVector) == 0):
                    self.getClusterMean(c, dataSet)
                for m in c.members:
                    dist += self.getDistance(dataSet[m].featureVector, c.headVector)                    
            return dist
    
        
        elif metricType == 'Silhouette_Coeff': #The silhouette ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters
            s = 0
            for c in self.clusters:
                for m1 in c.members:
                    a = self.calAvgDistances(m1, c, pairWiseDistances)
                    minDist = np.inf
                    minCluster = None
                    for othercluster in self.clusters:
                        if(c == othercluster):
                            continue
                        b = self.calAvgDistances(m1, othercluster, pairWiseDistances)
                        if(b < minDist):
                            minDist = b
                            minCluster = othercluster
                    
                    s += (minDist - a) / max(minDist, a)
                    
            s = s / float(len(dataSet))
            return s
                
        elif metricType == 'Mutual_Information_Gain': #should be between 0,1
            classCounts = {}
            clusterCounts = {}
            clusterClassCounts = {}
            for g in self.clusters:
                clusterCounts[str(g.id)] = len(g.members)
                for e in g.members:
                    if(dataSet[e].classLabel in classCounts):
                        classCounts[dataSet[e].classLabel] += 1
                    else:
                        classCounts[dataSet[e].classLabel] = 1
                    key = str(g.id)+','+str(dataSet[e].classLabel)
                    if(key in clusterClassCounts):
                        clusterClassCounts[key] += 1
                    else:
                        clusterClassCounts[key] = 1
            classEntropy = 0
            clusterEntropy = 0
            mutualInfo = 0
            for c in classCounts:
                probC = float(classCounts[c])/float(len(dataSet))
                classEntropy -= probC * math.log(probC,2)
                
            for g in clusterCounts:
                probG = float(clusterCounts[g])/float(len(dataSet))
                clusterEntropy -= probG * math.log(probG,2)
                
            for gc in clusterClassCounts:
                probGC = float(clusterClassCounts[gc])/float(len(dataSet))
                probG = float(clusterCounts[gc.split(',')[0]])/float(len(dataSet))
                probC = float(classCounts[gc.split(',')[1]])/float(len(dataSet))
                mutualInfo += probGC * math.log((probGC/probC/probG),2)
            
            mutualInfo = mutualInfo / (classEntropy + clusterEntropy)
            
            return mutualInfo
                    
    
    def getClusterMean(self, c, dataSet):
        vec = np.zeros(len(dataSet[0].featureVector))
        for m in c.members:
            vec += dataSet[m].featureVector
        vec = vec / float(len(c.members))
        #print vec
        c.headVector = vec
                     
    
    def doClustering(self, dataSet):
        pass
        
            
        
            