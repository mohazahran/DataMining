'''
Created on Apr 17, 2017

@author: mohame11
'''
from ClusteringAlgorithm import ClusteringAlgorithm
import numpy as np
import math

class Cluster():
    def __init__(self):
        self.id = None
        self.members = []
        self.headVector = []

class Kmeans(ClusteringAlgorithm):

    def __init__(self, distanceMeasure, k, maxIter):
        ClusteringAlgorithm.__init__(self, distanceMeasure)
        
        self.k = k
        self.maxIter = maxIter
        self.distanceMeasure = distanceMeasure
        self.clusters = []
        
    
    def pickCentroid(self, sampleIdx, dataSet):
        bestDist = np.inf
        bestCluster = None
        for c in self.clusters:
            dist = self.getDistance(dataSet[sampleIdx].featureVector, c.headVector)
            if(dist < bestDist):
                bestDist = dist
                bestCluster = c
        bestCluster.members.append(sampleIdx)
        #self.updateClusterMean(bestCluster, dataSet)
        bestCluster.headVector = bestCluster.headVector + (dataSet[sampleIdx].featureVector - bestCluster.headVector)/float(len(bestCluster.members))
    
    
    
    def updateClusterMean(self, c, dataSet):
        vec = np.zeros(len(dataSet[0].featureVector))
        for m in c.members:
            vec += dataSet[m].featureVector
        vec = vec / float(len(c.members))
        #print vec
        c.headVector = vec
                
    
    def clearMembers(self):
        for c in self.clusters:
            c.members = []
            
    
    def doClustering(self, dataSet):
        centroidsIdx = np.random.choice(len(dataSet), self.k, replace=False)
        centroidsIdx = {c for c in centroidsIdx}
        for i,c in enumerate(centroidsIdx):
            cluster = Cluster()
            cluster.id = i
            cluster.headVector = dataSet[c].featureVector
            self.clusters.append(cluster)
            
        for iter in range(self.maxIter):
            #print iter,
            self.clearMembers()
            for j in range(len(dataSet)): 
                self.pickCentroid(j, dataSet)
            #for c in self.clusters:
            #    self.updateClusterMean(c, dataSet)
               
    
    
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
            dist = 0.0
            for c in self.clusters:
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
                    
                    
                    
        
        
        