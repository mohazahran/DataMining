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
               
    
    
    
    
                    
        
        
        