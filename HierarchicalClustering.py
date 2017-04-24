'''
Created on Apr 19, 2017

@author: mohame11
'''
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from ClusteringAlgorithm import ClusteringAlgorithm
import Queue
import math
from Kmeans import Cluster


class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, distanceMeasure):
        ClusteringAlgorithm.__init__(self, distanceMeasure)
        self.distanceMeasure = distanceMeasure #'complete', 'average',  'euclidean'
        self.clusters = []
        
    def drawDendrogram(self, Z):
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Indices')
        plt.ylabel('distance')
        dendrogram(
                    Z,
                    leaf_rotation=90.,  # rotates the x axis labels
                    leaf_font_size=8.,  # font size for the x axis labels
                    )
        #plt.show()
        plt.savefig('dendrogram'+'_'+self.distanceMeasure+'.pdf', bbox_inches='tight')
        
    
    def doClustering(self, dataSet):
        dataMatrix = self.getDataMatrix(dataSet)
        Z = linkage(dataMatrix, method=self.distanceMeasure, metric='euclidean')
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i,0] and Z[i,1] are combined to form cluster n+i.
        Z[i] will tell us which clusters were merged in the i-th iteration, let's take a look at the first two points that were merged:
        Z[i]: [idx1, idx2, dist, sample_count] 
        I.e. in the ith iter, the algo merged the cluster idx1 with idx2 whose distance is dist, to form a new cluster whose index is n+i that has sample_count instances in them
        A cluster with an index less than n corresponds to one of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. 
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
        '''
        return Z
    
    def atDepth(self, Z, iterNum, currentDepth, requiredDepth, clusterList, n, k):
        if(currentDepth == requiredDepth):
            leftCid = int(Z[iterNum][0])
            rightCid = int(Z[iterNum][1])
            if(len(clusterList) < k):
                clusterList.append(leftCid)
            if(len(clusterList) < k):
                clusterList.append(rightCid)
            return
        
        leftCid = int(Z[iterNum][0])
        rightCid = int(Z[iterNum][1])
        
        if(len(clusterList) < k and leftCid < n):
            clusterList.append(leftCid)
            
        if(len(clusterList) < k and rightCid < n):
            clusterList.append(rightCid)
        
        if(leftCid >= n):
            leftIterNum = leftCid-n
            self.atDepth(Z, leftIterNum, currentDepth+1, requiredDepth, clusterList, n, k)
        
        if(rightCid >= n):
            rightIterNum = rightCid-n
            self.atDepth(Z, rightIterNum, currentDepth+1, requiredDepth, clusterList, n, k)
        
    
    def getAllPointsByClusterId(self, Z, iterNum, members, n):
        
        leftCid = int(Z[iterNum][0])
        rightCid = int(Z[iterNum][1])
        
        if(leftCid < n):
            members.append(leftCid)
        if(rightCid < n):
            members.append(rightCid)
        
        if(leftCid >= n):
            leftIterNum = leftCid-n
            self.getAllPointsByClusterId(Z, leftIterNum, members, n)
        
        if(rightCid >= n):
            rightIterNum = rightCid-n
            self.getAllPointsByClusterId(Z, rightIterNum, members, n)
        
        
    
    def getMoreClusters(self, Z, clusterList, dataSet, k):
        while len(clusterList) < k:
            clusterListCopy = list(clusterList)
            for cid in clusterListCopy:
                if(cid < len(dataSet)):
                    continue
                if len(clusterList) >= k:
                    break
                iterNum = cid - len(dataSet)
                clusterList.remove(cid)
                self.atDepth(Z, iterNum, 0, 1, clusterList, len(dataSet), k)
            
    
    def getClusters(self, Z, k, dataSet):
        if(k%2 !=0):
            return None
        clusterList = []
        requiredDepth = int(math.log(k,2))
        self.atDepth(Z, -1, 0, requiredDepth, clusterList, len(dataSet), k)
        self.getMoreClusters(Z, clusterList, dataSet, k)
        clusters = []
        for cid in clusterList:
            c = Cluster()
            if(cid < len(dataSet)):
                c.members.append(cid)
                clusters.append(c)
                continue
            members = []
            iterNum = cid - len(dataSet)
            self.getAllPointsByClusterId(Z, iterNum, members, len(dataSet))
            c.members = list(members)
            clusters.append(c)
        return clusters
                
            
            
            
            
            
            
        
        
        
        