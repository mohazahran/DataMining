'''
Created on Apr 19, 2017

@author: mohame11
'''
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from ClusteringAlgorithm import ClusteringAlgorithm


class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, distanceMeasure):
        ClusteringAlgorithm.__init__(self, distanceMeasure)
        self.distanceMeasure = distanceMeasure #'complete', 'average',  'euclidean'
        
        
        
        
        
    def evaluate(self, metricType, dataSet):
        pass
    
        
    
    def doClustering(self, dataSet):
        dataMatrix = self.getDataMatrix(dataSet)
        Z = linkage(dataMatrix, metric=self.distanceMeasure)
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i,0] and Z[i,1] are combined to form cluster n+i.
        Z[i] will tell us which clusters were merged in the i-th iteration, let's take a look at the first two points that were merged:
        Z[i]: [idx1, idx2, dist, sample_count] 
        I.e. in the ith iter, the algo merged the cluster idx1 with idx2 whose distance is dist, to form a new cluster whose index is n+i that has sample_count instances in them
        A cluster with an index less than n corresponds to one of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. 
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
        '''
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
                    Z,
                    leaf_rotation=90.,  # rotates the x axis labels
                    leaf_font_size=8.,  # font size for the x axis labels
                    )
        plt.show()
        