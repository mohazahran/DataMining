'''
Created on Apr 17, 2017

@author: mohame11
'''
from Kmeans import Kmeans
from HierarchicalClustering import HierarchicalClustering
from ClusteringAlgorithm import ClusteringAlgorithm
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import math


def main():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    distanceMeasure = 'euclidean'
    metricType = 'Silhouette_Coeff'
    k = 10
    maxIter = 5
    modelType = 'hc'
    
    if(modelType == 'kmeans'):
    
        kmeans = Kmeans(distanceMeasure, k, maxIter)
        dataSet = kmeans.parseData(dataPath)
        dataSet = dataSet[0:1000]
        kmeans.doClustering(dataSet)
        metricVal = kmeans.evaluate(metricType, dataSet)
        print metricType,metricVal
    
    else:
        hc = HierarchicalClustering(distanceMeasure)
        dataSet = hc.parseData(dataPath)
        dataSet = dataSet[0:1000]
        hc.doClustering(dataSet)
    


def selectFromData(dataSet, option):
    if(option == 'i'):
        return dataSet
    
    elif(option == 'ii'):
        filtered = []
        for inst in dataSet:
            if(inst.classLabel in ['2','4','6','7']):
                filtered.append(inst)
        return filtered
    
    elif(option == 'iii'):    
        filtered = []
        for inst in dataSet:
            if(inst.classLabel in ['6','7']):
                filtered.append(inst)
        return filtered

def QB():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    dataOptions = ['i', 'ii', 'iii']
    K = [2, 4, 8, 16, 32]
    metricTypes = ['Squared_Distances', 'Silhouette_Coeff']
    distanceMeasure = 'euclidean'
    maxIter = 50
    modelType = 'kmeans'
    repeatExp = 10
    
    files = {}
    for d in dataOptions:
        f = open('dataSet_v'+d,'w')
        f.write('k,Squared_Distances_mean,Squared_Distances_std,Silhouette_Coeff_mean,Silhouette_Coeff_std'+'\n')
        files[d] = f
    
    
            
    c = ClusteringAlgorithm(distanceMeasure)
    parsedDataSet = c.parseData(dataPath)
    #parsedDataSet = parsedDataSet[:1000]
    
    for option in dataOptions:
        print option
        dataSet = selectFromData(parsedDataSet, option)
        dataMatrix = c.getDataMatrix(dataSet)
        pairWiseDistances = squareform(pdist(dataMatrix, metric=distanceMeasure))
        
        for k in K:
            print k
            resDic = {}
            files[option].write(str(k)+',')
                
            for m in metricTypes:
                resDic[m] = []
                
            for i in range(repeatExp):
                kmeans = Kmeans(distanceMeasure, k, maxIter)
                kmeans.doClustering(dataSet)
                for metric in metricTypes:
                    metricVal = kmeans.evaluate(metric, dataSet, pairWiseDistances)
                    resDic[metric].append(metricVal)
            
            for m in metricTypes:
                avg = float(sum(resDic[m]))/float(len(resDic[m]))
                stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[m]]) / float(len(resDic[m])) )
                files[option].write(str(avg)+',')
                files[option].write(str(stdd)+',')
            files[option].write('\n')
            files[option].flush()
    
    
    



if __name__ == "__main__":
    QB()
    #main()