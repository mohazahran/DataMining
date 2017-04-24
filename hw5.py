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
import numpy as np
from PCA import *


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
    
    elif(option == 'c1'):
        filtered = []
        groupByLabel = {}
        for i,inst in enumerate(dataSet):
            if(inst.classLabel in groupByLabel):
                groupByLabel[inst.classLabel].append(i)
            else:
                groupByLabel[inst.classLabel] = [i]
        
        for label in groupByLabel:
            selectedIdx = np.random.choice(groupByLabel[label], 10, replace=False)
            for idx in selectedIdx:
                filtered.append(dataSet[idx])
        return filtered
    
    elif(option == 'a1'):
        filtered = []
        groupByLabel = {}
        for i,inst in enumerate(dataSet):
            if(inst.classLabel in groupByLabel):
                groupByLabel[inst.classLabel].append(i)
            else:
                groupByLabel[inst.classLabel] = [i]
        
        for label in groupByLabel:
            selectedIdx = np.random.choice(groupByLabel[label], 1, replace=False)
            for idx in selectedIdx:
                filtered.append(dataSet[idx])
        return filtered
            
            
            
                
        


def QA():
    '''
    print 'QA1'
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    distanceMeasure = 'euclidean'
    algo = ClusteringAlgorithm(distanceMeasure)
    parsedDataSet = algo.parseData(dataPath)
    
    dataSet = selectFromData(parsedDataSet, 'a1')
    for inst in dataSet:
        v = np.array(inst.featureVector, 'd')
        im = v.reshape(28,28)
        plt.imshow(im, cmap='gray')
        plt.savefig('QA1_digit'+str(inst.classLabel)+'.pdf', bbox_inches='tight')
        
    '''
    
    
    print 'QA2'
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    distanceMeasure = 'euclidean'
    algo = ClusteringAlgorithm(distanceMeasure)
    parsedDataSet = algo.parseData(dataPath)
    dataSet = selectFromData(parsedDataSet, 'c1')
    visualizeSamples(dataSet, 'QA2_1000RandomSamples_tSNE')
    


def QB4():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    option = 'iii'
    k = 2
    metric = 'Mutual_Information_Gain'
    distanceMeasure = 'euclidean'
    maxIter = 50
    
    
    c = ClusteringAlgorithm(distanceMeasure)
    parsedDataSet = c.parseData(dataPath)
    #parsedDataSet = parsedDataSet[:1000]
    
    print option,k,metric
    
    dataSet = selectFromData(parsedDataSet, option)
    dataMatrix = c.getDataMatrix(dataSet)
    pairWiseDistances = squareform(pdist(dataMatrix, metric=distanceMeasure))
   
    kmeans = Kmeans(distanceMeasure, k, maxIter)
    kmeans.doClustering(dataSet)
    
    metricVal = kmeans.evaluate(metric, dataSet, pairWiseDistances)
    print metricVal
    
    title = 'kmeans_k'+str(k)+'_dataVersion_'+option
    
    visualizeClusters(kmeans.clusters, title, dataSet)
    visualizeClustersClasses(kmeans.clusters, title+'_classes', dataSet)
    
    

def QB():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    dataOptions = ['i', 'ii', 'iii']
    K = [2, 4, 8, 16, 32]
    metricTypes = ['Squared_Distances', 'Silhouette_Coeff']
    distanceMeasure = 'euclidean'
    maxIter = 5
    modelType = 'kmeans'
    repeatExp = 10
    
    files = {}
    for d in dataOptions:
        f = open('dataSet_v'+d,'w')
        f.write('k,Squared_Distances_mean,Squared_Distances_std,Silhouette_Coeff_mean,Silhouette_Coeff_std'+'\n')
        files[d] = f
    
    
            
    c = ClusteringAlgorithm(distanceMeasure)
    parsedDataSet = c.parseData(dataPath)
    parsedDataSet = parsedDataSet[:1000]
    
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
    


def drawDenrodgrams():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    distances = ['single', 'complete', 'average']
    hc = HierarchicalClustering(None)
    parsedDataSet = hc.parseData(dataPath)
    dataSet = selectFromData(parsedDataSet, 'c1')
    #draw dendrograms
    for dist in distances:
        hc = HierarchicalClustering(dist)
        Z = hc.doClustering(dataSet)
        hc.drawDendrogram(Z)
        
    

def QC():
    #drawDenrodgrams()
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    #distances = ['single', 'complete', 'average']
    distanceMeasure = 'single'
    metricTypes = ['Squared_Distances', 'Silhouette_Coeff', 'Mutual_Information_Gain']
    hc = HierarchicalClustering(distanceMeasure)
    parsedDataSet = hc.parseData(dataPath)
    
    dataSet = selectFromData(parsedDataSet, 'i')
    #dataSet = parsedDataSet[200:250]
    
    dataMatrix = hc.getDataMatrix(dataSet)
    pairWiseDistances = squareform(pdist(dataMatrix, metric='euclidean'))
    Z = hc.doClustering(dataSet)
    #hc.drawDendrogram(Z)
    
    print 'k,'+','.join(metricTypes)
    for k in [2,4,8,16,32]:
    #for k in [32]:
        print k,',',
        clusters = hc.getClusters(Z, k, dataSet)
        hc.clusters = clusters
        print len(hc.clusters)
        #for c in hc.clusters:
        #    print c.members
        for metric in metricTypes:
            res = hc.evaluate(metric, dataSet, pairWiseDistances)
            print res,',',
        print
            
            

def bonus():
    #dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-embedding.csv'
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    distanceMeasure = 'euclidean'
    metricTypes = ['Squared_Distances', 'Silhouette_Coeff']
    k = 10
    PCA_dim = 10
    maxIter = 50
    l,w = 28,28
    sampleCount = 1000
    
    kmeans = Kmeans(distanceMeasure, k, maxIter)
    dataSet = kmeans.parseData(dataPath)
    
    #samples = dataSet[0:1000]
    #visualizeSamples(samples, 'sampleData_2D_t_SNE')
    
    matrix = kmeans.getDataMatrix(dataSet)
    
    reducedMatrix, evecs, evals = PCA(matrix, PCA_dim)
    
    plotEvecs(evecs, PCA_dim, l, w)
    
    #rows,cols = reducedMatrix.shape
    #for r in range(rows):
    #    dataSet[r].featureVector = reducedMatrix[r]
        
    #samples = np.random.choice(dataSet, sampleCount, replace=False)
    #samples = dataSet[0:1000]
    #visualizeSamples(samples,'sampleData_2D_PCA')
    
    #kmeans.doClustering(dataSet)
    

def bonus4():
    dataPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw5/digits-raw.csv'
    PCA_dim = 10
    dataOptions = ['i', 'ii', 'iii']
    K = [2, 4, 8, 16, 32]
    metricTypes = ['Squared_Distances', 'Silhouette_Coeff']
    distanceMeasure = 'euclidean'
    maxIter = 50
    repeatExp = 10
    
    c = ClusteringAlgorithm()
    parsedDataSet = c.parseData(dataPath)

    matrix = c.getDataMatrix(parsedDataSet)
    
    reducedMatrix, evecs, evals = PCA(matrix, PCA_dim)
    
    rows,cols = reducedMatrix.shape
    for r in range(rows):
        parsedDataSet[r].featureVector = reducedMatrix[r]
    
    
    
    files = {}
    for d in dataOptions:
        f = open('bonus_dataSet_v'+d,'w')
        f.write('k,Squared_Distances_mean,Squared_Distances_std,Silhouette_Coeff_mean,Silhouette_Coeff_std'+'\n')
        files[d] = f
    
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
    #QA()
    #QB()
    #QB4()
    #QC()
    #bonus()
    bonus4()
    #main()