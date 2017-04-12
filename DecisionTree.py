'''
Created on Apr 2, 2017

@author: mohame11
'''
from Classifier import *
import numpy as np
import math
import Queue

class node: 
    def __init__(self):
        self.children = [] # list of children
        self.featureIndex = -1 #feature index this node splits on
        self.label = -1 #the label this node will classify with in case it's a leaf node or reached a depth threshold.
        self.trainingData = [] # trainingData Subset
        self.featureIndexList = [] #list of possible feature to split on
        self.featureValue = -1 #the value of the previous split on feature (for debugging)
        self.parentFeatureIndex = -1 #the parent's split on feature (for debugging)
        self.depth = -1
    
    def addChild(self, child):
        self.children.append(child) 

class DecisionTree(Classifier):
    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.depthLimit = depthLimit
        self.samplesCountLimit = samplesCountLimit
        self.root = None
        #self.POS = 'POS'
        #self.NEG = 'NEG'
        if(self.isBinaryFeatures):
            self.allpossibleFeatureValues = [0,1]
        else:
            self.allpossibleFeatureValues = [0,1,2]
        
    def getFeaturesVector(self, parsedDataSet):
        for inst in parsedDataSet:
            words = inst.cleanedText
            inst.featureVector = np.zeros(len(self.filteredVocab))
            uniqueWords = set(words)
            for w in uniqueWords:
                if(w in self.filteredVocab):
                    idx = self.filteredVocab.index(w)
                    count = words.count(w)
                    if(count == 1):
                        inst.featureVector[idx] = 1
                    elif(count > 1):
                        if(self.isBinaryFeatures):
                            inst.featureVector[idx] = 1
                        else:
                            inst.featureVector[idx] = 2
                            
            
        
    def areSameLabel(self, trainingData): #checks if all the training examples have the same label
        '''
        label = -1
        for i in range(len(trainingData)):
            label = trainingData[i].classLabel
            if ((i+1) < len(trainingData)):            
                if(trainingData[i][-1] != trainingData[i+1][-1]):
                    return False, -1
        return True, label
        '''
        label = trainingData[0].classLabel
        for inst in trainingData:
            if(label != inst.classLabel):
                return False, -1
        return True, label
    
    def calEntropy(self, data): # given a dataset, it return the entropy of the labels
        NEGCount = 0
        POScount = 0        
        for inst in data:
            if(inst.classLabel == self.POS):
                POScount += 1
            else:
                NEGCount += 1
        pPOS = float(POScount)/float(len(data))        
        pNEG = float(NEGCount)/float(len(data))
        if(pPOS == 0 or pNEG == 0):
            h=0
        else:
            h = -(pPOS)*(math.log(pPOS,2)) - (pNEG)*(math.log(pNEG,2))
        return h
    
    def calGini(self, data): # given a dataset, it return the entropy of the labels
        NEGCount = 0
        POScount = 0        
        for inst in data:
            if(inst.classLabel == self.POS):
                POScount += 1
            else:
                NEGCount += 1
        pPOS = float(POScount)/float(len(data))        
        pNEG = float(NEGCount)/float(len(data))
        gini = 1-(pPOS**2 + pNEG**2)
        return gini
    
    
    def getCounts(self, f, trainingData, allpossibleFeatureValues):
        
        d = {v:dict([(self.POS,0), (self.NEG,0)]) for v in allpossibleFeatureValues}
        for inst in trainingData:
            d[int(inst.featureVector[f])][inst.classLabel] += 1
        return d
        
        '''
        d = {v:list([0,0]) for v in allpossibleFeatureValues}
        for inst in trainingData:
            if(inst.classLabel == self.POS):
                d[int(inst.featureVector[f])][0] += 1
            else:
                d[int(inst.featureVector[f])][1] += 1
        return d
        '''
    
    def chooseFeatureByInfoGain(self, currentNode): #input: 1)node about to form a branch. 2)the number of bins to divide the feature values domain. #returns: 1) the feature that maximizes the information gain. 2)all values that feature takes in the training data subset.
        highestInfoGain = -1000
        bestFeature = -1
        dataEntropy = self.calEntropy(currentNode.trainingData)
        bestFeaturePossibleValues = []
      
        for f in currentNode.featureIndexList:               
            expectedEntropy = 0
            dcounts = self.getCounts(f, currentNode.trainingData, self.allpossibleFeatureValues)
            featurePossibleValues = []
            for v in self.allpossibleFeatureValues:  
                NEGcount = dcounts[v][self.NEG]
                POScount = dcounts[v][self.POS]
                
                #NEGcount = dcounts[v][1]
                #POScount = dcounts[v][0]            
                           
                featureValueShare = POScount + NEGcount
                if(featureValueShare == 0): 
                    continue  
                else:
                    featurePossibleValues.append(v)      
                             
                pPOS = float(POScount)/float(featureValueShare)        
                pNEG = float(NEGcount)/float(featureValueShare)
                dataPortion = float(featureValueShare)/float(len(currentNode.trainingData))
                
                if(pPOS == 0 or pNEG == 0):
                    h=0
                else:
                    h = -(pPOS)*(math.log(pPOS,2)) - (pNEG)*(math.log(pNEG,2))
                    
                expectedEntropy += dataPortion*h
                
            infoGain = dataEntropy - expectedEntropy   
            if(infoGain > highestInfoGain):
                highestInfoGain = infoGain
                bestFeature = f
                bestFeaturePossibleValues = featurePossibleValues
                
        return bestFeature, bestFeaturePossibleValues
    
    def chooseFeatureByGiniGain(self, currentNode): #input: 1)node about to form a branch. 2)the number of bins to divide the feature values domain. #returns: 1) the feature that maximizes the information gain. 2)all values that feature takes in the training data subset.
        highestGiniGain = -1000
        bestFeature = -1
        dataGini = self.calGini(currentNode.trainingData)
        bestFeaturePossibleValues = []
      
        for f in currentNode.featureIndexList:               
            expectedGini = 0
            dcounts = self.getCounts(f, currentNode.trainingData, self.allpossibleFeatureValues)
            featurePossibleValues = []
            for v in self.allpossibleFeatureValues:  
                NEGcount = dcounts[v][self.NEG]
                POScount = dcounts[v][self.POS]
                
                #NEGcount = dcounts[v][1]
                #POScount = dcounts[v][0]            
                           
                featureValueShare = POScount + NEGcount
                if(featureValueShare == 0): 
                    continue  
                else:
                    featurePossibleValues.append(v)      
                             
                pPOS = float(POScount)/float(featureValueShare)        
                pNEG = float(NEGcount)/float(featureValueShare)
                dataPortion = float(featureValueShare)/float(len(currentNode.trainingData))
                
                gini = 1 - ((pPOS**2) + (pNEG**2))
                    
                expectedGini += dataPortion * gini
                
            giniGain = dataGini - expectedGini   
            if(giniGain > highestGiniGain):
                highestGiniGain = giniGain
                bestFeature = f
                bestFeaturePossibleValues = featurePossibleValues
                
        return bestFeature, bestFeaturePossibleValues
    
    def getMajorityVote (self, trainingData): #input: a data set. #returns the label with higher votes
        POSCount = 0
        NEGCount = 0
        for inst in trainingData:
            if(inst.classLabel == self.POS):
                POSCount += 1
            else:
                NEGCount += 1
        if(POSCount >= NEGCount):
            return self.POS
        else:
            return self.NEG
        
        
    def selectFromTrainingData (self, trainingData, splitOnFeature, featureValue): # input: 1)training data. 2)feature the will branch 3) specific value (bin) for this feature. #returns: filtered training data that has this feature bin.
        filteredTrainingDate = []
        for inst in trainingData:
            if(inst.featureVector[splitOnFeature] == featureValue):
                filteredTrainingDate.append(inst)
        return filteredTrainingDate
    
    '''
    def fixClassLabels(self, dataSet):
        for inst in trainingSet:
            if(inst.classLabel == 'POS'):
                inst.classLabel = self.POS
            else:
                inst.classLabel = self.NEG
    ''' 
                
    def train(self, trainingSet): #inputs: 1) a data set. 2) a feature selection criteria. 3) number of bins. #returns: root of the built decision tree
        # building the tree iteratively using a queue. Whenever possible, iterative solutions are better than recursive ones.
        
        featureCount = len(trainingSet[0].featureVector)
        featureIndexList = range(featureCount)
        root = node()
        root.featureIndexList = featureIndexList
        root.trainingData = trainingSet
        root.depth = 1
        
        Q = Queue.Queue()    
        Q.put(root)
        
        while not Q.empty():
            currentNode = Q.get()      
            flagSameLabel, label = self.areSameLabel(currentNode.trainingData)
            if(flagSameLabel):
                currentNode.label = label
                continue      
            if(currentNode.depth >= self.depthLimit):
                continue
            if(len(currentNode.trainingData) <= self.samplesCountLimit):
                continue
                             
            splitOnFeature, featurePossibleValues = self.chooseFeatureByGiniGain(currentNode)
            
            currentNode.featureIndex = splitOnFeature     
                   
            for featureValue in range(len(featurePossibleValues)):
                trainingSubset = self.selectFromTrainingData (currentNode.trainingData, currentNode.featureIndex, featurePossibleValues[featureValue])
                majorityVote = self.getMajorityVote (trainingSubset)
                
                newNode = node()
                newNode.depth = currentNode.depth + 1
                newNode.trainingData = list(trainingSubset)
                #newNode.trainingData = trainingSubset
                newNode.label = majorityVote
                newNode.featureValue = featurePossibleValues[featureValue]
                newNode.featureIndexList = list(currentNode.featureIndexList)
                newNode.featureIndexList.remove(splitOnFeature)   
                newNode.parentFeatureIndex = currentNode.featureIndex                                          
                currentNode.addChild(newNode)
                Q.put(newNode)
                
        self.root = root
    
    
    def classifyOne(self, inst):
        currentDepth = 0
        currectNode = self.root
        predictedLabel = -1
        while len(currectNode.children)>0 and currentDepth <= self.depthLimit: # till we reach a leaf or reach depth threshold           
            currentValue = inst.featureVector[currectNode.featureIndex] 
            foundChild = False
            for child in currectNode.children:  
                if(currentValue == child.featureValue): # in case a feature value has so subset of the training data, we use the parent's majority vote for classification
                    currectNode = child
                    currentDepth += 1
                    foundChild = True
                    break               
            predictedLabel = currectNode.label
            if(foundChild == False):
                break 
        return predictedLabel
        
    def classify(self, dataSet): #input: 1) the root of the decision tree. 2)data to classify. 3) search depth threshold. #returns a list of predicted labels
        for inst in dataSet:
            predictedLabel = self.classifyOne(inst)
            inst.predictedClassLabel = predictedLabel
        
    
  
