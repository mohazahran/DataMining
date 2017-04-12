'''
Created on Apr 5, 2017

@author: mohame11
'''
from DecisionTree import *

class RandomForestTree(DecisionTree):
    
    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit):
        DecisionTree.__init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
    
    
    
    def getFeaturesVector(self, parsedDataSet):
        for inst in parsedDataSet:
            words = inst.cleanedText
            inst.featureVector = np.zeros(len(self.filteredVocab)) #+1 for bias
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
                            
                            
    def chooseFeatureByInfoGain(self, currentNode): #input: 1)node about to form a branch. 2)the number of bins to divide the feature values domain. #returns: 1) the feature that maximizes the information gain. 2)all values that feature takes in the training data subset.
        highestInfoGain = -1000
        bestFeature = -1
        dataEntropy = self.calEntropy(currentNode.trainingData)
        bestFeaturePossibleValues = []
        
        #random forest random feature sampling
        if (len(currentNode.featureIndexList) > 1):
            sampledFeatures = np.random.choice(currentNode.featureIndexList, int(math.sqrt(len(currentNode.featureIndexList))), replace=False)  
      
        for f in sampledFeatures:               
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
        
        #random forest random feature sampling
        if (len(currentNode.featureIndexList) > 1):
            sampledFeatures = np.random.choice(currentNode.featureIndexList, int(math.sqrt(len(currentNode.featureIndexList))), replace=False)  
      
        for f in sampledFeatures:               
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
    
      
    def train(self, trainingSet):  
        
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
    
        
