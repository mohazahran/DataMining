'''
Created on Apr 3, 2017

@author: mohame11
'''
from Classifier import *
from DecisionTree import *

class AdaBoost(Classifier):

    def __init__(self, treesCount ,allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.trees = {}
        for i in range(treesCount):
            DT = DecisionTree(allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
            self.trees[DT] = 1.0
        
    
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
                            
    
    def train(self, trainingSet):
        probOne = 1.0 / float(len(trainingSet)) 
        instanceWeights = [probOne]*len(trainingSet)
        for cnt,dt in enumerate(self.trees):
            #print cnt,
            ts = np.random.choice(trainingSet, len(trainingSet), replace=True, p=instanceWeights)
            dt.train(ts)
            dtError = 0.0
            for i in range(len(trainingSet)):
                pred = dt.classifyOne(trainingSet[i])
                trainingSet[i].predictedClassLabel = pred
                if(pred != trainingSet[i].classLabel):
                    dtError += instanceWeights[i]
            
            alpha = (0.5) * math.log((1-dtError)/dtError)
            self.trees[dt] = alpha
            
            normalization = 2*math.sqrt((dtError * (1-dtError)))
            for i in range(len(trainingSet)):
                if(trainingSet[i].predictedClassLabel == trainingSet[i].classLabel):
                    instanceWeights[i] = instanceWeights[i] * math.exp(-1*alpha)/normalization
                else:
                    instanceWeights[i] = instanceWeights[i] * math.exp(alpha)/normalization
                 
                

    def classify(self, dataSet):
        for inst in dataSet:
            totPred = 0.0
            for dt in self.trees:
                pred = dt.classifyOne(inst)
                if(pred == self.POS):
                    val = 1
                else:
                    val = -1
                totPred += val * self.trees[dt]
                
            finalPrediction = int(np.sign(totPred))
            if(finalPrediction == 1 or finalPrediction == 0):
                inst.predictedClassLabel = self.POS
            else:
                inst.predictedClassLabel = self.NEG
            
                
                    
            