'''
Created on Apr 3, 2017

@author: mohame11
'''
from Classifier import *
from DecisionTree import *

class BaggedDecisionTrees(Classifier):

    def __init__(self, treesCount ,allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.trees = []
        for i in range(treesCount):
            DT = DecisionTree(allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
            self.trees.append(DT)
    
    
    
    def train(self, trainingSet): 
        for cnt,dt in enumerate(self.trees):
            ts = []
            ts = np.random.choice(trainingSet, len(trainingSet), replace=True)
            '''
            for i in range(len(trainingSet)):
                inst = random.choice(trainingSet)
                ts.append(inst)
            '''
            #print cnt,
            dt.train(ts)
                
    
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
    
    def classify(self, dataSet):
        for inst in dataSet:
            predPOS = 0
            predNEG = 0
            for dt in self.trees:
                pred = dt.classifyOne(inst)
                if(pred == self.POS):
                    predPOS += 1
                else:
                    predNEG += 1
            if(predPOS >= predNEG):
                inst.predictedClassLabel = self.POS
            else:
                inst.predictedClassLabel = self.NEG
                    
            