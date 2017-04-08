'''
Created on Apr 5, 2017

@author: mohame11
'''
from BaggedDecisionTrees import BaggedDecisionTrees
from Classifier import Classifier
from RandomForestTree import RandomForestTree
import numpy as np

class RandomForest(BaggedDecisionTrees):

    def __init__(self, treesCount ,allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.trees = []
        for i in range(treesCount):
            RFT = RandomForestTree(allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
            self.trees.append(RFT)
    
    
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
                            
           