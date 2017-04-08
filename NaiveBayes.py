'''
Created on Mar 3, 2017

@author: mohame11
'''
from Classifier import *
import math
class NaiveBayes(Classifier):
    '''
    classdocs
    '''


    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.NEG = 0
        self.POS = 1
        self.posFeatures = None
        self.negFeatures = None
        self.posCount = None
        self.negCount = None
        self.positiveProb = None
        self.negativeProb = None 
        self.posVocab = None
        self.negVocab = None
        
        
    
    def fixNegativeClassLabels(self, trainingSet):
        for inst in trainingSet:
            if(inst.classLabel == -1):
                inst.classLabel = self.NEG
                   
    def calculatedPriors(self, parsedDataSet):
        self.fixNegativeClassLabels(parsedDataSet)
        self.posCount = 0
        self.negCount = 0
        for inst in parsedDataSet:
            if(inst.classLabel == self.POS):
                self.posCount += 1
            elif (inst.classLabel == self.NEG):
                self.negCount += 1
            else:
                print('something wrong in calculatedPriors !')
        if(self.posCount+self.negCount != len(parsedDataSet)):
            print('something wrong in calculating Priors !')
    
        self.positiveProb = float(self.posCount)/float(self.posCount+self.negCount)
        self.negativeProb = float(self.negCount)/float(self.posCount+self.negCount) 
        
        
    def buildFeatures(self, parsedDataSet):
        self.fixNegativeClassLabels(parsedDataSet)
        vocab = {}
        for inst in parsedDataSet:
            uniqueWords = set(inst.cleanedText)
            for w in uniqueWords:
                if(w not in vocab):
                    vocab[w] = 1
                else:
                    vocab[w] += 1
        
        descendingFreqWords = sorted(vocab, key=lambda k: (-vocab[k], k), reverse=False)
        self.filteredVocab = descendingFreqWords[self.maxCountCutOff : self.maxCountCutOff+self.allowedVocabCount]
        
        if(self.isBinaryFeatures):
            self.posFeatures = [[0,0] for i in range(len(self.filteredVocab))]
            self.negFeatures = [[0,0] for i in range(len(self.filteredVocab))]
        else:
            self.posFeatures = [[0,0,0] for i in range(len(self.filteredVocab))]
            self.negFeatures = [[0,0,0] for i in range(len(self.filteredVocab))]
            
    
    def train(self, trainingSet):
        for inst in trainingSet:
            for i in range(len(inst.featureVector)):
                featureValue = inst.featureVector[i]
                if(inst.classLabel == self.POS):
                    self.posFeatures[i][featureValue] += 1
                else:
                    self.negFeatures[i][featureValue] += 1
                
                
        self.calculatedPriors(trainingSet)      
                    
    def getFeaturesVector(self, parsedDataSet):
        for inst in parsedDataSet:
            words = inst.cleanedText
            inst.featureVector = [0]*(len(self.filteredVocab))
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
                #else:
                #    print('word:',w,' is not found')
    def calculatedPosterior(self, inst, classLabel): #I'm not using the feature vector yet.
        
        if(classLabel == self.POS):
            prior = self.positiveProb
            featureCounts = self.posFeatures
            size = self.posCount
        else:
            prior = self.negativeProb
            featureCounts = self.negFeatures
            size = self.negCount
            
        posterior = math.log10(prior)
        for i in range(len(inst.featureVector)):
            featureValue = inst.featureVector[i]
            count = featureCounts[i][featureValue]
            prob = float(count + 1) / float(size + len(featureCounts[i]))
            posterior += math.log10(prob)
                    
        return posterior
    
    def classify(self, dataSet):
        self.fixNegativeClassLabels(dataSet)
        for inst in dataSet:
            posProb = self.calculatedPosterior(inst, self.POS)
            negProb = self.calculatedPosterior(inst, self.NEG)
            #print(posProb, negProb)
            if(posProb > negProb):
                inst.predictedClassLabel = self.POS
            elif(negProb > posProb):
                inst.predictedClassLabel = self.NEG
            else:
                inst.predictedClassLabel = self.POS
                print('A tie is found!', posProb, negProb)
                