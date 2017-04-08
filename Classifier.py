'''
Created on Mar 3, 2017

@author: mohame11
'''
import numpy as np
import random

class Instance():
    def __init__(self):
        self.text = ''
        self.classLabel = None
        self.perClassifierLabel = None
        self.predictedClassLabel = ''
        self.id = -1
        self.cleanedText = ''
        self.featureVector = None
        self.featureVectorDic = None
        
class Classifier(object):
    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures):
        self.NEG = 'NEG'
        self.POS = 'POS'
        self.allowedVocabCount = allowedVocabCount
        self.maxCountCutOff = maxCountCutOff
        self.filteredVocab = None
        self.isBinaryFeatures = isBinaryFeatures
    
    def cleanText(self, txt): #return clean list of words.     
        tmp = ' '.join(txt.split()) #remove contigous spaces
        cleaned = ''
        for c in tmp:
            if(c.isalpha() or c.isdigit()):
                cleaned += c
            elif(c == ' '):
                cleaned += c
            else:
                continue            
        cleaned = cleaned.lower()
        cleaned = ' '.join(cleaned.split()) 
        cleaned = [w.strip() for w in cleaned.split()]
        return cleaned
    
    def parseData(self, path):
        dataSet = []
        r = open(path, 'r')
        for line in r:
            parts = line.strip().split('\t')
            inst = Instance()
            inst.id = parts[0]
            if(parts[1] == '1'):
                inst.classLabel = self.POS
            elif(parts[1] == '0'):
                inst.classLabel = self.NEG
            inst.text = parts[2].strip()
            inst.cleanedText = self.cleanText(inst.text)
            dataSet.append(inst)
        r.close()
        return dataSet
    
    
    def buildFeatures(self, parsedDataSet):
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
        
        #only for NB
        #self.posVocab = dict.fromkeys(self.filteredVocab,0)
        #self.negVocab = dict.fromkeys(self.filteredVocab,0)
        
        
    def getFeaturesVector(self, parsedDataSet):
        for inst in parsedDataSet:
            words = inst.cleanedText
            inst.featureVector = np.zeros(len(self.filteredVocab)+1) #+1 for bias
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
                            
            inst.featureVector[-1] = 1 #bias term
                            
    
    def train(self, trainingSet): 
        pass
    
    
    def classify(self, dataSet):
        pass
    
    def fixClassLabels(self, dataSet):
        pass
            
    
    def evaluatePredictions(self, dataSet):
        zero_one_loss = 0.0
        for inst in dataSet:
            if(inst.predictedClassLabel != inst.classLabel):
                zero_one_loss += 1
        
        zero_one_loss = float(zero_one_loss)/float(len(dataSet)) 
        return zero_one_loss
    
    
            
        
       
   
   
   
   
   
   
   
   
        