'''
Created on Mar 3, 2017

@author: mohame11
'''
from Classifier import *
import numpy as np
import math

class LogisticRegression(Classifier):

    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        self.learningRate = None
        self.myLambda = None   #regularization coefficient
        self.maxIterations = None
        self.epsilon = None 
        self.weightsVector = []
    
    def fixClassLabels(self, dataSet):
        for inst in trainingSet:
            if(inst.classLabel == 'POS'):
                inst.perClassifierLabel = 1
            else:
                inst.perClassifierLabel = 0
                
    def classify(self, dataSet):
        for example in dataSet:
            prediction = self.sigmoid(np.dot(self.weightsVector, example.featureVector))
            if(prediction >= 0.5):
                example.predictedClassLabel = self.POS
            else:
                example.predictedClassLabel = self.NEG
    
    def sigmoid(self, x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / float(1 + z)
        else:
            z = math.exp(x)
            return z / float(1 + z)
        
    def train(self, trainingSet):
        vectorLength = len(trainingSet[0].featureVector)
        self.weightsVector = np.array([0.0]*(vectorLength))
      
        for i in range(self.maxIterations):
            #print ('>>iter:'+str(i)+'/'+str(self.maxIterations))
            weightsGradient = np.array([0.0]*(vectorLength))
           
            for k in range(len(trainingSet)):
                dot = np.dot(self.weightsVector , trainingSet[k].featureVector)
                prediction = self.sigmoid(dot)
                weightsGradient += (trainingSet[k].perClassifierLabel - prediction) * trainingSet[k].featureVector
                
            
            weightsGradient = weightsGradient - (self.myLambda * self.weightsVector)
            newWeights = self.weightsVector + (self.learningRate * weightsGradient)
            weightsDiff = math.sqrt(sum((self.weightsVector-newWeights)**2))
            #print(weightsDiff)
            self.weightsVector = newWeights
            if(weightsDiff < self.epsilon):
                print('Stopping criteria LR')
                break
               
    
   
            
            
           
         
        