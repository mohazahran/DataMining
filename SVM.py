'''
Created on Mar 3, 2017

@author: mohame11
'''
from Classifier import *
import numpy as np
import math
class SVM(Classifier):

    def __init__(self, allowedVocabCount, maxCountCutOff, isBinaryFeatures):
        Classifier.__init__(self,allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        #self.NEG = -1
        #self.POS = 1
        self.learningRate = None
        self.myLambda = None   #regularization coefficient
        self.maxIterations = None
        self.epsilon = None 
        self.weightsVector = []
    
    
    def classify(self, dataSet):
        for example in dataSet:
            dot = np.dot(self.weightsVector, example.featureVector)
            prediction = int(np.sign(dot))
            if(prediction == 1 or prediction == 0):
                example.predictedClassLabel = self.POS
            else:
                example.predictedClassLabel = self.NEG
      
    
    
    def fixClassLabels(self, dataSet):
        for inst in dataSet:
            if(inst.classLabel == 'POS'):
                inst.perClassifierLabel = 1
            else:
                inst.perClassifierLabel = -1
    
                
    def train(self, trainingSet):
        
        vectorLength = len(trainingSet[0].featureVector)
        self.weightsVector = np.array([0.0]*(vectorLength))
      
        for i in range(self.maxIterations):
            #print ('>>iter:'+str(i)+'/'+str(self.maxIterations))
            weightsGradient = np.array([0.0]*(vectorLength)) #weights without bias
           
            for k in range(len(trainingSet)):
                prediction = np.dot(self.weightsVector , trainingSet[k].featureVector)
                check = prediction * trainingSet[k].perClassifierLabel         
                if(check < 1): # it means to even consider the examples classified correctly but within the margin to be mistakes.
                    weightsGradient += self.myLambda * self.weightsVector - trainingSet[k].perClassifierLabel * trainingSet[k].featureVector
                else:
                    weightsGradient += self.myLambda * self.weightsVector
            
            weightsGradient = weightsGradient / float(len(trainingSet))
            newWeights = self.weightsVector - self.learningRate * weightsGradient
            weightsDiff = math.sqrt(sum((self.weightsVector-newWeights)**2))
            #print(weightsDiff)
            self.weightsVector = newWeights
            if(weightsDiff < self.epsilon):
                print('Stopping criteria SVM')
                break
               
    
    '''    
    def train(self, trainingSet):
        vectorLength = len(trainingSet[0].featureVector)
        self.weightsVector = np.array([0.0]*(vectorLength))
      
        for i in range(self.maxIterations):
            print ('>>iter:'+str(i)+'/'+str(self.maxIterations))
            weightsGradient = np.array([0.0]*(vectorLength)) #weights without bias
           
            for k in range(len(trainingSet)):
                prediction = np.dot(self.weightsVector , trainingSet[k].featureVector)
                check = prediction * trainingSet[k].classLabel         
                if(check < 1): # it means to even consider the examples classified correctly but within the margin to be mistakes.
                    weightsGradient += trainingSet[k].classLabel * trainingSet[k].featureVector
            
            weightsGradient = (self.myLambda * self.weightsVector) - (weightsGradient/float(len(trainingSet)))
            newWeights = self.weightsVector - self.learningRate * weightsGradient
            weightsDiff = sum(abs(self.weightsVector-newWeights))/float(len(self.weightsVector))
            print(weightsDiff)
            self.weightsVector = newWeights
            if(weightsDiff < self.epsilon):
                break
                
    '''     
            
            
           
         
        