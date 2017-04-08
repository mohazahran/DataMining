'''
Created on Apr 2, 2017

@author: mohame11
'''
import random
import math
import sys
from symbol import arglist
from SVM import *
from LogisticRegression import *
from NaiveBayes import *
from DecisionTree import *
from BaggedDecisionTrees import *
from RandomForest import *
from AdaBoost import *


def main():  
    trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    testingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    
    modelType = 'BT'
    isBinaryFeatures = True
    depthLimit = 10
    samplesCountLimit = 10
    numberOfTrees = 50
        
    if(modelType == 'LR'):
        myClassifier = LogisticRegression(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        myClassifier.learningRate = logReg_learningRate
        myClassifier.myLambda = myLambda   
        myClassifier.maxIterations = maxIteration
        myClassifier.epsilon = myEpsilon  
        
    elif(modelType == 'SVM'):
        myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        myClassifier.myLambda = myLambda   
        myClassifier.maxIterations = maxIteration
        myClassifier.epsilon = myEpsilon  
        myClassifier.learningRate = svm_learningRate
        
    elif(modelType == 'NB'):
        myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
        
    elif(modelType == 'DT'):
        myClassifier = DecisionTree(allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
        
    elif(modelType == 'BT'):
        myClassifier = BaggedDecisionTrees(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
        
    elif(modelType == 'RT'):
        myClassifier = RandomForest(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
    
    elif(modelType == 'BST'):
        myClassifier = AdaBoost(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
        
        
        
        
     
        
    parsedTrainingSet = myClassifier.parseData(trainingPath)
    pasredTestSet = myClassifier.parseData(testingPath)
    
    myClassifier.buildFeatures(parsedTrainingSet)
    
    myClassifier.getFeaturesVector(parsedTrainingSet)
    myClassifier.getFeaturesVector(pasredTestSet)
    
    myClassifier.train(parsedTrainingSet)
    myClassifier.classify(pasredTestSet)
    
    print '\nZERO-ONE-LOSS-'+modelType, myClassifier.evaluatePredictions(pasredTestSet)
    
    
def incremental_kfold():
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    trainingPath = 'yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    k = 10
    isBinaryFeatures= True
    trainingPortions =  [0.025, 0.05, 0.125, 0.25]
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    depthLimit = 10
    boostingDepth = 5
    samplesCountLimit = 10
    numberOfTrees = 50
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    trainingSetSizes = [int(tss * len(dataset)) for tss in trainingPortions]
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    
    #models = ['SVM', 'DT', 'BT', 'RF', 'BST']
    models = ['BT']
    
    header = 'tss'+','
    for m in models:
        header += m + ',' + m + '_std' + ','
    
    print header
    #print 'tss\tSVM\tSVM_std\tLR\tLR_std\tNB\tNB_std'
    
    for tss in trainingSetSizes: 
        resDic = {}
        for m in models:
            resDic[m] = []
               
        for fold in range(0,k):
            testStart = fold * foldSize
            testEnd = testStart + foldSize
            testSet = dataset[testStart : testEnd]
            possibleTrainingData = dataset[0:testStart] + dataset[testEnd :]
            currentTrainingSet = []
            
            #print str(tss)+'\t'+str(fold)+'\t',
            trainingIndexes = random.sample(xrange(len(possibleTrainingData)), tss)
            for j in trainingIndexes:
                currentTrainingSet.append(possibleTrainingData[j])

            for modelType in models:
                if(modelType == 'LR'):
                    myClassifier = LogisticRegression(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    myClassifier.learningRate = logReg_learningRate
                    myClassifier.myLambda = myLambda   
                    myClassifier.maxIterations = maxIteration
                    myClassifier.epsilon = myEpsilon  
                    
                elif(modelType == 'SVM'):
                    myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    myClassifier.myLambda = myLambda   
                    myClassifier.maxIterations = maxIteration
                    myClassifier.epsilon = myEpsilon  
                    myClassifier.learningRate = svm_learningRate
                    
                elif(modelType == 'NB'):
                    myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    
                elif(modelType == 'DT'):
                    myClassifier = DecisionTree(allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
                    
                elif(modelType == 'BT'):
                    myClassifier = BaggedDecisionTrees(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
                    
                elif(modelType == 'RT'):
                    myClassifier = RandomForest(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
                
                elif(modelType == 'BST'):
                    myClassifier = AdaBoost(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, boostingDepth, samplesCountLimit)
                    
                    
                
                myClassifier.buildFeatures(currentTrainingSet)
    
                myClassifier.getFeaturesVector(currentTrainingSet)
                myClassifier.fixClassLabels(currentTrainingSet)
                
                myClassifier.getFeaturesVector(testSet)
                myClassifier.fixClassLabels(testSet)
                
                myClassifier.train(currentTrainingSet)
                
                myClassifier.classify(testSet)
            
                zeroOneLoss = myClassifier.evaluatePredictions(testSet)
                
                resDic[modelType].append(zeroOneLoss)
            
                #print str(zeroOneLoss)+'\t',
            
            #print ''
            
        #print str(tss)+'\t'+str(fold)+'\t',
        print str(tss)+',',
        for modelType in models:
            avg = float(sum(resDic[modelType]))/float(len(resDic[modelType]))
            stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[modelType]]) / float(len(resDic[modelType])) )
            stdError = stdd / math.sqrt(k)
            #print str(avg)+'\t('+str(stdd)+')'+'\t('+str(stdError)+')'+'\t',
            print str(avg)+','+str(stdError)+',',
        print ''
   
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    incremental_kfold()
    #main()
   
    