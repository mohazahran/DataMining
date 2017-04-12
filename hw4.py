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
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    trainingPath = 'yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    isBinaryFeatures= True
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    depthLimit = 10
    boostingDepth = 10
    samplesCountLimit = 10
    numberOfTrees = 50
    modelType = 'DT'
    
    
    if(len(sys.argv) == 3): #experiments 
        trainingPath = sys.argv[1]
        if(sys.argv[2] == '1'):
            incremental_kfold_Q1(trainingPath)
            return
        elif(sys.argv[2] == '2'):
            incremental_kfold_Q2(trainingPath)
            return
        elif(sys.argv[2] == '3'):
            incremental_kfold_Q3(trainingPath)
            return
        elif(sys.argv[2] == '4'):
            incremental_kfold_Q4(trainingPath)
            return
        
    if(len(sys.argv) == 4): # default operation
        trainingPath = sys.argv[1]
        testingPath = sys.argv[2]
        if(sys.argv[3] == '1'):
            modelType = 'DT'
        elif(sys.argv[3] == '2'):
            modelType = 'BT'
        elif(sys.argv[3] == '3'):
            modelType = 'RF'
        elif(sys.argv[3] == '4'):
            modelType = 'BST'
        elif(sys.argv[3] == '5'):
            modelType = 'SVM'   
    
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
        
    elif(modelType == 'RF'):
        myClassifier = RandomForest(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, depthLimit, samplesCountLimit)
    
    elif(modelType == 'BST'):
        myClassifier = AdaBoost(numberOfTrees, allowedVocabCount, maxCountCutOff, isBinaryFeatures, boostingDepth, samplesCountLimit)
        
        
    parsedTrainingSet = myClassifier.parseData(trainingPath)
    pasredTestSet = myClassifier.parseData(testingPath)
        
    myClassifier.buildFeatures(parsedTrainingSet)

    myClassifier.getFeaturesVector(parsedTrainingSet)
    myClassifier.fixClassLabels(parsedTrainingSet)
    
    myClassifier.getFeaturesVector(pasredTestSet)
    myClassifier.fixClassLabels(pasredTestSet)
    
    myClassifier.train(parsedTrainingSet)
    
    myClassifier.classify(pasredTestSet)

    zeroOneLoss = myClassifier.evaluatePredictions(pasredTestSet)
    
    print '\nZERO-ONE-LOSS-'+modelType, myClassifier.evaluatePredictions(pasredTestSet)
    
    
def incremental_kfold_Q1(trainingPath):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = 'yelp_data.csv'
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
    boostingDepth = 10
    samplesCountLimit = 10
    numberOfTrees = 50
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    trainingSetSizes = [int(tss * len(dataset)) for tss in trainingPortions]
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    
    models = ['SVM', 'DT', 'BT', 'RF', 'BST']
    #models = ['BT']
    
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


def incremental_kfold_Q2(trainingPath):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = 'yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    k = 10
    isBinaryFeatures= True
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    depthLimit = 10
    boostingDepth = 10
    samplesCountLimit = 10
    numberOfTrees = 50
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    
    models = ['SVM', 'DT', 'BT', 'RF', 'BST']
    #models = ['BT']
    
    header = 'featureSize'+','
    for m in models:
        header += m + ',' + m + '_std' + ','
    
    print header
    #print 'tss\tSVM\tSVM_std\tLR\tLR_std\tNB\tNB_std'
    tss = 500
    if(0.25*len(dataset) < tss):
        tss = int(0.25*len(dataset))
    for allowedVocabCount in [200, 500, 1000, 1500]:
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
        print str(allowedVocabCount)+',',
        for modelType in models:
            avg = float(sum(resDic[modelType]))/float(len(resDic[modelType]))
            stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[modelType]]) / float(len(resDic[modelType])) )
            stdError = stdd / math.sqrt(k)
            #print str(avg)+'\t('+str(stdd)+')'+'\t('+str(stdError)+')'+'\t',
            print str(avg)+','+str(stdError)+',',
        print ''   
    
        
def incremental_kfold_Q3(trainingPath):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = 'yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    k = 10
    isBinaryFeatures= True
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    depthLimit = 10
    boostingDepth = 10
    samplesCountLimit = 10
    numberOfTrees = 50
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    
    models = ['DT', 'BT', 'RF', 'BST']
    #models = ['BT']
    
    header = 'depth'+','
    for m in models:
        header += m + ',' + m + '_std' + ','
    
    print header
    #print 'tss\tSVM\tSVM_std\tLR\tLR_std\tNB\tNB_std'
    tss = 500
    if(0.25*len(dataset) < tss):
        tss = int(0.25*len(dataset))
    for depthLimit in [5, 10, 15, 20]:
        boostingDepth = depthLimit
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
        print str(depthLimit)+',',
        for modelType in models:
            avg = float(sum(resDic[modelType]))/float(len(resDic[modelType]))
            stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[modelType]]) / float(len(resDic[modelType])) )
            stdError = stdd / math.sqrt(k)
            #print str(avg)+'\t('+str(stdd)+')'+'\t('+str(stdError)+')'+'\t',
            print str(avg)+','+str(stdError)+',',
        print ''        
        
        
def incremental_kfold_Q4(trainingPath):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = 'yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    k = 10
    isBinaryFeatures= True
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 1000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    depthLimit = 10
    boostingDepth = 10
    samplesCountLimit = 10
    numberOfTrees = 50
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    
    models = ['DT', 'BT', 'RF', 'BST']
    
    header = 'numberOfTrees'+','
    for m in models:
        header += m + ',' + m + '_std' + ','
    
    print header
    #print 'tss\tSVM\tSVM_std\tLR\tLR_std\tNB\tNB_std'
    tss = 500
    if(0.25*len(dataset) < tss):
        tss = int(0.25*len(dataset))
    for numberOfTrees in [10, 25, 50, 100]:
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
        print str(numberOfTrees)+',',
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
    #incremental_kfold_Q4('yelp_train0.txt')
    main()
   
    
