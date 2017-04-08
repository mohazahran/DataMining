'''
Created on Mar 3, 2017

@author: mohame11
'''
import random
import math
import sys
from symbol import arglist
from SVM import *
from LogisticRegression import *
from NaiveBayes import *

def incremental_kfold(k, trainingPortions, trainingPath, isBinaryFeatures):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 4000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    trainingSetSizes = [int(tss * len(dataset)) for tss in trainingPortions]
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    print 'tss\tSVM\tSVM_std\tLR\tLR_std\tNB\tNB_std'
    
    for tss in trainingSetSizes: 
        models = ['SVM', 'LR', 'NB']
        resDic = {}
        resDic['SVM']=[]
        resDic['LR']=[]
        resDic['NB']=[]           
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
                elif(modelType == 'SVM'):
                    myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    myClassifier.learningRate = svm_learningRate
                elif(modelType == 'NB'):
                    myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    
                myClassifier.myLambda = myLambda   
                myClassifier.maxIterations = maxIteration
                myClassifier.epsilon = myEpsilon 
                
                myClassifier.buildFeatures(currentTrainingSet)
    
                myClassifier.getFeaturesVector(currentTrainingSet)
                myClassifier.getFeaturesVector(testSet)
                
                myClassifier.train(currentTrainingSet)
                
                myClassifier.classify(testSet)
            
                zeroOneLoss = myClassifier.evaluatePredictions(testSet)
                
                resDic[modelType].append(zeroOneLoss)
            
                #print str(zeroOneLoss)+'\t',
            
            #print ''
            
        #print str(tss)+'\t'+str(fold)+'\t',
        print str(tss)+'\t',
        for modelType in models:
            avg = float(sum(resDic[modelType]))/float(len(resDic[modelType]))
            stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[modelType]]) / float(len(resDic[modelType])) )
            stdError = stdd / math.sqrt(k)
            #print str(avg)+'\t('+str(stdd)+')'+'\t('+str(stdError)+')'+'\t',
            print str(avg)+'\t'+str(stdError)+'\t',
        print ''
            
            
def main():  
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 4000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    isBinaryFeatures = True
    
    if(len(sys.argv) >= 4): # default operation
        trainingPath = sys.argv[1]
        testingPath = sys.argv[2]
        if(sys.argv[3] == '1'):
            modelType = 'LR'
        elif(sys.argv[3] == '2'):
            modelType = 'SVM'
        else:
            modelType = 'NB'
        if(len(sys.argv) == 5):
            if(sys.argv[4] == 'ter'):
                isBinaryFeatures = False
                
        
        if(modelType == 'LR'):
            myClassifier = LogisticRegression(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            myClassifier.learningRate = logReg_learningRate
        elif(modelType == 'SVM'):
            myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            myClassifier.learningRate = svm_learningRate
        elif(modelType == 'NB'):
            myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            
            
        myClassifier.myLambda = myLambda   
        myClassifier.maxIterations = maxIteration
        myClassifier.epsilon = myEpsilon 
            
        parsedTrainingSet = myClassifier.parseData(trainingPath)
        pasredTestSet = myClassifier.parseData(testingPath)
        
        myClassifier.buildFeatures(parsedTrainingSet)
        
        myClassifier.getFeaturesVector(parsedTrainingSet)
        myClassifier.getFeaturesVector(pasredTestSet)
        
        myClassifier.train(parsedTrainingSet)
        myClassifier.classify(pasredTestSet)
        print 'ZERO-ONE-LOSS-'+modelType, myClassifier.evaluatePredictions(pasredTestSet)
        
    elif(len(sys.argv)  == 3):
        trainingPath = sys.argv[1]
        work = sys.argv[2]
        if(work == 'bin' or work == 'ter'):
            if(work == 'ter'):
                isBinaryFeatures = False
            else:
                isBinaryFeatures = True
            incremental_kfold(10, [0.01, 0.03, 0.05, 0.08, 0.1, 0.15], trainingPath, isBinaryFeatures)
        else:
            binaryVSternary(10, [0.01, 0.03, 0.05, 0.08, 0.1, 0.15], trainingPath, work)
            
            
        
    
def binaryVSternary(k, trainingPortions, trainingPath, model):
    #trainingPath = '/Users/mohame11/Documents/myFiles/Career/Work/Purdue/PhD_courses/Spring2017_sem5/CS57300_DataMining/hw/hw2/yelp_data.csv'
    #trainingPath = '/Users/mohame11/Downloads/training_dataset.csv'
    #testingPath = '/Users/mohame11/Downloads/testing_dataset.csv'
    maxCountCutOff = 100 #top maxCountCutOff frequent words will be discarded
    allowedVocabCount = 4000 #the total number of words allowed in the vocab
    maxIteration = 100
    myEpsilon = 1e-6
    svm_learningRate = 0.5
    logReg_learningRate = 0.01
    myLambda = 0.01 #regularization coefficient
    isBinaryFeatures = True

    k = 10
    tmp = Classifier(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
    dataset = tmp.parseData(trainingPath)
    trainingSetSizes = [int(tss * len(dataset)) for tss in trainingPortions]
    random.shuffle(dataset)
    foldSize = len(dataset)/k
    models = [model+'_bin', model+'_ter']
    print 'tss\t'+models[0]+'\t'+models[0]+'_std'+'\t'+models[1]+'\t'+models[1]+'_std'
    
    for tss in trainingSetSizes: 
        resDic = {}
        resDic[models[0]]=[]
        resDic[models[1]]=[]
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
                if('bin' in modelType):
                    isBinaryFeatures = True
                else:
                    isBinaryFeatures = False
                    
                if('LR' in modelType):
                    myClassifier = LogisticRegression(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    myClassifier.learningRate = logReg_learningRate
                elif('SVM' in modelType):
                    myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
                    myClassifier.learningRate = svm_learningRate
                elif('NB' in modelType):
                    myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
               
                
                    
                myClassifier.myLambda = myLambda   
                myClassifier.maxIterations = maxIteration
                myClassifier.epsilon = myEpsilon 
                
                myClassifier.buildFeatures(currentTrainingSet)
    
                myClassifier.getFeaturesVector(currentTrainingSet)
                myClassifier.getFeaturesVector(testSet)
                
                myClassifier.train(currentTrainingSet)
                
                myClassifier.classify(testSet)
            
                zeroOneLoss = myClassifier.evaluatePredictions(testSet)
                
                resDic[modelType].append(zeroOneLoss)
            
                #print str(zeroOneLoss)+'\t',
            
            #print ''
            
        #print str(tss)+'\t'+str(fold)+'\t',
        print str(tss)+'\t',
        for modelType in models:
            avg = float(sum(resDic[modelType]))/float(len(resDic[modelType]))
            stdd = math.sqrt( sum([(x-avg)**2 for x in resDic[modelType]]) / float(len(resDic[modelType])) )
            stdError = stdd / math.sqrt(k)
            #print str(avg)+'\t('+str(stdd)+')'+'\t('+str(stdError)+')'+'\t',
            print str(avg)+'\t'+str(stdError)+'\t',
        print ''
    
    
    
def debugging():
    '''
    self.text = ''
        self.classLabel = None
        self.predictedClassLabel = ''
        self.id = -1
        self.cleanedText = ''
        self.featureVector = None
        self.featureVectorDic = None
    '''
    trainingSet = []
    i1 = Instance()
    i1.classLabel = 1
    i1.featureVector = np.array([1,0,0,1])
    trainingSet.append(i1)
    
    i2 = Instance()
    i2.classLabel = 1
    i2.featureVector = np.array([1,1,1,1])
    trainingSet.append(i2)
    
    i3 = Instance()
    i3.classLabel = 0
    i3.featureVector = np.array([0,1,0,1])
    trainingSet.append(i3)
    
    i4 = Instance()
    i4.classLabel = 0
    i4.featureVector = np.array([0,1,1,1])
    trainingSet.append(i4)
    
    models = ['SVM']
    for modelType in models:
   
        if(modelType == 'LR'):
            myClassifier = LogisticRegression(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            myClassifier.learningRate = logReg_learningRate
        elif(modelType == 'SVM'):
            myClassifier = SVM(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            myClassifier.learningRate = svm_learningRate
        elif(modelType == 'NB'):
            myClassifier = NaiveBayes(allowedVocabCount, maxCountCutOff, isBinaryFeatures)
            
            
        myClassifier.myLambda = myLambda   
        myClassifier.maxIterations = maxIteration
        myClassifier.epsilon = myEpsilon 
            
        myClassifier.train(trainingSet)
        myClassifier.classify(trainingSet)
        print(modelType, myClassifier.evaluatePredictions(trainingSet))
        
    
    
    
    
if __name__ == "__main__":
    #binaryVSternary(10, [0.01, 0.03, 0.05, 0.08, 0.1, 0.15], 'NB')
    #incremental_kfold(10, [0.01, 0.03, 0.05, 0.08, 0.1, 0.15])
    main()
    #debugging()