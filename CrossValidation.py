#Import required libraries
import numpy as np
import math
from ModelAnalyser import ModelAnalyser

class CrossValidation:   
    def __init__(self):
        self.ma = ModelAnalyser()
    
    def crossValidate(self, model, x, y, numFolds, inputShape=False, epochs=5):#Here we will implement 5 fold cross validation      
        #Split training data into blocks.
        dataBlocks, targetBlocks, blockSize = self._generateFolds(x, y, numFolds)
        
        #Permutate through all possible block/fold combinations returning the trained model and test data for each block
        foldInfo = self._permutationTrainer(model, dataBlocks, targetBlocks, numFolds, epochs, inputShape)
        
        #Benchmark Models
        foldScores = self._BenchmarkModelFolds(foldInfo)
        
        #Aggregated scores
        aggScores = self._aggregateFoldMetrics(foldScores)
        
        return aggScores, foldScores
        
    def _reshapeDesignMatrix(self, data, shape: tuple):
        """Scales design matrix to desired shape dynamically calculating the number samples for the new shape."""
        product = math.prod(shape)
        numSamplesInShape = tuple([math.floor((len(data)*product)/product)])
        data = np.reshape(data, numSamplesInShape+shape)
        return data
    
    def _aggregateFoldMetrics(self, outputMetricDict):
        """Uses output of bencharkModelFoldDict and computes overall statistics."""
        accArr = []
        precArr = []
        recArr = []
        f1Arr = []
        
        for key in list(outputMetricDict.keys()):
            accArr.append(outputMetricDict[key]["Accuracy"])
            precArr.append(outputMetricDict[key]["Precision"])
            recArr.append(outputMetricDict[key]["Recall"])
            f1Arr.append(outputMetricDict[key]["F1"])
        
        aggAcc = sum(accArr)/len(accArr)
        aggPrec = sum(precArr)/len(precArr)
        aggRec = sum(recArr)/len(recArr)
        aggF1 = sum(f1Arr)/len(f1Arr)
        
        return {"Accuracy":aggAcc, "Precision": aggPrec, "Recall":aggRec, "F1":aggF1}
    
    def _BenchmarkModelFolds(self, foldDict):
        """Uses test data, labels and trained model produced in each fold to produce performance metrics."""
        scoreDict = {}
        for key in list(foldDict.keys()):
            scoreDict[key] = self.ma.benchmarkModel(foldDict[key]["model"], foldDict[key]["testData"], foldDict[key]["testLabels"])
        return scoreDict
    
    def _permutationTrainer(self, model, dataBlocks, targetBlocks, numFolds, epochs, input_shape=False):
        """Trains model for all combinations of folds as the test data."""
        trainData, testData, testLabels, trainLabels = [], [], [], []
        testIndex =-1
        foldInfo = {}
   
        #Train models permutatiing through all folds.
        for k in range(numFolds):
            testIndex+=1
            for i in range(numFolds):
                #Create training dataset.
                for j in range(len(dataBlocks[0])):
                    if(i == testIndex):
                        testData.append(dataBlocks[i][j])
                        testLabels.append(targetBlocks[i][j])
   
                    else:
                        trainData.append(dataBlocks[i][j])
                        trainLabels.append(targetBlocks[i][j])
                        
            trainData, trainLabels, testData, testLabels = np.asarray(trainData), np.asarray(trainLabels),np.asarray(testData),np.asarray(testLabels)
            
            #If design .trix requires scaling do so.
            if input_shape:
                trainData = self._reshapeDesignMatrix(trainData, input_shape)
                testData = self._reshapeDesignMatrix(testData, input_shape)
            
            tmpModel = model()
            tmpModel.fit(trainData, trainLabels, epochs=epochs, batch_size=32)
            foldInfo["F"+str(testIndex)] = {"model":tmpModel, "testData":testData, "testLabels":testLabels}
            trainData, trainLabels, testData, testLabels = [], [], [], []
           
        return foldInfo
 
    def _generateFolds(self, x, y, numFolds):
        """Generates 'numFolds' number of blocks of data and corresponding labels."""
        #Assemble folds.
        tmpData, tmpTarget = [], []
        dataBlocks, targetBlocks = [], []    
        blockSize = math.floor((len(x)/numFolds))
        indexHandler = 0
        
        #Generate "numFolds" blocks of data.
        for _ in range(numFolds):    
            for j in range(blockSize):
                tmpData.append(x[j+indexHandler])
                tmpTarget.append(y[j+indexHandler])
            
            dataBlocks.append(tmpData)
            targetBlocks.append(tmpTarget)
            indexHandler = indexHandler+blockSize-1 #Adjust pointer to point at next block of data
            tmpData = []
            tmpTarget = []
        
        return dataBlocks, targetBlocks, blockSize
