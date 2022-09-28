#Import required libraries
import numpy as np

class ModelAnalyser:
    def benchmarkModel(self, model, x_test, y_test):
        '''Given a trained model and testing data produces the following performance metrics: Accuracy, Precision, Recall, F1.'''
        predictions = model.predict(([x_test]))
        predictions = [np.argmax(i) for i in predictions] #Takes the argmax of the maximum value in the array revealing the actual predicted classes.
        accuracy = self.arrayAccuracy(predictions, y_test)
        precision = self.arrayPrecision(predictions, y_test)
        recall = self.arrayRecall(predictions, y_test)
        f1 = (2*precision*recall)/(precision+recall)
        outputMetrics = {'Accuracy':accuracy, 'Precision': precision, 'Recall': recall, 'F1':f1}
        return outputMetrics
    
    def confusionMatrix(self, predictions:list, evidence:list, numClasses:int) -> np.array:
        '''Computes confusion matrix given a set of predictions, true labels, and number of classes.'''
        confusionMatrix = np.zeros((numClasses, numClasses)) #Intialise confusion matrix.
    
        #Populate confusion matrix where rows represent predictions and columns represent true values/
        for i in range(len(predictions)):
            confusionMatrix[predictions[i]][evidence[i]] = confusionMatrix[predictions[i]][evidence[i]] + 1
    
        #Display confusion matrix.
        for i in range(numClasses):
            for j in range(numClasses):
                print(str(int(confusionMatrix[i][j])) + " ", end = "")
            print()
    
        return 
    
    def arrayAccuracy(self, pred:list, actual:list):
        '''Computes model accuracy given a list of predictions and actual values, returns as percentage.'''
        count = 0
        for i in range(len(pred)):
            if(pred[i] == actual[i]):
                count = count + 1
                
        dataSize = len(pred)
        accuracy = (count/dataSize)*100
        return accuracy
    
    
    def arrayPrecision(self, pred:list, actual:list):
        """Computes model precision given a list of predictions and actual values, returns as percentage."""
        tp = 0
        fp = 0
        numClasses = self._unique(pred)   
        precisionList = []
        
        for _class in numClasses:
            for i in range(len(pred)):
                if pred[i]==_class and actual[i]==pred[i]:
                    tp+=1
                if pred[i]==_class and actual[i]!=pred[i]:
                    fp+=1
                    
            precision = tp/(tp+fp)
            precisionList.append(precision)
            tp = 0
            fp = 0
        
        return sum(precisionList)/len(precisionList)
    
    def arrayRecall(self, pred, actual):
        """Computes model recall given a list of predictions and actual values, returns as percentage."""
        tp = 0
        fn = 0
        numClasses = self._unique(pred)   
        recallList = []
        
        for _class in numClasses:
            for i in range(len(pred)):
                if pred[i]==_class and actual[i]==pred[i]:
                    tp+=1
                if pred[i]!=_class and actual[i]==_class:
                    fn+=1
                    
            recall = tp/(tp+fn)
            recallList.append(recall)
            tp = 0
            fn = 0
        
        return sum(recallList)/len(recallList)

    def _unique(self, _list):
        """Given a list returns list of unique values within that list."""
        arr = np.array(_list)
        uniqueArr = np.unique(arr)
        return uniqueArr.tolist()
    