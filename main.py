#Import required libraries 
from ModelInvoker import ModelInvoker
from CrossValidation import CrossValidation
from sklearn import datasets

#Note that since there is no stratification metrics acheived can vary.
def main(epochs=5, numFolds=5):
    #Load in dataset
    data = datasets.load_digits()
    
    #Separate data and labels for data - Into x and y
    x = data.data
    y = data.target
    
    #Instantiate appropriate objects.
    invoker = ModelInvoker()
    cnn = invoker.getModel('CNN')
    mlp = invoker.getModel('MLP')
    cv = CrossValidation()
    
    #Train models using cross validation to assess their performance.
    cnnMetrics = cv.crossValidate(cnn, x, y, 5, (8,8,1))
    mlpMetrics = cv.crossValidate(mlp, x, y, 5, False)
    
    print('------------------------------------------------------------')
    print('CNN Metrics for Each Fold: '+str(cnnMetrics[1]))
    print('CNN Aggregated Metrics: '+str(cnnMetrics[0]))
    print('------------------------------------------------------------')
    print()
    print('------------------------------------------------------------')
    print('MLP Metrics for Each Fold: '+str(mlpMetrics[1]))
    print('MLP Aggregated Metrics: '+str(mlpMetrics[0]))
    print('------------------------------------------------------------')
    
main()
