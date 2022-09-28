#Import required libraries
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
   
class ModelInvoker:
    def getModel(self, modelName:str):
        modelDict = {'CNN': self._configureCnn, 'MLP': self._configureMlp}
        return modelDict[modelName]
        
    def _configureCnn(self):
        '''Configures and compiles CNN, returning the model object.'''
        #Convolutional Neural Network
        model = Sequential() #Neural network propagates forwards as it is 'sequential'
        model.add(Conv2D(128,(3,3)))#Add first convolutional layer consisting of 64 nodes and a window size of 3x3
        model.add(Activation('relu')) #Activation function for this layer
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Conv2D(64,(1,1)))#Add first convolutional layer consisting of 64 nodes and a window size of 3x3
        model.add(Activation('relu')) #Activation function for this layer
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('sigmoid'))#Use of signmoid function, giving a value between 0 and 1 better than a simple stepper function.
        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model
        
    def _configureMlp(self):
        '''Configures and compiles MLP, returning the model object.'''
        #Standard Neural Network
        model = Sequential()
        model.add(Flatten(input_shape=(64, )))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(10,activation = 'sigmoid'))
        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model