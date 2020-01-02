from keras.layers import Layer
import keras.backend as K

class _RandomWeightedAverage(Layer):
    
    def __init__(self, **kwargs):
    	
    	super(_RandomWeightedAverage, self).__init__(**kwargs)

    def call(self, inputs):
    	batch_size = K.shape(inputs[0])[0]
    	weights = K.random_uniform((batch_size, 1, 1, 1))
    	return (weights * inputs[0]) + ((1 - weights) * inputs[1])
        
    