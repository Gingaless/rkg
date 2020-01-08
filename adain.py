from keras.layers import Layer, Input
from keras.models import Model
from keras import backend as K
from keras.backend import reshape
import numpy as np
from keras_layer_normalization import LayerNormalization
from keras.layers import Reshape

#Input b and g should be 1x1xC
class AdaIN(Layer):
    def __init__(self, input_dim, units,
             axis=-1,
             momentum=0.99,
             epsilon=1e-7,
             center=True,
             scale=True,
             **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.input_dim = input_dim
        self.units = units
        
        self.w_gamma = self.add_weight(shape=(input_dim, units,), initializer='he_normal',dtype='float', trainable=True,name='adain_w_gamma')
        self.b_gamma = self.add_weight(shape=(units,), initializer='zeros',dtype='float' ,trainable=True,name='adain_b_gamma')
        self.w_beta = self.add_weight(shape=(input_dim, units,), initializer='he_normal',dtype='float', trainable=True,name='adain_w_beta')
        self.b_beta = self.add_weight(shape=(units,), initializer='zeros',dtype='float', trainable=True,name='adain_b_beta')
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaIN, self).build(input_shape) 
    
    def call(self, inputs):
    	
        input_shape = K.int_shape(inputs[0])
        batch_size = K.shape(inputs[0])[0:1]
        shp = K.variable([1,1,self.units])
        shp = K.cast(shp, dtype='int32')
        shp = K.concatenate([batch_size, shp])
        reduction_axes = list(range(1, len(input_shape)-1))
        style = inputs[1]
        
        beta = K.dot(style, self.w_beta) + self.b_beta
        gamma = K.dot(style, self.w_gamma) + self.b_gamma
        beta = K.reshape(beta, shp)
        gamma = K.reshape(gamma, shp)
        
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean)/stddev
        
        
        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'input_dim' : self.input_dim,
            'units' : self.units,
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]







#Input b and g should be HxWxC
class SPADE(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(SPADE, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(SPADE, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        
        beta = inputs[1]
        gamma = inputs[2]

        reduction_axes = [0, 1, 2]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(SPADE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]
        
        
        
        
'''
a = K.random_normal(mean=0,stddev=1,shape=[3,2,2,4])
b = K.random_normal(mean=0,stddev=1,shape=[2,2,2])
r = K.random_uniform(minval=0, maxval=1.0,shape=[3,2])
print(a)
print(b)

      
c = K.dot(a,b)
f = K.function([],[c])
print(f(1))

styinp=Input(shape=[2])
inp = Input(shape=[2,2,4])
ad = AdaIN(2,4)
out = ad([inp,styinp])
f = K.function([inp,styinp],out)
print(f([a,r]),np.shape(f([a,r])))
inp2 = Input(shape=[2,2])
inp3 = LayerNormalization()(inp2)
arr = np.arange(0,4).reshape((2,2))
mu = np.mean(arr)
sigma = np.std(arr)
print((arr - mu)/sigma)
f=K.function([inp2],[inp3])
print(f(arr))
'''
