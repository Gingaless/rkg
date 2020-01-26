from keras.layers import Layer
import keras.backend as K


class MiniBatchStandardDeviation(Layer):
	
	def __init__(self, epsilon=1e-8, **kwargs):
		
		self.epsilon = epsilon
		super(MiniBatchStandardDeviation, self).__init__(**kwargs)
		
		
	def call(self, inputs):
		
		mean = K.mean(inputs, axis = 0, keepdims = True)
		var = K.square(inputs - mean)
		var = K.mean(var, axis=0, keepdims=True)
		var += self.epsilon
		stdev = K.sqrt(var)
		mean_std = K.mean(stdev, keepdims=True)
		shape = K.shape(inputs)
		output = K.tile(mean_std, (shape[0], shape[1], shape[2], 1))
		combined = K.concatenate([inputs, output], axis=-1)
		return combined
		
		
	def compute_output_shape(self,input_shape):
		
		inp_shape = list(input_shape)
		inp_shape[-1] += 1
		return tuple(inp_shape)
		
		
if __name__=='__main__':
	
	from keras.models import Model
	from keras.layers import Input
	import numpy as np
	
	a = np.arange(2*4*4*2).reshape((2,4,4,2))
	inp = Input(shape=(4,4,2))
	m = Model(inp, MiniBatchStandardDeviation()(inp))
	out = m.predict(a)
	print(out)
	print(out.shape)