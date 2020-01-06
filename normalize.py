
from keras.layers import Layer, Input
from keras import backend as K
from keras.models import Model
import numpy as np


class Normalize(Layer):
	
	def call(self,inputs):
		
		axis=np.arange(1,len(K.int_shape(inputs)))
		mean = K.mean(inputs, axis=axis, keepdims=True)
		stddev = K.std(inputs, axis=axis, keepdims=True)
		
		out = (inputs - mean)/stddev
		return out
	
	def compute_output_shape(self, input_shape):
		return input_shape
	
	
if __name__=="__main__":
	
	inp = Input(shape=[10])
	a = np.random.normal(0,1,(2,10))
	no = Normalize()(inp)
	m = Model(inp, no)
	r = m.predict(a)
	print(r)
		