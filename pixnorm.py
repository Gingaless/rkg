from keras.layers import Layer
import keras.backend as K


class PixelNormalization(Layer):
	
	
	def __init__(self, epsilon = 1.0e-8, **kwargs):
		
		self.epsilon = epsilon
		super(PixelNormalization, self).__init__(**kwargs)
		
		
	def call(self, inputs):
		
		sqr = inputs ** 2.0
		sqr_mean = K.mean(sqr, axis=-1, keepdims=True)
		sqr_mean += self.epsilon
		l2_norm = K.sqrt(sqr_mean)
		normalized = inputs/l2_norm
		return normalized
		
		
	def compute_output_shape(self, input_shape):
		
		return input_shape
		
		


if __name__=='__main__':
	
	import numpy as np
	from keras.models import Model
	from keras.layers import Input
	
	a = np.arange(0,2*4*4*3).reshape((2,4,4,3))
	inp = Input(shape = [4,4,3])
	m = Model(inputs=inp, outputs=PixelNormalization()(inp))
	out = m.predict(a)
	print(out.shape)
	print(out)

		
		
		
		