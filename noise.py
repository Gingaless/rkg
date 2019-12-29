from keras.layers import Layer, Input
from keras import backend as K
#from keras.backend import Tensor
import numpy as np
from keras.layers import Activation


'''
This layer requires noise generating function with 1 argument to initialize.
The inputs for call function are the shapes of noises.
'''

class Noise(Layer):
	
	def __init__(self, noise_generating_rule, **kwargs):
		
		super(Noise, self).__init__(**kwargs)
		self.noise_generating_rule = noise_generating_rule


		
	def call(self,inputs, training=None):
		
		return self.noise_generating_rule(inputs)
		
	def get_config(self):
		config = {'noise_generating_rule' : self.noise_generating_rule}
		base_config = super(Noise, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))




if __name__=='__main__':

	mu = 0
	sigma = 1.0
	a = Input([2])
	ngr = (lambda shape : K.random_normal(mu, sigma, shape=shape))
	layer = Noise(ngr)(a)
	mdl = Activation('tanh')(layer)
	print(mdl([5,5]))