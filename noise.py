from keras.layers import Layer, Input
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.layers import Lambda
import copy
from functools import partial


'''
This layer requires noise generating function with 1 argument to initialize.
The inputs for call function are the shapes of noises.
'''

class Noise(Layer):
	
	def __init__(self, noise_generating_rule,shape, **kwargs):
		
		super(Noise, self).__init__(**kwargs)
		self.noise_generating_rule = noise_generating_rule
		


		
	def call(self,inputs, training=None):
		
		return self.noise_generating_rule(inputs)
		
	def get_config(self):
		config = {'noise_generating_rule' : self.noise_generating_rule}
		base_config = super(Noise, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))





class ApplyNoise(Layer):
	
	def __init__(self, noise_generating_rule, fils, **kwargs):
		
		super(ApplyNoise, self).__init__(**kwargs)
		if type(noise_generating_rule) != partial:
			noise_generating_rule = partial(eval(noise_generating_rule['func']), *noise_generating_rule['args'], **noise_generating_rule['kwargs'])
		self.noise_generating_rule = noise_generating_rule
		self.fils = self.add_weight(shape=(1,fils),dtype='float32', initializer='he_normal',trainable=True,name='noise_ratio_per_channel')
		
	def build(self, input_shape):
		
		assert len(input_shape)==4
		super(ApplyNoise, self).build(input_shape)
		
	
	def get_config(self):
		noise_func = str.split(str(self.noise_generating_rule.func),' ')[1]
		noise_args = self.noise_generating_rule.args
		noise_keywords = self.noise_generating_rule.keywords
		config = {'noise_generating_rule' : {'func' : noise_func, 'args' : noise_args, 'kwargs' : noise_keywords},
		'number of channels' : self.fils}
		base_config = super(ApplyNoise, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
		
	
#input shape should be (batch_size, H, W, C)
	def call(self, inputs):
		
		input_shape = K.shape(inputs)
		noise = self.noise_generating_rule((input_shape[0],input_shape[1],input_shape[2],1))
		
		noise = K.dot(noise, self.fils)
		
		out = inputs + noise
		
		return out
		
	'''
	def compute_output_shape(self, input_shape):
		return input_shape
	'''





if __name__=='__main__':
	
	mu = 0
	sigma = 1.0
	shape = (4,4,1)
	ngr = lambda shape : K.random_normal(mean=mu, stddev = sigma, shape=shape)
	
	inp=Input(shape=(4,4,2))
	a = np.ones(shape=(2,4,4,2)).astype('float32')
	an = ApplyNoise(ngr,2)
	m=Model(inp,an(inp))
	m.summary()
	print(m.predict(a))
	print(ngr([2,5]), dtype='float32')