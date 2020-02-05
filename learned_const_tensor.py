#pylint:disable=E0001

from keras.layers import Layer, Input
from keras import backend as K
from keras.models import Model
import numpy as np


#You must set inputs as np.ones(shape=[batch_size, 1])
class LearnedConstTensor(Layer):
	
	def __init__(self, shape, **kwargs):
		
		super(LearnedConstTensor, self).__init__(**kwargs)
		self.shape = tuple(shape)
		n_w = np.prod(shape)
		self.w = self.add_weight(shape=(1,n_w), dtype='float32', initializer='he_normal',trainable=True, name='learnt_const_tensor')
		
		
	def get_config(self):
		config = {'shape' : np.array(self.shape)}
		base_config = super(LearnedConstTensor, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
		
		
	def call(self, inputs):
		
		batch_size = K.shape(inputs)[0:1]
		batch_size = K.cast(batch_size, dtype='int32')
		shp = K.variable(self.shape)
		shp = K.cast(shp,dtype='int32')
		shp = K.concatenate([batch_size, shp])
		out = K.dot(inputs,self.w)
		out = K.reshape(out, self.compute_output_shape(K.shape(inputs)))
		return out
		
	
	def compute_output_shape(self, input_shape):
		
		return (input_shape[0],) + self.shape
	
		



if __name__=='__main__':
	
	a = np.ones(shape=(2,1))
	print(a)
	pinp = Input(shape=[1])
	ct = LearnedConstTensor((4,4,3))(pinp)
	m = Model(pinp, ct)
	r = m.predict(a)
	print(r.shape)
		