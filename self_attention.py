
import keras.backend as K
import numpy as np
from keras.layers import Layer
from keras.constraints import max_norm


data_format = "channels_last"
initializer='glorot_uniform'
constraint = max_norm(1.0)
ggl_scale_hw = 2

class SelfAttention(Layer):
	
	
	def __init__(self, attn_filters, **kwargs):
		
		super(SelfAttention, self).__init__(**kwargs)
		self.filters = attn_filters
		self.Wf = None
		self.Wg = None
		self.Wh = None
		self.Wv = None
		self.n = 0
		self.channels = 0
		self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', constraint=constraint, dtype = 'float32')
		
		
	def build(self, input_shape):
		
		shape1 = [1,1] + [input_shape[-1], self.filters]
		shape2 = [1,1] + [input_shape[-1]] * 2
		self.Wf = self.add_weight(name='Wf', shape=shape1,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wg = self.add_weight(name='Wg', shape = shape1,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wh = self.add_weight(name='Wh', shape = shape2,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wv = self.add_weight(name='Wh', shape = shape2, 
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.channels= input_shape[-1]
		self.n = input_shape[1]*input_shape[2]
		super(SelfAttention, self).build(input_shape)
        
        
	def call(self, inputs):
		
		bs = K.shape(inputs)[0:1]
		shape1 = K.variable(np.array([self.n, self.filters]), dtype='int32')
		shape2 = K.variable(np.array([self.n, self.channels]), dtype='int32')
		shape1 = K.concatenate([bs, shape1])
		shape2 = K.concatenate([bs, shape2])
		f = K.conv2d(inputs, kernel = self.Wf, data_format = data_format)
		g = K.conv2d(inputs, kernel = self.Wg, data_format = data_format)
		h = K.conv2d(inputs, kernel = self.Wh, data_format = data_format)
		ff = K.reshape(f, shape1)
		gf = K.reshape(g, shape1)
		hf = K.reshape(h,shape2)#bs × n x c
		s = K.batch_dot(ff, gf, axes=(2,2))#bs x n x n
		beta = K.softmax(s)
		o = K.batch_dot(beta, hf, axes=(2,1))#bs x n x c
		o = K.reshape(o, K.shape(inputs))
		o = K.conv2d(o, kernel = self.Wv, data_format = data_format)
		y = self.gamma * o + inputs
		return y
		
		
	def get_config(self):
		
		config = {'attn_filters' : self.filters}
		base_config = super(SelfAttention, self).get_config()
		return dict(list(config.items()) + list(base_config.items()))





class GoogleAttention(Layer):


	def __init__(self, scale_channels = (8,2), **kwargs):

		self.scale_channels = scale_channels
		self.Wf = None
		self.Wg = None
		self.Wh = None
		self.Wv = None
		self.n1 = 0
		self.n2 = 0
		self.channels = 0
		super(GoogleAttention, self).__init__(**kwargs)
		
		self.gamma = self.add_weight(name='google_gamma', shape=[1], initializer='zeros', constraint=constraint, dtype = 'float32')


	def build(self, input_shape):
		
		self.channels= input_shape[-1]
		shape1 = [1,1] + [input_shape[-1], self.channels // self.scale_channels[0]]
		shape2 = [1,1] + [input_shape[-1], self.channels // self.scale_channels[1]]
		shape3 = [1,1] + [self.channels // self.scale_channels[1], self.channels]
		self.Wf = self.add_weight(name='Wf', shape=shape1,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wg = self.add_weight(name='Wg', shape = shape1,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wh = self.add_weight(name='Wh', shape = shape2,
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.Wv = self.add_weight(name='Wh', shape = shape3, 
		initializer=initializer, constraint=constraint, dtype = 'float32')
		self.n1 = (input_shape[1] // ggl_scale_hw)*(input_shape[2] // ggl_scale_hw)
		self.n2 = input_shape[1] * input_shape[2]
		super(GoogleAttention, self).build(input_shape)


	def call(self, inputs):

		#c2 : num of input channels // 2, c1 : num of input channels // 8
		#n1 : (h/2) * (w/2), n2 : h * w, C : num of input channels

		bs = K.shape(inputs)[0:1]
		hw = K.shape(inputs)[1:3]
		#c1 = K.shape(inputs)[-1:] // self.scale_channels[0]
		c2 =  K.shape(inputs)[-1:] // self.scale_channels[1]
		shape1 = K.variable(np.array([self.n1, self.channels // self.scale_channels[0]]), dtype='int32')
		shape2 = K.variable(np.array([self.n2, self.channels // self.scale_channels[0]]), dtype='int32')
		shape3 = K.variable(np.array([self.n1, self.channels // self.scale_channels[1]]), dtype='int32')
		
		shape1 = K.concatenate([bs, shape1])
		shape2 = K.concatenate([bs, shape2])
		shape3 = K.concatenate([bs, shape3])
		shape4 = K.concatenate([bs, hw, c2])
		f = K.conv2d(inputs, kernel = self.Wf, data_format = data_format)
		f = K.pool2d(f, pool_size = (ggl_scale_hw, ggl_scale_hw), 
		strides = (ggl_scale_hw, ggl_scale_hw), padding='same', pool_mode = 'max') # h/2 , w/2 , c1

		g = K.conv2d(inputs, kernel = self.Wg, data_format = data_format) # h, w, c1

		h = K.conv2d(inputs, kernel = self.Wh, data_format = data_format)
		h = K.pool2d(h, pool_size = (ggl_scale_hw, ggl_scale_hw), 
		strides = (ggl_scale_hw, ggl_scale_hw), padding='same', pool_mode = 'max') # h/2, w/2, c2


		ff = K.reshape(f, shape1) # bs, n1, c1
		gf = K.reshape(g, shape2) # bs, n2, c1
		hf = K.reshape(h, shape3) # bs, n1, c2

		s = K.batch_dot(ff, gf, axes=(2,2)) # bs, n1, n2
		beta = K.softmax(s) #bs, n1, n2
		o = K.batch_dot(beta, hf, axes=(1,1)) #bs, n2, c2
		o = K.reshape(o, shape4) #bs, h, w, c2
		o = K.conv2d(o, kernel = self.Wv, data_format = data_format) #bs, h, w, C
		return self.gamma*o + inputs


	def get_config(self):
		
		config = {'scale_channels' : self.scale_channels}
		base_config = super(GoogleAttention, self).get_config()
		return dict(list(config.items()) + list(base_config.items()))

		


						
if __name__ == '__main__':
	
	from keras.layers import Input
	from keras.models import Model
	from keras.optimizers import Adam
	
	'''
	inp = Input([4,4,3])
	img = np.random.normal(0.0,1.0, (2,4,4,3))
	obj = np.random.normal(0.0, 1.0, (2,4,4,3))
	out = SelfAttention(2)(inp)
	m = Model(inp, out)
	m.compile(optimizer=Adam(0.01), loss = 'mse')
	p = m.predict(img)
	print(img)
	print('----')
	print(obj)
	print('----')
	print(p)
	print(p.shape)
	m.train_on_batch(img, obj)
	print('----')
	print(m.predict(img))
	'''
	
	inp = Input([4,4,16])
	img = np.random.normal(0.0,1.0, (2,4,4,16))
	obj = np.random.normal(0.0, 1.0, (2,4,4,16))
	out = GoogleAttention()(inp)
	m = Model(inp, out)
	m.compile(optimizer=Adam(0.01), loss = 'mse')
	p = m.predict(img)
	print(img)
	print('----')
	print(obj)
	print('----')
	print(p)
	print(p.shape)
	m.train_on_batch(img, obj)
	print('----')
	print(m.predict(img))
	
