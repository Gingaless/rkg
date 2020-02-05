
from keras.layers import Layer
import keras.backend as K
import numpy as np
from keras.models import Model

	


class MixStyle(Layer):
	
	def __init__(self, step, n_inp_w, n_g_block, **kwargs):
		
		super(MixStyle, self).__init__(**kwargs)
		self.n_inp_w = n_inp_w
		self.n_g_block = n_g_block
		self.mixing_matrix = [K.variable(np.zeros((n_inp_w, 1, 1))) for _ in range(n_g_block)]
		self._n_inp_w = K.variable([self.n_inp_w], dtype='int32')
		self.step = step
		self.trainable = False
		
			
	def call(self, inputs):
		
		if not isinstance(inputs, list):
			inputs = [inputs]
		input_shape = K.shape(inputs[0])
		input_shape = K.concatenate([self._n_inp_w, input_shape])
		concat = K.concatenate(inputs,axis=0)
		res = K.reshape(concat, input_shape)
		out = self.mixing_matrix[self.step]*res
		s = K.sum(out,axis=0)
		return s
		
	'''
	def compute_output_shape(self, input_shape):
		
		return input_shape[0]
	'''
			
	def get_config(self):
		
		config = {
		'step' : self.step,
		'n_inp_w' : self.n_inp_w
		}
		base_config = super(MixStyle, self).get_config()
		return dict(list(config.items()) + list(base_config.items()))
		


				
def mk_mix_mat(dist_list, n_inp_w):
	
	mix_mat = [np.zeros((len(dist_list),n_inp_w,1,1)) for _ in dist_list]
	for i in range(len(dist_list)):
		mix_mat[i][dist_list[i],0,0] = 1.0
	return mix_mat		
			

def mk_random_mix_mat(n_inp_w, n_g_block):
	
	rnd_dist = mk_rnd_dist(n_inp_w, n_g_block)
	return mk_mix_mat(rnd_dist, n_inp_w)
	
	
def mk_rnd_dist(n_inp_w, n_g_block):
	
	rnd_dist = np.random.randint(0, n_inp_w, n_g_block)
	rnd_dist = np.sort(rnd_dist)
	return rnd_dist
	
def ini_mixing_matrix(n_w_inps, n_g_block):
	
	return [np.zeros((n_w_inps, 1, 1)) for _ in range(n_g_block)]
	
	
def switch_styles(model, mix_mat):
	
	n_g_block = len(mix_mat)
	for layer in model.layers:
		if isinstance(layer, MixStyle):
			for i in range(n_g_block):
				K.set_value(layer.mixing_matrix[i], mix_mat[i])
		else:
			if isinstance(layer, Model):
				switch_styles(layer, mix_mat)
				
			
					
							
											
if __name__ == '__main__':
	from keras.layers import Input, Dense
	from keras.optimizers import Adam
	from keras.models import Model
	l = np.arange(60).reshape((3,4,5))
	print(l)
	l2 = list(l)
	inps = [Input([5]) for _ in range(1)]
	dens = [Dense(2)(inp) for inp in inps]
	out = MixStyle(0,3,5)
	out = out(dens)
	m = Model(inps, out)
	print(m.predict(l2))
	m.compile(optimizer=Adam(lr=0.01), loss='mse')
	m.summary()
	print(m.train_on_batch(l2, np.ones((4,2))))