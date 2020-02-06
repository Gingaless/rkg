
from functools import partial
from mypggan import MyPGGAN, kernel_cond
import keras.backend as K
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D as Conv2D
from keras.models import Model
from learned_const_tensor import LearnedConstTensor
from normalize import Normalize
from adain import AdaIN
from noise import ApplyNoise
from mixstyle import MixStyle, ini_mixing_matrix, mk_random_mix_mat, switch_styles
import numpy as np


K.set_image_data_format('channels_last')
init_depth = 512
default_depth_G = [init_depth, int(init_depth/2), 
int(init_depth/4),int(init_depth/8), int(init_depth/8), 
int(init_depth/8), int(init_depth/8)]

class PGStyleGAN(MyPGGAN):
	
	def __init__(self, n_layers_of_mn=8, 
	img_noise_func = K.random_normal, img_noise_args = [], img_noise_kwargs = {'mean' : 0, 'stddev' : 0.5}, 
	style_mixing = (1,3), **kwargs):
		
		super(PGStyleGAN, self).__init__(**kwargs)
		
		self.n_layers_of_mn = n_layers_of_mn
		self.img_noise_func = img_noise_func
		self.img_noise_args = img_noise_args
		self.img_noise_kwargs = img_noise_kwargs
		self.style_mixing = style_mixing[0]
		self.start_to_mix_style = style_mixing[1]
		self.style_dist = [0]*(self.style_mixing)
		self.mixing_matrices = None
		self.img_noise_generator = partial(img_noise_func, *img_noise_args, **img_noise_kwargs)
		self.custom_layers = dict(list(self.custom_layers.items()) + list({'ApplyNoise' : ApplyNoise, 'AdaIN' : AdaIN, 'MixStyle' : MixStyle, 'LearnedConstTensor' : LearnedConstTensor, 'Normalize' : Normalize}.items()))
		
	
	def mk_input_layers_for_G(self, step):
		
		n_sty_inp = 1 if step < self.start_to_mix_style else min(step+1, self.style_mixing+1)
		self.mixing_matrices = ini_mixing_matrix(n_sty_inp, step)
		mn_inps = [Input([self.latent_size]) for _ in range(n_sty_inp)]
		lct_fake_inp = Input([1])
		dens = [Dense(self.latent_size, **kernel_cond) for _ in range(self.n_layers_of_mn)]
		nors = [Normalize()(mn_inps[i]) for i in range(n_sty_inp)]
		d = [nors[i] for i in range(n_sty_inp)]
		for i in range(n_sty_inp):
			for j in range(self.n_layers_of_mn):
				d[i] = dens[j](d[i])
		lct = LearnedConstTensor(self.img_shape[0][:2] + (self.latent_size,))(lct_fake_inp)
		sty_out = [MixStyle(i, n_sty_inp, step+1) for i in range(step+1)]
		for i in range(step+1):
			sty_out[i] = sty_out[i](d)
		return Model(inputs = [lct_fake_inp] + mn_inps, outputs = [lct] + sty_out, name = 'input_layers_{}_for_G'.format(str(step)))
	
			
	def mk_G_block(self, step, depth=init_depth, scale=2):
	
		inps = [Input(self.img_shape[0][:2] + (self.latent_size,)),
		Input([self.latent_size])]
		out = inps[0]
		sty = inps[1]
		if step>0:
			inps[0] = Input(self.generators[step-1].output_shape[1:])
			out = UpSampling2D(scale)(inps[0])
			out = Conv2D(depth, 3, padding='same', **kernel_cond)(out)
		out = ApplyNoise(self.img_noise_generator, K.int_shape(out)[-1])(out)
		out = AdaIN(self.latent_size, K.int_shape(out)[-1])([out, sty])
		out = Conv2D(K.int_shape(out)[-1], 3, padding='same', **kernel_cond)(out)
		out = ApplyNoise(self.img_noise_generator, K.int_shape(out)[-1])(out)
		out = AdaIN(self.latent_size, K.int_shape(out)[-1])([out, sty])
		return Model(inputs=inps, outputs=out, name='G_chain_' + str(step))
		
		
				
	def mk_output_layers_for_G(self, step):
		
		inp = Input(shape=self.generators[step].layers[-1].output_shape[1:])
		out = inp
		out = Conv2D(3,1,padding='same', **kernel_cond)(out)
		out = Activation('tanh')(out)
		return Model(inp, out, name='output_layers_' + str(step) + '_for_G')
		
	
	def initialize_D_chains(self,scale=2):

		for i in range(self.num_steps):
			
			self.discriminators[self.num_steps - 1 - i] = self.mk_D_block(self.num_steps - 1 - i,int(init_depth/2),scale)	
			
					
	def build_G(self, step, input_layers=None, output_layers=None, merged_old_output_layers=None):
		
		n_sty_inp = 1 if step < self.start_to_mix_style else min(step+1, self.style_mixing+1)
		self.mixing_matrices = ini_mixing_matrix(n_sty_inp, step+1)
		G = input_layers
		if G == None:
			G = self.mk_input_layers_for_G(step)
		inps = G.input
		G = G(inps)
		styles = G[1:]
		G = G[0]
		
		if self.generators[0]==None:
			self.generators[0] = self.mk_G_block(0, default_depth_G[0])

		for i in range(step):
			if self.generators[i]==None:
				self.generators[i] = self.mk_G_block(step, default_depth_G[i])
			G = self.generators[i](G, styles[i])

		old_G = G
		G = self.generators[step]([old_G, styles[step]])
		if output_layers == None:
			output_layers = self.mk_output_layers_for_G(step)
		G = output_layers(G)

		if merged_old_output_layers != None:
			G = self.mk_merge_layers_for_G(step, merged_old_output_layers)([old_G, G])

		self.G = Model(inputs=inps, outputs=G)
		
	def train_AM(self, batch_size):
		
		self.mixing_matrices = mk_random_mix_mat(len(self.G.input) - 1, len(self.mixing_matrices) )
		switch_styles(self.G, self.mixing_matrices)
		return super(PGStyleGAN, self).train_AM(batch_size)
		
	def train_DM(self, real_samples, batch_size):
		
		self.mixing_matrices = mk_random_mix_mat(len(self.G.input)-1, len(self.mixing_matrices))
		switch_styles(self.G, self.mixing_matrices)
		return super(PGStyleGAN, self).train_DM(real_samples, batch_size)
		
	def random_input_vector_for_G(self, batch_size):
		fake_x = np.ones([batch_size,1])
		return [fake_x] + [self.noise_func(batch_size, self.latent_size) for _ in range(len(self.G.input)-1)]




if __name__=='__main__':
	
	gan = PGStyleGAN(latent_size=512)
	gan.build_G(0)
	gan.G.summary()
	gan.initialize_D_chains()
	gan.build_D(0)
	gan.D.summary()
	gan.compile()
	gan.AM.summary()
	gan.DM.summary()
	print(gan.AM_optimizer)
	print(gan.DM_optimizer)
	print(gan.AM_loss)
	print(gan.DM_loss)
	gan.train(0,1,20,1,True)