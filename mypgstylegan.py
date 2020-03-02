
from functools import partial
from mypggan import MyPGGAN, kernel_cond, init, const
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
	style_mixing = (1,2), **kwargs):
		
		super(PGStyleGAN, self).__init__(**kwargs)
		
		self.n_layers_of_mn = n_layers_of_mn
		self.img_noise_func = img_noise_func
		self.img_noise_args = img_noise_args
		self.img_noise_kwargs = img_noise_kwargs
		self.style_mixing = style_mixing[0]
		self.start_to_mix_style = style_mixing[1]
		self.style_dist = [0]*(self.style_mixing)
		self.mixing_matrices = []
		self.img_noise_generator = partial(img_noise_func, *img_noise_args, **img_noise_kwargs)
		self.custom_layers = dict(list(self.custom_layers.items()) + list({'ApplyNoise' : ApplyNoise, 'AdaIN' : AdaIN, 'MixStyle' : MixStyle, 'LearnedConstTensor' : LearnedConstTensor, 'Normalize' : Normalize}.items()))
		
		
	def get_n_inp_sty(self, step):
		
		return 1 if step < self.start_to_mix_style else min(step+1, self.style_mixing+1)
		
	
	def mk_input_layers_for_G(self, step):
		
		n_sty_inp = self.get_n_inp_sty(step)
		self.mixing_matrices = ini_mixing_matrix(n_sty_inp, step+1)
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
		
		if hasattr(depth, '__len__'):
			depth = depth[step]
		inps = [Input(self.img_shape[0][:2] + (self.latent_size,)),
		Input([self.latent_size])]
		out = inps[0]
		sty = inps[1]
		if step>0:
			inps[0] = Input(self.generators[step-1].output_shape[1:])
			out = UpSampling2D(scale)(inps[0])
			out = Conv2D(depth, 3, padding='same', **kernel_cond)(out)
		out = ApplyNoise(self.img_noise_generator, K.int_shape(out)[-1], initializer=init, constraint=const)(out)
		out = AdaIN(self.latent_size, K.int_shape(out)[-1], initializer=init, constraint=const)([out, sty])
		out = Conv2D(K.int_shape(out)[-1], 3, padding='same', **kernel_cond)(out)
		out = ApplyNoise(self.img_noise_generator, K.int_shape(out)[-1], initializer=init, constraint=const)(out)
		out = AdaIN(self.latent_size, K.int_shape(out)[-1], initializer=init, constraint=const)([out, sty])
		return Model(inputs=inps, outputs=out, name='G_chain_' + str(step))
		
		
				
	def mk_output_layers_for_G(self, step):
		
		inp = Input(shape=self.generators[step].layers[-1].output_shape[1:])
		out = inp
		out = Conv2D(3,1,padding='same', **kernel_cond)(out)
		out = Activation('tanh')(out)
		return Model(inp, out, name='output_layers_' + str(step) + '_for_G')
		
	
	def initialize_D_chains(self,scale=2):

		for i in range(self.num_steps):
			
			self.discriminators[self.num_steps - 1 - i] = self.mk_D_block(self.num_steps - 1 - i,default_depth_G,scale)	
			
					
	def build_G(self, step, input_layers=None, output_layers=None, merged_old_output_layers=None):
		
		n_sty_inp = self.get_n_inp_sty(step)
		self.mixing_matrices = ini_mixing_matrix(n_sty_inp, step+1)
		G = input_layers
		if G == None:
			G = self.mk_input_layers_for_G(step)
		elif len(G.output) < step+2:
			G.name = 'input_layers_{}_for_G'.format(step-1)
			print('rebuild input layers... from {} to {}.'.format(step-1,step))
			self.load_weights_by_name(G)
			n_sty_inp = self.get_n_inp_sty(step)
			lct_inp = Input([1])
			lct = None
			sty_inps = [Input([self.latent_size]) for _ in range(n_sty_inp)]
			nors = [Normalize()(inp) for inp in sty_inps]
			dens = []
			d = nors
			for layer in G.layers:
				if isinstance(layer, Dense):
					dens.append(layer)
				if isinstance(layer, LearnedConstTensor):
					lct = layer(lct_inp)
			for i in range(n_sty_inp):
				for j in range(self.n_layers_of_mn):
					d[i] = dens[j](d[i])
			sty_mix = [MixStyle(i, n_sty_inp, step+1)(d) for i in range(step+1)]
			G = Model(inputs=[lct_inp] + sty_inps, outputs = [lct] + sty_mix,
			name = 'input_layers_{}_for_G'.format(str(step)))
					
					
		inps = G.input
		G = G(inps)
		styles = G[1:]
		G = G[0]
		
		if self.generators[0]==None:
			self.generators[0] = self.mk_G_block(0, default_depth_G[0])

		for i in range(step):
			if self.generators[i]==None:
				self.generators[i] = self.mk_G_block(i, default_depth_G[i])
			G = self.generators[i]([G, styles[i]])

		old_G = G
		if self.generators[step]==None:
			self.generators[step]=self.mk_G_block(step, default_depth_G[step])
		G = self.generators[step]([old_G, styles[step]])
		if output_layers == None:
			output_layers = self.mk_output_layers_for_G(step)
		G = output_layers(G)

		if merged_old_output_layers != None:
			G = self.mk_merge_layers_for_G(step, merged_old_output_layers)([old_G, G])

		self.G = Model(inputs=inps, outputs=G)
		self.mix_reg()
	
			
	def mix_reg(self):
		
		self.mixing_matrices = mk_random_mix_mat(len(self.G.inputs)-1, len(self.mixing_matrices))
		switch_styles(self.G, self.mixing_matrices)
		
		
	def train_AM(self, batch_size):
		
		self.mix_reg()
		return super(PGStyleGAN, self).train_AM(batch_size)
		
	def train_DM(self, real_samples, batch_size):
		self.mix_reg()
		return super(PGStyleGAN, self).train_DM(real_samples, batch_size)
		
	def random_input_vector_for_G(self, batch_size):
		fake_x = np.ones([batch_size,1])
		return [fake_x] + [self.noise_func(batch_size, self.latent_size) for _ in range(len(self.G.inputs)-1)]

				
	def generate_samples(self, num_samples):
		
		self.mix_reg()
		return super(PGStyleGAN, self).generate_samples(num_samples)




if __name__=='__main__':
	
	from PIL import Image
	
	gan = PGStyleGAN(latent_size=512)
	
	gan.build_G(4)
	gan.initialize_D_chains()
	gan.build_D(4)
	
	#gan.load(2,merge=True)
	gan.compile()
	gan.G.summary()
	gan.D.summary()
	print(gan.D.layers[0].input_shape)
	'''
	im = gan.generate_samples(40).astype('uint8')
	img = Image.fromarray(im)
	img.save('sample1.jpg')
	'''
	
	#gan.save(False)
	
	gan.train(4,1,32,1,True)
	
	im = gan.generate_samples(100).astype('uint8')
	im = Image.fromarray(im)
	im.save('sample2.jpg')
	gan.save(False)