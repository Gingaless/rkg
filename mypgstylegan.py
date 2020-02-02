
from mypggan import MyPGGAN
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from learned_const_tensor import LearnedConstTensor
from normalize import Normalize
from noise import ApplyNoise


init_depth = 512

class PGStyleGAN(MyPGGAN):
	
	def __init__(self, *args, n_layers_of_mn=8, 
	img_noise_func = K.random_normal, img_noise_args = None, img_noise_kwargs = {'mean' : 0, 'stddev' : 0.5}, 
	style_mixing = (1,3), **kwargs):
		
		super(PGStyleGAN, self).__init__(*args, **kwargs)
		
		self.n_layers_of_mn = n_layers_of_mn
		self.img_noise_func = img_noise_func
		self.img_noise_args = img_noise_args
		self.img_noise_kwargs = img_noise_kwargs
		self.style_mixing = style_mixing[0]
		self.start_to_mix_style = style_mixing[1]
		
	
	def mk_input_layers_for_G(self, step):
		
		n_sty_inp = 1 if step < self.start_mix_style else min(step+1, self.style_mixing+1)
		mn_inps = [Input([n_sty_inp, self.latent_size]) for _ in range(n_sty_inp)]
		lct_fake_inp = Input([1])
		dens = [Dense(self.latent_size) for _ in range(self.n_layers_of_mn)]
		nors = [Normalize()(mn_inps[i]) for i in range(n_stp_inp)]
		sty_out = [None]*n_sty_inp
		for i in range(n_sty_inp):
			d = nor[i]
			for j in range(self.n_layers_of_mn):
				d = dens[j](d)
			sty_out[i] = d
		lct = LearnedConstTensor([self.img_shape[0][:2] + (self.latent_size,)])(lct_fake_inp)
		return Model(inputs = [lct_fake_inp, mn_inps], outputs = [lct, sty_out], name = 'input_layers_{}_for_G'.format(str(step)))
	
			
	def mk_G_block(self, step, depth, style, scale=2):

		
		
				


if __name__=='main':
	print()
		