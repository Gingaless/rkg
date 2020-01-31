
from mypggan import MyPGGAN
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from learned_const_tensor import LearnedConstTensor
from normalize import Normalize
from noise import ApplyNoise


init_depth = 512

class PGStyleGAN(MyPGGAN):
	
	def __init__(self, *args, n_layers_of_mn=8, img_noise_func = K.random_normal, img_noise_args = None, img_noise_kwargs = {'mean' : 0, 'stddev' : 0.5}, **kwargs):
		
		super(PGStyleGAN, self).__init__(*args, **kwargs)
		
		self.n_layers_of_mn = n_layers_of_mn
		self.img_noise_func = img_noise_func
		self.img_noise_args = img_noise_args
		self.img_noise_kwargs = img_noise_kwargs
		
	
	def mk_input_layers_for_G(self, step):
		
		mn_inp = Input([self.latent_size])
		lct_fake_inp = Input([1])
		d = Normalize()(mn_inp)
		for d in range(self.n_layers_of_mn):
			d = Dense(self.latent_size)(d)
		lct = LearnedConstTensor([self.img_shape[0][:2] + (self.latent_size,)])(lct_fake_inp)
		return Model(inputs = [lct_fake_inp, mn_inp], outputs = [lct, d], name = 'input_layers_{}_for_G'.format(str(step)))
	
			
	def mk_G_block(self, step, depth, style, scale=2):
		
		
		
				
								
gan = PGStyleGAN()
gan.initialize_DnG_chains()
gan.build_D(0)
gan.build_G(0)
gan.compile()
gan.G.summary()
gan.D.summary()
		