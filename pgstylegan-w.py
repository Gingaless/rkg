
from functools import partial
from mypgstylegan import PGStyleGAN
from mywgan import wasserstein_loss, gradient_penalty_loss
from manage_model import set_model_trainable
from keras.models import Model
from keras.layers import Input
from rwa import _RandomWeightedAverage
import numpy as np


class PGStyleWGAN(PGStyleGAN):
	
	def __init__(self, gradient_panelty_weight = 1, **kwargs):
		
		super(PGStyleWGAN, self).__init__(**kwargs)
		
		self.gradient_penalty_weight = gradient_panelty_weight
		
		self.DM_loss = [wasserstein_loss, wasserstein_loss, None]
		self.AM_loss = wasserstein_loss
		
		
	def compile_DM(self):
		
		set_model_trainable(self.G, False)
		set_model_trainable(self.D, True)
		
		real_samples = Input(shape=self.D.layers[0].input_shape[1:])
		generator_input_for_discriminator = [Input([1])] + [Input([self.latent_size]) for _ in range(len(self.G.input) - 1)]
		inps = [real_samples] + generator_input_for_discriminator
		
		generated_samples = self.G(generator_input_for_discriminator)
		averaged_samples = _RandomWeightedAverage()([real_samples, generated_samples])
		partial_gp_loss = partial(gradient_penalty_loss, 
		averaged_samples = averaged_samples, gradient_penalty_weight = self.gradient_penalty_weight)
		partial_gp_loss.__name__ = 'gradient_penalty'
		
		outs = [self.D(real_samples), self.D(generated_samples), self.D(averaged_samples)]
		self.DM_loss[2] = partial_gp_loss
		

		self.DM = Model(inputs=inps, outputs=outs)
		self.DM.compile(loss=self.DM_loss, optimizer=self.DM_optimizer)
		
		
	def y_for_GM(self, batch_size):
		
		return np.ones([batch_size, 1])
		
	def real_y_for_DM(self, batch_size):
		
		return -np.ones([batch_size, 1])
		
	def fake_y_for_DM(self, batch_size):
		
		return np.ones([batch_size, 1])
		
	def dummy_y_for_DM(self, batch_size):
		
		return np.zeros([batch_size, 1])
	

	def train_DM(self, real_samples, batch_size):
		
		DM_loss = []
		latent = self.random_input_vector_for_G(batch_size)
		real_y = self.real_y_for_DM(batch_size)
		fake_y = self.fake_y_for_DM(batch_size)
		dummy_y = self.dummy_y_for_DM(batch_size)
		DM_loss.append(self.DM.train_on_batch([real_samples] + latent, [real_y, fake_y, dummy_y]))
		return DM_loss
		
		
if __name__=='__main__':
	
	gan = PGStyleWGAN(latent_size=512)
	gan.initialize_D_chains()
	gan.build_D(1)
	gan.build_G(1)
	gan.compile()
	gan.train(1,1,32,1,True)