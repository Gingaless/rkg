import numpy as np
from functools import partial




from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, add,Activation, Lambda
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D, UpSampling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
import mywgan
from mywgan import MyWGAN
from mywgan import RandomWeightedAverage
from adain import AdaIN
from noise import ApplyNoise
from learned_const_tensor import LearnedConstTensor
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras_layer_normalization import LayerNormalization
from normalize import Normalize
from os import listdir
import zipfile
from PIL import Image
from rwa import _RandomWeightedAverage
from kds import KianaDataSet

K.set_image_data_format('channels_last')




def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

	
	
def d_block1(fil, inp, p = True):
    
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    route2 = LeakyReLU(0.01)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(route2)
    out = LeakyReLU(0.01)(route2)
    
    return out






#noise generating rule requires 1 argument(noise shape) in this class and its type should be partial.
class MyStyleGAN(MyWGAN):
	
	def __init__(self,const_tensor_shape = (4,4,256) , noise_arguments=dict(), **kwargs):
		
		self.const_tensor_shape=const_tensor_shape
		super(MyStyleGAN, self).__init__(**kwargs)
		
		
		self.MN = Model() #mapping network
		self.SN = Model() #synthesis network
		self.AM = Model()
		self.DM = Model()
		self.generate_latent_vector = lambda shape : K.eval(self.noise_generating_rule(shape))
		
		self.positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
		self.negative_y = -self.positive_y
		self.dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)
		
		
		
		
	def construct_mn(self,n_layers,n_neurons, alpha=0.1):
		mn_inp = Input(shape=[self.noise_size])
		mn = Normalize()(mn_inp)
		mn = Dense(n_neurons, kernel_initializer='he_normal')(mn)
		for _ in range(1,n_layers):
			mn = Dense(n_neurons, kernel_initializer = 'he_normal')(mn)
			mn = LeakyReLU(alpha)(mn)
		mn = Model(inputs=mn_inp,outputs=mn)
		return mn
		
		
		
		
		#noise + adain / conv2d / noise + adain / upsample + conv2d /
	def construct_g_block1(self, gout, wlen,n_fils1, n_fils2, u=True,size=(2,2)):
		
		inp = gout[0]
		w = gout[1]
		out = ApplyNoise(noise_generating_rule=self.noise_generating_rule, fils=n_fils1)(inp)
		out = AdaIN(wlen,n_fils1)([out,w])
		out = Conv2D(n_fils1, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(out)
		
		out = ApplyNoise(noise_generating_rule=self.noise_generating_rule, fils=n_fils1)(out)
		out = AdaIN(wlen,n_fils1)([out,w])
						
		if u:
			
			out = UpSampling2D(interpolation='bilinear',size=size,data_format='channels_last')(out)
			
			out = Conv2D(n_fils2, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(out)
		else:
			out = Activation('linear')(inp)
			
		return [out, w]
		
				
				

	def set_G_trainable(self, tr=True):
		self.MN.trainable = tr
		for layer in self.MN.layers:
			layer.trainable = tr
		self.SN.trainable = tr
		for layer in self.SN.layers:
			layer.trainable = tr

	def set_D_trainable(self, tr=True):
		self.D.trainable = tr
		for layer in self.D.layers:
			layer.trainable = tr
			
			
			
	def construct_generator_input(self, pseudo_input_for_const_tensor, latent_vector):
	
	
		w = self.MN(latent_vector)
		
		const_tensor = LearnedConstTensor(self.const_tensor_shape)(pseudo_input_for_const_tensor)
		
		return [const_tensor, w]
		
		
	def compile_generator(self):
		
		self.set_D_trainable(False)
		self.set_G_trainable()
		
		
		inp_sty = Input(shape=[self.noise_size])
		pseudo_input_for_const_tensor= Input(shape=[1])
		
		gminp = [pseudo_input_for_const_tensor, inp_sty]
		
		geneinp = self.construct_generator_input(pseudo_input_for_const_tensor, inp_sty)
		
		gene = self.SN(geneinp)
		self.G = Model(inputs=gminp, outputs=gene)
		gout = self.D(self.G(gminp))
		
		self.AM = Model(inputs=gminp, outputs=gout)
		self.AM.compile(optimizer=self.optimizer, loss=MyWGAN.wasserstein_loss)
		
	
		
	
	def compile_discriminator(self):
		
		self.set_G_trainable(False)
		self.set_D_trainable(True)
		
		real_samples = Input(shape=self.img_shape)
		
		pseudo_input = Input(shape=[1])
		
		latent_vector = Input(shape=[self.noise_size])
		
		
		generator_input_for_discriminator = [pseudo_input, latent_vector]
		
		generated_samples_for_discriminator = self.G(generator_input_for_discriminator)
		
		
		discriminator_output_from_generator = self.D(generated_samples_for_discriminator)
		discriminator_output_from_real_samples =self.D(real_samples)
		
		averaged_samples = _RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
		averaged_samples_out = self.D(averaged_samples)
		partial_gp_loss = partial(MyWGAN.gradient_penalty_loss,averaged_samples=averaged_samples, gradient_penalty_weight=self.gradient_penalty_weight)
		partial_gp_loss.__name__ = 'gradient_penalty'
		
		self.DM = Model(inputs=[real_samples, pseudo_input, latent_vector], outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
		
		self.DM.compile(optimizer=self.optimizer, loss = [MyWGAN.wasserstein_loss, MyWGAN.wasserstein_loss, partial_gp_loss])
		
	def compile(self):
		self.compile_generator()
		self.compile_discriminator()
		
	def generate_samples(self, n_samples):
		
		latent_vector = self.generate_latent_vector([n_samples, self.noise_size])
		samples = self.G.predict([np.ones([n_samples,1]),latent_vector])
		samples = ((samples + 1)*127.5).astype('uint8')
		
		for i in range(n_samples):
			Image.fromarray(samples[i]).save('{}.jpg'.format(i))
		
		
	def train_epoch(self, data, print_term=10, print_samples=0):
		
		np.random.shuffle(data)
		positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
		negative_y = -positive_y
		dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)
		
		discriminator_loss = []
		generator_loss = []
		minibatches_size = self.batch_size*self.n_critic
		iter_per_epoch_g = int(np.shape(data)[0]//minibatches_size)
		for i in range(iter_per_epoch_g):
			discriminator_minibatches = data[i*minibatches_size:(i+1)*minibatches_size]
			for j in range(self.n_critic):
				latent_vector = self.generate_latent_vector([self.batch_size, self.noise_size])
				image_batch = discriminator_minibatches[j*self.batch_size:(j+1)*self.batch_size]
				#for real -1, for generated sample 1.
				discriminator_loss.append(self.DM.train_on_batch([image_batch, positive_y, latent_vector], [negative_y, positive_y, dummy_y]))
			latent_vector = self.generate_latent_vector([self.batch_size, self.noise_size])
			generator_loss.append(self.AM.train_on_batch([positive_y, latent_vector], negative_y))
			if i%print_term == 0:
				print('generator iteration per epoch : ', i+1, '/',iter_per_epoch_g, '\ndiscriminator iteration per epoch : ', (i+1)*self.n_critic, '/', iter_per_epoch_g*self.n_critic)
				print('D loss : ', discriminator_loss[-1])
				print('G loss : ', generator_loss[-1])
			
		return discriminator_loss, generator_loss
		
		
		
		
	def get_mn_model_file_name(self):
		return '{}-mn.json'.format(self.model_file_name)
		
	def get_mn_weight_file_name(self):
		return '{}-mn.h5'.format(self.weight_file_name)

	def get_sn_weight_file_name(self):
		return '{}-sn.h5'.format(self.weight_file_name)

	def get_sn_model_file_name(self):
		return '{}-sn.json'.format(self.model_file_name)
		
	def save_models(self):
		sn_json = self.get_sn_model_file_name()
		mn_json = self.get_mn_model_file_name()
		d_json =self.get_d_model_file_name()

		self.save_model(sn_json, self.SN)
		self.save_model(mn_json, self.MN)
		self.save_model(d_json, self.D)
		print('save complete.')
		
	def load_models(self,
	custom_layers={'LayerNormalization':LayerNormalization, 'ApplyNoise':ApplyNoise,
	'LearnedConstTensor' : LearnedConstTensor, 'Normalize': Normalize, 'AdaIN':AdaIN,
	'_RandomWeightedAverage' : _RandomWeightedAverage}):
		mn_json_file = self.get_mn_model_file_name()
		sn_json_file = self.get_sn_model_file_name()
		d_json_file = self.get_d_model_file_name()

		self.MN = self.load_model(mn_json_file, custom_layers)
		self.SN = self.load_model(sn_json_file, custom_layers)
		self.D = self.load_model(d_json_file, custom_layers)

		self.compile()
		
	def save_weights(self):
		
		sn_w = self.get_sn_weight_file_name()
		mn_w = self.get_mn_weight_file_name()
		d_w = self.get_d_weight_file_name()

		self.SN.save_weights(sn_w)
		self.MN.save_weights(mn_w)
		self.D.save_weights(d_w)
		print('save weights complete.')
		
	def load_weights(self):
		
		mn_w_file = self.get_mn_weight_file_name()
		sn_w_file = self.get_sn_weight_file_name()
		d_w_file = self.get_d_weight_file_name()
		self.load_weight(self.MN, mn_w_file)
		self.load_weight(self.SN, sn_w_file)
		self.load_weight(self.D, d_w_file)
		
	def zip(self, zipname):
		
		zipname = '{}.zip'.format(zipname)
		zf = zipfile.ZipFile(zipname, 'w')
		sf = [self.get_d_weight_file_name(),self.get_d_model_file_name(),
		 self.get_mn_model_file_name(), self.get_mn_weight_file_name(),
		 self.get_sn_model_file_name(), self.get_sn_weight_file_name()]
		for f in sf:
			self.write_zip(zf,f)
		zf.close()
		print('zip complete.')
		
	def load(self):

		self.load_models()
		self.load_weights()
		print('load complete.')
		
		

if __name__=='__main__':
	
	D = Input(shape=(256,256,3))
	ngr = partial(K.random_uniform, minval=-1.0, maxval=1.0)
	ngr.__name__ = 'noise_random_uniform'
	gan1 = MyStyleGAN(img_shape=(256,256,3),optimizer=Adam(lr=0.001, beta_1 = 0, beta_2=0.99), noise_size=256,
	noise_generating_rule= ngr)
	gan1.MN = gan1.construct_mn(8,256)
	lconst_tensor = Input(shape=(4,4,256))
	in_sty = Input(shape=[256])
	G = gan1.construct_g_block1([lconst_tensor, in_sty],256,256,256)#8
	G = gan1.construct_g_block1(G, 256,256,192)#16
	G = gan1.construct_g_block1(G,256,192,128)#32
	G = gan1.construct_g_block1(G,256,128,64)#64
	G = gan1.construct_g_block1(G,256,64,48)#128
	G = gan1.construct_g_block1(G,256,48,32)#256
	G = Conv2D(3,1,padding='same', kernel_initializer='he_normal')(G[0])
	G = Model(inputs=[lconst_tensor, in_sty], outputs=G)
	
	D_input = Input(shape=(256,256,3))
	D = Conv2D(32, kernel_size=3 , padding='same',kernel_initializer = 'he_normal')(D_input)
	D = LeakyReLU(0.1)(D)
	D = d_block1(64,D)
	D = d_block1(128, D)
	D = d_block1(256, D)
	D = d_block1(256, D)
	D = Flatten()(D)
	D = Dense(1)(D)
	D = Model(inputs=D_input, outputs=D)
	
	gan1.SN = G
	gan1.D = D
	
	gan1.compile()
	kds1 = KianaDataSet(folder='kfcp256',load_from_zip=True)
	gan1.generate_samples(20)
	gan1.train(kds1.normalized, 2, 0, 1, False)