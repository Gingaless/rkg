import numpy as np
from functools import partial




from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, add,Activation
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






#noise generating rule requires 1 argument(noise shape) in this class.
class MyStyleGAN(MyWGAN):
	
	def __init__(self,const_tensor_shape = (4,4,256) ,**kwargs):
		
		self.const_tensor_shape=const_tensor_shape
		super(MyStyleGAN, self).__init__(**kwargs)
		
		
		self.MN = Model() #mapping network
		self.AM = Model()
		self.DM = Model()
		
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
		self.G.trainable = tr
		for layer in self.G.layers:
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
		
		gene = self.G(geneinp)
		gout = self.D(gene)
		
		self.AM = Model(inputs=gminp, outputs=gout)
		self.AM.compile(optimizer=self.optimizer, loss=MyWGAN.wasserstein_loss)
		
	
		
	
	def compile_discriminator(self):
		
		for layer in self.MN.layers:
			layer.trainable = True
		self.MN.trainable = True
		for layer in self.G.layers:
			layer.trainable = False
		self.G.trainable = False
		for layer in self.D.layers:
			layer.trainable = True
		self.D.trainable = True
		
		real_samples = Input(shape=self.img_shape)
		
		pseudo_input = Input(shape=[1])
		
		latent_vector = Input(shape=[self.noise_size])
		
		
		generator_input_for_discriminator = self.construct_generator_input(pseudo_input, latent_vector)
		
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
		
		samples = self.G.predict(np.ones(n_samples))
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
				latent_vector = self.noise_generating_rule([self.batch_size, self.noise_size])
				image_batch = discriminator_minibatches[j*self.batch_size:(j+1)*self.batch_size]
				discriminator_loss.append(self.DM.train_on_batch([image_batch, positive_y, latent_vector], [positive_y, negative_y, dummy_y]))
			latent_vector = self.noise_generating_rule([self.batch_size, self.noise_size])
			generator_loss.append(self.AM.train_on_batch([positive_y, latent_vector], positive_y))
			if i%print_term == 0:
				print('generator iteration per epoch : ', i+1, '/',iter_per_epoch_g, '\ndiscriminator iteration per epoch : ', (i+1)*self.n_critic, '/', iter_per_epoch_g*self.n_critic)
				print('D loss : ', discriminator_loss[-1])
				print('G loss : ', generator_loss[-1])
			
		return discriminator_loss, generator_loss
		
		
		
		
	def get_mn_model_file_name(self):
		return '{}-mn.json'.format(self.model_file_name)
		
	def get_mn_weight_file_name(self):
		return '{}-mn.h5'.format(self.weight_file_name)
		
	def save_models(self):
		mn_json = self.MN.to_json() 
		with open(self.get_mn_model_file_name(), "w") as json_file:
			json_file.write(mn_json)
			json_file.close()
			
		super(MyStyleGAN, self).save_models()
		
	def load_models(self,
	custom_layers={'LayerNormalization':LayerNormalization, 'ApplyNoise':ApplyNoise,
	'LearnedConstTensor' : LearnedConstTensor, 'Normalize': Normalize, 'AdaIN':AdaIN,
	'_RandomWeightedAverage' : _RandomWeightedAverage}):
		mn_json_file = open(self.get_mn_model_file_name(), "r")
		mn_model = mn_json_file.read()
		mn_json_file.close()
		with CustomObjectScope(custom_layers):
			self.MN = model_from_json(mn_model)
		
		super(MyStyleGAN, self).load_models(custom_layers)
		
	def save_weights(self):
		
		self.MN.save_weights(self.get_mn_weight_file_name())
		
		super(MyStyleGAN, self).save_weights()
		
	def load_weights(self):
		
		self.MN.load_weights(self.get_mn_weight_file_name())
		
		super(MyStyleGAN, self).load_weights()
		
	def zip(self, zipname):
		
		zipname = '{}.zip'.format(zipname)
		mw_zip = zipfile.ZipFile(zipname, 'w')
		mw_zip.write(self.get_mn_model_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_mn_weight_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_d_model_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_g_model_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_d_weight_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_g_weight_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.close()
		print('zip complete.')
		
	def load(self):
		if self.get_d_model_file_name() in listdir() and self.get_g_model_file_name() in listdir() and self.get_mn_model_file_name() in listdir():
			self.load_models()
		else:
			print('there exist no model files.')
		
		self.compile()
		
		if self.get_d_weight_file_name() in listdir() and self.get_g_weight_file_name() in listdir() and self.get_mn_weight_file_name() in listdir():
			self.load_weights()
		else:
			print('there exist no weight files.')
		print('load complete.')
		
		

if __name__=='__main__':
	
	D = Input(shape=(256,256,3))
	
	gan1 = MyStyleGAN(img_shape=(256,256,3),optimizer=Adam(lr=0.001, beta_1 = 0, beta_2=0.99), noise_size=256,
	noise_generating_rule= (lambda shape : K.random_uniform(shape, -1.0,1.0)))
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
	
	gan1.G = G
	gan1.D = D
	
	gan1.compile()
	kds1 = KianaDataSet(folder='kfcp256',load_from_zip=True)
	gan1.train(kds1.normalized, 2, 0, 1, False)