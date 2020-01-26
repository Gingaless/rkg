import numpy as np
import os
from functools import partial
from keras.models import Model, Sequential, save_model
from keras.layers import Input, Dense, Reshape, Flatten, add,Activation, Lambda
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D, UpSampling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.engine.input_layer import InputLayer
from keras.optimizers import RMSprop
from keras import backend as K
from mywgan import MyWGAN
import mywgan
from mywgan import RandomWeightedAverage
from mywgan import gradient_penalty_loss
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

def cbl_block1(input, channel, filter_size,stride,alpha=0.2):
	out = Conv2D(channel, filter_size, strides = stride, padding = 'same', kernel_initializer='he_normal')(input)
	out = BatchNormalization()(out)
	out = LeakyReLU(alpha)(out)
	return out
	
def dbr_block1(input, channel, filter_size, stride):
	out = Conv2DTranspose(channel, filter_size, strides=stride, padding='same', kernel_initializer='he_normal')(input)
	out = BatchNormalization()(out)
	out = Activation('relu')(out)
	return out



class MyProGAN:
	
	
	def __init__(self, 
	input_args_for_G, 
	input_args_for_DM, 
	h_size_growth = [8,16,32,64,128,256],
	w_size_growth = [8,16,32,64,128,256],
	img_channel = 3,
	batch_size = 64,
	iter_ratio = 1,
	dirname = 'myprogan1',
	optimizer = Adam(lr=0.0002, beta_1 = 0, beta_2 = 0.9),
	DM_loss = ['mse', 'mse'],
	AM_loss = 'mse',
	gradient_penalty_weight = 0,
	valid = 1.0, fake = 0.0,
	valid_for_fake = 1.0,
	custom_layers = {
	'LayerNormalization':LayerNormalization, 
	'ApplyNoise':ApplyNoise,
	'LearnedConstTensor' : LearnedConstTensor, 
	'Normalize': Normalize, 
	'AdaIN':AdaIN,
	'_RandomWeightedAverage' : _RandomWeightedAverage
	}
	):
		assert len(h_size_growth) == len(w_size_growth)
		self.img_data = dict()
		self.img_folder_path = None
		self.model_folder_path = None
		self.input_args_for_G = input_args_for_G
		self.input_args_for_DM = input_args_for_DM
		self.h_size_growth = h_size_growth
		self.w_size_growth = w_size_growth
		self.n_growth = len(self.h_size_growth)
		self.img_channel = img_channel
		self.img_shape = [(self.h_size_growth[i], self.w_size_growth[i], self.img_channel) for i in range(self.n_growth)]
		self.batch_size = batch_size
		self.iter_ratio = iter_ratio
		self.dirname = dirname
		self.optimizer = optimizer
		self.DM_loss = DM_loss
		self.AM_loss = AM_loss
		self.gradient_penalty_weight = gradient_penalty_weight
		self.valid = valid*np.ones((self.batch_size,1), dtype=np.float)
		self.fake = fake*np.ones((self.batch_size,1), dtype=np.float)
		self.valid_for_fake = valid_for_fake*np.ones((self.batch_size,1), dtype=np.float)
		self.custom_layers = custom_layers
		self.generators =  [Model() for i in range(self.n_growth)]
		self.discriminators = [Model() for i in range(self.n_growth)]
		self.AM = Model()
		self.DM = Model()
		
	def generate_inputs_for_G(self, latent_size, noise_function,clip=None,num=None):
		noi = None
		if not num:
			noi = noise_function((self.batch_size, latent_size))
		else:
			noi = noise_function((num, latent_size))
		if clip:
			noi = np.clip(noi, *clip)
		return noi
		
	def generate_inputs_for_DM(self, i, **kwargs):
		
		np.random.shuffle(self.img_data)
		real = self.img_data[i]
		fake = self.generators[i].predict(self.generate_inputs_for_G(**kwargs))
		return [real,fake]
		
	def mk_output_layers_for_G(self, idx):
		
		lc = Conv2D(3, 1, padding = 'same', kernel_initializer = 'he_normal', name = 'last_conv2d_G')(self.generators[idx])
		actanh = Activation('tanh', name = 'tanh_G')(lc)
		return actanh
		
	def mk_output_layers_for_D(self, idx,alpha=0.1):
		last_channels = K.int_shape(self.discriminators[idx])[-1]
		lc = Conv2D(last_channels,5,strides=2,padding='same', kernel_initializer='he_normal',name = 'last_conv2d_D')(self.discriminators[idx])
		relu = LeakyReLU(alpha, name='last_lrelu_D')(lc)
		fl = Flatten(name = 'last_flatten')(relu)
		out = Dense(128,name='second_last_Dense')(fl)
		out = Dense(1,name = 'last_D')(out)
		out = Activation('linear', name = 'last_D_Act')(out)
		return out
	
			
	def compile_AM(self, idx):
		G = self.mk_generator(idx)
		D = self.mk_discriminator(idx)
		D = Model(outputs=D)
		self.AM = Model(D(G))
		self.AM.compile(optimizer=self.optimizer, loss = self.AM_loss)
	
			
	def compile_DM(self,idx):
		D = self.mk_discriminator(idx)
		self.DM = Model(outputs=D)
		self.DM.compile(optimizer=self.optimizer, loss = self.DM_loss)
		
		
	def compile(self,idx):
		self.compile_AM(idx)
		self.compile_DM(idx)
		
		
	def chain_generators(self, idx):
		return Model(inputs=self.generators[0], outputs=self.generators[idx])(self.chain_generators(idx-1)) if idx>0 else self.generators[idx]
		
		
	def chain_discriminators(self,idx):
		return self.discriminators[idx](self.chain_discriminators(idx-1)) if idx>0 else self.discriminators[idx]
			
			
	def mk_generator(self,idx):
		self.chain_generators(idx)
		return self.mk_output_layers_for_G(idx)
		
		
	def mk_discriminator(self,idx):
		self.chain_discriminators(idx)
		return self.mk_output_layers_for_D(idx)
		
		
	
	def load_img_data(self, img_folder_path, idx,unzip=True):
		
		if unzip:
			zipfilename = '{}.zip'.format(img_folder_path)
			assert os.path.exists(zipfilename)
			with zipfile.ZipFile(zipfilename, 'r') as zf:
				print('start to extract...')
				zf.extractall(img_folder_path)
				print('extract all!')
				
		self.img_folder_path = img_folder_path
		assert os.path.exists(os.path.join(img_folder_path, str(idx)))
		path = os.path.join(img_folder_path, str(idx))
		files = [Image.open(os.path.join(path,f)) for f in listdir(path)]
		img_arr = np.zeros((len(files),) + tuple(self.img_shape[idx]))
		for i, f in enumerate(files):
			img_arr[i,:,:,:] = f
		self.img_data[idx] = img_arr
		
		
	def train_dis(self, idx, real_samples):
		samples = self.generate_inputs_for_DM(idx, **self.input_args_for_DM)
		loss = 0
		loss += self.DM.train_on_batch(samples, self.valid)
		return loss
		
	def train_gen(self, idx):
		return self.AM.train_on_batch(self.generate_inputs_for_G(**self.input_args_for_G))
		
		
		
	def train_epoch(self,idx, print_term):
		
		r_iter = 0
		np.random.shuffle(self.img_data)
		minibatch_size = int(self.batch_size*self.iter_ratio)
		iter_per_epoch_g = int(np.shape(self.img_data)[0] // minibatch_size)
		total_iter = iter_per_epoch_g*int(self.iter_ratio+1)
		total_d_loss = []
		total_g_loss = []
		iter_g_loss = []
		iter_d_loss = []
		
		for i in range(iter_per_epoch_g):
			discriminator_minibatch = self.img_data[i*minibatch_size:(i+1)*minibatch_size]
			for j in range(self.iter_ratio):
				total_d_loss.append(self.train_dis(idx, self.img_data[i]))
				iter_g_loss.append(total_d_loss[-1])
				r_iter+=1
				if r_iter%print_term==0:
					print(r_iter, ' / ' , total_iter, 'iter per epoch : ' , 'average D loss - ', np.mean(iter_d_loss, axis=0), ', average G loss - ', np.mean(iter_g_loss, axis = 0))
					iter_d_loss = []
					iter_g_loss = []
			self.train_gen(idx)
			r_iter+=1
			if r_iter%print_term==0:
				print(r_iter, ' / ' , total_iter, 'iter per epoch : ' , 'average D loss - ', np.mean(iter_d_loss, axis=0), ', average G loss - ', np.mean(iter_g_loss, axis = 0))
				iter_d_loss = []
				iter_g_loss = []
		return total_d_loss, total_g_loss
		
		
		
	def train(self, idx, epoch, print_term=1):
		
		
		for i in range(epoch):
			(tt_d_loss, tt_g_loss) = self.train_epoch(idx, print_term)
			print(i, 'th epoch / ', epoch, ' epoches : ')
			print('average G loss = ' ,np.mean(tt_g_loss,axis=0))
			print('average D loss = ', np.mean(tt_d_loss, axis=0))
			
			
	def generate_samples(self,idx, num=None):
		
		return self.generators[idx].predict(self.generate_inputs_for_G(**dict(list(self.input_args_for_G.items()) + list({'num' : num}.items()))))





		
if __name__ == "__main__":
	
	D0 = Input(shape=[8,8,3])
	D1 = cbl_block1(D0, 32, 5, 2)
	G0 = Input(shape=[1024])
	G1 = Dense(4*4*512)(G0)
	G1 = Reshape([4,4,512])(G1)
	G1 = dbr_block1(G1, 512, 5, 2)
	kwargs = {'latent_size' : [1024], 'noise_function' : (lambda shape : np.random.normal(0, 1, size=shape))}
	gan1 = MyProGAN(kwargs, kwargs)
	gan1.generators[0] = G0
	gan1.discriminators[0] = D0
	gan1.generators[1] = G1
	gan1.discriminators[1] = D1
	gan1.compile(1)