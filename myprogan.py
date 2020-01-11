import numpy as np
import os
from functools import partial




from keras.models import Model, Sequential, Graph, save_model
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
        self.custom_layers = self.custom_layers
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
    	
    def output_layers_for_G(self, idx):
    	lc = Conv2D(3, 1, padding = 'same', kernel_initializer = 'he_normal', name = 'last_conv2d_G')(self.generators[idx])
    	actanh = Activation('tanh', name = 'tanh_G')(lc)
    	return actanh
    	
    def output_layers_for_D(self, idx,alpha=0.1):
    	last_channels = K.int_shape(self.discriminators[idx])[-1]
    	lc = Conv2D(last_channels,5,2,padding='same', kernel_initializer='he_normal',name = 'last_conv2d_D')(self.discriminators[idx])
    	relu = LeakyReLU(alpha, name='last_lrelu_D')(lc)
    	fl = Flatten(name = 'last_flatten')(relu)
    	out = Dense(1,name = 'last_D')(fl)
    	
    	return out
    	
    def compile_AM(self, idx):
    	G = self.mk_generator(idx)
    	D = self.mk_discriminator(idx)
    	out = D(G)
    	self.AM = Model(outputs=out)
    	self.AM.compile(optimizer=self.optimizer, loss = self.AM_loss)
    	
    def compile_DM(self,idx):
    	D = self.mk_discriminator(idx)
    	self.DM = Model(outputs=D)
    	self.DM.compile(optimizer=self.optimizer, loss = self.DM_loss)
    	
    def compile(self,idx):
    	self.compile_AM(idx)
    	self.compile_DM(idx)
    	
    def chain_generators(self, idx):
    	return self.generators[idx](self.chain_generators(idx-1)) if idx>0 else self.generators[idx]
    	
    def chain_discriminators(self,idx):
    	return self.discriminators[idx](self.chain_discriminators(idx-1)) if idx>0 else self.discriminators[idx]
    	
    def mk_generator(self,idx):
    	return self.output_layers_for_G(self.chain_generators(idx))
    	
    def mk_discriminator(self,idx):
    	return self.output_layers_for_D(self.chain_discriminators(idx))
    	
    	
    	
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
    	loss = 0
    	loss += self.DM.train_on_batch(real_samples, self.valid)
    	fakes = self.generate_samples(idx)
    	loss += self.DM.train_on_batch(fakes, self.fake)
    	return loss
   
    def train_gen(idx):
    	return self.AM.train_on_batch(self.generate_inputs_for_G(**self.input_args_for_G))
    	
    def generate_samples(self,idx, num=None):
    	return self.generators[idx].predict(self.generate_inputs_for_G(**dict(list(self.input_args_for_G.items()) + list({'num' : num}.items())))
    	
    	
    	
    	
    		
    	
    	
    	
    	


