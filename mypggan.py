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
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from pixnorm import PixelNormalization
from minibatchstdev import MiniBatchStandardDeviation
from weightedsum import WeightedSum




class MyPGGAN(object):
	
	init = RandomNormal(stddev=0.02)
	const = max_norm(1.0)
	kernel_cond = {'kernel_initializer' : init, 'kernel_constraint' : const}
	
	
	def __init__(self,
	latent_size = 1024,
	heights = [8,16,32,64,128,256],
	widths = [8,16,32,64,128,256]):
		self.generators = []
		self.discriminators =[]
		self.latent_size = latent_size
		self.channels = 3
		self.heights = heights
		self.widths = widths
		self.img_shape = [(heights[i], widths[i], self.channels) 
		for i in range(len(heights))]
		self.training_block = 0
		self.G = Model()
		self.D = Model()
		self.DM = Model()
		self.AM = Model()
		
		
	def mk_input_layers_for_G(self, depth=128):
		
		in_latent = Input(shape=self.latent_size)
		g = Dense(np.prod(self.img_shape[0]), 
		kernel_initializer = MyPGGAN.init, 
		kernel_constraint = MyPGGAN.const)(in_latent)
		g = Reshape(self.img_shape[0][0:2] + (self.channels,))(g)
		g = Conv2D(depth, 4, padding='same',**MyPGGAN.kernel_cond)(g)
		g = PixelNormalization()(g)
		g = LeakyReLU(0.2)(g)
		g = Conv2D(depth, 3,padding='same', **MyPGGAN.kernel_cond)(g)
		g = PixelNormalization()(g)
		g = LeakyReLU(0.2)(g)
		return Model(inputs = in_latent, outputs = g, name = 'input_layers_for_G')
		
		
	def mk_G_block(self, step, depth=128,scale=2):
		
		assert step>0
		
		block_end = Input(shape=K.int_shape(self.generators[step-1].output)[1:])
		upsampling = UpSampling2D(scale)(block_end)
		g = Conv2D(depth, 3, padding='same', **MyPGGAN.kernel_cond)(upsampling)
		g = PixelNormalization()(g)
		g = LeakyReLU(0.2)(g)
		
		return Model(block_end, g, name='g_chain_' + str(step))
		
		
	def mk_output_layers_for_G(self, step):
		
		inp = Input(shape=self.generators[step].output_shape)
		out = Conv2D(3,1,**MyPGGAN.kernel_cond)(inp)
		out = Activation('linear')(out)
		
		return Model(inp, out, name='output_layer_' + str(step) + '_for_G')
	
			
	def mk_input_layers_for_D(self,step,depth):
		
		inp = Input(shape=self.img_shape[step])
		d = Conv2D(depth, 1, **MyPGGAN.kernel_cond)(inp)
		d = LeakyReLU(0.2)(d)
		d = MiniBatchStandardDeviation()(d)
		d = Conv2D(depth, 3, padding='same', **MyPGGAN.kernel_cond)(d)
		d = LeakyReLU(0.2)(d)
		d = Conv2D(depth, 4, padding='same', **MyPGGAN.kernel_cond)(d)
		d = LeakyReLU(0.2)(d)
		
		return Model(inputs=inp, outputs=d, name='input_layers_' + str(step) + '_for_D')
		
		
			
		