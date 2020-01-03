
from __future__ import print_function, division

import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from functools import partial
from keras.models import model_from_json
from os import listdir
import zipfile
from keras_layer_normalization import LayerNormalization
from keras.utils import CustomObjectScope


import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

K.set_image_data_format('channels_last')


class MyWGAN:
	
	def __init__(self, img_shape=(256,256,3), noise_size=100, batch_size = 64, n_ciritic=5, gradient_penalty_weight = 10, optimizer = RMSprop(lr=0.00005), noise_generating_rule = (lambda batchsize, noisesize : np.random.uniform(-1.0, 1.0, size = [batchsize, noisesize])), weight_file_name = 'mywgan1', model_file_name = 'mywgan1'):
		
		self.img_shape = img_shape
		self.img_rows = img_shape[0]
		self.img_cols = img_shape[1]
		self.channel = img_shape[2]
		self.noise_size = noise_size
		self.batch_size = batch_size
		self.gradient_penalty_weight = gradient_penalty_weight
		self.n_critic = n_ciritic
		self.weight_file_name = weight_file_name
		self.model_file_name = model_file_name
		self.optimizer = optimizer
		self.G = Sequential()
		self.D = Sequential()
		self.generator_model = Model()
		self.discriminator_model = Model()
		self.noise_generating_rule = noise_generating_rule
		
		
	def wasserstein_loss(y_true, y_pred):
			return K.mean(y_true*y_pred)
			
	def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
		gradients = K.gradients(y_pred, averaged_samples)[0]
		gradients_sqr = K.square(gradients)
		gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
		return K.mean(gradient_penalty)
		
	
	def compile_model(self,D, G):
		self.D = D
		self.G = G
		self.compile()
				
	def compile(self):
		
		#build generator model
		
		for layer in self.D.layers:
			layer.trainable = False
		self.D.trainable = False
		generator_input = Input(shape=(self.noise_size,))
		generator_layers = self.G(generator_input)
		discriminator_layers_for_generator = self.D(generator_layers)
		self.generator_model = Model(inputs=[generator_input], outputs= [discriminator_layers_for_generator])
		self.generator_model.compile(optimizer=self.optimizer, loss = MyWGAN.wasserstein_loss)
		
		#build discriminator model
		for layer in self.G.layers:
			layer.trainabe = False
		for layer in self.D.layers:
			layer.trainable = True
		self.G.trainable = False
		self.D.trainable = True
		
		real_samples = Input(shape=self.img_shape)
		
		generator_input_for_discriminator = Input(shape=(self.noise_size,))
		
		generated_samples_for_discriminator = self.G(generator_input_for_discriminator)
		discriminator_output_from_generator = self.D(generated_samples_for_discriminator)
		discriminator_output_from_real_samples =self.D(real_samples)
		
		averaged_samples = RandomWeightedAverage(self.batch_size)([real_samples, generated_samples_for_discriminator])
		averaged_samples_out = self.D(averaged_samples)
		partial_gp_loss = partial(MyWGAN.gradient_penalty_loss,averaged_samples=averaged_samples, gradient_penalty_weight=self.gradient_penalty_weight)
		partial_gp_loss.__name__ = 'gradient_penalty'
		
		self.discriminator_model = Model(inputs=[real_samples,generator_input_for_discriminator], outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
		self.discriminator_model.compile(optimizer=self.optimizer, loss=[MyWGAN.wasserstein_loss, MyWGAN.wasserstein_loss, partial_gp_loss])
		
	def train(self, data, epoches, print_samples=0, print_term=10, saving=True):
		
		for i in range(epoches):
			print(i, 'th epoch train start.')
			discriminator_loss, generator_loss = self.train_epoch(data, print_samples=print_samples, print_term=print_term)
			
			discriminator_loss = abs(np.array(discriminator_loss))
			generator_loss = abs(np.array(generator_loss))
			
			print('\nepoch : ', i+1, '/', epoches)
			print('average abs G loss : ',  np.sum(abs(generator_loss))/len(generator_loss))
			print('average abs D loss : ', np.sum(abs(discriminator_loss),axis=0)/len(discriminator_loss))
			print()
			
		if saving:
			self.save_weights()
		
		
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
				image_batch = discriminator_minibatches[j*self.batch_size:(j+1)*self.batch_size]
				noise = self.noise_generating_rule(self.batch_size, self.noise_size)
				#for real -1, for fake 1.
				discriminator_loss.append(self.discriminator_model.train_on_batch([image_batch, noise], [negative_y, positive_y, dummy_y]))
			generator_loss.append(self.generator_model.train_on_batch(self.noise_generating_rule(self.batch_size, self.noise_size), negative_y))
			if i%print_term == 0:
				print('generator iteration per epoch : ', i+1, '/',iter_per_epoch_g, '\ndiscriminator iteration per epoch : ', (i+1)*self.n_critic, '/', iter_per_epoch_g*self.n_critic)
				print('D loss : ', discriminator_loss[-1])
				print('G loss : ', generator_loss[-1])
			
		return discriminator_loss, generator_loss
	
			
	def get_d_weight_file_name(self):
		return '{}-d.h5'.format(self.weight_file_name)
		
	def get_g_weight_file_name(self):
		return '{}-g.h5'.format(self.weight_file_name)
		
	def get_g_model_file_name(self):
		return '{}-g.json'.format(self.model_file_name)
		
	def get_d_model_file_name(self):
		return '{}-d.json'.format(self.model_file_name)

	def save_model(self, file_name, model):
		model_json = model.to_json()
		with open(file_name, "w") as json_file:
			json_file.write(model_json)
			json_file.close()
		print('save model ', file_name, ' complete.')

		
	def save_models(self):
		
		d_json = self.D.to_json() 
		with open(self.get_d_model_file_name(), "w") as json_file:
			json_file.write(d_json)
			json_file.close()
			
		g_json = self.G.to_json() 
		with open(self.get_g_model_file_name(), "w") as json_file:
			json_file.write(g_json)
			json_file.close()
		print('save models complete.')

	def load_model(self, file_name, custom_layers):
		assert file_name in listdir()
		model = None
		json_file = open(file_name, "r")
		read_model = json_file.read()
		json_file.close()
		with CustomObjectScope(custom_layers):
			model = model_from_json(read_model)
		print("load model ", json_file, " complete")
		return model
			
	def load_models(self,custom_layers={'LayerNormalization': LayerNormalization}):
		d_json_file = open(self.get_d_model_file_name(), "r")
		g_json_file = open(self.get_g_model_file_name(),"r")
		d_model = d_json_file.read()
		g_model = g_json_file.read()
		d_json_file.close()
		g_json_file.close()
		with CustomObjectScope(custom_layers):
			self.D = model_from_json(d_model)
			self.G = model_from_json(g_model)
		print("load models complete.")
		
	def save_weights(self):
		self.D.save_weights(self.get_d_weight_file_name())
		self.G.save_weights(self.get_g_weight_file_name())
		print('save weights complete.')
		
	def load_weights(self):
		self.D.load_weights(self.get_d_weight_file_name())
		self.G.load_weights(self.get_g_weight_file_name())
		print('load weights complete.')

	def load_weight(self, model, file_name):
		assert file_name in listdir()
		model.load_weights(file_name)
		print('load model ', file_name, 'complete')

	def save(self):
		self.save_models()
		self.save_weights()
		print('save complete.')

	def write_zip(self, zip_file, write_file):
		zip_file.write(write_file, compress_type=zipfile.ZIP_DEFLATED)
		print('write the content of ', write_file, ' in the zip file.')

		
	def zip(self,zipname):
		zipname = '{}.zip'.format(zipname)
		mw_zip = zipfile.ZipFile(zipname, 'w')
		mw_zip.write(self.get_d_model_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_g_model_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_d_weight_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.write(self.get_g_weight_file_name(), compress_type=zipfile.ZIP_DEFLATED)
		mw_zip.close()
		print('zip complete.')
		
	def save_and_zip(self, zipname):
		self.save()
		self.zip(zipname)
		print('save and zip complete')
		
	def load(self):
		if self.get_d_model_file_name() in listdir() and self.get_g_model_file_name() in listdir():
			self.load_models()
		else:
			print('there exist no model files.')
		
		self.compile()
		
		if self.get_d_weight_file_name() in listdir() and self.get_g_weight_file_name() in listdir():
			self.load_weights()
		else:
			print('there exist no weight files.')
		print('load complete.')
		
	def unzip(self, zipname):
		zipname = '{}.zip'.format(zipname)
		if zipname in listdir():
			mw_zip = zipfile.ZipFile(zipname, 'r')
			mw_zip.extractall()
			print('unzip complete')
		else:
			print('there exist no zip file.')
			
	def unzip_and_load(self,zipname):
		self.unzip(zipname)
		self.load()
		print('unzip and load complete.')
                                            
		
		
		
		
class RandomWeightedAverage(_Merge):
    
    def __init__(self, batch_size):
    	super(RandomWeightedAverage, self).__init__()
    	self.batch_size=batch_size

    def _merge_function(self, inputs):
    	batch_size = K.shape(inputs[0])[0]
    	weights = K.random_uniform((batch_size, 1, 1, 1))
    	return (weights * inputs[0]) + ((1 - weights) * inputs[1])
        
    