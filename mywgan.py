
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


import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np


class MyWGAN:
	
	def __init__(self, img_shape=(256,256,3), noise_size=100, batch_size = 64, n_ciritic=5, gradient_penalty_weight = 10, optimizer = RMSprop(lr=0.00005), noise_generating_rule = (lambda batchsize, noisesize : np.random.uniform(-1.0, 1.0, size = [batchsize, noisesize]))):
		
		self.img_shape = img_shape
		self.img_rows = img_shape[0]
		self.img_cols = img_shape[1]
		self.channel = img_shape[2]
		self.noise_size = noise_size
		self.batch_size = batch_size
		self.gradient_penalty_weight = gradient_penalty_weight
		self.n_critic = n_ciritic
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
		
		
	def compile_model(self, D, G):
		
		self.D = D
		self.G = G
		
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
		generator_input_for_discriminator = Input(shape=(100,))
		generated_samples_for_discriminator = self.G(generator_input_for_discriminator)
		discriminator_output_from_generator = self.D(generated_samples_for_discriminator)
		discriminator_output_from_real_samples =self.D(real_samples)
		
		averaged_samples = RandomWeightedAverage(self.batch_size)([real_samples,generated_samples_for_discriminator])
		averaged_samples_out = self.D(averaged_samples)
		partial_gp_loss = partial(MyWGAN.gradient_penalty_loss,averaged_samples=averaged_samples, gradient_penalty_weight=self.gradient_penalty_weight)
		partial_gp_loss.__name__ = 'gradient_penalty'
		
		self.discriminator_model = Model(inputs=[real_samples,generator_input_for_discriminator], outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
		self.discriminator_model.compile(optimizer=self.optimizer, loss=[MyWGAN.wasserstein_loss, MyWGAN.wasserstein_loss, partial_gp_loss])
		
	def train(self, data, print_samples=0):
		
		discriminator_loss, generator_loss = self.train_epoch(data, print_samples=print_samples)
		
		print('average G loss : ',  np.sum(generator_loss)/self.batch_size)
		print('average D loss : ', np.sum(discriminator_loss)/(self.batch_size*self.n_critic))
		
	def train_epoch(self, data, print_term=10, print_samples=0):
		
		np.random.shuffle(data)
		positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
		negative_y = -positive_y
		dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)
		
		discriminator_loss = []
		generator_loss = []
		minibatches_size = self.batch_size*self.n_critic
		for i in range(int(np.shape(data)[0]//minibatches_size)):
			discriminator_minibatches = data[i*minibatches_size:(i+1)*minibatches_size]
			for j in range(self.n_critic):
				image_batch = discriminator_minibatches[j*self.batch_size:(j+1)*self.batch_size]
				noise = self.noise_generating_rule(self.batch_size, self.noise_size)
				discriminator_loss.append(self.discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y]))
			generator_loss.append(self.generator_model.train_on_batch(self.noise_generating_rule(self.batch_size, self.noise_size), positive_y))
			
		return discriminator_loss, generator_loss
                                            
		
		
		
		
class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""
    
    def __init__(self, batchsize):
    	super(_Merge)
    	self.batch_size = batchsize

    def _merge_function(self, inputs):
        weights = K.random_uniform((self.batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
        
    