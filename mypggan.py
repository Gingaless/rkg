import numpy as np
import os
from functools import partial
from keras.models import Model, Sequential, save_model
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Activation, Layer
from keras.layers.convolutional import Conv2D, UpSampling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
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
import manage_data
from manage_data import save_model, load_model, zip, unzip, load_image_batch, generate_sample_image




class MyPGGAN(object):
	
	init = RandomNormal(stddev=0.02)
	const = max_norm(1.0)
	kernel_cond = {'kernel_initializer' : init, 'kernel_constraint' : const}
	
	
	def __init__(self,
	latent_size = 1024,
	heights = [8,16,32,64,128,256],
	widths = [8,16,32,64,128,256],
	AM_loss = 'mse', DM_loss = 'mse',
	AM_optimizer = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
	DM_optimizer = Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
	custom_layers = 
	{'PixelNormalization' : PixelNormalization,
	'MiniBatchStandardDeviation' : MiniBatchStandardDeviation,
	'WeightedSum' : WeightedSum},
	model_info_dir = 'mypggan1',
	img_src = 'kfcp256fp',
	noise_func = lambda num, size : np.clip(np.random.normal(1,0.5,(num,size)),-2.0,2.0)):
		
		assert len(heights)==len(widths)
		self.latent_size = latent_size
		self.channels = 3
		self.heights = heights
		self.widths = widths
		self.num_steps = len(self.heights)
		self.img_shape = [(heights[i], widths[i], self.channels) 
		for i in range(self.num_steps)]
		self.model_info_dir = model_info_dir
		self.img_src = img_src
		self.generators = np.empty(self.num_steps, dtype=Model)
		self.discriminators = np.empty(self.num_steps, dtype=Model)
		self.training_block = 0
		self.AM_optimizer = AM_optimizer
		self.DM_optimizer = DM_optimizer
		self.AM_loss = AM_loss
		self.DM_loss = DM_loss
		self.G = Model()
		self.D = Model()
		self.DM = Model()
		self.AM = Model()
		self.custom_layers = custom_layers
		self.noise_func = noise_func
		
		
	def mk_input_layers_for_G(self, step, depth=128):
		
		in_latent = Input(shape=[self.latent_size])
		g = Dense(np.prod(self.img_shape[0][0:2] + (depth,)), 
		kernel_initializer = MyPGGAN.init, 
		kernel_constraint = MyPGGAN.const)(in_latent)
		g = Reshape(self.img_shape[0][0:2] + (depth,))(g)

		return Model(inputs = in_latent, outputs = g, name = 'input_layers_' + str(step) + '_for_G')
		
		
	def mk_G_block(self, step, depth=128,scale=2):

		inp = Input(shape=self.img_shape[0][:2] + (depth,))
		g = inp
		if step>0:
			block_end = Input(shape=K.int_shape(self.generators[step-1].output)[1:])
			inp = block_end
			upsampling = UpSampling2D(scale)(block_end)
			g = upsampling
		g = Conv2D(depth, 4, padding='same',**MyPGGAN.kernel_cond)(g)
		g = PixelNormalization()(g)
		g = LeakyReLU(0.2)(g)
		g = Conv2D(depth, 3, padding='same', **MyPGGAN.kernel_cond)(g)
		g = PixelNormalization()(g)
		g = LeakyReLU(0.2)(g)
		
		return Model(inp, g, name='G_chain_' + str(step))



	#output layer model 뒤에 붇이면 댐
	def mk_merge_layers_for_G(self,step,old_output_layers, scale=2):
		pv_block_end = Input(shape=self.generators[step-1].output_shape[1:])
		old_block_end = old_output_layers(pv_block_end)
		new_image = Input(shape=self.img_shape[step])
		old_img_upsampling = UpSampling2D(scale)(old_block_end)
		merged = WeightedSum()([old_img_upsampling, new_image])
		return Model(inputs=[pv_block_end, new_image], outputs = merged, name = 'merge_layers_' + str(step) + '_for_G')
		
		
	def mk_output_layers_for_G(self, step):
		
		inp = Input(shape=self.generators[step].output_shape[1:])
		out = Conv2D(3,1,**MyPGGAN.kernel_cond)(inp)
		out = Activation('tanh')(out)
		
		return Model(inp, out, name='output_layers_' + str(step) + '_for_G')


	
			
	def mk_input_layers_for_D(self,step,depth=128):
		
		inp = Input(shape=self.img_shape[step])
		d = Conv2D(depth, 1, **MyPGGAN.kernel_cond)(inp)
		d = LeakyReLU(0.2)(d)
		
		return Model(inputs=inp, outputs=d, name='input_layers_' + str(step) + '_for_D')


	def mk_D_block(self, step, depth=128, scale=2):

		
		inp = Input(shape=self.img_shape[step][:2] + (depth,))
		d = inp
		d = Conv2D(depth, (3,3), padding='same', **MyPGGAN.kernel_cond)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(depth, (4,4), padding='same', **MyPGGAN.kernel_cond)(d)
		d = LeakyReLU(alpha=0.2)(d)
		if step>0:
			d = AveragePooling2D(scale)(d)
		return Model(inputs=inp, outputs=d, name='D_chain_' + str(step))



	#input layer model 뒤에 붙이면 댐
	def mk_merge_layers_for_D(self,step, old_input_layers, depth=128,scale=2):

		raw_inp = Input(shape=self.img_shape[step])
		new_d_block_pass_inp = Input(shape=self.img_shape[step-1][:2] + (depth,))
		raw_inp_pooling = AveragePooling2D(scale)(raw_inp)
		old_inp_block_pass = old_input_layers(raw_inp_pooling)
		merged = WeightedSum()([old_inp_block_pass, new_d_block_pass_inp])
		return Model(inputs=[raw_inp, new_d_block_pass_inp], outputs = merged, name = 'merge_layers_' + str(step) + '_for_D')




	def mk_output_layers_for_D(self,step,depth=128):

		inp = Input(shape=self.discriminators[0].output_shape[1:])
		d = MiniBatchStandardDeviation()(inp)
		d = Flatten()(d)
		d = Dense(1, **MyPGGAN.kernel_cond)(d)
		d = Activation('linear')(d)

		return Model(inputs = inp, outputs=d, name='output_layers_' + str(step) + '_for_D')


	def initialize_DnG_chains(self,depth=128,scale=2):

		for i in range(self.num_steps):
			self.generators[i] = self.mk_G_block(i,depth,scale)
			self.discriminators[i] = self.mk_D_block(i,depth,scale)



	def build_D(self,step,input_layers = None, output_layers = None, merged_old_input_layers=None):

		D = input_layers
		if not D:
			D = self.mk_input_layers_for_D(step)
		inp = D.input
		D = D(inp)
		D = self.discriminators[step](D)

		if merged_old_input_layers != None:
			old_D = inp
			new_D = D
			D = self.mk_merge_layers_for_D(step, merged_old_input_layers)([old_D, new_D])

		for i in range(1,step+1):
			D = self.discriminators[step-i](D)

		if not output_layers:
			D = self.mk_output_layers_for_D(step)(D)
		else:
			D = output_layers(D)
		
		self.D = Model(inputs=inp, outputs=D)


	def build_G(self,step,input_layers = None, output_layers = None,merged_old_output_layers=None):

		G = input_layers
		if G == None:
			G = self.mk_input_layers_for_G(step)
		inp = G.input
		G = G(inp)

		for i in range(step):
			G = self.generators[i](G)

		old_G = G
		G = self.generators[step](old_G)
		if output_layers == None:
			output_layers = self.mk_output_layers_for_G(step)
		G = output_layers(G)

		if merged_old_output_layers != None:
			G = self.mk_merge_layers_for_G(step, merged_old_output_layers)([old_G, G])

		self.G = Model(inputs=inp, outputs=G)



	def set_model_trainable(self, model, trainable):

		model.trainable = trainable
		for layer in model.layers:
			layer.trainable = trainable
			if isinstance(layer, Layer):
				continue
			else:
				self.set_model_trainable(layer,trainable)

	
	def compile_DM(self):

		inp = self.D.layers[0].input
		self.set_model_trainable(self.G, False)
		self.set_model_trainable(self.D, True)
		out = self.D(inp)

		self.DM = Model(inputs=inp, outputs=out)
		self.DM.compile(loss=self.DM_loss, optimizer=self.DM_optimizer)

	def compile_AM(self):

		self.set_model_trainable(self.D,False)
		self.set_model_trainable(self.G,True)
		out = self.D(self.G.output)

		self.AM = Model(inputs=self.G.input, outputs=out)
		self.AM.compile(loss=self.AM_loss, optimizer=self.AM_optimizer)

	def compile(self):

		self.compile_AM()
		self.compile_DM()

	def save_models(self):

		if not os.path.exists(self.model_info_dir):
			os.mkdir(self.model_info_dir)
		path = os.path.join(self.model_info_dir,'models')
		if not os.path.exists(path):
			os.mkdir(path)
		for layer in gan.D.layers:
			if not isinstance(layer, InputLayer):
				save_model(layer,os.path.join(path,layer.name))
		for layer in gan.G.layers:
			if not isinstance(layer, InputLayer):
				save_model(layer, os.path.join(path,layer.name))


	def load_models(self,step):

		input_layers_for_D = None
		input_layers_for_G = None
		output_layers_for_D = None
		output_layers_for_G = None
		path = os.path.join(self.model_info_dir, 'models')

		for i in range(self.num_steps):
			path_g = os.path.join(path, 'G_chain_' + str(i))
			path_d = os.path.join(path, 'D_chain_' + str(i))
			if os.path.exists(path_g + '.json'):
				self.generators[i] = load_model(path_g, custom_layers=self.custom_layers)
			if os.path.exists(path_d + '.json'):
				self.discriminators[i] = load_model(path_d, self.custom_layers)

		path_inD = os.path.join(path, 'input_layers_{}_for_D'.format(step))
		path_inG = os.path.join(path, 'input_layers_{}_for_G'.format(step))
		path_outD = os.path.join(path, 'output_layers_{}_for_D'.format(step))
		path_outG = os.path.join(path, 'output_layers_{}_for_G'.format(step))
		if os.path.exists(path_inD + '.json'):
			input_layers_for_D = load_model(path_inD, self.custom_layers)
		if os.path.exists(path_inG + '.json'):
			input_layers_for_G = load_model(path_inG, self.custom_layers)
		if os.path.exists(path_outD + '.json'):
			output_layers_for_D = load_model(path_outD, self.custom_layers)
		if os.path.exists(path_outG + '.json'):
			output_layers_for_G = load_model(path_outG, self.custom_layers)

		return input_layers_for_D, input_layers_for_G, output_layers_for_D, output_layers_for_G

	def save_weights(self):

		if not os.path.exists(self.model_info_dir):
			os.mkdir(self.model_info_dir)
		path = os.path.join(self.model_info_dir,'weights')
		if not os.path.exists(path):
			os.mkdir(path)
		for layer in gan.D.layers:
			if not isinstance(layer, InputLayer):
				layer.save_weights(os.path.join(path, layer.name + '.h5'))
		for layer in gan.G.layers:
			if not isinstance(layer, InputLayer):
				layer.save_weights(os.path.join(path, layer.name + '.h5'))

	def load_weights_by_name(self,layer):
		if layer==None:
			return
		path = os.path.join(self.model_info_dir, 'weights', layer.name + '.h5')
		if os.path.exists(path):
			layer.load_weights(path)
			print('load weights of {}, complete.'.format(layer.name))


	def load_weights(self, input_layers_for_D, input_layers_for_G, output_layers_for_D, output_layers_for_G):

		self.load_weights_by_name(input_layers_for_D)
		self.load_weights_by_name(input_layers_for_G)
		self.load_weights_by_name(output_layers_for_D)
		self.load_weights_by_name(output_layers_for_G)

		for i in range(self.num_steps):
			self.load_weights_by_name(self.generators[i])
			self.load_weights_by_name(self.discriminators[i])


	def save(self, zipQ=False):
		self.save_models()
		self.save_weights()
		print('save complete.')
		if zipQ:
			zip(self.model_info_dir)
			print('zip complete.')

	def load(self, step, merge=False, unzipQ=False):

		if unzipQ:
			unzip(self.model_info_dir)
			print('unzip complete.')

		input_layers_for_D, input_layers_for_G, output_layers_for_D, output_layers_for_G = self.load_models(step)
		merged_old_output_layers_for_G = None
		merged_old_input_layers_for_D = None

		if merge and step>0:
			merged_old_output_layers_for_G = load_model(
				os.path.join(self.model_info_dir, 'models', 'output_layers_{}_for_G'.format(step-1)), 
				self.custom_layers)
			merged_old_input_layers_for_D = load_model(
				os.path.join(self.model_info_dir, 'models', 'input_layers_{}_for_D'.format(step-1)), 
				self.custom_layers)
			self.load_weights_by_name(merged_old_input_layers_for_D)
			self.load_weights_by_name(merged_old_output_layers_for_G)

		if self.discriminators[step]==None:
			self.discriminators[step] = self.mk_D_block(step)
		if self.generators[step]==None:
			self.generators[step] = self.mk_G_block(step)

		self.load_weights(input_layers_for_D, input_layers_for_G, output_layers_for_D, output_layers_for_G)

		self.build_D(step,input_layers_for_D, output_layers_for_D, merged_old_input_layers_for_D)
		self.build_G(step, input_layers_for_G, output_layers_for_G, merged_old_output_layers_for_G)

	def generate_fake(self,batch_size):
		latent_vectors = self.noise_func(batch_size, self.latent_size)
		fake = self.G.predict(latent_vectors)
		return fake

	def train_DM(self, real_samples, batch_size):
		
		DM_loss = []
		fake = self.generate_fake(batch_size)
		real_y = np.ones([batch_size,1])
		fake_y = np.zeros([batch_size,1])
		DM_loss.append(self.DM.train_on_batch(real_samples, real_y))
		DM_loss.append(self.DM.train_on_batch(fake, fake_y))
		return DM_loss

	def train_AM(self, batch_size):

		AM_loss = 0
		latent_vectors = self.noise_func(batch_size, self.latent_size)
		fake_y = np.ones([batch_size,1])
		AM_loss += self.AM.train_on_batch(latent_vectors,fake_y)
		return AM_loss

	def train_on_epoch(self, step, batch_size, print_term=0):

		path = os.path.join(self.img_src, str(self.heights[step]) + 'x' + str(self.widths[step]))
		AM_loss = []
		DM_loss = []
		num_iter = 0
		iter_per_epoch = len([f for f in os.listdir(path) if ('jpg' in f or 'jpeg' in f)]) // batch_size
		for real_samples in load_image_batch(path, self.img_shape[step][:2], batch_size):

			DM_loss.append(self.train_DM(real_samples,batch_size))
			AM_loss.append(self.train_AM(batch_size))
			num_iter += 1

			if print_term>0:
				if num_iter % print_term==0 and num_iter > 0:
					mean_DM_loss = np.mean(DM_loss[-print_term:], axis=0)
					mean_AM_loss = np.mean(AM_loss[-print_term:])
					print('iteration_per_epoch : {}/{}'.format(num_iter, iter_per_epoch))
					print('mean_of_DM : ', mean_DM_loss)
					print('mean_of_AM : ', mean_AM_loss)
					print()

		return DM_loss, AM_loss

	def generate_samples(self, num_samples):
		samples = self.generate_fake(num_samples)
		samples = generate_sample_image(samples)
		return samples

	
	def train(self, step, epoches, batch_size, print_term=0, unzip_images=False):

		if unzip_images:
			unzip(self.img_src)

		for i in range(epoches):
			DM_loss, AM_loss = self.train_on_epoch(step, batch_size, print_term)
			mean_DM_loss = np.mean(DM_loss, axis=0)
			mean_AM_loss = np.mean(AM_loss)
			print()
			print('epoch : {}/{}'.format(i+1, epoches))
			print('mean_of_DM : ', mean_DM_loss)
			print('mean_of_AM : ', mean_AM_loss)
			print()
		print()
		print('train complete.')
		print('\n\n')

		


		





		

if __name__=='__main__':

	gan = MyPGGAN()
	gan.initialize_DnG_chains()
	gan.build_D(0)
	gan.build_G(0)
	gan.compile()
	gan.save(True)
	gan.train(0, 2, 16, 0,True)
	sample_img = gan.generate_samples(30)
	sample_img = Image.fromarray(sample_img.astype('uint8'))
	sample_img.show()