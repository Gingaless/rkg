import os.path
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image


K.set_image_data_format('channels_last')

class MyDCGAN:
	
	def __init__(self,noise_size = 100, G = None, D =None, D_optimizer=Adam(lr=0.001), G_optimizer = Adam(lr=0.0005),loss='binary_crossentropy',epoch=5,batchsize = 100,print_term=20,img_size=256,channel=3, G_lr_num = 2,Dfname='gan_d_weights.h5',Gfname='gan_g_weights.h5',save_per_epoch=True):
		
		self.img_data = None
		self.input_shape = (img_size, img_size, channel)
		self.img_rows = img_size
		self.img_cols = img_size
		self.channel = channel
		self.noise_size = noise_size
		self.epoch = epoch
		self.batchsize = batchsize
		self.loss = loss
		self.print_term = print_term
		self.G_lr_nim = G_lr_num
		self.Dfname = Dfname
		self.Gfname = Gfname
		self.save_per_epoch = save_per_epoch
		
		if (D==None):
			self.D=MyDCGAN.create_default_d(self.input_shape)
		
		if (G==None):
			self.G=MyDCGAN.create_default_g(channel,noise_size)
			
		self.D=D
		self.G=G
		# Build model to train D.
		self.D. compile(loss=loss, optimizer=D_optimizer)
		# Build model to train G.
		self.D.trainable = False
		self.AM = Sequential()
		self.AM.add(self.G)
		self.AM.add(self.D)
		self.AM.compile(loss=loss, optimizer=G_optimizer)

		
	def create_default_g(channel,noise_size):
		G = Sequential()
		dropout = 0.4
		depth = 64+64+64+64
		dim = 8
		G.add(Dense(dim*dim*depth, input_dim=noise_size))
		G.add(BatchNormalization(momentum=0.9))
		G.add(Activation('relu'))
		G.add(Reshape((dim, dim, depth)))
		G.add(Dropout(dropout))
		G.add(UpSampling2D())
		G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
		G.add(BatchNormalization(momentum=0.9))
		G.add(Activation('relu'))
		G.add(UpSampling2D())
		G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
		G.add(BatchNormalization(momentum=0.9))
		G.add(Activation('relu'))
		G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
		G.add(BatchNormalization(momentum=0.9))
		G.add(Activation('relu'))
		G.add(Conv2DTranspose(channel, 5, padding='same'))
		G.add(Activation('sigmoid'))
		G.summary()
		return G
		
	def create_default_d(input_shape):
		D = Sequential()
		depth = 64
		dropout = 0.4
		D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		add(Conv2D(depth*4, 5, strides=2, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
		D.add(LeakyReLU(alpha=0.2))
		D.add(Dropout(dropout))
		D.add(Flatten())
		D.add(Dense(1))
		D.add(Activation('sigmoid'))
		D.summary()
		return D
		
	# Convolution, BatchNormalization, LeakyReLU
	def add_cbl(D, depth, kernel_size, strides, alpha, bn=True):
		
		D.add(Conv2D(depth, kernel_size=kernel_size, strides=strides,padding='same'))
		if (bn):
			D.add(BatchNormalization())
		D.add(LeakyReLU(alpha=alpha))
		
		
	#deconv, batch normalization, relu
	def add_dbr(G,  depth, kernel_size, strides, bn=True, relu=True,bn_momentum=0.9):
		
		G.add(Conv2DTranspose(depth,kernel_size,strides=strides,padding='same'))
		if bn:
			G.add(BatchNormalization(momentum=bn_momentum))
		if relu:
			G.add(Activation('relu'))
		
		
		
	def save(self):
		self.D.save_weights(self.Dfname)
		self.G.save_weights(self.Gfname)
		print('weights saved')
		
	def load(self):
		if os.path.isfile(self.Dfname):
			self.D.load_weights(self.Dfname)
			print('load the weights of the discriminator')
		if os.path.isfile(self.Dfname):
			self.G.load_weights(self.Gfname)
			print('load the weights of the generator')
			
		if os.path.isfile(Gfname):
			self.G.load_weights(Gfname)
			print('load the weights of the generator')
		
	def data_shuffle(self, data):
		
		ndata = len(data)
		indices = np.arange(ndata)
		np.random.shuffle(indices)
		new_data = np.zeros((len(data ),) + self.input_shape)
		
		for i, j in enumerate(range(len(data))):
			new_data[i] = data[j]
			
		return new_data
		
		
	def train_batch(self, shuffle_D = True):
		#print('load train images')
		images_train = self.img_data[np.random.randint(0,np.shape(self.img_data)[0], size=self.batchsize),:,:,:]
		#print('load image ok')
		noise = np.random.uniform(-1.0,1.0,size=[self.batchsize, self.noise_size])
		images_fake = self.G.predict(noise)
		
		#Train D
		x = np.concatenate([images_train, images_fake])
		x = self.data_shuffle(x)
		y = np.ones([2*self.batchsize,1])
		y[self.batchsize:,:] = 0
		self.D.trainable = True
		d_loss = self.D.train_on_batch(x,y)
		
		#Train G
		self.D.trainable = False
		y = np.ones([self.batchsize,1])
		a_loss = 0
		for i in range(self.G_lr_nim):
			noise = np.random.uniform(-1.0,1.0,size=[self.batchsize,self.noise_size])
			a_loss += self.AM.train_on_batch(noise,y)
			#print('batch train ok')
		
		return d_loss, a_loss
		
	def create_samples(self,num_sample):
		noise = np.random.uniform(-1.0,1.0,size=[num_sample, self.noise_size])
		samples=self.G.predict(noise)
		samples=((samples+1.0)*127.5).astype('uint8')
		for i in range(num_sample):
			Image.fromarray(samples[i]).save('{}.jpg'.format(str(i)))
		
		
	def train(self,print_sample=0):
		
		print('train start.')
		
		train_per_epoch = np.shape(self.img_data)[0] // self.batchsize
		
		for epoch in range(0,self.epoch):
			total_d_loss = 0.0
			total_a_loss = 0.0
			
			for batch in range(0,train_per_epoch):
				d_loss, a_loss = self.train_batch()
				total_d_loss += d_loss
				total_a_loss += a_loss
				
				if (batch%self.print_term == 0) or (batch==self.batchsize-1):
					
					print("Epoch : {}/{}, batch : {}/{}, D_loss : {}, A_loss : {}".format(epoch+1, self.epoch, batch+1, train_per_epoch, d_loss, a_loss))
					
					if print_sample>0:
						self.create_samples(print_sample)
			
			print("Epoch : {}/{}, total_d_loss : {}, total_a_loss : {}".format(epoch+1, self.epoch, total_d_loss,total_a_loss))
			if self.save_per_epoch: self.save()
			
		print('training  complete.')
			