import os.path
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import matplotlib.pyplot as plt
import mydcgan
from mydcgan import MyDCGAN
from kds import KianaDataSet


#random kiana generator
class RKG:
	
	def __init__(self,input_shape,model=1,noise_size=100,kdsfromzip=False):
		
		self.input_shape=input_shape
		self.D = Sequential()
		self.G = Sequential()
		self.noise_size = noise_size
		
		if model==1:
			self.create_d1()
			self.create_g1()
		if model==2:
			self.create_d2()
			self.create_g2()
		self.kdata = KianaDataSet(load_from_zip=kdsfromzip)
		self.gan = MyDCGAN(self.noise_size,D=self.D,G=self.G,batchsize=2,print_term=1,Dfname = 'k_d_w.h5', Gfname='k_g_w.h5')
		self.gan.img_data = self.kdata.normalized
	
	def create_d1(self):
		depth = 64
		alpha = 0.2
		channel = 3
		self.D.add(Conv2D(depth*channel,4,strides=2,input_shape=(256,256,3),padding='same'))
		self.D.add(LeakyReLU(alpha=alpha))
		for i in range(1,5):
			MyDCGAN.add_cbl(self.D, 64*int(np.power(2,i))*channel,4, 2, alpha=0.2, bn=True)
		
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))
		self.D.summary()
		
	def create_g1(self):
		
		depth=64*int(np.power(2,4))
		channel = 3
		dim=8
		momentum = 0.5
		
		self.G.add(Dense(dim*dim*channel*depth, input_dim=self.noise_size))
		self.G.add(BatchNormalization(momentum=momentum))
		self.G.add(Reshape((dim,dim,channel)))
		
		for i in range(1,4):
			MyDCGAN.add_dbr(self.G,int(depth/np.power(2,i)),4,2,momentum)
		
		self.G.add(Conv2DTranspose(int(depth/np.power(2,4)),4,2,padding='same'))
		self.G.add(Activation('tanh'))
		self.G.summary()
	
	def create_d2(self):
		
		depth=128
		alpha = 0.2
		channel = 3
		dropout=0.4
		
		self.D.add(Conv2D(depth,3,input_shape=self.input_shape,padding='same'))
		self.D.add(LeakyReLU(alpha=alpha))
		MyDCGAN.add_cbl(self.D,depth, 4, 2, 0.2)
		MyDCGAN.add_cbl(self.D,depth, 4, 2, 0.2)
		MyDCGAN.add_cbl(self.D,depth, 4, 2, 0.2)
		self.D.add(Flatten())
		self.D.add(Dropout(dropout))
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))
		self.D.summary()
			
				
	def create_g2(self):
		
		depth=128
		dim = 64
		alpha = 0.2
		channel=3
		
		
		self.G.add(Dense(depth*dim*dim,input_dim=self.noise_size))
		self.G.add(Reshape((dim,dim,depth)))
		MyDCGAN.add_cbl(self.G, depth*2, 5, 1, 0.2)
		MyDCGAN.add_dbr(self.G,depth*2,4,2)
		MyDCGAN.add_cbl(self.G,depth*4,4,1,0.2)
		MyDCGAN.add_dbr(self.G,depth*4,4,2)
		MyDCGAN.add_cbl(self.G,depth*4,5,1,0.2)
		self.G.add(Conv2D(channel, 7, strides=1,padding='same'))
		self.G.summary()
		
		

rkg1 = RKG((256,256,3),2,64, kdsfromzip=True)
rkg1.gan.train(print_sample=10)
