import numpy as np
from functools import partial




from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, add, UpSampling2D, AveragePooling2D, Activation
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
import mywgan
from mywgan import MyWGAN
from mywgan import RandomWeightedAverage
from adain import AdaIN
from noise import Noise
from keras.models import model_from_json
from os import listdir
import zipfile
from PIL import Image

K.set_image_data_format('channels_last')




def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)


def set_adain(inp, style,noise, fil,alpha=0.01):

	b = Dense(fil)(style)
	b = Reshape([1,1,fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1,1,fil])(g)

	n = Conv2D(filters = fil, kernel_size=1, padding='same', kernel_initializer='he_normal')

	out = AdaIN()([inp, b, g])
	out = add([out, n])
	out = LeakyReLU(alpha=alpha)(out)



def g_block1(style, noise, inp, fil, u = 2, kernel_size=3):
	
	b = Dense(fil)(style)
	b = Reshape([1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, fil])(g)
	
	n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
	
	if u>1:
		out = UpSampling2D(size=u, interpolation = 'bilinear')(inp)
		out = Conv2D(filters = fil, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal')(out)
	else:
		out = Activation('linear')(inp)
		
	out = AdaIN()([out, b, g])
	out = add([out, n])
	out = LeakyReLU(0.01)(out)
	
	b = Dense(fil)(style)
	b = Reshape([1, 1, fil])(b)
	g = Dense(fil)(style)
	g = Reshape([1, 1, fil])(g)
	
	n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
	
	out = Conv2D(filters = fil, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal')(out)
	out = AdaIN()([out, b, g])
	out = add([out, n])
	out = LeakyReLU(0.01)(out)
	
	return out
	
	
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
	
	def __init__(self, **kwargs):
		
		super(MyStyleGAN, self).__init__(**kwargs)
		
		
		self.MN = Model() #mapping network
		self.AM = Model()
		self.DM = Model()
		
		self.positive_y = np.ones((self.batch_size, 1), dtype=np.float32)
		self.negative_y = -self.positive_y
		self.dummy_y = np.zeros((self.batch_size, 1), dtype=np.float32)

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
		
		
	def compile_generator(self):
		
		self.set_D_trainable(False)
		self.set_G_trainable()
		
		
		inp_s = Noise(self.noise_generating_rule,[self.noise_size])
		sty = self.MN(inp_s)
		inp_n = Noise(self.noise_generating_rule, [self.noise_size])
		noi = Activation('linear')(inp_n)
		inp = Input(shape=[1])
		
		ginp = [inp]
		gene = self.G([sty, noi, inp])
		gout = [self.D(gene)]
		
		self.AM = Model(inputs=ginp, outputs=gout)
		self.AM.compile(optimizer=self.optimizer, loss=[MyWGAN.wasserstein_loss])
		
	def construct_mn(self,n_layers,n_neurons, alpha=0.1):
		mn_inp = Input(shape=[self.noise_size])
		mn = Dense(n_neurons, kernel_initializer='he_normal')(mn_inp)
		for _ in range(1,n_layers):
			mn = Dense(n_neurons, kernel_initializer = 'he_normal')(mn)
			mn = LeakyReLU(alpha)(mn)
		mn = Model(inputs=[mn_inp],outputs=[mn])
		return mn
		
	
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
		
		generator_input_for_discriminator = Input(shape=[1])
		
		generated_samples_for_discriminator = self.G(generator_input_for_discriminator)
		discriminator_output_from_generator = self.D(generated_samples_for_discriminator)
		discriminator_output_from_real_samples =self.D(real_samples)
		
		averaged_samples = RandomWeightedAverage(self.batch_size)([real_samples, generated_samples_for_discriminator])
		averaged_samples_out = self.D(averaged_samples)
		partial_gp_loss = partial(MyWGAN.gradient_penalty_loss,averaged_samples=averaged_samples, gradient_penalty_weight=self.gradient_penalty_weight)
		partial_gp_loss.__name__ = 'gradient_penalty'
		
		self.DM = Model(inputs=[real_samples], outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])
		
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
				image_batch = discriminator_minibatches[j*self.batch_size:(j+1)*self.batch_size]
				discriminator_loss.append(self.DM.train_on_batch([image_batch, positive_y], [positive_y, negative_y, dummy_y]))
			generator_loss.append(self.AM.train_on_batch(positive_y, positive_y))
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
		
	def load_models(self):
		mn_json_file = open(self.get_mn_model_file_name(), "r")
		mn_model = mn_json_file.read()
		mn_json_file.close()
		
		self.MN = model_from_json(mn_model)
		
		super(MyStyleGAN, self).load_models()
		
	def save_weights(self):
		
		self.MN.save_weights(self.get_mn_weight_file_name())
		
		super(MyStyleGAN, self).save_weights()
		
	def load_weights(self):
		
		self.MN.load_weights(self.get_mn_weight_file_name())
		
		super(MyStyleGAN, self).load_weights()
		
	def zip(self, zipname):
		
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
		
		

D = Input(shape=(256,256,3))
gan1 = MyStyleGAN(optimizer=Adam(lr=0.001, beta_1 = 0, beta_2=0.99), noise_size=256, noise_generating_rule= (lambda size : K.random.uniform(-1.0,1.0,size=size)))

gan1.MN = gan1.construct_mn(8,256)
G_inputs=[Input(shape=[256]), Input(shape=[256,256,1]),Input(shape=[1])] #sty, noi, inp
G_sty = G_inputs[0]
G_noi = G_inputs[1]
G_inp = G_inputs[2]
inp2 = Dense(4*4*256)(G_inp)
inp3 = Reshape((4,4,256))(inp2)
noi = G_noi
noi128 = AveragePooling2D()(noi)
noi64 = AveragePooling2D()(noi128)
noi32 = AveragePooling2D()(noi64)
noi16 = AveragePooling2D()(noi32)
G = g_block1(G_sty, noi16, inp3, 256,4,kernel_size=5)#up 16
G = g_block1(G_sty, noi64, G, 256, 4,kernel_size=5)#up 64
G = g_block1(G_sty, noi128, G, 128)#up 128
G = g_block1(G_sty, noi, G,64)#up 256
G = Conv2D(3,kernel_size=3, padding='same',kernel_initializer='he_normal')(G)
G = Activation('tanh')(G)
G = Model(inputs=G_inputs, outputs=[G])


D_input = Input(shape=(256,256,3))
D = Conv2D(32, kernel_size=3 , padding='same',kernel_initializer = 'he_normal')(D_input)
D = d_block1(64,D)
D = d_block1(128, D)
D = d_block1(256, D)
D = d_block1(3, D)
D = Model(inputs=[D_input], outputs=[D])

gan1.G = G
gan1.D = D

gan1.compile()