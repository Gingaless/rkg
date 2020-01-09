import numpy as np
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
    input_args,
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
    vaild = 1.0, fake = 0.0,
    vaild_for_fake = 1.0,
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
        self.valid = vaild*np.ones((self.batch_size,1), dtype=np.float)
        self.fake = fake*np.ones((self.batch_size,1), dtype=np.float)
        self.valid_for_fake = vaild_for_fake*np.ones((self.batch_size,1), dtype=np.float)
        self.custom_layers = self.custom_layers
        self.generators =  [Model() for i in range(self.n_growth)]
        self.discriminators = [Model() for i in range(self.n_growth)]
        self.AM = Model()
        self.DM = Model()




