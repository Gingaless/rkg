
from keras.layers import Input, Dense, Layer
from keras.models import Model
import numpy as np
import keras.backend as K
from keras.optimizers import Adam

class Ls(Layer):

    def __init__(self, **kwargs):

        super(Ls, self).__init__(**kwargs)
        self.w = [self.add_weight(shape=(1,1), initializer='he_normal') for _ in range(2)]


    def call(self, inputs):

        inp = inputs[0]
        return K.dot(inp, self.w[0])


inp1 = Input([1])
inp2 = Input([1])
inps = [inp1,inp2]
out = Ls()(inps)
m = Model(inps, out)
r = m.predict([[1],[1]])
print(r)
m.compile(optimizer=Adam(0.01), loss='mse')
m.train_on_batch([np.ones([10,1]), np.ones([1,1])], np.ones([10,1]))
