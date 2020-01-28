from keras.layers import Add, Input, Dense
from keras.models import Model
import keras.backend as K
import numpy as np
from keras.optimizers import Adam


#앞에께 old고 뒤에께 new
class WeightedSum(Add):
	
	def __init__(self, alpha=0.0, alpha_step = 0.001, **kwargs):
		
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = alpha
		self.alpha_step = alpha_step
		self.alpha = K.variable(alpha)
		self.alpha_step = alpha_step
		self.trainable = False
		
		
	def _merge_function(self, inputs):
		
		assert len(inputs) == 2
		
		output = (1.0 - self.alpha)*inputs[0] + self.alpha*inputs[1]
		
		return output
		
	def get_config(self):
		
		base_config = list(super(WeightedSum, self).get_config().items())
		config = list({'alpha' : float(K.eval(self.alpha)), 'alpha_step' : self.alpha_step, 'class_name' : 'WeightedSum'}.items())
		return dict(config + base_config)
		

def update_fadein(model):
	
	for layer in model.layers:
		if isinstance(layer, WeightedSum):
			K.set_value(layer.alpha, np.clip(K.eval(layer.alpha) + layer.alpha_step,0.0,1.0))
		elif isinstance(layer, Model):
			update_fadein(layer)

def set_fadein(model, alpha, alpha_step, name=None):

	for layer in model.layers:
		if isinstance(layer, WeightedSum):
			if name==None:
				K.set_value(layer.alpha, alpha)
				layer.alpha_step = alpha_step
			else:
				if layer.name==name:
					layer.alpha = alpha
					layer.alpha_step = alpha_step
		elif isinstance(layer, Model):
			set_fadein(layer,alpha,alpha_step,name)
		
		
		


if __name__=='__main__':
	
	inp1 = Input(shape=[1])
	inp2 = Input(shape=[1])
	out = WeightedSum(0.6)([inp1,inp2])
	d = Dense(1)(out)
	m = Model(inputs=[inp1, inp2], outputs=d)
	m.summary()
	print(m.layers[2].get_config())
	m.compile(optimizer=Adam(lr=0.01), loss = 'mse')
	print(m.predict([[[1],[2]], [[3],[4]]]))
	print(m.layers[2].get_config())
	x = np.ones([4,1])
	y = np.ones([4,1])
	m.train_on_batch([x,x], y)
	update_fadein(m)
	print(K.eval(m.layers[2].alpha))
	