from keras.layers import Add, Input
from keras.models import Model
import keras.backend as K
import numpy as np


#앞에께 old고 뒤에께 new
class WeightedSum(Add):
	
	def __init__(self, alpha=0.0, alpha_step = 0.01, **kwargs):
		
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = alpha
		self.alpha_step = alpha_step
		#self.alpha = K.variable(alpha, name = 'ws_alpha')
		
		
	def _merge_function(self, inputs):
		
		assert len(inputs) == 2
		
		output = (1.0 - self.alpha)*inputs[0] + self.alpha*inputs[1]
		
		self.alpha = np.clip(self.alpha + self.alpha_step, 0.0, 1.0)
		
		return output
		
	def get_config(self):
		
		base_config = list(super(WeightedSum, self).get_config().items())
		config = list({'alpha' : self.alpha, 'alpha_step' : self.alpha_step}.items())
		return dict(config + base_config)
		
		
		


if __name__=='__main__':
	
	inp1 = Input(shape=[1])
	inp2 = Input(shape=[1])
	out = WeightedSum(0.6)
	m = Model(inputs=[inp1, inp2], outputs=WeightedSum(0.6)([inp1, inp2]))
	m.summary()
	print(m.predict([[[1],[2]], [[3],[4]]]))
	print(m.layers[2].get_config())
	print(out.get_config())
	